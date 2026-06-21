from __future__ import annotations

import csv

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.config import ExperimentConfig
from src.models.dcgan import Discriminator, Generator
from src.utils.checkpoint import ensure_run_dirs, save_named_checkpoint
from src.utils.images import save_image_grid


class WGANGPTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        params = config.model_params
        self.generator = Generator(params.latent_dim, params.image_channels, params.feature_maps_g)
        self.critic = Discriminator(
            params.image_channels,
            params.feature_maps_d,
            use_sigmoid=False,
        )
        self._prefix_parameter_names(self.generator, "generator")
        self._prefix_parameter_names(self.critic, "critic")
        self.optimizer_g = nn.Adam(
            self.generator.trainable_params(),
            learning_rate=config.train.learning_rate_g,
            beta1=config.train.beta1,
            beta2=config.train.beta2,
        )
        self.optimizer_c = nn.Adam(
            self.critic.trainable_params(),
            learning_rate=config.train.learning_rate_d,
            beta1=config.train.beta1,
            beta2=config.train.beta2,
        )
        self.paths = ensure_run_dirs(config.train.output_dir)
        self.fixed_noise = ops.randn(
            config.train.fixed_noise_size, params.latent_dim, 1, 1,
        ).astype(ms.float32)

    @staticmethod
    def _prefix_parameter_names(network: nn.Cell, prefix: str) -> None:
        for param in network.get_parameters():
            if not param.name.startswith(f"{prefix}."):
                param.name = f"{prefix}.{param.name}"

    def _sample_noise(self, batch_size: int):
        return ops.randn(batch_size, self.config.model_params.latent_dim, 1, 1).astype(ms.float32)

    def _gradient_norms(self, real_images, fake_images):
        batch_size = real_images.shape[0]
        epsilon = ops.rand(batch_size, 1, 1, 1).astype(ms.float32)
        interpolated = epsilon * real_images + (1.0 - epsilon) * fake_images

        def critic_sum(images):
            return self.critic(images).sum()

        gradients = ops.grad(critic_sum, grad_position=0)(interpolated)
        gradients = ops.reshape(gradients, (batch_size, -1))
        return ops.sqrt(ops.sum(ops.square(gradients), 1) + 1e-12)

    def _gradient_penalty_stats(self, real_images, fake_images):
        slopes = self._gradient_norms(real_images, fake_images)
        penalty = ops.mean(ops.square(slopes - 1.0))
        grad_norm_mean = ops.mean(slopes)
        grad_norm_std = ops.sqrt(ops.mean(ops.square(slopes - grad_norm_mean)))
        return penalty, grad_norm_mean, grad_norm_std

    def _gradient_penalty(self, real_images, fake_images):
        slopes = self._gradient_norms(real_images, fake_images)
        return ops.mean(ops.square(slopes - 1.0))

    def _critic_loss(self, real_images):
        batch_size = real_images.shape[0]
        fake_images = ops.stop_gradient(self.generator(self._sample_noise(batch_size)))
        real_scores = self.critic(real_images)
        fake_scores = self.critic(fake_images)
        wasserstein = ops.mean(fake_scores) - ops.mean(real_scores)
        penalty = self._gradient_penalty(real_images, fake_images)
        return wasserstein + self.config.wgan_gp.lambda_gp * penalty

    def _generator_loss(self, real_images):
        batch_size = real_images.shape[0]
        fake_images = self.generator(self._sample_noise(batch_size))
        fake_scores = self.critic(fake_images)
        return -ops.mean(fake_scores)

    def _pynative_diagnostics(self, real_images):
        """Run one forward pass in PyNative mode to get per-step diagnostics."""
        prev_mode = ms.get_context("mode")
        generator_was_training = self.generator.training
        critic_was_training = self.critic.training
        try:
            ms.set_context(mode=ms.PYNATIVE_MODE)
            self.generator.set_train(False)
            self.critic.set_train(False)
            batch_size = real_images.shape[0]
            fake_images = self.generator(self._sample_noise(batch_size))
            real_score = float(ops.mean(self.critic(real_images)).asnumpy())
            fake_score = float(ops.mean(self.critic(fake_images)).asnumpy())
            wasserstein_gap = real_score - fake_score
            gp, grad_norm_mean, grad_norm_std = self._gradient_penalty_stats(real_images, fake_images)
            gp = float(gp.asnumpy())
            grad_norm_mean = float(grad_norm_mean.asnumpy())
            grad_norm_std = float(grad_norm_std.asnumpy())
            return real_score, fake_score, wasserstein_gap, gp, grad_norm_mean, grad_norm_std
        finally:
            self.generator.set_train(generator_was_training)
            self.critic.set_train(critic_was_training)
            ms.set_context(mode=prev_mode)

    def _load_checkpoint(self, epoch: int) -> None:
        ckpt_dir = self.paths["checkpoints"]
        g_path = ckpt_dir / f"generator_epoch_{epoch:03d}.ckpt"
        c_path = ckpt_dir / f"critic_epoch_{epoch:03d}.ckpt"
        if not g_path.exists() or not c_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for epoch {epoch} not found: {g_path}, {c_path}"
            )
        ms.load_checkpoint(str(g_path), self.generator)
        ms.load_checkpoint(str(c_path), self.critic)

    def train(self, dataset, start_epoch: int = 1) -> None:
        if start_epoch > 1:
            self._load_checkpoint(start_epoch - 1)

        # value_and_grad does not switch Cell mode the way TrainOneStepCell does.
        self.generator.set_train(True)
        self.critic.set_train(True)

        critic_grad_fn = ops.value_and_grad(
            self._critic_loss, None, self.optimizer_c.parameters,
        )
        generator_grad_fn = ops.value_and_grad(
            self._generator_loss, None, self.optimizer_g.parameters,
        )

        log_path = self.paths["logs"] / "losses.csv"
        mode = "w" if start_epoch == 1 else "a"

        with log_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if start_epoch == 1:
                writer.writerow([
                    "epoch", "step",
                    "critic_loss",
                    "real_score_mean", "fake_score_mean", "wasserstein_gap",
                    "gradient_penalty_raw", "gradient_norm_mean", "gradient_norm_std",
                    "g_loss",
                ])
            for epoch in range(start_epoch, self.config.train.epochs + 1):
                print(f"[wgan_gp] epoch {epoch}/{self.config.train.epochs} start", flush=True)
                for step, batch in enumerate(dataset.create_tuple_iterator(), start=1):
                    real_images = batch[0]

                    critic_loss, critic_grads = critic_grad_fn(real_images)
                    self.optimizer_c(critic_grads)

                    # diagnostics every critic_steps (same cadence as generator update)
                    real_score = fake_score = wasserstein_gap = gp = grad_norm_mean = grad_norm_std = ""
                    if step % self.config.wgan_gp.critic_steps == 0:
                        (
                            real_score,
                            fake_score,
                            wasserstein_gap,
                            gp,
                            grad_norm_mean,
                            grad_norm_std,
                        ) = self._pynative_diagnostics(real_images)
                        g_loss, g_grads = generator_grad_fn(real_images)
                        self.optimizer_g(g_grads)
                        g_loss_value = float(g_loss.asnumpy())
                    else:
                        g_loss_value = ""

                    writer.writerow([
                        epoch, step,
                        float(critic_loss.asnumpy()),
                        real_score, fake_score, wasserstein_gap,
                        gp, grad_norm_mean, grad_norm_std,
                        g_loss_value,
                    ])

                self._after_epoch(epoch)
                print(f"[wgan_gp] epoch {epoch}/{self.config.train.epochs} done", flush=True)

    def _after_epoch(self, epoch: int) -> None:
        if epoch % self.config.train.sample_interval == 0:
            generator_was_training = self.generator.training
            try:
                self.generator.set_train(False)
                fake_images = self.generator(self.fixed_noise).asnumpy()
            finally:
                self.generator.set_train(generator_was_training)
            save_image_grid(fake_images, self.paths["samples"] / f"epoch_{epoch:03d}.png")
        if epoch % self.config.train.checkpoint_interval == 0:
            save_named_checkpoint(self.generator, self.paths["checkpoints"], "generator", epoch)
            save_named_checkpoint(self.critic, self.paths["checkpoints"], "critic", epoch)
