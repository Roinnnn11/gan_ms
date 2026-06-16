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
            (config.train.fixed_noise_size, params.latent_dim, 1, 1),
            ms.float32,
        )

    def _sample_noise(self, batch_size: int):
        return ops.randn((batch_size, self.config.model_params.latent_dim, 1, 1), ms.float32)

    def _gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.shape[0]
        epsilon = ops.rand((batch_size, 1, 1, 1), ms.float32)
        interpolated = epsilon * real_images + (1.0 - epsilon) * fake_images

        def critic_sum(images):
            return self.critic(images).sum()

        gradients = ms.grad(critic_sum, grad_position=0)(interpolated)
        gradients = ops.reshape(gradients, (batch_size, -1))
        slopes = ops.sqrt(ops.sum(ops.square(gradients), axis=1) + 1e-12)
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

    def train(self, dataset) -> None:
        critic_grad_fn = ms.value_and_grad(
            self._critic_loss,
            None,
            self.optimizer_c.parameters,
        )
        generator_grad_fn = ms.value_and_grad(
            self._generator_loss,
            None,
            self.optimizer_g.parameters,
        )
        log_path = self.paths["logs"] / "losses.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "critic_loss", "g_loss"])
            for epoch in range(1, self.config.train.epochs + 1):
                for step, batch in enumerate(dataset.create_tuple_iterator(), start=1):
                    real_images = batch[0]
                    critic_loss, critic_grads = critic_grad_fn(real_images)
                    self.optimizer_c(critic_grads)

                    g_loss_value = None
                    if step % self.config.wgan_gp.critic_steps == 0:
                        g_loss, g_grads = generator_grad_fn(real_images)
                        self.optimizer_g(g_grads)
                        g_loss_value = float(g_loss.asnumpy())

                    writer.writerow(
                        [
                            epoch,
                            step,
                            float(critic_loss.asnumpy()),
                            "" if g_loss_value is None else g_loss_value,
                        ]
                    )

                self._after_epoch(epoch)

    def _after_epoch(self, epoch: int) -> None:
        if epoch % self.config.train.sample_interval == 0:
            fake_images = self.generator(self.fixed_noise).asnumpy()
            save_image_grid(fake_images, self.paths["samples"] / f"epoch_{epoch:03d}.png")
        if epoch % self.config.train.checkpoint_interval == 0:
            save_named_checkpoint(self.generator, self.paths["checkpoints"], "generator", epoch)
            save_named_checkpoint(self.critic, self.paths["checkpoints"], "critic", epoch)
