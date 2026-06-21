from __future__ import annotations

import csv

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.communication import get_rank, get_group_size

from src.config import ExperimentConfig
from src.models.dcgan import Discriminator, Generator
from src.utils.checkpoint import ensure_run_dirs, save_named_checkpoint
from src.utils.images import save_image_grid


class _CriticLoss(nn.Cell):
    def __init__(self, generator, critic, latent_dim, lambda_gp):
        super().__init__(auto_prefix=False)
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp

    def construct(self, real_images):
        batch_size = real_images.shape[0]
        noise = ops.randn(batch_size, self.latent_dim, 1, 1).astype(ms.float32)
        fake_images = ops.stop_gradient(self.generator(noise))
        real_scores = self.critic(real_images)
        fake_scores = self.critic(fake_images)
        wasserstein = ops.mean(fake_scores) - ops.mean(real_scores)

        epsilon = ops.rand(batch_size, 1, 1, 1).astype(ms.float32)
        interpolated = epsilon * real_images + (1.0 - epsilon) * fake_images

        def critic_sum(images):
            return self.critic(images).sum()

        gradients = ops.grad(critic_sum, grad_position=0)(interpolated)
        gradients = ops.reshape(gradients, (batch_size, -1))
        slopes = ops.sqrt(ops.sum(ops.square(gradients), axis=1) + 1e-12)
        penalty = ops.mean(ops.square(slopes - 1.0))
        return wasserstein + self.lambda_gp * penalty


class _GeneratorLoss(nn.Cell):
    def __init__(self, generator, critic, latent_dim):
        super().__init__(auto_prefix=False)
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim

    def construct(self, real_images):
        batch_size = real_images.shape[0]
        noise = ops.randn(batch_size, self.latent_dim, 1, 1).astype(ms.float32)
        fake_images = self.generator(noise)
        fake_scores = self.critic(fake_images)
        return -ops.mean(fake_scores)


class _DistTrainStep(nn.Cell):
    """Wraps a loss cell with gradient reduction across devices."""
    def __init__(self, net, optimizer):
        super().__init__(auto_prefix=False)
        self.net = net
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(net, None, optimizer.parameters)
        self.reducer = nn.DistributedGradReducer(optimizer.parameters)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        grads = self.reducer(grads)
        self.optimizer(grads)
        return loss


class WGANGPDistTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rank = get_rank()
        self.world_size = get_group_size()
        params = config.model_params

        self.generator = Generator(params.latent_dim, params.image_channels, params.feature_maps_g)
        self.critic = Discriminator(params.image_channels, params.feature_maps_d, use_sigmoid=False)

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

        c_net = _CriticLoss(self.generator, self.critic, params.latent_dim, config.wgan_gp.lambda_gp)
        g_net = _GeneratorLoss(self.generator, self.critic, params.latent_dim)
        self._critic_train = _DistTrainStep(c_net, self.optimizer_c)
        self._generator_train = _DistTrainStep(g_net, self.optimizer_g)
        self._critic_train.set_train()
        self._generator_train.set_train()

    def train(self, dataset) -> None:
        log_path = self.paths["logs"] / f"losses_rank{self.rank}.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "critic_loss", "g_loss"])
            for epoch in range(1, self.config.train.epochs + 1):
                for step, batch in enumerate(dataset.create_tuple_iterator(), start=1):
                    real_images = batch[0]
                    critic_loss = self._critic_train(real_images)

                    g_loss_value = None
                    if step % self.config.wgan_gp.critic_steps == 0:
                        g_loss = self._generator_train(real_images)
                        g_loss_value = float(g_loss.asnumpy())

                    writer.writerow([
                        epoch, step,
                        float(critic_loss.asnumpy()),
                        "" if g_loss_value is None else g_loss_value,
                    ])

                if self.rank == 0:
                    self._after_epoch(epoch)

    def _after_epoch(self, epoch: int) -> None:
        if epoch % self.config.train.sample_interval == 0:
            fake_images = self.generator(self.fixed_noise).asnumpy()
            save_image_grid(fake_images, self.paths["samples"] / f"epoch_{epoch:03d}.png")
        if epoch % self.config.train.checkpoint_interval == 0:
            save_named_checkpoint(self.generator, self.paths["checkpoints"], "generator", epoch)
            save_named_checkpoint(self.critic, self.paths["checkpoints"], "critic", epoch)
