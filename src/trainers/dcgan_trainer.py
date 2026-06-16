from __future__ import annotations

import csv

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.config import ExperimentConfig
from src.models.dcgan import Discriminator, Generator
from src.utils.checkpoint import ensure_run_dirs, save_named_checkpoint
from src.utils.images import save_image_grid


class DCGANTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        params = config.model_params
        self.generator = Generator(params.latent_dim, params.image_channels, params.feature_maps_g)
        self.discriminator = Discriminator(
            params.image_channels,
            params.feature_maps_d,
            use_sigmoid=True,
        )
        self.loss_fn = nn.BCELoss(reduction="mean")
        self.optimizer_g = nn.Adam(
            self.generator.trainable_params(),
            learning_rate=config.train.learning_rate_g,
            beta1=config.train.beta1,
            beta2=config.train.beta2,
        )
        self.optimizer_d = nn.Adam(
            self.discriminator.trainable_params(),
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

    def _discriminator_loss(self, real_images):
        batch_size = real_images.shape[0]
        fake_images = self.generator(self._sample_noise(batch_size))
        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(ops.stop_gradient(fake_images))
        real_loss = self.loss_fn(real_scores, ops.ones_like(real_scores))
        fake_loss = self.loss_fn(fake_scores, ops.zeros_like(fake_scores))
        return real_loss + fake_loss

    def _generator_loss(self, real_images):
        batch_size = real_images.shape[0]
        fake_images = self.generator(self._sample_noise(batch_size))
        fake_scores = self.discriminator(fake_images)
        return self.loss_fn(fake_scores, ops.ones_like(fake_scores))

    def train(self, dataset) -> None:
        d_grad_fn = ops.value_and_grad(
            self._discriminator_loss,
            None,
            self.optimizer_d.parameters,
        )
        g_grad_fn = ops.value_and_grad(
            self._generator_loss,
            None,
            self.optimizer_g.parameters,
        )
        log_path = self.paths["logs"] / "losses.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "d_loss", "g_loss"])
            for epoch in range(1, self.config.train.epochs + 1):
                for step, batch in enumerate(dataset.create_tuple_iterator(), start=1):
                    real_images = batch[0]
                    d_loss, d_grads = d_grad_fn(real_images)
                    self.optimizer_d(d_grads)
                    g_loss, g_grads = g_grad_fn(real_images)
                    self.optimizer_g(g_grads)
                    writer.writerow([epoch, step, float(d_loss.asnumpy()), float(g_loss.asnumpy())])

                self._after_epoch(epoch)

    def _after_epoch(self, epoch: int) -> None:
        if epoch % self.config.train.sample_interval == 0:
            fake_images = self.generator(self.fixed_noise).asnumpy()
            save_image_grid(fake_images, self.paths["samples"] / f"epoch_{epoch:03d}.png")
        if epoch % self.config.train.checkpoint_interval == 0:
            save_named_checkpoint(self.generator, self.paths["checkpoints"], "generator", epoch)
            save_named_checkpoint(self.discriminator, self.paths["checkpoints"], "discriminator", epoch)
