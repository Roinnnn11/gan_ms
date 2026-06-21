from __future__ import annotations

import csv

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.config import ExperimentConfig
from src.models.dcgan import Discriminator, Generator
from src.utils.checkpoint import ensure_run_dirs, save_named_checkpoint
from src.utils.images import save_image_grid


class _DLoss(nn.Cell):
    def __init__(self, generator, discriminator, loss_fn, latent_dim):
        super().__init__(auto_prefix=False)
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim

    def construct(self, real_images):
        batch_size = real_images.shape[0]
        noise = ops.randn(batch_size, self.latent_dim, 1, 1).astype(ms.float32)
        fake_images = ops.stop_gradient(self.generator(noise))
        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)
        real_loss = self.loss_fn(real_scores, ops.ones_like(real_scores))
        fake_loss = self.loss_fn(fake_scores, ops.zeros_like(fake_scores))
        return real_loss + fake_loss


class _GLoss(nn.Cell):
    def __init__(self, generator, discriminator, loss_fn, latent_dim):
        super().__init__(auto_prefix=False)
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim

    def construct(self, real_images):
        batch_size = real_images.shape[0]
        noise = ops.randn(batch_size, self.latent_dim, 1, 1).astype(ms.float32)
        fake_images = self.generator(noise)
        fake_scores = self.discriminator(fake_images)
        return self.loss_fn(fake_scores, ops.ones_like(fake_scores))


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
        loss_fn = nn.BCELoss(reduction="mean")
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
            config.train.fixed_noise_size, params.latent_dim, 1, 1,
        ).astype(ms.float32)
        self._d_net = _DLoss(self.generator, self.discriminator, loss_fn, params.latent_dim)
        self._g_net = _GLoss(self.generator, self.discriminator, loss_fn, params.latent_dim)

    def train(self, dataset, start_epoch: int = 1) -> None:
        d_grad_fn = nn.TrainOneStepCell(self._d_net, self.optimizer_d)
        g_grad_fn = nn.TrainOneStepCell(self._g_net, self.optimizer_g)
        d_grad_fn.set_train()
        g_grad_fn.set_train()

        log_path = self.paths["logs"] / "losses.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "d_loss", "g_loss"])
            for epoch in range(1, self.config.train.epochs + 1):
                for step, batch in enumerate(dataset.create_tuple_iterator(), start=1):
                    real_images = batch[0]
                    d_loss = d_grad_fn(real_images)
                    g_loss = g_grad_fn(real_images)
                    writer.writerow([epoch, step, float(d_loss.asnumpy()), float(g_loss.asnumpy())])

                self._after_epoch(epoch)

    def _after_epoch(self, epoch: int) -> None:
        if epoch % self.config.train.sample_interval == 0:
            fake_images = self.generator(self.fixed_noise).asnumpy()
            save_image_grid(fake_images, self.paths["samples"] / f"epoch_{epoch:03d}.png")
        if epoch % self.config.train.checkpoint_interval == 0:
            save_named_checkpoint(self.generator, self.paths["checkpoints"], "generator", epoch)
            save_named_checkpoint(self.discriminator, self.paths["checkpoints"], "discriminator", epoch)
