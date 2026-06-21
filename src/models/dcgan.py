from __future__ import annotations

import mindspore.nn as nn
from mindspore.common.initializer import Normal


def conv_weight() -> Normal:
    return Normal(sigma=0.02)


class Generator(nn.Cell):
    def __init__(
        self,
        latent_dim: int = 100,
        image_channels: int = 3,
        feature_maps: int = 64,
    ) -> None:
        super().__init__()
        ngf = feature_maps
        self.net = nn.SequentialCell(
            nn.Conv2dTranspose(
                latent_dim,
                ngf * 8,
                kernel_size=4,
                stride=1,
                pad_mode="valid",
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            nn.Conv2dTranspose(
                ngf * 8,
                ngf * 4,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            nn.Conv2dTranspose(
                ngf * 4,
                ngf * 2,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.Conv2dTranspose(
                ngf * 2,
                ngf,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2dTranspose(
                ngf,
                image_channels,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.Tanh(),
        )

    def construct(self, noise):
        return self.net(noise)


class Discriminator(nn.Cell):
    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 64,
        use_sigmoid: bool = True,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        ndf = feature_maps
        layers: list[nn.Cell] = [
            nn.Conv2d(
                image_channels,
                ndf,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                ndf,
                ndf * 2,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                ndf * 2,
                ndf * 4,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                ndf * 4,
                ndf * 8,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
                weight_init=conv_weight(),
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                ndf * 8,
                1,
                kernel_size=4,
                stride=1,
                pad_mode="valid",
                has_bias=False,
                weight_init=conv_weight(),
            ),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.SequentialCell(layers)
        self.flatten = nn.Flatten()

    def construct(self, images):
        logits = self.net(images)
        return self.flatten(logits)
