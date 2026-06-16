import numpy as np

import mindspore as ms
from mindspore import Tensor

from src.models.dcgan import Discriminator, Generator


def test_generator_output_shape() -> None:
    generator = Generator(latent_dim=100, image_channels=3, feature_maps=16)
    noise = Tensor(np.random.randn(2, 100, 1, 1), ms.float32)

    images = generator(noise)

    assert images.shape == (2, 3, 64, 64)


def test_discriminator_output_shape() -> None:
    discriminator = Discriminator(image_channels=3, feature_maps=16)
    images = Tensor(np.random.randn(2, 3, 64, 64), ms.float32)

    scores = discriminator(images)

    assert scores.shape == (2, 1)
