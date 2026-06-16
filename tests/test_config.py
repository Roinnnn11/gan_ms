from pathlib import Path

from src.config import load_config


def test_load_dcgan_config() -> None:
    config = load_config(Path("configs/dcgan_celeba.yaml"))

    assert config.model == "dcgan"
    assert config.data.image_size == 64
    assert config.data.batch_size == 128
    assert config.model_params.latent_dim == 100
    assert config.train.output_dir == "outputs/dcgan"
