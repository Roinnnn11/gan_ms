from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    image_size: int = 64
    batch_size: int = 64
    num_parallel_workers: int = 4
    shuffle_buffer: int = 10000


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 100
    image_channels: int = 3
    feature_maps_g: int = 64
    feature_maps_d: int = 64


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 25
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    sample_interval: int = 1
    checkpoint_interval: int = 5
    output_dir: str = "outputs/run"
    fixed_noise_size: int = 64


@dataclass(frozen=True)
class WganGpConfig:
    critic_steps: int = 5
    lambda_gp: float = 10.0


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model: str
    device_target: str
    device_id: int
    seed: int
    data: DataConfig
    model_params: ModelConfig
    train: TrainConfig
    wgan_gp: WganGpConfig


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping.")

    data = DataConfig(**_section(raw, "data"))
    model_params = ModelConfig(**_section(raw, "model_params"))
    train = TrainConfig(**_section(raw, "train"))
    wgan_gp = WganGpConfig(**_section(raw, "wgan_gp"))

    return ExperimentConfig(
        experiment_name=str(raw.get("experiment_name", "gan_run")),
        model=str(raw.get("model", "dcgan")),
        device_target=str(raw.get("device_target", "GPU")),
        device_id=int(raw.get("device_id", 0)),
        seed=int(raw.get("seed", 42)),
        data=data,
        model_params=model_params,
        train=train,
        wgan_gp=wgan_gp,
    )
