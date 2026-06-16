from __future__ import annotations

from pathlib import Path

import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter

from src.config import DataConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageBytesDataset:
    def __init__(self, data_root: Path) -> None:
        self.paths = sorted(
            path for path in data_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No images found under: {data_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray]:
        return (np.fromfile(self.paths[index], dtype=np.uint8),)


def create_celeba_dataset(config: DataConfig) -> ds.Dataset:
    data_root = Path(config.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"CelebA image directory not found: {data_root}. "
            "Set data.data_root in the YAML config."
        )

    dataset = ds.GeneratorDataset(
        source=ImageBytesDataset(data_root),
        column_names=["image"],
        shuffle=True,
    )

    transforms = [
        vision.Decode(),
        vision.Resize((config.image_size, config.image_size), interpolation=Inter.BILINEAR),
        vision.HWC2CHW(),
        vision.Rescale(1.0 / 127.5, -1.0),
    ]

    dataset = dataset.map(
        operations=transforms,
        input_columns="image",
        num_parallel_workers=config.num_parallel_workers,
    )
    dataset = dataset.project(["image"])
    dataset = dataset.shuffle(buffer_size=config.shuffle_buffer)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    return dataset
