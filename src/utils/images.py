from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import numpy as np
from PIL import Image


def tensor_to_uint8_images(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError("Expected images with shape [N, C, H, W].")
    if images.shape[1] not in (1, 3):
        raise ValueError("Expected 1 or 3 image channels.")

    clipped = np.clip(images, -1.0, 1.0)
    scaled = ((clipped + 1.0) * 127.5).astype(np.uint8)
    return np.transpose(scaled, (0, 2, 3, 1))


def make_image_grid(images: np.ndarray, nrow: int | None = None) -> np.ndarray:
    uint8_images = tensor_to_uint8_images(images)
    count, height, width, channels = uint8_images.shape
    if count == 0:
        raise ValueError("Cannot create a grid from zero images.")

    cols = nrow or int(ceil(sqrt(count)))
    rows = int(ceil(count / cols))
    grid = np.zeros((rows * height, cols * width, channels), dtype=np.uint8)

    for idx, image in enumerate(uint8_images):
        row = idx // cols
        col = idx % cols
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = image

    if channels == 1:
        return grid[:, :, 0]
    return grid


def save_image_grid(images: np.ndarray, path: str | Path, nrow: int | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_image_grid(images, nrow=nrow)
    Image.fromarray(grid).save(output_path)
