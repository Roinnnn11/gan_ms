from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.images import tensor_to_uint8_images


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def discover_image_paths(source_dir: str | Path, limit: int | None = None) -> list[Path]:
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Image source directory not found: {source}")
    paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found under: {source}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when provided.")
        if len(paths) < limit:
            raise ValueError(f"Requested {limit} images, but only found {len(paths)} under {source}.")
        paths = paths[:limit]
    return paths


def _create_empty_output_directory(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_real_images(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    image_size: int = 64,
    center_crop_size: int | None = 178,
    num_images: int | None = None,
) -> int:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if center_crop_size is not None and center_crop_size <= 0:
        raise ValueError("center_crop_size must be positive when provided.")
    paths = discover_image_paths(source_dir, limit=num_images)
    output = _create_empty_output_directory(output_dir)
    resampling = getattr(Image, "Resampling", Image)

    for index, path in enumerate(paths):
        with Image.open(path) as image:
            converted = image.convert("RGB")
            if center_crop_size is not None:
                width, height = converted.size
                left = max((width - center_crop_size) // 2, 0)
                top = max((height - center_crop_size) // 2, 0)
                right = min(left + center_crop_size, width)
                bottom = min(top + center_crop_size, height)
                converted = converted.crop((left, top, right, bottom))
            resized = converted.resize((image_size, image_size), resampling.BILINEAR)
            resized.save(output / f"{index:06d}.png")
    return len(paths)


def save_generated_images(
    images: np.ndarray,
    output_dir: str | Path,
    *,
    start_index: int = 0,
) -> int:
    if start_index < 0:
        raise ValueError("start_index must not be negative.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    converted = tensor_to_uint8_images(np.asarray(images))

    for offset, image in enumerate(converted):
        target = output / f"{start_index + offset:06d}.png"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite generated sample: {target}")
        Image.fromarray(image).save(target)
    return start_index + len(converted)
