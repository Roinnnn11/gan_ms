from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.metrics.core import (
    activation_statistics,
    frechet_distance,
    inception_score_from_logits,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _list_images(path: str | Path, max_samples: int | None = None) -> list[Path]:
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Evaluation image directory not found: {directory}")
    paths = sorted(item for item in directory.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")
        paths = paths[:max_samples]
    if not paths:
        raise ValueError(f"No supported images found in evaluation image directory: {directory}")
    return paths


def _preprocess_images(paths: list[Path]) -> np.ndarray:
    images: list[np.ndarray] = []
    resampling = getattr(Image, "Resampling", Image)
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((299, 299), resampling.BILINEAR)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        images.append(np.transpose(array, (2, 0, 1)))
    return np.stack(images).astype(np.float32, copy=False)


def metrics_from_features_and_logits(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    fake_logits: np.ndarray,
    *,
    splits: int = 10,
    shuffle: bool = True,
    seed: int = 2020,
) -> dict[str, float | str]:
    real_mean, real_covariance = activation_statistics(real_features)
    fake_mean, fake_covariance = activation_statistics(fake_features)
    fid = frechet_distance(real_mean, real_covariance, fake_mean, fake_covariance)
    is_mean, is_std = inception_score_from_logits(
        fake_logits,
        splits=splits,
        shuffle=shuffle,
        seed=seed,
    )
    return {
        "backend": "mindspore-mindcv",
        "fid": fid,
        "is_mean": is_mean,
        "is_std": is_std,
    }


def _extract_features_and_logits(
    paths: list[Path],
    model,
    *,
    batch_size: int,
    include_logits: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    import mindspore as ms
    from mindspore import ops

    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    reduce_mean = ops.ReduceMean(keep_dims=False)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        inputs = ms.Tensor(_preprocess_images(batch_paths), ms.float32)
        feature_maps = model.forward_features(inputs)
        pooled = reduce_mean(feature_maps, (2, 3))
        features.append(pooled.asnumpy())
        if include_logits:
            logits.append(model.classifier(pooled).asnumpy())

    all_features = np.concatenate(features, axis=0)
    all_logits = np.concatenate(logits, axis=0) if logits else None
    return all_features, all_logits


def evaluate_with_mindspore(
    real_dir: str | Path,
    fake_dir: str | Path,
    *,
    batch_size: int = 32,
    splits: int = 10,
    seed: int = 2020,
    max_samples: int | None = None,
    device_target: str = "GPU",
    device_id: int = 0,
) -> dict[str, float | str | int]:
    real_paths = _list_images(real_dir, max_samples=max_samples)
    fake_paths = _list_images(fake_dir, max_samples=max_samples)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise ValueError("FID requires at least two real and two generated images.")
    if splits > len(fake_paths):
        raise ValueError("splits cannot exceed the generated image count.")

    try:
        import mindcv
        import mindspore as ms
    except ImportError as exc:
        raise RuntimeError(
            "MindSpore evaluation backend is not installed. "
            "Install requirements-eval-mindspore.txt in the evaluation environment."
        ) from exc

    ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target, device_id=device_id)
    model = mindcv.create_model("inception_v3", pretrained=True)
    model.set_train(False)

    real_features, _ = _extract_features_and_logits(
        real_paths,
        model,
        batch_size=batch_size,
        include_logits=False,
    )
    fake_features, fake_logits = _extract_features_and_logits(
        fake_paths,
        model,
        batch_size=batch_size,
        include_logits=True,
    )
    assert fake_logits is not None
    result = metrics_from_features_and_logits(
        real_features,
        fake_features,
        fake_logits,
        splits=splits,
        shuffle=True,
        seed=seed,
    )
    result["real_samples"] = len(real_paths)
    result["fake_samples"] = len(fake_paths)
    return result
