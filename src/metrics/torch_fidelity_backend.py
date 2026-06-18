from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _validate_image_directory(path: str | Path) -> Path:
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Evaluation image directory not found: {directory}")
    if not any(item.suffix.lower() in IMAGE_EXTENSIONS for item in directory.rglob("*")):
        raise ValueError(f"No supported images found in evaluation image directory: {directory}")
    return directory


def evaluate_with_torch_fidelity(
    real_dir: str | Path,
    fake_dir: str | Path,
    *,
    batch_size: int = 64,
    cuda: bool = True,
    splits: int = 10,
    seed: int = 2020,
    verbose: bool = True,
) -> dict[str, float | str]:
    real_path = _validate_image_directory(real_dir)
    fake_path = _validate_image_directory(fake_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if splits <= 0:
        raise ValueError("splits must be positive.")

    try:
        import torch_fidelity
    except ImportError as exc:
        raise RuntimeError(
            "torch-fidelity backend is not installed. "
            "Install requirements-eval-torch.txt in the evaluation environment."
        ) from exc

    raw = torch_fidelity.calculate_metrics(
        input1=str(fake_path),
        input2=str(real_path),
        cuda=cuda,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=verbose,
        batch_size=batch_size,
        isc_splits=splits,
        samples_shuffle=True,
        rng_seed=seed,
        samples_find_deep=True,
    )
    return {
        "backend": "torch-fidelity",
        "fid": float(raw["frechet_inception_distance"]),
        "is_mean": float(raw["inception_score_mean"]),
        "is_std": float(raw["inception_score_std"]),
    }
