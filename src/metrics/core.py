from __future__ import annotations

import numpy as np


def _as_finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return matrix


def activation_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = _as_finite_matrix(features, "features")
    if matrix.shape[0] < 2:
        raise ValueError("features must contain at least two samples.")
    return np.mean(matrix, axis=0), np.cov(matrix, rowvar=False)


def frechet_distance(
    mean_real: np.ndarray,
    covariance_real: np.ndarray,
    mean_fake: np.ndarray,
    covariance_fake: np.ndarray,
) -> float:
    mean_real = np.asarray(mean_real, dtype=np.float64)
    mean_fake = np.asarray(mean_fake, dtype=np.float64)
    covariance_real = _as_finite_matrix(covariance_real, "covariance_real")
    covariance_fake = _as_finite_matrix(covariance_fake, "covariance_fake")

    if mean_real.ndim != 1 or mean_real.shape != mean_fake.shape:
        raise ValueError("Mean vectors must be one-dimensional and have matching shapes.")
    if covariance_real.shape != covariance_fake.shape:
        raise ValueError("Covariance matrices must have matching shapes.")
    expected_shape = (mean_real.size, mean_real.size)
    if covariance_real.shape != expected_shape:
        raise ValueError("Covariance dimensions must match the mean vector length.")
    if not np.isfinite(mean_real).all() or not np.isfinite(mean_fake).all():
        raise ValueError("Mean vectors contain NaN or infinite values.")

    product_eigenvalues = np.linalg.eigvals(covariance_real @ covariance_fake)
    if np.max(np.abs(product_eigenvalues.imag)) > 1e-6:
        raise ValueError("Covariance product has a significant imaginary component.")
    real_eigenvalues = product_eigenvalues.real
    if np.min(real_eigenvalues) < -1e-6:
        raise ValueError("Covariance product has a negative eigenvalue.")

    trace_covariance_mean = np.sqrt(np.clip(real_eigenvalues, 0.0, None)).sum()
    difference = mean_real - mean_fake
    score = float(
        difference @ difference
        + np.trace(covariance_real)
        + np.trace(covariance_fake)
        - 2.0 * trace_covariance_mean
    )
    return max(score, 0.0) if score > -1e-7 else score


def inception_score_from_logits(
    logits: np.ndarray,
    splits: int = 10,
    shuffle: bool = True,
    seed: int = 2020,
) -> tuple[float, float]:
    matrix = _as_finite_matrix(logits, "logits")
    sample_count = matrix.shape[0]
    if splits <= 0 or splits > sample_count:
        raise ValueError("splits must be positive and no greater than the sample count.")

    if shuffle:
        matrix = matrix[np.random.default_rng(seed).permutation(sample_count)]

    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)

    scores: list[float] = []
    for chunk in np.array_split(np.arange(sample_count), splits):
        chunk_probabilities = probabilities[chunk]
        chunk_log_probabilities = log_probabilities[chunk]
        marginal = chunk_probabilities.mean(axis=0, keepdims=True)
        kl = chunk_probabilities * (chunk_log_probabilities - np.log(marginal))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))

    return float(np.mean(scores)), float(np.std(scores))
