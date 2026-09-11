from __future__ import annotations

import numpy as np


def gaussian_trajectory_metrics(
    truth: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    kappa_levels: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute trajectory accuracy, Gaussian NLL, and empirical ellipse coverage."""
    truth, mean, covariance = (
        np.asarray(truth, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(covariance, dtype=float),
    )
    if truth.shape != mean.shape or truth.ndim != 3:
        raise ValueError("truth and mean must have shape (samples, horizon, dimension)")
    if covariance.shape != truth.shape + (truth.shape[-1],):
        raise ValueError("covariance has an incompatible shape")
    error = truth - mean
    distances = np.linalg.norm(error, axis=-1)
    inverses = np.linalg.pinv(covariance)
    mahalanobis = np.einsum("nhd,nhde,nhe->nh", error, inverses, error)
    _, log_determinant = np.linalg.slogdet(covariance)
    dimension = truth.shape[-1]
    nll = 0.5 * (dimension * np.log(2 * np.pi) + log_determinant + mahalanobis)
    result = {
        "ade": float(np.mean(distances)),
        "fde": float(np.mean(distances[:, -1])),
        "nll": float(np.mean(nll)),
    }
    levels = kappa_levels or {
        "coverage_90": 2.146,
        "coverage_95": 2.448,
        "coverage_99": 3.035,
    }
    for name, kappa in levels.items():
        result[name] = float(np.mean(mahalanobis <= kappa**2))
    return result
