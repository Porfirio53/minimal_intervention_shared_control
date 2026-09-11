from __future__ import annotations

import numpy as np


class CovarianceCalibrator:
    """Scalar covariance inflation chosen from held-out Mahalanobis errors."""

    def __init__(self, target_coverage: float = 0.95, kappa: float = 2.326347874):
        if not 0 < target_coverage < 1:
            raise ValueError("target_coverage must be in (0, 1)")
        self.target_coverage = target_coverage
        self.kappa = kappa
        self.scale = 1.0

    def fit(self, errors: np.ndarray, covariances: np.ndarray) -> float:
        errors, covariances = (
            np.asarray(errors, dtype=float),
            np.asarray(covariances, dtype=float),
        )
        if errors.ndim != 2 or covariances.shape != (
            len(errors),
            errors.shape[1],
            errors.shape[1],
        ):
            raise ValueError("errors/covariances have incompatible shapes")
        distances = np.array(
            [e @ np.linalg.pinv(c) @ e for e, c in zip(errors, covariances)]
        )
        quantile = float(np.quantile(distances, self.target_coverage))
        self.scale = max(1.0, quantile / (self.kappa**2))
        return self.scale

    def transform(self, covariance: np.ndarray) -> np.ndarray:
        return np.asarray(covariance, dtype=float) * self.scale

    def coverage(self, errors: np.ndarray, covariance: np.ndarray) -> float:
        errors, covariance = np.asarray(errors), np.asarray(covariance)
        distances = np.einsum(
            "ni,nij,nj->n", errors, np.linalg.pinv(covariance), errors
        )
        return float(np.mean(distances <= self.kappa**2 * self.scale))
