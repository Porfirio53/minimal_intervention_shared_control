from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    samples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired arrays must be non-empty and one-dimensional")
    differences = a - b
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        differences, size=(samples, len(differences)), replace=True
    ).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
    }
