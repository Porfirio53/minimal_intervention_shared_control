from __future__ import annotations

import numpy as np

from ..types import GaussianTrajectory


def trajectory_conflict_and_uncertainty(
    human_intent: GaussianTrajectory,
    autonomous_positions: np.ndarray,
    position_scales: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute C_i, U_i, bar(C), and bar(U) from paper equations (7)-(8)."""
    autonomous = np.asarray(autonomous_positions, dtype=float)
    scales = np.asarray(position_scales, dtype=float)
    if autonomous.shape != human_intent.mean.shape or autonomous.shape[1] != 2:
        raise ValueError(
            "human and autonomous position trajectories must both be (N, 2)"
        )
    if scales.shape != (2,) or np.any(scales <= 0):
        raise ValueError("position_scales must contain two positive lengths")
    horizon = len(autonomous)
    normalized_weights = (
        np.full(horizon, 1.0 / horizon)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if (
        normalized_weights.shape != (horizon,)
        or np.any(normalized_weights < 0)
        or not np.isclose(np.sum(normalized_weights), 1.0)
    ):
        raise ValueError("weights must be non-negative and sum to one")
    difference = (human_intent.mean - autonomous) / scales
    conflict = np.einsum("nd,nd->n", difference, difference)
    inverse_scale = np.diag(1.0 / scales)
    uncertainty = np.array(
        [
            np.trace(inverse_scale @ covariance @ inverse_scale.T)
            for covariance in human_intent.covariance
        ]
    )
    return (
        conflict,
        uncertainty,
        float(normalized_weights @ conflict),
        float(normalized_weights @ uncertainty),
    )


def direction_cosine(human: np.ndarray, autonomous: np.ndarray) -> float:
    human, autonomous = (
        np.asarray(human, dtype=float),
        np.asarray(autonomous, dtype=float),
    )
    denominator = np.linalg.norm(human) * np.linalg.norm(autonomous)
    if denominator < 1e-12:
        return 1.0
    return float(np.dot(human, autonomous) / denominator)


def disagreement(
    human: np.ndarray, autonomous: np.ndarray, scales: np.ndarray | None = None
) -> float:
    scales = np.ones(2) if scales is None else np.asarray(scales, dtype=float)
    return float(np.linalg.norm((np.asarray(autonomous) - np.asarray(human)) / scales))
