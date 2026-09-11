from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics.diff_drive import rollout
from ..types import GaussianTrajectory


@dataclass(frozen=True)
class PredictionRiskTube:
    """Typed representation of the distinct objects in paper equation (6)."""

    human_intent: GaussianTrajectory
    autonomous_positions: np.ndarray
    execution: GaussianTrajectory
    obstacles: tuple[GaussianTrajectory, ...]
    conflict: np.ndarray
    intent_uncertainty: np.ndarray
    human_margins: np.ndarray
    autonomous_margins: np.ndarray
    execution_margins: np.ndarray

    def __post_init__(self) -> None:
        horizon = len(self.human_intent.mean)
        arrays = (
            self.autonomous_positions,
            self.conflict,
            self.intent_uncertainty,
            self.human_margins,
            self.autonomous_margins,
            self.execution_margins,
        )
        if np.asarray(arrays[0]).shape != (horizon, 2):
            raise ValueError("autonomous_positions must have shape (N, 2)")
        if any(np.asarray(value).shape[0] != horizon for value in arrays[1:]):
            raise ValueError("risk-tube arrays must share one prediction horizon")


def blend_controls(
    human: np.ndarray, autonomous: np.ndarray, alpha: np.ndarray | float
) -> np.ndarray:
    human, autonomous = (
        np.asarray(human, dtype=float),
        np.asarray(autonomous, dtype=float),
    )
    weights = np.asarray(alpha, dtype=float)
    return human + weights[..., None] * (autonomous - human)


def trajectory_positions(
    initial_state: np.ndarray, controls: np.ndarray, dt: float
) -> np.ndarray:
    return rollout(initial_state, controls, dt)[1:, :2]


def tube_radius(
    covariance: np.ndarray, kappa: float, base_radius: float = 0.0
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    largest = np.linalg.eigvalsh(covariance).max(axis=-1)
    return base_radius + kappa * np.sqrt(np.maximum(largest, 0.0))
