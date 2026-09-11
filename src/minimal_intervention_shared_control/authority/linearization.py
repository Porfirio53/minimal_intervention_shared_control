from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics.diff_drive import jacobians, rollout
from ..risk.prediction_tube import blend_controls, trajectory_positions


@dataclass(frozen=True)
class AffineStatePrediction:
    """Condensed local model x(alpha) = offset + S alpha from equations (14)-(16)."""

    offset: np.ndarray
    sensitivity: np.ndarray
    reference_states: np.ndarray
    reference_controls: np.ndarray
    reference_alpha: np.ndarray

    def states(self, alpha: np.ndarray) -> np.ndarray:
        weights = np.asarray(alpha, dtype=float)
        if weights.shape != self.reference_alpha.shape:
            raise ValueError("alpha must match the prediction horizon")
        return self.offset + np.einsum("ijd,j->id", self.sensitivity, weights)

    def positions(self, alpha: np.ndarray) -> np.ndarray:
        return self.states(alpha)[:, :2]


def affine_state_prediction(
    initial_state: np.ndarray,
    human_controls: np.ndarray,
    autonomous_controls: np.ndarray,
    reference_alpha: np.ndarray,
    dt: float,
) -> AffineStatePrediction:
    """Condense the Euler dynamics linearized along one fixed reference trajectory."""
    human = np.asarray(human_controls, dtype=float)
    autonomous = np.asarray(autonomous_controls, dtype=float)
    reference = np.asarray(reference_alpha, dtype=float)
    if human.shape != autonomous.shape or human.ndim != 2 or human.shape[1] != 2:
        raise ValueError("candidate controls must both have shape (N, 2)")
    horizon = len(human)
    if reference.shape != (horizon,):
        raise ValueError("reference_alpha must match the horizon")
    reference_controls = blend_controls(human, autonomous, reference)
    reference_states = rollout(initial_state, reference_controls, dt)
    control_difference = autonomous - human
    sensitivity = np.zeros((horizon, horizon, 3), dtype=float)
    propagated = np.zeros((3, horizon), dtype=float)
    for index in range(horizon):
        a_matrix, b_matrix = jacobians(
            reference_states[index], reference_controls[index], dt
        )
        propagated = a_matrix @ propagated
        propagated[:, index] += b_matrix @ control_difference[index]
        sensitivity[index] = propagated.T
    reference_future = reference_states[1:]
    offset = reference_future - np.einsum("ijd,j->id", sensitivity, reference)
    return AffineStatePrediction(
        offset,
        sensitivity,
        reference_future,
        reference_controls,
        reference,
    )


def finite_difference_sensitivity(
    initial_state: np.ndarray,
    human_controls: np.ndarray,
    autonomous_controls: np.ndarray,
    dt: float,
    epsilon: float = 1e-4,
    reference_alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Numerically construct S around a warm-start authority sequence."""
    human_controls, autonomous_controls = (
        np.asarray(human_controls),
        np.asarray(autonomous_controls),
    )
    horizon = len(human_controls)
    reference = (
        np.zeros(horizon)
        if reference_alpha is None
        else np.asarray(reference_alpha, dtype=float)
    )
    if reference.shape != (horizon,):
        raise ValueError("reference_alpha must match the horizon")
    baseline = trajectory_positions(
        initial_state,
        blend_controls(human_controls, autonomous_controls, reference),
        dt,
    )
    sensitivity = np.zeros((horizon, horizon, 2), dtype=float)
    for j in range(horizon):
        direction = epsilon if reference[j] <= 1.0 - epsilon else -epsilon
        alpha = reference.copy()
        alpha[j] += direction
        perturbed = trajectory_positions(
            initial_state,
            blend_controls(human_controls, autonomous_controls, alpha),
            dt,
        )
        sensitivity[:, j] = (perturbed - baseline) / direction
    return sensitivity
