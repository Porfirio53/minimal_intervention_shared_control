from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linearization import AffineStatePrediction


@dataclass(frozen=True)
class AuthorityConstraintSet:
    """Condensed G alpha >= b representation of paper equation (18)."""

    matrix: np.ndarray
    lower: np.ndarray
    labels: tuple[str, ...]


def build_authority_constraints(
    human_controls: np.ndarray,
    autonomous_controls: np.ndarray,
    chance_matrix: np.ndarray,
    chance_lower: np.ndarray,
    control_lower: np.ndarray,
    control_upper: np.ndarray,
    *,
    previous_control: np.ndarray | None = None,
    control_rate_limit: np.ndarray | None = None,
    previous_alpha: float | None = None,
    alpha_rate_limit: float | None = None,
    first_interval: float | None = None,
    prediction_interval: float | None = None,
    wheelbase: float | None = None,
    wheel_speed_limit: float | None = None,
    affine_prediction: AffineStatePrediction | None = None,
    corridors: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    state_trust_radius: np.ndarray | None = None,
    control_trust_radius: np.ndarray | None = None,
) -> AuthorityConstraintSet:
    """Assemble every configured affine part of A_k without adding relaxation."""
    human = np.asarray(human_controls, dtype=float)
    autonomous = np.asarray(autonomous_controls, dtype=float)
    if human.shape != autonomous.shape or human.ndim != 2 or human.shape[1] != 2:
        raise ValueError("candidate controls must both have shape (N, 2)")
    horizon = len(human)
    difference = autonomous - human
    control_lower = np.asarray(control_lower, dtype=float)
    control_upper = np.asarray(control_upper, dtype=float)
    if control_lower.shape != (2,) or control_upper.shape != (2,):
        raise ValueError("control bounds must have shape (2,)")
    chance_matrix = np.asarray(chance_matrix, dtype=float).reshape(-1, horizon)
    chance_lower = np.asarray(chance_lower, dtype=float)
    if chance_lower.shape != (len(chance_matrix),):
        raise ValueError("chance constraint dimensions do not match")

    rows = [row.copy() for row in chance_matrix]
    bounds = chance_lower.tolist()
    labels = ["chance"] * len(chance_matrix)

    def append(row: np.ndarray, lower: float, label: str) -> None:
        rows.append(np.asarray(row, dtype=float))
        bounds.append(float(lower))
        labels.append(label)

    for index in range(horizon):
        for axis in range(2):
            row = np.zeros(horizon)
            row[index] = difference[index, axis]
            append(row, control_lower[axis] - human[index, axis], "control_lower")
            append(-row, human[index, axis] - control_upper[axis], "control_upper")

    if control_rate_limit is not None:
        if (
            previous_control is None
            or first_interval is None
            or prediction_interval is None
        ):
            raise ValueError(
                "control-rate limits require previous control and both intervals"
            )
        rate = np.asarray(control_rate_limit, dtype=float)
        previous = np.asarray(previous_control, dtype=float)
        if rate.shape != (2,) or previous.shape != (2,) or np.any(rate < 0):
            raise ValueError("invalid control-rate inputs")
        for index in range(horizon):
            interval = first_interval if index == 0 else prediction_interval
            previous_human = previous if index == 0 else human[index - 1]
            previous_difference = np.zeros(2) if index == 0 else difference[index - 1]
            base = human[index] - previous_human
            for axis in range(2):
                row = np.zeros(horizon)
                row[index] = difference[index, axis]
                if index > 0:
                    row[index - 1] = -previous_difference[axis]
                append(row, -rate[axis] * interval - base[axis], "control_rate_lower")
                append(-row, base[axis] - rate[axis] * interval, "control_rate_upper")

    if alpha_rate_limit is not None:
        if (
            previous_alpha is None
            or first_interval is None
            or prediction_interval is None
        ):
            raise ValueError(
                "alpha-rate limits require previous alpha and both intervals"
            )
        if alpha_rate_limit < 0:
            raise ValueError("alpha_rate_limit must be non-negative")
        for index in range(horizon):
            interval = first_interval if index == 0 else prediction_interval
            row = np.zeros(horizon)
            row[index] = 1.0
            if index == 0:
                append(
                    row,
                    previous_alpha - alpha_rate_limit * interval,
                    "alpha_rate_lower",
                )
                append(
                    -row,
                    -previous_alpha - alpha_rate_limit * interval,
                    "alpha_rate_upper",
                )
            else:
                row[index - 1] = -1.0
                append(row, -alpha_rate_limit * interval, "alpha_rate_lower")
                append(-row, -alpha_rate_limit * interval, "alpha_rate_upper")

    if wheel_speed_limit is not None:
        if wheelbase is None or wheelbase <= 0 or wheel_speed_limit <= 0:
            raise ValueError(
                "wheel constraints require positive wheelbase and speed limit"
            )
        for index in range(horizon):
            for sign in (-1.0, 1.0):
                base = human[index, 0] + sign * wheelbase * human[index, 1] / 2.0
                delta = (
                    difference[index, 0] + sign * wheelbase * difference[index, 1] / 2.0
                )
                row = np.zeros(horizon)
                row[index] = delta
                append(row, -wheel_speed_limit - base, "wheel_lower")
                append(-row, base - wheel_speed_limit, "wheel_upper")

    if corridors is not None:
        if affine_prediction is None or len(corridors) != horizon:
            raise ValueError("corridors require one affine state constraint per step")
        for index, (corridor_matrix, corridor_upper) in enumerate(corridors):
            matrix = np.asarray(corridor_matrix, dtype=float)
            upper = np.asarray(corridor_upper, dtype=float)
            if (
                matrix.ndim != 2
                or matrix.shape[1] != 3
                or upper.shape != (len(matrix),)
            ):
                raise ValueError("each corridor must be H(N,3), h(N)")
            for h_row, h_upper in zip(matrix, upper):
                append(
                    -(h_row @ affine_prediction.sensitivity[index].T),
                    h_row @ affine_prediction.offset[index] - h_upper,
                    "corridor",
                )

    if state_trust_radius is not None:
        if affine_prediction is None:
            raise ValueError("state trust regions require an affine prediction")
        radius = np.asarray(state_trust_radius, dtype=float)
        if radius.shape != (3,) or np.any(radius < 0):
            raise ValueError("state_trust_radius must have shape (3,)")
        for index in range(horizon):
            base = (
                affine_prediction.offset[index]
                - affine_prediction.reference_states[index]
            )
            for axis in range(3):
                row = affine_prediction.sensitivity[index, :, axis]
                append(row, -radius[axis] - base[axis], "state_trust_lower")
                append(-row, base[axis] - radius[axis], "state_trust_upper")

    if control_trust_radius is not None:
        radius = np.asarray(control_trust_radius, dtype=float)
        if radius.shape != (2,) or np.any(radius < 0):
            raise ValueError("control_trust_radius must have shape (2,)")
        if affine_prediction is None:
            raise ValueError("control trust regions require a reference prediction")
        for index in range(horizon):
            base = human[index] - affine_prediction.reference_controls[index]
            for axis in range(2):
                row = np.zeros(horizon)
                row[index] = difference[index, axis]
                append(row, -radius[axis] - base[axis], "control_trust_lower")
                append(-row, base[axis] - radius[axis], "control_trust_upper")

    return AuthorityConstraintSet(
        np.asarray(rows, dtype=float).reshape(-1, horizon),
        np.asarray(bounds, dtype=float),
        tuple(labels),
    )
