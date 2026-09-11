from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import Control
from .robust_cbf_qp import RelativeStateObservation


def observations_from_current_estimates(
    state: np.ndarray,
    previous_control: Control,
    obstacles: Sequence[tuple[np.ndarray, float, np.ndarray]],
    ages: float | Sequence[float],
    *,
    lookahead: float,
    position_std: float,
    velocity_std: float,
) -> list[RelativeStateObservation]:
    """Build timestamped safety observations for the deterministic simulator.

    The simulator has exact current obstacle states rather than a sensor buffer. It therefore
    back-propagates each current relative estimate with the constant-velocity model so that the
    execution filter must recover it through the paper's age propagation equation (24).
    """
    state_array = np.asarray(state, dtype=float)
    if state_array.shape != (3,):
        raise ValueError("state must have shape (3,)")
    requested_ages = (
        np.full(len(obstacles), float(ages))
        if np.isscalar(ages)
        else np.asarray(ages, dtype=float)
    )
    if requested_ages.shape != (len(obstacles),) or np.any(requested_ages < 0):
        raise ValueError("ages must be non-negative with one value per obstacle")
    if position_std < 0 or velocity_std < 0:
        raise ValueError("standard deviations must be non-negative")

    heading = np.array([np.cos(state_array[2]), np.sin(state_array[2])])
    lateral = np.array([-np.sin(state_array[2]), np.cos(state_array[2])])
    input_map = np.column_stack([heading, lookahead * lateral])
    lookahead_point = state_array[:2] + lookahead * heading
    point_velocity = input_map @ previous_control.as_array()
    observations: list[RelativeStateObservation] = []
    for (position, radius, velocity), age in zip(obstacles, requested_ages):
        obstacle_position = np.asarray(position, dtype=float)
        obstacle_velocity = np.asarray(velocity, dtype=float)
        relative_velocity = point_velocity - obstacle_velocity
        current_relative_position = lookahead_point - obstacle_position
        observations.append(
            RelativeStateObservation(
                source_relative_position=current_relative_position
                - float(age) * relative_velocity,
                source_relative_velocity=relative_velocity,
                obstacle_velocity=obstacle_velocity,
                obstacle_radius=float(radius),
                obstacle_speed_bound=float(np.linalg.norm(obstacle_velocity)),
                age=float(age),
                position_covariance=np.eye(2) * position_std**2,
                position_velocity_covariance=np.zeros((2, 2)),
                velocity_position_covariance=np.zeros((2, 2)),
                velocity_covariance=np.eye(2) * velocity_std**2,
            )
        )
    return observations
