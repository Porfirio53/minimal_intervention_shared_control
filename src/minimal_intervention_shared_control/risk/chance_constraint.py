from __future__ import annotations

import numpy as np


def relative_position_covariance(
    execution_covariance: np.ndarray,
    obstacle_covariance: np.ndarray,
    cross_covariance: np.ndarray | None = None,
    *,
    assume_independent: bool = False,
) -> np.ndarray:
    """Build Sigma_Z from paper equation (9) without silently assuming independence."""
    execution = np.asarray(execution_covariance, dtype=float)
    obstacle = np.asarray(obstacle_covariance, dtype=float)
    if execution.shape != (2, 2) or obstacle.shape != (2, 2):
        raise ValueError("position covariance matrices must have shape (2, 2)")
    if cross_covariance is None:
        if not assume_independent:
            raise ValueError(
                "cross covariance is required unless independence is explicit"
            )
        cross = np.zeros((2, 2))
    else:
        cross = np.asarray(cross_covariance, dtype=float)
        if cross.shape != (2, 2):
            raise ValueError("cross_covariance must have shape (2, 2)")
    return execution + obstacle - cross - cross.T


def chance_margin(
    robot_position: np.ndarray,
    obstacle_mean: np.ndarray,
    relative_covariance: np.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    clearance: float,
    kappa: float,
    normal: np.ndarray | None = None,
) -> float:
    """Evaluate the fixed-normal half-space margin in paper equation (10)."""
    delta = np.asarray(robot_position, dtype=float) - np.asarray(
        obstacle_mean, dtype=float
    )
    direction = (
        delta / max(float(np.linalg.norm(delta)), 1e-9)
        if normal is None
        else np.asarray(normal, dtype=float)
    )
    if direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1.0):
        raise ValueError("normal must be a two-dimensional unit vector")
    covariance = np.asarray(relative_covariance, dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("relative_covariance must have shape (2, 2)")
    sigma = float(np.sqrt(max(0.0, direction @ covariance @ direction)))
    projected_separation = float(direction @ delta)
    return (
        projected_separation
        - robot_radius
        - obstacle_radius
        - clearance
        - kappa * sigma
    )


def linearized_constraints(
    robot_positions: np.ndarray,
    obstacle_means: np.ndarray,
    obstacle_covariances: np.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    clearance: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rows G and right hand side b for G alpha >= b."""
    rows, rhs = [], []
    for position, obstacle, covariance in zip(
        robot_positions, obstacle_means, obstacle_covariances
    ):
        delta = np.asarray(position) - np.asarray(obstacle)
        normal = delta / max(np.linalg.norm(delta), 1e-9)
        margin = chance_margin(
            position,
            obstacle,
            covariance,
            robot_radius,
            obstacle_radius,
            clearance,
            kappa,
        )
        rows.append(normal)
        rhs.append(-margin)
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)
