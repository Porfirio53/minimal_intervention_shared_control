from __future__ import annotations

import numpy as np

from ..types import FloatArray, GaussianTrajectory


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def step(state: FloatArray, control: FloatArray, dt: float) -> FloatArray:
    """Apply the Euler-discretized differential-drive model from paper equation (3)."""
    x = np.asarray(state, dtype=float)
    u = np.asarray(control, dtype=float)
    if x.shape != (3,) or u.shape != (2,):
        raise ValueError("state and control must have shapes (3,) and (2,)")
    if dt <= 0:
        raise ValueError("dt must be positive")
    px, py, heading = x
    v, omega = u
    next_px = px + dt * v * np.cos(heading)
    next_py = py + dt * v * np.sin(heading)
    return np.array([next_px, next_py, wrap_angle(heading + omega * dt)], dtype=float)


def rollout(state: FloatArray, controls: FloatArray, dt: float) -> FloatArray:
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError("controls must have shape (horizon, 2)")
    states = np.empty((len(controls) + 1, 3), dtype=float)
    states[0] = state
    for index, control in enumerate(controls):
        states[index + 1] = step(states[index], control, dt)
    return states


def jacobians(
    state: FloatArray, control: FloatArray, dt: float
) -> tuple[FloatArray, FloatArray]:
    """Euler-linearized discrete dynamics, used for diagnostics and MPC extensions."""
    heading = float(np.asarray(state)[2])
    v = float(np.asarray(control)[0])
    a = np.eye(3)
    a[0, 2] = -dt * v * np.sin(heading)
    a[1, 2] = dt * v * np.cos(heading)
    b = np.array(
        [[dt * np.cos(heading), 0.0], [dt * np.sin(heading), 0.0], [0.0, dt]],
        dtype=float,
    )
    return a, b


def rollout_gaussian_controls(
    state: FloatArray,
    controls: FloatArray,
    joint_control_covariance: FloatArray,
    dt: float,
) -> GaussianTrajectory:
    """Propagate a joint Gaussian control forecast to positional ``Sigma_H``.

    The mean follows the nonlinear Euler vehicle model. Covariance uses a first-order
    linearization along that mean and retains cross-time control correlations.
    """
    controls_array = np.asarray(controls, dtype=float)
    horizon = len(controls_array)
    joint = np.asarray(joint_control_covariance, dtype=float)
    if controls_array.ndim != 2 or controls_array.shape[1] != 2:
        raise ValueError("controls must have shape (horizon, 2)")
    if joint.shape != (2 * horizon, 2 * horizon):
        raise ValueError("joint control covariance has an incompatible shape")
    if not np.all(np.isfinite(joint)) or not np.allclose(joint, joint.T, atol=1e-10):
        raise ValueError("joint control covariance must be finite and symmetric")
    minimum_eigenvalue = float(np.linalg.eigvalsh(joint).min()) if horizon else 0.0
    if minimum_eigenvalue < -1e-9:
        raise ValueError("joint control covariance must be positive semidefinite")

    states = rollout(state, controls_array, dt)
    covariance = np.empty((horizon, 2, 2))
    sensitivity = np.zeros((3, 2 * horizon))
    for index, control in enumerate(controls_array):
        a, b = jacobians(states[index], control, dt)
        sensitivity = a @ sensitivity
        sensitivity[:, 2 * index : 2 * index + 2] += b
        position_sensitivity = sensitivity[:2]
        block = position_sensitivity @ joint @ position_sensitivity.T
        covariance[index] = 0.5 * (block + block.T)
    return GaussianTrajectory(states[1:, :2], covariance)
