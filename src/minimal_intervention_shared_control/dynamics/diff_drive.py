from __future__ import annotations

import numpy as np

from ..types import FloatArray


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
