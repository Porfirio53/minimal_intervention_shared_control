from __future__ import annotations

import numpy as np

from ..types import FloatArray, GaussianTrajectory


class ConstantVelocityPredictor:
    """Constant-velocity obstacle predictor with covariance growth."""

    def __init__(self, process_std: float = 0.35, measurement_std: float = 0.03):
        self.process_std = process_std
        self.measurement_std = measurement_std

    def fit_velocity(self, positions: FloatArray, dt: float) -> FloatArray:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2:
            raise ValueError("positions must be (samples, 2)")
        return (positions[-1] - positions[0]) / (dt * (len(positions) - 1))

    def predict(
        self,
        position: FloatArray,
        velocity: FloatArray,
        horizon: int,
        dt: float,
        initial_std: float = 0.03,
    ) -> GaussianTrajectory:
        position, velocity = (
            np.asarray(position, dtype=float),
            np.asarray(velocity, dtype=float),
        )
        times = np.arange(1, horizon + 1, dtype=float) * dt
        mean = position[None, :] + times[:, None] * velocity[None, :]
        covariance = np.stack(
            [
                np.eye(2)
                * (
                    initial_std**2
                    + (self.process_std * max(t, dt)) ** 2
                    + self.measurement_std**2
                )
                for t in times
            ]
        )
        return GaussianTrajectory(mean, covariance)
