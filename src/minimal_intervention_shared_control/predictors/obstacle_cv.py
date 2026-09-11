from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import FloatArray, GaussianTrajectory


class ConstantVelocityPredictor:
    """Constant-velocity obstacle predictor with covariance growth."""

    def __init__(
        self,
        process_std: float = 0.35,
        measurement_std: float = 0.03,
        initial_std: float = 0.03,
    ):
        parameters = np.array([process_std, measurement_std, initial_std], dtype=float)
        if not np.all(np.isfinite(parameters)) or np.any(parameters < 0):
            raise ValueError("obstacle standard deviations must be non-negative")
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.initial_std = initial_std

    @classmethod
    def from_e0_metrics(cls, path: str | Path) -> ConstantVelocityPredictor:
        """Load THOR parameters fitted on train and calibrated on validation data."""
        with Path(path).open(encoding="utf-8") as stream:
            result = json.load(stream)
        scale = float(result["calibration_scale"])
        if scale < 1:
            raise ValueError("calibration scale must not deflate covariance")
        multiplier = np.sqrt(scale)
        return cls(
            process_std=float(result["process_std"]) * multiplier,
            measurement_std=0.0,
            initial_std=float(result["initial_std"]) * multiplier,
        )

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
        initial_std: float | None = None,
    ) -> GaussianTrajectory:
        position, velocity = (
            np.asarray(position, dtype=float),
            np.asarray(velocity, dtype=float),
        )
        times = np.arange(1, horizon + 1, dtype=float) * dt
        mean = position[None, :] + times[:, None] * velocity[None, :]
        initial = self.initial_std if initial_std is None else float(initial_std)
        if initial < 0:
            raise ValueError("initial_std must be non-negative")
        covariance = np.stack(
            [
                np.eye(2)
                * (
                    initial**2
                    + (self.process_std * max(t, dt)) ** 2
                    + self.measurement_std**2
                )
                for t in times
            ]
        )
        return GaussianTrajectory(mean, covariance)
