from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import FloatArray


@dataclass
class DynamicObstacle:
    position: FloatArray
    velocity: FloatArray
    radius: float = 0.28

    def step(self, dt: float) -> None:
        self.position = np.asarray(self.position, dtype=float) + dt * np.asarray(
            self.velocity, dtype=float
        )


@dataclass
class TrajectoryObstacle:
    """Obstacle replaying timestamped ground-truth positions without interpolation drift."""

    timestamps: FloatArray
    positions: FloatArray
    radius: float = 0.28
    elapsed: float = 0.0

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        if (
            self.timestamps.ndim != 1
            or self.positions.shape != (len(self.timestamps), 2)
            or len(self.timestamps) < 2
            or np.any(np.diff(self.timestamps) <= 0)
            or not np.all(np.isfinite(self.timestamps))
            or not np.all(np.isfinite(self.positions))
            or self.radius < 0
        ):
            raise ValueError("invalid obstacle trajectory")
        self.timestamps = self.timestamps - self.timestamps[0]
        self.position = self.positions[0].copy()
        self.velocity = self._velocity_at(0.0)

    def _position_at(self, elapsed: float) -> FloatArray:
        return np.array(
            [
                np.interp(elapsed, self.timestamps, self.positions[:, axis])
                for axis in range(2)
            ]
        )

    def _velocity_at(self, elapsed: float, history: float = 0.5) -> FloatArray:
        if elapsed <= 0:
            return np.zeros(2)
        start = max(0.0, elapsed - history)
        duration = elapsed - start
        return (self._position_at(elapsed) - self._position_at(start)) / duration

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.elapsed = min(self.elapsed + dt, float(self.timestamps[-1]))
        self.position = self._position_at(self.elapsed)
        self.velocity = (
            self._velocity_at(self.elapsed)
            if self.elapsed < self.timestamps[-1]
            else np.zeros(2)
        )


@dataclass(frozen=True)
class ObstacleSpec:
    position: tuple[float, float]
    velocity: tuple[float, float]
    radius: float = 0.28

    def instantiate(self) -> DynamicObstacle:
        return DynamicObstacle(
            np.array(self.position, dtype=float),
            np.array(self.velocity, dtype=float),
            self.radius,
        )


@dataclass(frozen=True)
class TrajectoryObstacleSpec:
    timestamps: FloatArray
    positions: FloatArray
    radius: float = 0.28

    def instantiate(self) -> TrajectoryObstacle:
        return TrajectoryObstacle(
            self.timestamps.copy(), self.positions.copy(), self.radius
        )
