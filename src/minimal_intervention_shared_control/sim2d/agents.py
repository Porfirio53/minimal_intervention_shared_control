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
