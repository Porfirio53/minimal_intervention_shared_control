from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import FloatArray
from .agents import DynamicObstacle, TrajectoryObstacle


@dataclass
class World:
    state: FloatArray
    obstacles: list[DynamicObstacle | TrajectoryObstacle]
    robot_radius: float

    def advance_obstacles(self, dt: float) -> None:
        for obstacle in self.obstacles:
            obstacle.step(dt)

    def minimum_clearance(self) -> float:
        if not self.obstacles:
            return float("inf")
        return min(
            float(
                np.linalg.norm(self.state[:2] - obstacle.position)
                - self.robot_radius
                - obstacle.radius
            )
            for obstacle in self.obstacles
        )

    def collision(self) -> bool:
        return self.minimum_clearance() <= 0.0
