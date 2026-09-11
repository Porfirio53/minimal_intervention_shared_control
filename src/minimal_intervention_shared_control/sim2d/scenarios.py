from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..autonomy.reference_path import PolylinePath
from .agents import ObstacleSpec


@dataclass(frozen=True)
class Scenario:
    name: str
    initial_state: np.ndarray
    goal: np.ndarray
    human_path: PolylinePath
    autonomous_path: PolylinePath
    obstacles: tuple[ObstacleSpec, ...]
    network_condition: str = "N1"
    sensor_age: float = 0.02


def make_scenario(name: str) -> Scenario:
    start = np.array([0.0, 0.0, 0.0])
    if name == "bend":
        path = np.array([[0, 0], [1.8, 0], [2.5, 0.7], [2.5, 2.8]], dtype=float)
        return Scenario(
            name,
            start,
            path[-1],
            PolylinePath(path),
            PolylinePath(path),
            (ObstacleSpec((1.2, 1.2), (0, 0), 0.32),),
        )
    if name == "crossing":
        human = np.array([[0, 0], [4.0, 0]], dtype=float)
        autonomous = np.array([[0, 0], [1.2, 1.3], [2.4, 1.3], [4.0, 0]], dtype=float)
        return Scenario(
            name,
            start,
            human[-1],
            PolylinePath(human),
            PolylinePath(autonomous),
            (ObstacleSpec((1.5, -1.2), (0, 0.45), 0.27),),
        )
    if name == "conflict":
        human = np.array([[0, 0], [1.3, 0.8], [2.7, 0.8], [4, 0]], dtype=float)
        autonomous = np.array([[0, 0], [1.3, -0.8], [2.7, -0.8], [4, 0]], dtype=float)
        return Scenario(
            name,
            start,
            human[-1],
            PolylinePath(human),
            PolylinePath(autonomous),
            (ObstacleSpec((2.0, 0.75), (0, 0), 0.3),),
        )
    if name == "network_anomaly":
        human = np.array([[0, 0], [1.5, 0], [3.0, 0]], dtype=float)
        autonomous = np.array([[0, 0], [1.2, 1.15], [2.2, 1.15], [3.0, 0]], dtype=float)
        return Scenario(
            name,
            start,
            human[-1],
            PolylinePath(human),
            PolylinePath(autonomous),
            (ObstacleSpec((1.55, -0.9), (0, 0.55), 0.3),),
            "NJ",
            0.15,
        )
    raise ValueError(f"unknown scenario: {name}")


SCENARIO_NAMES = ("bend", "crossing", "conflict", "network_anomaly")
