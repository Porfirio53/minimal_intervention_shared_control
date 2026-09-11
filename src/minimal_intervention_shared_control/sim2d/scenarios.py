from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..autonomy.reference_path import PolylinePath
from ..config import project_root
from ..datasets.thor import load_thor_tsv
from .agents import ObstacleSpec, TrajectoryObstacleSpec


@dataclass(frozen=True)
class Scenario:
    name: str
    initial_state: np.ndarray
    goal: np.ndarray
    human_path: PolylinePath
    autonomous_path: PolylinePath
    obstacles: tuple[ObstacleSpec | TrajectoryObstacleSpec, ...]
    network_condition: str = "N1"
    sensor_age: float = 0.02
    obstacle_source: str = "synthetic"


_THOR_CROSSING_RECORDING = "Exp_2_run_3.tsv"
_THOR_CROSSING_TRACK = "Exp_2_run_3:Helmet_10:003"
_THOR_CROSSING_TIME = 2.7


def thor_crossing_obstacle(path: str | Path) -> TrajectoryObstacleSpec:
    """Rigidly place one held-out THOR track into the 2-D crossing scene.

    Rotation and translation preserve the recorded trajectory's timing, curvature,
    speed, and covariance-relevant prediction difficulty.
    """
    tracks = load_thor_tsv(path)
    try:
        track = next(item for item in tracks if item.track_id == _THOR_CROSSING_TRACK)
    except StopIteration as error:
        raise ValueError(
            f"THOR track {_THOR_CROSSING_TRACK!r} was not found"
        ) from error
    relative_time = track.timestamps - track.timestamps[0]
    if relative_time[-1] < _THOR_CROSSING_TIME + 0.5:
        raise ValueError("THOR crossing track is too short")
    crossing = np.array(
        [
            np.interp(_THOR_CROSSING_TIME, relative_time, track.positions[:, axis])
            for axis in range(2)
        ]
    )
    before = np.array(
        [
            np.interp(
                _THOR_CROSSING_TIME - 0.5, relative_time, track.positions[:, axis]
            )
            for axis in range(2)
        ]
    )
    after = np.array(
        [
            np.interp(
                _THOR_CROSSING_TIME + 0.5, relative_time, track.positions[:, axis]
            )
            for axis in range(2)
        ]
    )
    direction = after - before
    angle = np.pi / 2.0 - np.arctan2(direction[1], direction[0])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    positions = (track.positions - crossing) @ rotation.T + np.array([1.5, 0.0])
    return TrajectoryObstacleSpec(relative_time, positions, 0.27)


def _crossing_obstacle() -> tuple[ObstacleSpec | TrajectoryObstacleSpec, str]:
    path = project_root() / "datasets" / "thor" / "processed" / _THOR_CROSSING_RECORDING
    if path.exists():
        return thor_crossing_obstacle(path), f"THOR:{_THOR_CROSSING_TRACK}"
    return ObstacleSpec((1.5, -1.2), (0, 0.45), 0.27), "synthetic-fallback"


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
        obstacle, source = _crossing_obstacle()
        return Scenario(
            name,
            start,
            human[-1],
            PolylinePath(human),
            PolylinePath(autonomous),
            (obstacle,),
            obstacle_source=source,
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
