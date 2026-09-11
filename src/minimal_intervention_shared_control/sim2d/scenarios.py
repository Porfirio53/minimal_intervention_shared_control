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


def _accelerating_crossing_obstacle() -> TrajectoryObstacleSpec:
    timestamps = np.arange(0.0, 12.0 + 0.05, 0.05)
    acceleration_end = 2.5
    accelerated_time = np.minimum(timestamps, acceleration_end)
    initial_y = -0.9
    initial_speed = 0.1
    acceleration = 0.2
    accelerated_y = (
        initial_y
        + initial_speed * accelerated_time
        + 0.5 * acceleration * accelerated_time**2
    )
    terminal_speed = initial_speed + acceleration * acceleration_end
    positions = np.column_stack(
        [
            np.full_like(timestamps, 1.55),
            accelerated_y
            + terminal_speed * np.maximum(timestamps - acceleration_end, 0.0),
        ]
    )
    return TrajectoryObstacleSpec(timestamps, positions, 0.3)


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
    conflict_variant = "conflict_human_unsafe" if name == "conflict" else name
    if conflict_variant.startswith("conflict_"):
        human = np.array([[0, 0], [1.3, 1.6], [2.7, 1.6], [4, 0]], dtype=float)
        autonomous = np.array([[0, 0], [1.3, -1.6], [2.7, -1.6], [4, 0]], dtype=float)
        obstacle_positions = {
            "conflict_both_safe": (2.0, 3.2),
            "conflict_human_unsafe": (2.0, 1.55),
            "conflict_autonomy_unsafe": (2.0, -1.55),
            "conflict_blend_unsafe": (2.0, 0.0),
        }
        if conflict_variant not in obstacle_positions:
            raise ValueError(f"unknown scenario: {conflict_variant}")
        return Scenario(
            name,
            start,
            human[-1],
            PolylinePath(human),
            PolylinePath(autonomous),
            (ObstacleSpec(obstacle_positions[conflict_variant], (0, 0), 0.3),),
        )
    if name == "network_anomaly":
        human = np.array([[0, 0], [1.0, 0], [1.55, -0.35], [3.0, 0]], dtype=float)
        autonomous = np.array([[0, 0], [1.2, 1.15], [2.2, 1.15], [3.0, 0]], dtype=float)
        return Scenario(
            name,
            start,
            human[-1],
            PolylinePath(human),
            PolylinePath(autonomous),
            (_accelerating_crossing_obstacle(),),
            "NJ",
            0.15,
            "synthetic-accelerating",
        )
    raise ValueError(f"unknown scenario: {name}")


SCENARIO_NAMES = (
    "bend",
    "crossing",
    "conflict",
    "conflict_both_safe",
    "conflict_human_unsafe",
    "conflict_autonomy_unsafe",
    "conflict_blend_unsafe",
    "network_anomaly",
)
