from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.sim2d.agents import TrajectoryObstacle
from minimal_intervention_shared_control.sim2d.scenarios import thor_crossing_obstacle


def test_thor_crossing_placement_is_a_rigid_transform(tmp_path: Path) -> None:
    timestamps = np.arange(0.0, 4.1, 0.1)
    positions = np.column_stack([timestamps, 0.2 * timestamps**2])
    path = tmp_path / "track.tsv"
    rows = ["id\ttime\tx\ty"] + [
        f"Exp_2_run_3:Helmet_10:003\t{time}\t{x}\t{y}"
        for time, (x, y) in zip(timestamps, positions, strict=True)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    placed = thor_crossing_obstacle(path)
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(placed.positions, axis=0), axis=1),
        np.linalg.norm(np.diff(positions, axis=0), axis=1),
    )
    crossing = np.array(
        [
            np.interp(2.7, placed.timestamps, placed.positions[:, axis])
            for axis in range(2)
        ]
    )
    np.testing.assert_allclose(crossing, [1.5, 0.0], atol=1e-12)


def test_trajectory_obstacle_velocity_uses_only_past_positions() -> None:
    obstacle = TrajectoryObstacle(
        np.array([0.0, 0.5, 1.0]),
        np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 0.0]]),
    )
    np.testing.assert_allclose(obstacle.velocity, 0.0)
    obstacle.step(0.5)
    np.testing.assert_allclose(obstacle.velocity, [1.0, 0.0])
