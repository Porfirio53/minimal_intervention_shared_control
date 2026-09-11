from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.datasets.scand import (
    load_scand_csv,
    split_runs,
)
from minimal_intervention_shared_control.datasets.thor import load_thor_tsv


def test_scand_loader_keeps_whole_runs(tmp_path: Path) -> None:
    path = tmp_path / "scand.csv"
    path.write_text(
        "run_id,driver_id,timestamp,x,y,yaw,v,omega\n"
        "r1,d1,0,0,0,0,0.5,0\n"
        "r1,d1,1,0.5,0,0,0.5,0\n"
        "r2,d2,0,0,0,0,0.4,0.1\n",
        encoding="utf-8",
    )
    runs = load_scand_csv(path)
    train, validation, test = split_runs(
        runs, test_driver="d2", validation_fraction=0.0
    )
    assert [run.run_id for run in train] == ["r1"]
    assert validation == []
    assert [run.driver_id for run in test] == ["d2"]


def test_thor_loader_groups_and_sorts_tracks(tmp_path: Path) -> None:
    path = tmp_path / "thor.tsv"
    path.write_text(
        "id\ttime\tx\ty\na\t1\t1\t0\na\t0\t0\t0\nb\t0\t2\t3\n", encoding="utf-8"
    )
    tracks = load_thor_tsv(path)
    assert len(tracks) == 2
    np.testing.assert_allclose(tracks[0].positions, [[0, 0], [1, 0]])
