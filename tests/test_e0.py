import importlib.util
from pathlib import Path

import numpy as np


def _load_run_e0():
    path = Path(__file__).parents[1] / "scripts" / "run_e0.py"
    spec = importlib.util.spec_from_file_location("run_e0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(path: Path, run_id: str, driver: str, offset: float) -> None:
    rows = ["run_id,driver_id,timestamp,x,y,yaw,v,omega"]
    for index in range(35):
        value = offset + 0.05 * np.sin(index / 3)
        rows.append(f"{run_id},{driver},{index / 10},0,0,0,{value},{value / 2}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_causal_resampling_never_uses_a_future_value() -> None:
    module = _load_run_e0()
    values = module._causal_resample(
        np.array([0.0, 0.11, 0.21]), np.array([[0.0], [1.0], [2.0]]), 0.1
    )
    np.testing.assert_allclose(values[:, 0], [0.0, 0.0, 1.0])


def test_scand_calibration_and_model_fit_do_not_use_test_driver(tmp_path: Path) -> None:
    module = _load_run_e0()
    data = tmp_path / "data"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    data.mkdir()
    first_output.mkdir()
    second_output.mkdir()
    for index in range(3):
        _write_run(data / f"a{index}.csv", f"a{index}", "A", 0.2 + index / 10)
    _write_run(data / "b.csv", "b", "B", 0.5)
    first = module._run_scand(
        data,
        first_output,
        seed=7,
        test_driver="B",
        interval=0.1,
        horizon=3,
        order=2,
    )
    _write_run(data / "b.csv", "b", "B", 5.0)
    second = module._run_scand(
        data,
        second_output,
        seed=7,
        test_driver="B",
        interval=0.1,
        horizon=3,
        order=2,
    )
    assert first["train_runs"] == second["train_runs"]
    assert first["validation_runs"] == second["validation_runs"]
    assert first["test_runs"] == second["test_runs"] == ["b"]
    assert first["calibration_scale"] == second["calibration_scale"]
    assert (first_output / "scand_human_ar.json").read_bytes() == (
        second_output / "scand_human_ar.json"
    ).read_bytes()
    assert first["metrics"] != second["metrics"]


def test_thor_split_is_deterministic_and_has_no_recording_leakage() -> None:
    module = _load_run_e0()
    paths = [Path(f"run-{index}.tsv") for index in range(13)]
    first = module._split_recordings(paths, seed=17)
    second = module._split_recordings(paths, seed=17)
    assert first == second
    sets = [set(split) for split in first]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(paths)
