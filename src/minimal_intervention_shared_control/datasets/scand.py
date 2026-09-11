from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SCANDRun:
    run_id: str
    driver_id: str
    timestamps: np.ndarray
    states: np.ndarray
    commands: np.ndarray


REQUIRED_COLUMNS = ("run_id", "driver_id", "timestamp", "x", "y", "yaw", "v", "omega")


def load_scand_csv(path: str | Path) -> list[SCANDRun]:
    """Load the normalized output produced by scripts/preprocess_scand.py."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"SCAND file is missing columns: {sorted(missing)}")
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in reader:
            grouped.setdefault((row["run_id"], row["driver_id"]), []).append(row)
    runs = []
    for (run_id, driver_id), rows in grouped.items():
        rows.sort(key=lambda item: float(item["timestamp"]))
        timestamps = np.array([float(row["timestamp"]) for row in rows])
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"timestamps are not strictly increasing in run {run_id}")
        states = np.array(
            [[float(row[name]) for name in ("x", "y", "yaw")] for row in rows]
        )
        commands = np.array(
            [[float(row[name]) for name in ("v", "omega")] for row in rows]
        )
        runs.append(SCANDRun(run_id, driver_id, timestamps, states, commands))
    return runs


def split_runs(
    runs: list[SCANDRun],
    test_driver: str | None = None,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[SCANDRun], list[SCANDRun], list[SCANDRun]]:
    """Split whole runs, optionally leaving one driver out as the test set."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    test = [
        run for run in runs if test_driver is not None and run.driver_id == test_driver
    ]
    remaining = [
        run for run in runs if test_driver is None or run.driver_id != test_driver
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(remaining))
    validation_count = round(validation_fraction * len(remaining))
    validation_indices = set(order[:validation_count].tolist())
    train = [run for i, run in enumerate(remaining) if i not in validation_indices]
    validation = [run for i, run in enumerate(remaining) if i in validation_indices]
    return train, validation, test
