from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class THORTrack:
    track_id: str
    timestamps: np.ndarray
    positions: np.ndarray


def load_thor_tsv(
    path: str | Path,
    track_column: str = "id",
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
) -> list[THORTrack]:
    """Load THOR tracks while allowing release-specific column names."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {track_column, time_column, x_column, y_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"THOR file is missing columns: {sorted(missing)}")
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            grouped.setdefault(row[track_column], []).append(row)
    tracks = []
    for track_id, rows in grouped.items():
        rows.sort(key=lambda item: float(item[time_column]))
        timestamps = np.array([float(row[time_column]) for row in rows])
        positions = np.array(
            [[float(row[x_column]), float(row[y_column])] for row in rows]
        )
        tracks.append(THORTrack(track_id, timestamps, positions))
    return tracks
