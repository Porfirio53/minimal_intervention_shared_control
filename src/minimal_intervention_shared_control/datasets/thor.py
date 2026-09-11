from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np


@dataclass(frozen=True)
class THORTrack:
    track_id: str
    timestamps: np.ndarray
    positions: np.ndarray


_MARKER_COLUMN = re.compile(r"^(?P<agent>.+?)\s+-\s+(?P<marker>\d+)\s+(?P<axis>[XYZ])$")


@dataclass(frozen=True)
class _TrackState:
    segment: int
    last_timestamp: float | None


def _read_layout(stream: TextIO) -> tuple[dict[str, str], list[str]]:
    """Read THOR metadata and return metadata plus the data-column header."""
    metadata: dict[str, str] = {}
    while line := stream.readline():
        fields = line.rstrip("\r\n").split("\t")
        if not fields or not fields[0].strip():
            continue
        key = fields[0].strip()
        if key == "Frame":
            if len(fields) < 3 or fields[1].strip() != "Time":
                raise ValueError("THOR data header must start with Frame and Time")
            header = [field.strip() for field in fields]
            while header and not header[-1]:
                header.pop()
            return metadata, header
        metadata[key] = "\t".join(field.strip() for field in fields[1:])
    raise ValueError("THOR file does not contain a Frame/Time data header")


def _marker_columns(header: list[str]) -> dict[str, list[dict[str, int]]]:
    grouped: dict[str, dict[str, dict[str, int]]] = {}
    for index, name in enumerate(header):
        match = _MARKER_COLUMN.match(name)
        if match is None:
            continue
        agent, marker, axis = (
            match.group("agent"),
            match.group("marker"),
            match.group("axis"),
        )
        grouped.setdefault(agent, {}).setdefault(marker, {})[axis] = index
    groups = {
        agent: [
            axes for _, axes in sorted(markers.items()) if set(axes) == {"X", "Y", "Z"}
        ]
        for agent, markers in grouped.items()
        if agent.startswith("Helmet_")
    }
    groups = {agent: markers for agent, markers in groups.items() if markers}
    if not groups:
        raise ValueError("THOR file contains no complete X/Y/Z marker groups")
    return groups


def _as_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def convert_thor_tsv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    scale: float = 1e-3,
    min_markers: int = 1,
    max_gap_factor: float = 1.5,
    track_prefix: str | None = None,
) -> int:
    """Convert a THOR Qualisys wide TSV and return output row count.

    ``scale`` converts the Qualisys coordinate unit to metres. A track is
    closed when fewer than ``min_markers`` are visible or when its timestamp
    gap exceeds ``max_gap_factor / FREQUENCY``. The next valid observation
    starts a new segment, so no missing interval is silently bridged.
    """
    if scale <= 0 or min_markers < 1 or max_gap_factor <= 0:
        raise ValueError("scale, min_markers, and max_gap_factor must be positive")
    input_path, output_path = Path(input_path), Path(output_path)
    with input_path.open(newline="", encoding="utf-8") as stream:
        metadata, header = _read_layout(stream)
        groups = _marker_columns(header)
        frequency = _as_float(metadata.get("FREQUENCY", ""))
        if frequency is None or frequency <= 0:
            raise ValueError("THOR metadata must contain a positive FREQUENCY")
        time_index = header.index("Time")
        gap_limit = max_gap_factor / frequency
        states = {agent: _TrackState(0, None) for agent in groups}
        prefix = track_prefix or input_path.stem
        rows_written = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target, delimiter="\t")
            writer.writerow(["id", "time", "x", "y"])
            reader = csv.reader(stream, delimiter="\t")
            for fields in reader:
                if not fields or len(fields) < len(header):
                    continue
                timestamp = _as_float(fields[time_index])
                if timestamp is None:
                    continue
                for agent, markers in groups.items():
                    points: list[tuple[float, float, float]] = []
                    for axes in markers:
                        values = [_as_float(fields[axes[axis]]) for axis in "XYZ"]
                        if any(value is None for value in values):
                            continue
                        point = tuple(float(value) for value in values)
                        if not np.all(np.asarray(point) == 0.0):
                            points.append(point)
                    state = states[agent]
                    valid = len(points) >= min_markers
                    if (
                        valid
                        and state.last_timestamp is not None
                        and timestamp - state.last_timestamp > gap_limit
                    ):
                        state = _TrackState(state.segment + 1, None)
                    if not valid:
                        if state.last_timestamp is not None:
                            states[agent] = _TrackState(state.segment + 1, None)
                        continue
                    position = np.mean(np.asarray(points), axis=0) * scale
                    track_id = f"{prefix}:{agent}:{state.segment:03d}"
                    writer.writerow([track_id, timestamp, position[0], position[1]])
                    rows_written += 1
                    states[agent] = _TrackState(state.segment, timestamp)
    return rows_written


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
