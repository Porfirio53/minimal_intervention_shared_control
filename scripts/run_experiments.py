#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from minimal_intervention_shared_control.config import load_yaml
from minimal_intervention_shared_control.sim2d.scenarios import SCENARIO_NAMES
from minimal_intervention_shared_control.sim2d.simulation import (
    METHODS,
    SimulationConfig,
    run_simulation,
)

NETWORKS = ("N0", "N1", "N2", "NJ")


@dataclass(frozen=True)
class ExperimentSpec:
    scenario: str
    method: str
    seed: int
    network_condition: str
    duration: float

    @property
    def key(self) -> tuple[str, str, int, str, float]:
        return (
            self.scenario,
            self.method,
            self.seed,
            self.network_condition,
            self.duration,
        )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _status_counts(records: list[dict[str, object]], key: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get(key, "missing")) for row in records).items())
    )


def _require_finite(value: object, path: str = "row") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        raise ValueError(f"non-finite experiment output at {path}: {value}")


def _run_one(spec: ExperimentSpec) -> dict[str, object]:
    result = run_simulation(
        spec.scenario,
        SimulationConfig(
            duration=spec.duration,
            method=spec.method,
            seed=spec.seed,
            network_condition=spec.network_condition,
        ),
    )
    upper_records = [row for row in result.records if bool(row["upper_updated"])]
    row: dict[str, object] = {
        **asdict(spec),
        "steps": len(result.records),
        **result.metrics,
        "upper_status_counts": _status_counts(upper_records, "qp_status"),
        "stage1_status_counts": _status_counts(upper_records, "stage1_status"),
        "stage2_status_counts": _status_counts(upper_records, "stage2_status"),
        "filter_status_counts": _status_counts(result.records, "filter_status"),
        "human_prediction_sources": _status_counts(
            result.records, "human_prediction_source"
        ),
        "obstacle_prediction_sources": _status_counts(
            result.records, "obstacle_prediction_source"
        ),
        "scenario_obstacle_sources": _status_counts(
            result.records, "scenario_obstacle_source"
        ),
    }
    _require_finite(row)
    return row


def _seed_values(value: object, seed_start: int) -> list[int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError("seed count must be positive")
        return list(range(seed_start, seed_start + value))
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, int) for item in value)
    ):
        return value
    raise TypeError("seeds must be a positive count or a non-empty integer list")


def _existing_keys(path: Path) -> set[tuple[str, str, int, str, float]]:
    keys: set[tuple[str, str, int, str, float]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                keys.add(
                    (
                        str(row["scenario"]),
                        str(row["method"]),
                        int(row["seed"]),
                        str(row["network_condition"]),
                        float(row["duration"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid resume row {path}:{line_number}") from error
    return keys


def _resolve_settings(args: argparse.Namespace) -> dict[str, object]:
    configured: dict[str, Any] = load_yaml(args.config) if args.config else {}
    scenarios = args.scenarios or configured.get("scenarios") or list(SCENARIO_NAMES)
    methods = args.methods or configured.get("methods") or list(METHODS)
    configured_networks = configured.get("network_conditions")
    if configured_networks is None and configured.get("network") is not None:
        configured_networks = [configured["network"]]
    networks = args.networks or ([args.network] if args.network else None)
    networks = networks or configured_networks or ["N1"]
    duration = (
        args.duration if args.duration is not None else configured.get("duration", 10.0)
    )
    seed_setting = (
        args.seeds if args.seeds is not None else configured.get("seeds", [0, 1, 2])
    )
    settings = {
        "scenarios": list(scenarios),
        "methods": list(methods),
        "network_conditions": list(networks),
        "seeds": _seed_values(seed_setting, args.seed_start),
        "duration": float(duration),
    }
    unknown_scenarios = set(settings["scenarios"]) - set(SCENARIO_NAMES)
    unknown_methods = set(settings["methods"]) - set(METHODS)
    unknown_networks = set(settings["network_conditions"]) - set(NETWORKS)
    if unknown_scenarios or unknown_methods or unknown_networks:
        raise ValueError(
            "unknown experiment values: "
            f"scenarios={sorted(unknown_scenarios)}, "
            f"methods={sorted(unknown_methods)}, networks={sorted(unknown_networks)}"
        )
    if settings["duration"] <= 0:
        raise ValueError("duration must be positive")
    return settings


def _write_manifest(
    path: Path,
    settings: dict[str, object],
    *,
    requested: int,
    completed: int,
    skipped: int,
    started_at: str,
) -> None:
    manifest = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "requested_runs": requested,
        "completed_runs": completed,
        "skipped_existing_runs": skipped,
        "settings": settings,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired 2-D shared-control experiments"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_NAMES)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--network", choices=NETWORKS)
    parser.add_argument("--networks", nargs="+", choices=NETWORKS)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/experiments.jsonl")
    )
    args = parser.parse_args()
    if args.network and args.networks:
        parser.error("use only one of --network and --networks")
    if args.jobs < 0:
        parser.error("--jobs must be non-negative; use 0 for all CPUs")

    settings = _resolve_settings(args)
    specs = [
        ExperimentSpec(scenario, method, seed, network, settings["duration"])
        for scenario in settings["scenarios"]
        for network in settings["network_conditions"]
        for seed in settings["seeds"]
        for method in settings["methods"]
    ]
    existing = _existing_keys(args.output) if args.resume else set()
    pending = [spec for spec in specs if spec.key not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    started_at = datetime.now(UTC).isoformat()
    completed = 0
    mode = "a" if args.resume and args.output.exists() else "w"
    worker_count = (os.cpu_count() or 1) if args.jobs == 0 else args.jobs
    print(
        f"requested={len(specs)} pending={len(pending)} skipped={len(specs) - len(pending)} "
        f"workers={worker_count} output={args.output}"
    )
    _write_manifest(
        manifest_path,
        settings,
        requested=len(specs),
        completed=0,
        skipped=len(specs) - len(pending),
        started_at=started_at,
    )
    try:
        with args.output.open(mode, encoding="utf-8") as stream:
            if worker_count == 1:
                for spec in pending:
                    row = _run_one(spec)
                    stream.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    completed += 1
                    print(
                        f"[{completed}/{len(pending)}] {spec.scenario:24s} "
                        f"{spec.network_condition} seed={spec.seed:3d} "
                        f"{spec.method:16s} collision={row['collision']} "
                        f"clearance={row['minimum_clearance']:.3f}"
                    )
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = {
                        executor.submit(_run_one, spec): spec for spec in pending
                    }
                    for future in as_completed(futures):
                        spec = futures[future]
                        row = future.result()
                        stream.write(
                            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                        )
                        stream.flush()
                        completed += 1
                        print(
                            f"[{completed}/{len(pending)}] {spec.scenario:24s} "
                            f"{spec.network_condition} seed={spec.seed:3d} "
                            f"{spec.method:16s} collision={row['collision']} "
                            f"clearance={row['minimum_clearance']:.3f}"
                        )
    finally:
        _write_manifest(
            manifest_path,
            settings,
            requested=len(specs),
            completed=completed,
            skipped=len(specs) - len(pending),
            started_at=started_at,
        )


if __name__ == "__main__":
    main()
