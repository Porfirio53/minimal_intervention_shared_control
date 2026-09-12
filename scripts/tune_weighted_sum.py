#!/usr/bin/env python3
"""Select the weighted-sum baseline weight without using formal E1 seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.config import load_yaml
from minimal_intervention_shared_control.sim2d.simulation import (
    SimulationConfig,
    run_simulation,
)


@dataclass(frozen=True)
class SweepSpec:
    weight: float
    scenario: str
    network_condition: str
    seed: int
    duration: float


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run(spec: SweepSpec) -> dict[str, object]:
    result = run_simulation(
        spec.scenario,
        SimulationConfig(
            duration=spec.duration,
            method="weighted_sum",
            seed=spec.seed,
            network_condition=spec.network_condition,
            risk_overrides={"weighted_sum_intervention_weight": spec.weight},
        ),
    )
    return {**asdict(spec), **result.metrics}


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for weight in sorted({float(row["weight"]) for row in rows}):
        selected = [row for row in rows if float(row["weight"]) == weight]

        def mean(key: str, group: list[dict[str, object]] = selected) -> float:
            return float(np.mean([float(row[key]) for row in group]))

        output.append(
            {
                "weight": weight,
                "runs": len(selected),
                "collision_rate": mean("collision"),
                "success_rate": mean("success"),
                "filter_infeasible_rate": mean("filter_infeasible_rate"),
                "upper_infeasible_rate": mean("upper_infeasible_rate"),
                "intervention_budget": mean("intervention_budget"),
                "nominal_modification": mean("nominal_modification"),
                "filter_modification": mean("filter_modification"),
                "authority_total_variation": mean("authority_total_variation"),
            }
        )
    return output


def _ranking(row: dict[str, object]) -> tuple[float, ...]:
    return (
        float(row["collision_rate"]),
        -float(row["success_rate"]),
        float(row["filter_infeasible_rate"]),
        float(row["upper_infeasible_rate"]),
        float(row["intervention_budget"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/paper.yaml")
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0]
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[920001, 920002]
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper/weighted_sum_sensitivity.json"),
    )
    args = parser.parse_args()
    if args.jobs < 0:
        parser.error("--jobs must be non-negative; use 0 for all CPUs")
    if not args.weights or any(weight <= 0 for weight in args.weights):
        parser.error("all weights must be positive")
    if set(args.seeds) & set(range(50)):
        parser.error("formal E1 seeds 0-49 cannot be used for baseline selection")

    config = load_yaml(args.config)
    scenarios = [str(value) for value in config["scenarios"]]
    networks = [str(value) for value in config["network_conditions"]]
    duration = float(config["duration"])
    specs = [
        SweepSpec(weight, scenario, network, seed, duration)
        for weight in args.weights
        for scenario in scenarios
        for network in networks
        for seed in args.seeds
    ]
    rows: list[dict[str, object]] = []
    workers = (os.cpu_count() or 1) if args.jobs == 0 else args.jobs
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run, spec) for spec in specs]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 20 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}]", flush=True)

    candidates = _aggregate(rows)
    selected = min(candidates, key=_ranking)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "formal_seeds_used": False,
        "development_seeds": args.seeds,
        "selection_order": [
            "minimum collision rate",
            "maximum success rate",
            "minimum filter infeasible rate",
            "minimum upper infeasible rate",
            "minimum intervention budget",
        ],
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: float(row["weight"])),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
