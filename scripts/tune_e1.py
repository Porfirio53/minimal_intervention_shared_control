#!/usr/bin/env python3
"""Tune paper-permitted E1 secondary parameters on designated development seeds."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.sim2d.simulation import (
    SimulationConfig,
    run_simulation,
)


@dataclass(frozen=True)
class TuneSpec:
    tolerance: float
    smooth_weight: float
    input_weight: float
    task_weight: float
    retry_alpha: float
    scenario: str
    network: str
    seed: int
    duration: float


def _run(spec: TuneSpec) -> dict[str, object]:
    result = run_simulation(
        spec.scenario,
        SimulationConfig(
            duration=spec.duration,
            method="ours",
            seed=spec.seed,
            network_condition=spec.network,
            risk_overrides={
                "lexicographic_tolerance": spec.tolerance,
                "secondary_smooth_weight": spec.smooth_weight,
                "secondary_input_weight": spec.input_weight,
                "secondary_task_weight": spec.task_weight,
                "fallback_relinearization_alpha": spec.retry_alpha,
            },
        ),
    )
    return {
        "lexicographic_tolerance": spec.tolerance,
        "secondary_smooth_weight": spec.smooth_weight,
        "secondary_input_weight": spec.input_weight,
        "secondary_task_weight": spec.task_weight,
        "fallback_relinearization_alpha": spec.retry_alpha,
        "scenario": spec.scenario,
        "network_condition": spec.network,
        "seed": spec.seed,
        "duration": spec.duration,
        **result.metrics,
    }


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    settings = sorted(
        {
            (
                float(row["lexicographic_tolerance"]),
                float(row["secondary_smooth_weight"]),
                float(row["secondary_input_weight"]),
                float(row["secondary_task_weight"]),
                float(row["fallback_relinearization_alpha"]),
            )
            for row in rows
        }
    )
    for tolerance, smooth_weight, input_weight, task_weight, retry_alpha in settings:
        selected = [
            row
            for row in rows
            if row["lexicographic_tolerance"] == tolerance
            and row["secondary_smooth_weight"] == smooth_weight
            and row["secondary_input_weight"] == input_weight
            and row["secondary_task_weight"] == task_weight
            and row["fallback_relinearization_alpha"] == retry_alpha
        ]

        candidates.append(
            {
                "lexicographic_tolerance": tolerance,
                "secondary_smooth_weight": smooth_weight,
                "secondary_input_weight": input_weight,
                "secondary_task_weight": task_weight,
                "fallback_relinearization_alpha": retry_alpha,
                "runs": len(selected),
                "collision_rate": float(
                    np.mean([row["collision"] for row in selected])
                ),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "filter_infeasible_rate": float(
                    np.mean([row["filter_infeasible_rate"] for row in selected])
                ),
                "upper_infeasible_rate": float(
                    np.mean([row["upper_infeasible_rate"] for row in selected])
                ),
                "secondary_fallback_rate": float(
                    np.mean([row["secondary_fallback_rate"] for row in selected])
                ),
                "intervention_budget": float(
                    np.mean([row["intervention_budget"] for row in selected])
                ),
                "nominal_modification": float(
                    np.mean([row["nominal_modification"] for row in selected])
                ),
                "filter_modification": float(
                    np.mean([row["filter_modification"] for row in selected])
                ),
                "authority_total_variation": float(
                    np.mean([row["authority_total_variation"] for row in selected])
                ),
                "completion_time": float(
                    np.mean([row["completion_time"] for row in selected])
                ),
            }
        )
    return candidates


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
        "--tolerances", nargs="+", type=float, default=[0.001, 0.01, 0.03, 0.05]
    )
    parser.add_argument("--smooth-weights", nargs="+", type=float, default=[0.2])
    parser.add_argument("--input-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--task-weights", nargs="+", type=float, default=[0.0, 0.2, 1.0]
    )
    parser.add_argument("--retry-alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["conflict_both_safe", "conflict_human_unsafe"],
    )
    parser.add_argument("--networks", nargs="+", default=["N0", "N1", "N2", "NJ"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[920001])
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/development/e1_parameter_search.json"),
    )
    args = parser.parse_args()
    specs = [
        TuneSpec(
            tolerance,
            smooth_weight,
            input_weight,
            task_weight,
            retry_alpha,
            scenario,
            network,
            seed,
            args.duration,
        )
        for tolerance in args.tolerances
        for smooth_weight in args.smooth_weights
        for input_weight in args.input_weights
        for task_weight in args.task_weights
        for retry_alpha in args.retry_alphas
        for scenario in args.scenarios
        for network in args.networks
        for seed in args.seeds
    ]
    rows: list[dict[str, object]] = []
    workers = (os.cpu_count() or 1) if args.jobs == 0 else args.jobs
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run, spec) for spec in specs]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"[{index}/{len(specs)}]", flush=True)
    candidates = _aggregate(rows)
    selected = min(candidates, key=_ranking)
    result = {
        "selection_order": [
            "minimum collision rate",
            "maximum success rate",
            "minimum filter infeasible rate",
            "minimum upper infeasible rate",
            "minimum intervention budget",
        ],
        "selected": selected,
        "candidates": sorted(candidates, key=_ranking),
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
