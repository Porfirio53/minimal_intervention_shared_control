"""Small paired E1 batches; retain every trial, configuration and source hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from minimal_intervention_shared_control.config import load_default, load_yaml
from minimal_intervention_shared_control.sim2d.simulation import (
    SimulationConfig,
    run_simulation,
)


def run_one(spec: dict) -> dict:
    result = run_simulation(
        spec["scenario"],
        SimulationConfig(
            duration=spec["duration"],
            method=spec["method"],
            seed=spec["seed"],
            network_condition=spec["network_condition"],
            risk_overrides=spec["overrides"],
        ),
    )
    row = {**spec, **result.metrics}
    if spec["trace"]:
        row["records"] = result.records
    return row


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "n": len(selected),
                "collisions": sum(row["collision"] for row in selected),
                "successes": sum(row["success"] for row in selected),
                **{
                    metric: float(np.mean([row[metric] for row in selected]))
                    for metric in (
                        "intervention_budget",
                        "executed_modification",
                        "filter_modification",
                        "authority_total_variation",
                        "upper_infeasible_rate",
                    )
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[940001])
    parser.add_argument("--networks", nargs="+", default=["N0", "NJ"])
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "delay_agreement",
            "single_step",
            "weighted_sum",
            "human_filter",
            "ours",
        ],
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument(
        "--overrides", default="{}", help="JSON risk overrides shared by methods"
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", help="Development sweep; weighted_sum only"
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new name to preserve trial history")
    if set(args.seeds) & set(range(50)):
        parser.error("formal seeds 0--49 are reserved")
    scenarios = (
        args.scenarios or load_yaml("configs/experiments/paper.yaml")["scenarios"]
    )
    overrides = json.loads(args.overrides)
    if args.weights and args.methods != ["weighted_sum"]:
        parser.error("--weights requires --methods weighted_sum")
    specs = [
        {
            "scenario": s,
            "network_condition": n,
            "seed": seed,
            "method": m,
            "duration": args.duration,
            "overrides": overrides,
            "trace": args.trace,
        }
        for s in scenarios
        for n in args.networks
        for seed in args.seeds
        for m in args.methods
    ]
    if args.weights:
        specs = [
            {
                **spec,
                "overrides": {**overrides, "weighted_sum_intervention_weight": weight},
            }
            for weight in args.weights
            for spec in specs
        ]
    source_paths = (
        sorted(Path("src").rglob("*.py"))
        + sorted(Path("configs").rglob("*.yaml"))
        + sorted(Path("artifacts").rglob("*.json"))
    )
    manifest = {
        "specs": specs,
        "risk": {**load_default("risk"), **overrides},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    started = perf_counter()
    rows = []
    with (
        args.output.open("w") as stream,
        ProcessPoolExecutor(max_workers=args.jobs) as executor,
    ):
        futures = [executor.submit(run_one, spec) for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    print(
        json.dumps(
            {"seconds": perf_counter() - started, "summary": summarize(rows)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
