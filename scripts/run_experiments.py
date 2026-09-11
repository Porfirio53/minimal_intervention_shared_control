#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_intervention_shared_control.sim2d.scenarios import SCENARIO_NAMES
from minimal_intervention_shared_control.sim2d.simulation import (
    METHODS,
    SimulationConfig,
    run_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired 2-D shared-control experiments"
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIO_NAMES, default=list(SCENARIO_NAMES)
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--network", choices=("N0", "N1", "N2", "NJ"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/experiments.jsonl")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for scenario in args.scenarios:
            for seed in args.seeds:
                for method in args.methods:
                    config = SimulationConfig(
                        duration=args.duration,
                        method=method,
                        seed=seed,
                        network_condition=args.network,
                    )
                    result = run_simulation(scenario, config)
                    row = {
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        **result.metrics,
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(
                        f"{scenario:16s} seed={seed:3d} {method:16s} collision={row['collision']} clearance={row['minimum_clearance']:.3f}"
                    )


if __name__ == "__main__":
    main()
