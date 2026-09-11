#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_intervention_shared_control.sim2d.safety_experiment import (
    FILTER_MODES,
    replay_safety_filter,
)
from minimal_intervention_shared_control.sim2d.scenarios import SCENARIO_NAMES
from minimal_intervention_shared_control.sim2d.simulation import (
    SimulationConfig,
    run_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-command E2 safety-filter isolation"
    )
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="network_anomaly")
    parser.add_argument("--tau-s", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/safety_isolation.jsonl")
    )
    args = parser.parse_args()
    source = run_simulation(
        args.scenario,
        SimulationConfig(duration=args.duration, method="human_filter", seed=args.seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for mode in FILTER_MODES:
            ages = [0.0] if mode != "robust" else args.tau_s
            for age in ages:
                result = replay_safety_filter(args.scenario, source.records, mode, age)
                row = {
                    "scenario": args.scenario,
                    "filter": mode,
                    "tau_s": age,
                    **result.metrics,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
