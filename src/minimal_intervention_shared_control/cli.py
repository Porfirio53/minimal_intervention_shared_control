from __future__ import annotations

import argparse
import json

from .sim2d.scenarios import SCENARIO_NAMES
from .sim2d.simulation import METHODS, SimulationConfig, run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a 2-D delayed shared-control simulation"
    )
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="crossing")
    parser.add_argument("--method", choices=METHODS, default="ours")
    parser.add_argument("--network", choices=("N0", "N1", "N2", "NJ"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--trajectory", action="store_true", help="include per-step records"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_simulation(
        args.scenario,
        SimulationConfig(
            duration=args.duration,
            method=args.method,
            network_condition=args.network,
            seed=args.seed,
        ),
    )
    output: dict[str, object] = {
        "scenario": result.scenario,
        "method": result.method,
        "seed": result.seed,
        "metrics": result.metrics,
    }
    if args.trajectory:
        output["records"] = result.records
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
