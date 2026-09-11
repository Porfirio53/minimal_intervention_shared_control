#!/usr/bin/env python3
"""Run and summarize the paper-scale WSL E1 experiment with resume support."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/paper.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/full"))
    args = parser.parse_args()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e0_directory = args.output_dir / "e0"
    e0_directory.mkdir(parents=True, exist_ok=True)
    selection = Path("results/development/e0_selection.json")
    if selection.exists():
        shutil.copy2(selection, e0_directory / "model_selection.json")
    e1_rows = args.output_dir / "e1_runs.jsonl"
    _run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--config",
            str(args.config),
            "--jobs",
            str(args.jobs),
            "--resume",
            "--output",
            str(e1_rows),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            str(e1_rows),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    print(f"complete: paper-ready E0/E1 outputs are under {args.output_dir}")


if __name__ == "__main__":
    main()
