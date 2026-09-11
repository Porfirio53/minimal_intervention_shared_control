#!/usr/bin/env python3
"""Select compact E0 model hyperparameters without inspecting held-out test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_e0 import (
    _causal_resample,
    _cv_variances,
    _human_examples,
    _isotropic_covariance,
    _split_recordings,
    _thor_examples,
)

from minimal_intervention_shared_control.datasets.scand import (
    load_scand_csv,
    split_runs,
)
from minimal_intervention_shared_control.evaluation.prediction_metrics import (
    gaussian_trajectory_metrics,
)
from minimal_intervention_shared_control.predictors.human_ar import HumanAR


def _scand_selection(
    directory: Path,
    *,
    seed: int,
    test_driver: str,
    interval: float,
    horizon: int,
    orders: list[int],
) -> tuple[int, list[dict[str, float | int]]]:
    runs = [
        run for path in sorted(directory.glob("*.csv")) for run in load_scand_csv(path)
    ]
    remaining_count = sum(run.driver_id != test_driver for run in runs)
    train, _, _ = split_runs(
        runs,
        test_driver=test_driver,
        validation_fraction=max(0.2, 1.0 / remaining_count),
        seed=seed,
    )
    if len(train) < 3:
        raise ValueError("SCAND tuning requires at least three training runs")

    def sequence(run):
        return _causal_resample(run.timestamps, run.commands, interval)

    sequences = [sequence(run) for run in train]
    candidates: list[dict[str, float | int]] = []
    for order in orders:
        fold_truth: list[np.ndarray] = []
        fold_mean: list[np.ndarray] = []
        fold_covariance: list[np.ndarray] = []
        for selection_index in range(len(train)):
            fitting_sequences = (
                sequences[:selection_index] + sequences[selection_index + 1 :]
            )
            model = HumanAR(order=order).fit_sequences(fitting_sequences)
            truth, mean, covariance = _human_examples(
                [sequences[selection_index]], model, horizon
            )
            fold_truth.append(truth)
            fold_mean.append(mean)
            fold_covariance.append(covariance)
        truth = np.concatenate(fold_truth)
        mean = np.concatenate(fold_mean)
        covariance = np.concatenate(fold_covariance)
        metrics = gaussian_trajectory_metrics(truth, mean, covariance)
        error = truth - mean
        candidates.append(
            {
                "order": order,
                "selection_examples": len(truth),
                "nll": metrics["nll"],
                "mae_v": float(np.mean(np.abs(error[..., 0]))),
                "mae_omega": float(np.mean(np.abs(error[..., 1]))),
                "selection_runs": len(train),
            }
        )
    selected = min(candidates, key=lambda row: (float(row["nll"]), int(row["order"])))
    return int(selected["order"]), candidates


def _thor_selection(
    directory: Path,
    *,
    seed: int,
    interval: float,
    horizon: int,
    histories: list[float],
    stride: int,
) -> tuple[float, list[dict[str, object]]]:
    train, _, _ = _split_recordings(sorted(directory.glob("*.tsv")), seed)
    if len(train) < 4:
        raise ValueError("THOR tuning requires at least four training recordings")
    shuffled = list(np.random.default_rng(seed + 1).permutation(train))
    fitting, selection = shuffled[:-2], shuffled[-2:]
    candidates: list[dict[str, object]] = []
    for history in histories:
        parameters = {
            "interval": interval,
            "history_seconds": history,
            "horizon": horizon,
            "stride": stride,
        }
        train_truth, train_mean = _thor_examples(fitting, **parameters)
        _, _, variances = _cv_variances(train_truth, train_mean, interval)
        truth, mean = _thor_examples(selection, **parameters)
        covariance = _isotropic_covariance(len(truth), variances)
        metrics = gaussian_trajectory_metrics(truth, mean, covariance)
        candidates.append(
            {
                "history_seconds": history,
                "selection_examples": len(truth),
                "nll": metrics["nll"],
                "ade": metrics["ade"],
                "fde": metrics["fde"],
                "selection_recordings": [path.stem for path in selection],
            }
        )
    selected = min(
        candidates,
        key=lambda row: (float(row["nll"]), float(row["history_seconds"])),
    )
    return float(selected["history_seconds"]), candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scand", type=Path, default=Path("datasets/scand/processed"))
    parser.add_argument("--thor", type=Path, default=Path("datasets/thor/processed"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/development/e0_selection.json")
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--test-driver", default="B")
    parser.add_argument("--human-orders", nargs="+", type=int, default=[1, 3, 5, 8, 10])
    parser.add_argument(
        "--thor-histories", nargs="+", type=float, default=[0.3, 0.5, 0.8, 1.0]
    )
    parser.add_argument("--thor-stride", type=int, default=5)
    args = parser.parse_args()
    human_order, human_candidates = _scand_selection(
        args.scand,
        seed=args.seed,
        test_driver=args.test_driver,
        interval=args.interval,
        horizon=args.horizon,
        orders=args.human_orders,
    )
    thor_history, thor_candidates = _thor_selection(
        args.thor,
        seed=args.seed,
        interval=args.interval,
        horizon=args.horizon,
        histories=args.thor_histories,
        stride=args.thor_stride,
    )
    result = {
        "selection_rule": "minimum uncalibrated validation NLL; test partitions untouched",
        "seed": args.seed,
        "selected_human_order": human_order,
        "selected_thor_history_seconds": thor_history,
        "scand_candidates": human_candidates,
        "thor_candidates": thor_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
