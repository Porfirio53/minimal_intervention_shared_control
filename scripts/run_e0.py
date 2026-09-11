#!/usr/bin/env python3
"""Run leakage-controlled SCAND and THOR prediction/calibration experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.datasets.scand import (
    SCANDRun,
    load_scand_csv,
    split_runs,
)
from minimal_intervention_shared_control.datasets.thor import load_thor_tsv
from minimal_intervention_shared_control.evaluation.prediction_metrics import (
    gaussian_trajectory_metrics,
)
from minimal_intervention_shared_control.predictors.calibration import (
    CovarianceCalibrator,
)
from minimal_intervention_shared_control.predictors.human_ar import HumanAR


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _causal_resample(
    timestamps: np.ndarray, values: np.ndarray, interval: float
) -> np.ndarray:
    if interval <= 0 or len(timestamps) == 0 or len(timestamps) != len(values):
        raise ValueError("invalid causal-resampling inputs")
    count = int(np.floor((timestamps[-1] - timestamps[0]) / interval + 1e-9)) + 1
    targets = timestamps[0] + np.arange(count) * interval
    indices = np.searchsorted(timestamps, targets, side="right") - 1
    return np.asarray(values, dtype=float)[indices]


def _human_examples(
    sequences: list[np.ndarray], model: HumanAR, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth: list[np.ndarray] = []
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for sequence in sequences:
        for origin in range(model.order, len(sequence) - horizon + 1):
            prediction = model.predict(sequence[origin - model.order : origin], horizon)
            truth.append(sequence[origin : origin + horizon])
            means.append(prediction.mean)
            covariances.append(prediction.covariance)
    if not truth:
        raise ValueError("split has no complete human-prediction windows")
    return np.stack(truth), np.stack(means), np.stack(covariances)


def _component_metrics(truth: np.ndarray, mean: np.ndarray) -> dict[str, float]:
    error = truth - mean
    return {
        "mae_v": float(np.mean(np.abs(error[..., 0]))),
        "mae_omega": float(np.mean(np.abs(error[..., 1]))),
        "rmse_v": float(np.sqrt(np.mean(error[..., 0] ** 2))),
        "rmse_omega": float(np.sqrt(np.mean(error[..., 1] ** 2))),
    }


def _run_scand(
    directory: Path,
    output: Path,
    *,
    seed: int,
    test_driver: str,
    interval: float,
    horizon: int,
    order: int,
) -> dict[str, object]:
    paths = sorted(directory.glob("*.csv"))
    runs = [run for path in paths for run in load_scand_csv(path)]
    remaining_count = sum(run.driver_id != test_driver for run in runs)
    if remaining_count < 2:
        raise ValueError(
            "SCAND requires at least two non-test runs for train/validation"
        )
    validation_fraction = max(0.2, 1.0 / remaining_count)
    train, validation, test = split_runs(
        runs,
        test_driver=test_driver,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if not train or not validation or not test:
        raise ValueError("SCAND split must contain train, validation, and test runs")

    def sequences(selected: list[SCANDRun]) -> list[np.ndarray]:
        return [
            _causal_resample(run.timestamps, run.commands, interval) for run in selected
        ]

    train_sequences = sequences(train)
    validation_sequences = sequences(validation)
    test_sequences = sequences(test)
    model = HumanAR(order=order).fit_sequences(train_sequences)
    validation_truth, validation_mean, validation_covariance = _human_examples(
        validation_sequences, model, horizon
    )
    calibrator = CovarianceCalibrator(target_coverage=0.95, kappa=2.448)
    calibrator.fit(
        (validation_truth - validation_mean).reshape(-1, 2),
        validation_covariance.reshape(-1, 2, 2),
    )
    test_truth, test_mean, test_covariance = _human_examples(
        test_sequences, model, horizon
    )
    test_covariance = calibrator.transform(test_covariance)
    metrics = gaussian_trajectory_metrics(test_truth, test_mean, test_covariance)
    metrics.update(_component_metrics(test_truth, test_mean))
    np.savez_compressed(
        output / "scand_predictions.npz",
        truth=test_truth,
        mean=test_mean,
        covariance=test_covariance,
    )
    np.savez_compressed(
        output / "scand_human_ar.npz",
        coefficients=model.coef_,
        residual_covariance=model.residual_cov_,
        order=model.order,
        interval=interval,
        horizon=horizon,
        calibration_scale=calibrator.scale,
    )
    result: dict[str, object] = {
        "status": "pilot" if len(runs) < 10 else "dataset-run",
        "prediction_space": "control [v_m_per_s, omega_rad_per_s]",
        "seed": seed,
        "interval": interval,
        "horizon": horizon,
        "order": order,
        "training_windows": model.training_samples_,
        "validation_examples": len(validation_truth),
        "test_examples": len(test_truth),
        "calibration_scale": calibrator.scale,
        "train_runs": [run.run_id for run in train],
        "validation_runs": [run.run_id for run in validation],
        "test_runs": [run.run_id for run in test],
        "metrics": metrics,
        "limitations": [
            f"This is a selected {len(runs)}-run Jackal subset, not full SCAND.",
            "ADE/FDE are L2 distances in mixed control coordinates; component metrics retain physical units.",
            "SCAND has no network-delay labels; closed-loop experiments inject delays separately.",
        ],
    }
    _write_json(output / "scand_metrics.json", result)
    _write_json(
        output / "scand_human_ar.json",
        {
            "model": "HumanAR",
            "order": model.order,
            "interval": interval,
            "horizon": horizon,
            "coefficients": model.coef_.tolist(),
            "residual_covariance": model.residual_cov_.tolist(),
            "calibration_scale": calibrator.scale,
        },
    )
    return result


def _split_recordings(
    paths: list[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    if len(paths) < 5:
        raise ValueError("THOR requires at least five recordings")
    order = np.random.default_rng(seed).permutation(len(paths))
    test_count = max(1, round(0.2 * len(paths)))
    validation_count = max(1, round(0.2 * len(paths)))
    test_indices = set(order[:test_count].tolist())
    validation_indices = set(order[test_count : test_count + validation_count].tolist())
    train = [
        path
        for index, path in enumerate(paths)
        if index not in test_indices | validation_indices
    ]
    validation = [
        path for index, path in enumerate(paths) if index in validation_indices
    ]
    test = [path for index, path in enumerate(paths) if index in test_indices]
    return train, validation, test


def _thor_examples(
    paths: list[Path],
    *,
    interval: float,
    history_seconds: float,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_steps = round(history_seconds / interval)
    if history_steps < 1 or stride < 1:
        raise ValueError("THOR history and stride must be positive")
    truth: list[np.ndarray] = []
    means: list[np.ndarray] = []
    future_times = np.arange(1, horizon + 1) * interval
    for path in paths:
        for track in load_thor_tsv(path):
            positions = _causal_resample(track.timestamps, track.positions, interval)
            for origin in range(history_steps, len(positions) - horizon, stride):
                history = positions[origin - history_steps : origin + 1]
                velocity = (history[-1] - history[0]) / (history_steps * interval)
                means.append(history[-1] + future_times[:, None] * velocity)
                truth.append(positions[origin + 1 : origin + horizon + 1])
    if not truth:
        raise ValueError("THOR split has no complete prediction windows")
    return np.stack(truth), np.stack(means)


def _cv_variances(
    truth: np.ndarray, mean: np.ndarray, interval: float
) -> tuple[float, float, np.ndarray]:
    errors = truth - mean
    per_coordinate_mse = np.mean(errors**2, axis=(0, 2))
    times = np.arange(1, truth.shape[1] + 1) * interval
    design = np.column_stack([np.ones(len(times)), times**2])
    intercept, slope = np.linalg.lstsq(design, per_coordinate_mse, rcond=None)[0]
    initial_variance = max(float(intercept), 1e-8)
    process_variance = max(float(slope), 0.0)
    return (
        initial_variance,
        process_variance,
        initial_variance + process_variance * times**2,
    )


def _isotropic_covariance(sample_count: int, variances: np.ndarray) -> np.ndarray:
    covariance = np.zeros((sample_count, len(variances), 2, 2))
    covariance[..., 0, 0] = variances
    covariance[..., 1, 1] = variances
    return covariance


def _run_thor(
    directory: Path,
    output: Path,
    *,
    seed: int,
    interval: float,
    history_seconds: float,
    horizon: int,
    stride: int,
) -> dict[str, object]:
    paths = sorted(directory.glob("*.tsv"))
    train, validation, test = _split_recordings(paths, seed)
    parameters = {
        "interval": interval,
        "history_seconds": history_seconds,
        "horizon": horizon,
        "stride": stride,
    }
    train_truth, train_mean = _thor_examples(train, **parameters)
    initial_variance, process_variance, variances = _cv_variances(
        train_truth, train_mean, interval
    )
    validation_truth, validation_mean = _thor_examples(validation, **parameters)
    validation_covariance = _isotropic_covariance(len(validation_truth), variances)
    calibrator = CovarianceCalibrator(target_coverage=0.95, kappa=2.448)
    calibrator.fit(
        (validation_truth - validation_mean).reshape(-1, 2),
        validation_covariance.reshape(-1, 2, 2),
    )
    test_truth, test_mean = _thor_examples(test, **parameters)
    test_covariance = calibrator.transform(
        _isotropic_covariance(len(test_truth), variances)
    )
    metrics = gaussian_trajectory_metrics(test_truth, test_mean, test_covariance)
    np.savez_compressed(
        output / "thor_predictions.npz",
        truth=test_truth,
        mean=test_mean,
        covariance=test_covariance,
    )
    result: dict[str, object] = {
        "prediction_space": "position [m, m]",
        "seed": seed,
        **parameters,
        "train_examples": len(train_truth),
        "validation_examples": len(validation_truth),
        "test_examples": len(test_truth),
        "initial_std": float(np.sqrt(initial_variance)),
        "process_std": float(np.sqrt(process_variance)),
        "calibration_scale": calibrator.scale,
        "train_recordings": [path.stem for path in train],
        "validation_recordings": [path.stem for path in validation],
        "test_recordings": [path.stem for path in test],
        "metrics": metrics,
    }
    _write_json(output / "thor_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scand", type=Path, default=Path("datasets/scand/processed"))
    parser.add_argument("--thor", type=Path, default=Path("datasets/thor/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/e0"))
    parser.add_argument("--artifact-output", type=Path, default=Path("artifacts/e0"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--human-order", type=int, default=5)
    parser.add_argument("--test-driver", default="B")
    parser.add_argument("--thor-history", type=float, default=0.5)
    parser.add_argument("--thor-stride", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scand = _run_scand(
        args.scand,
        args.output,
        seed=args.seed,
        test_driver=args.test_driver,
        interval=args.interval,
        horizon=args.horizon,
        order=args.human_order,
    )
    thor = _run_thor(
        args.thor,
        args.output,
        seed=args.seed,
        interval=args.interval,
        history_seconds=args.thor_history,
        horizon=args.horizon,
        stride=args.thor_stride,
    )
    args.artifact_output.mkdir(parents=True, exist_ok=True)
    for name in ("scand_human_ar.json", "scand_metrics.json", "thor_metrics.json"):
        shutil.copy2(args.output / name, args.artifact_output / name)
    print(json.dumps({"scand": scand, "thor": thor}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
