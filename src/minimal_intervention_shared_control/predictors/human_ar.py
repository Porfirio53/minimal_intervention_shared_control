from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import Control, FloatArray, GaussianTrajectory


class HumanAR:
    """Small autoregressive Gaussian model for joystick commands.

    It is intentionally transparent: the fitted coefficients can be inspected and the
    model is suitable for the first SCAND calibration pass.
    """

    def __init__(
        self,
        order: int = 5,
        ridge: float = 1e-3,
        min_std: float = 0.02,
        calibration_scale: float = 1.0,
        sample_interval: float | None = None,
    ):
        if order < 1 or not np.isfinite(ridge) or not np.isfinite(min_std):
            raise ValueError("order must be positive")
        if ridge < 0 or min_std < 0:
            raise ValueError("ridge and min_std must be non-negative")
        if (
            not np.isfinite(calibration_scale)
            or calibration_scale < 1
            or (sample_interval is not None and sample_interval <= 0)
            or (sample_interval is not None and not np.isfinite(sample_interval))
        ):
            raise ValueError("invalid calibration scale or sample interval")
        self.order, self.ridge, self.min_std = order, ridge, min_std
        self.calibration_scale = float(calibration_scale)
        self.sample_interval = sample_interval
        self.coef_: FloatArray | None = None
        self.residual_cov_: FloatArray = np.eye(2) * 0.05**2
        self.training_samples_: int = 0

    def fit(self, controls: FloatArray) -> HumanAR:
        return self.fit_sequences([controls])

    def fit_sequences(self, sequences: list[FloatArray]) -> HumanAR:
        """Fit complete control sequences without creating cross-run windows."""
        predictors: list[FloatArray] = []
        targets: list[FloatArray] = []
        for controls in sequences:
            data = np.asarray(controls, dtype=float)
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("each control sequence must have shape (samples, 2)")
            predictors.extend(
                data[i - self.order : i].reshape(-1)
                for i in range(self.order, len(data))
            )
            targets.extend(data[self.order :])
        if len(predictors) < 2:
            raise ValueError(
                "sequences must provide at least two autoregressive windows"
            )
        x, y = np.stack(predictors), np.stack(targets)
        design = np.column_stack([x, np.ones(len(x))])
        regularizer = self.ridge * np.eye(design.shape[1])
        regularizer[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        residual = y - design @ self.coef_
        self.residual_cov_ = (
            np.atleast_2d(np.cov(residual.T)) + np.eye(2) * self.min_std**2
        )
        self.training_samples_ = len(x)
        return self

    @classmethod
    def load(cls, path: str | Path) -> HumanAR:
        """Load the transparent E0 artifact written by ``scripts/run_e0.py``."""
        artifact_path = Path(path)
        if artifact_path.suffix == ".json":
            with artifact_path.open(encoding="utf-8") as stream:
                artifact = json.load(stream)
            order = int(artifact["order"])
            model = cls(
                order=order,
                calibration_scale=float(artifact["calibration_scale"]),
                sample_interval=float(artifact["interval"]),
            )
            coefficients = np.asarray(artifact["coefficients"], dtype=float)
            residual = np.asarray(artifact["residual_covariance"], dtype=float)
        else:
            with np.load(artifact_path, allow_pickle=False) as artifact:
                order = int(artifact["order"])
                model = cls(
                    order=order,
                    calibration_scale=float(artifact["calibration_scale"]),
                    sample_interval=float(artifact["interval"]),
                )
                coefficients = np.asarray(artifact["coefficients"], dtype=float)
                residual = np.asarray(artifact["residual_covariance"], dtype=float)
        if coefficients.shape != (2 * order + 1, 2):
            raise ValueError("HumanAR artifact has incompatible coefficients")
        if (
            residual.shape != (2, 2)
            or not np.all(np.isfinite(coefficients))
            or not np.all(np.isfinite(residual))
            or not np.allclose(residual, residual.T, atol=1e-10)
            or float(np.linalg.eigvalsh(residual).min()) < -1e-10
        ):
            raise ValueError("HumanAR artifact has invalid residual covariance")
        model.coef_ = coefficients
        model.residual_cov_ = residual
        return model

    def joint_prediction_covariance(self, horizon: int) -> FloatArray:
        """Return the joint covariance of all future controls.

        The lifted covariance retains correlations between prediction times. Those
        correlations matter when command uncertainty is propagated through vehicle
        dynamics to obtain the positional intent covariance ``Sigma_H``.
        """
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        dimension = 2 * self.order
        transition = np.zeros((dimension, dimension))
        if self.order > 1:
            transition[:-2, 2:] = np.eye(dimension - 2)
        for lag in range(self.order):
            transition[-2:, 2 * lag : 2 * lag + 2] = self.coef_[2 * lag : 2 * lag + 2].T
        selector = np.zeros((2, dimension))
        selector[:, -2:] = np.eye(2)
        state_sensitivity = np.zeros((dimension, horizon * 2))
        output_sensitivity = np.zeros((horizon * 2, horizon * 2))
        for index in range(horizon):
            state_sensitivity = transition @ state_sensitivity
            current = slice(2 * index, 2 * index + 2)
            state_sensitivity[:, current] += selector.T
            output_sensitivity[current] = selector @ state_sensitivity
        innovation_covariance = np.kron(np.eye(horizon), self.residual_cov_)
        joint = output_sensitivity @ innovation_covariance @ output_sensitivity.T
        return 0.5 * (joint + joint.T) * self.calibration_scale

    def _prediction_covariance(self, horizon: int) -> FloatArray:
        joint = self.joint_prediction_covariance(horizon)
        return np.stack(
            [
                joint[2 * index : 2 * index + 2, 2 * index : 2 * index + 2]
                for index in range(horizon)
            ]
        )

    def predict(self, history: FloatArray, horizon: int) -> GaussianTrajectory:
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        values = np.asarray(history, dtype=float).copy()
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or len(values) < self.order
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("history must contain at least order controls")
        mean = np.empty((horizon, 2), dtype=float)
        for i in range(horizon):
            features = np.r_[values[-self.order :].reshape(-1), 1.0]
            next_value = features @ self.coef_
            mean[i] = next_value
            values = np.vstack([values, next_value])
        covariance = self._prediction_covariance(horizon)
        return GaussianTrajectory(mean, covariance)

    def predict_constant(self, control: Control, horizon: int) -> GaussianTrajectory:
        mean = np.repeat(control.as_array()[None, :], horizon, axis=0)
        covariance = np.repeat(
            (self.residual_cov_ * self.calibration_scale)[None, :, :], horizon, axis=0
        )
        return GaussianTrajectory(mean, covariance)
