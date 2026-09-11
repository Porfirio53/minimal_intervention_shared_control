from __future__ import annotations

import numpy as np

from ..types import Control, FloatArray, GaussianTrajectory


class HumanAR:
    """Small autoregressive Gaussian model for joystick commands.

    It is intentionally transparent: the fitted coefficients can be inspected and the
    model is suitable for the first SCAND calibration pass.
    """

    def __init__(self, order: int = 5, ridge: float = 1e-3, min_std: float = 0.02):
        if order < 1:
            raise ValueError("order must be positive")
        self.order, self.ridge, self.min_std = order, ridge, min_std
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

    def _prediction_covariance(self, horizon: int) -> FloatArray:
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        dimension = 2 * self.order
        transition = np.zeros((dimension, dimension))
        if self.order > 1:
            transition[:-2, 2:] = np.eye(dimension - 2)
        for lag in range(self.order):
            transition[-2:, 2 * lag : 2 * lag + 2] = self.coef_[2 * lag : 2 * lag + 2].T
        process_covariance = np.zeros((dimension, dimension))
        process_covariance[-2:, -2:] = self.residual_cov_
        state_covariance = np.zeros_like(process_covariance)
        covariance = np.empty((horizon, 2, 2), dtype=float)
        for index in range(horizon):
            state_covariance = (
                transition @ state_covariance @ transition.T + process_covariance
            )
            covariance[index] = state_covariance[-2:, -2:]
        return covariance

    def predict(self, history: FloatArray, horizon: int) -> GaussianTrajectory:
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        values = np.asarray(history, dtype=float).copy()
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < self.order:
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
        covariance = np.repeat(self.residual_cov_[None, :, :], horizon, axis=0)
        return GaussianTrajectory(mean, covariance)
