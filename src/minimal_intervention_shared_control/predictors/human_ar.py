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

    def fit(self, controls: FloatArray) -> HumanAR:
        data = np.asarray(controls, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2 or len(data) <= self.order:
            raise ValueError(
                "controls must be (samples, 2) with more samples than order"
            )
        x = np.stack(
            [data[i - self.order : i].reshape(-1) for i in range(self.order, len(data))]
        )
        y = data[self.order :]
        design = np.column_stack([x, np.ones(len(x))])
        regularizer = self.ridge * np.eye(design.shape[1])
        regularizer[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        residual = y - design @ self.coef_
        self.residual_cov_ = (
            np.atleast_2d(np.cov(residual.T)) + np.eye(2) * self.min_std**2
        )
        return self

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
        covariance = np.stack(
            [self.residual_cov_ * (1.0 + 0.15 * i) for i in range(horizon)]
        )
        return GaussianTrajectory(mean, covariance)

    def predict_constant(self, control: Control, horizon: int) -> GaussianTrajectory:
        mean = np.repeat(control.as_array()[None, :], horizon, axis=0)
        covariance = np.stack(
            [self.residual_cov_ * (1.0 + 0.15 * i) for i in range(horizon)]
        )
        return GaussianTrajectory(mean, covariance)
