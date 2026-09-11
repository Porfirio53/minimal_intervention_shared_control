from __future__ import annotations

import numpy as np

from ..types import FloatArray, GaussianTrajectory


class HumanBehaviorCloner:
    """Feature-to-control ridge model for SCAND-style demonstrations."""

    def __init__(self, horizon: int = 15, ridge: float = 1e-3):
        self.horizon = horizon
        self.ridge = ridge
        self.coef_: FloatArray | None = None
        self.residual_cov_ = np.eye(2) * 0.06**2

    @staticmethod
    def _features(state: FloatArray, goal: FloatArray) -> FloatArray:
        state, goal = np.asarray(state, dtype=float), np.asarray(goal, dtype=float)
        delta = goal[:2] - state[:2]
        distance = np.linalg.norm(delta)
        bearing = np.arctan2(delta[1], delta[0]) - state[2]
        return np.array(
            [
                distance,
                np.sin(bearing),
                np.cos(bearing),
                state[0],
                state[1],
                state[2],
                1.0,
            ]
        )

    def fit(
        self, states: FloatArray, goals: FloatArray, commands: FloatArray
    ) -> HumanBehaviorCloner:
        x = np.stack([self._features(s, g) for s, g in zip(states, goals)])
        y = np.asarray(commands, dtype=float)
        if y.shape != (len(x), 2):
            raise ValueError("commands must have shape (samples, 2)")
        reg = self.ridge * np.eye(x.shape[1])
        reg[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(x.T @ x + reg, x.T @ y)
        residual = y - x @ self.coef_
        self.residual_cov_ = np.atleast_2d(np.cov(residual.T)) + np.eye(2) * 1e-4
        return self

    def predict(
        self, state: FloatArray, goal: FloatArray, horizon: int | None = None
    ) -> GaussianTrajectory:
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        h = self.horizon if horizon is None else horizon
        command = self._features(state, goal) @ self.coef_
        return GaussianTrajectory(
            np.repeat(command[None, :], h, axis=0), np.stack([self.residual_cov_] * h)
        )
