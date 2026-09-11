from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Control:
    v: float
    omega: float

    def as_array(self) -> FloatArray:
        return np.array([self.v, self.omega], dtype=float)

    @classmethod
    def from_array(cls, value: FloatArray) -> Control:
        array = np.asarray(value, dtype=float)
        if array.shape != (2,):
            raise ValueError("a control must have shape (2,)")
        return cls(float(array[0]), float(array[1]))


@dataclass(frozen=True)
class GaussianTrajectory:
    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if mean.ndim != 2:
            raise ValueError("trajectory mean must have shape (horizon, dimension)")
        if covariance.shape != (len(mean), mean.shape[1], mean.shape[1]):
            raise ValueError("trajectory covariance has an incompatible shape")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
