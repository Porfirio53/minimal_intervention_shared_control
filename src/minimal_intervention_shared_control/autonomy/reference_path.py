from __future__ import annotations

import numpy as np

from ..types import FloatArray


class PolylinePath:
    def __init__(self, points: FloatArray):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("a path needs at least two 2-D points")
        self.points = points
        self.segment = np.diff(points, axis=0)
        self.lengths = np.linalg.norm(self.segment, axis=1)
        self.cumulative = np.r_[0.0, np.cumsum(self.lengths)]

    @property
    def total_length(self) -> float:
        return float(self.cumulative[-1])

    def project(self, position: FloatArray) -> tuple[float, FloatArray, int]:
        position = np.asarray(position, dtype=float)[:2]
        best_distance = float("inf")
        best_s, best_point, best_segment = 0.0, self.points[0], 0
        for i, (start, vector, length) in enumerate(
            zip(self.points[:-1], self.segment, self.lengths)
        ):
            if length < 1e-12:
                continue
            fraction = np.clip(np.dot(position - start, vector) / length**2, 0.0, 1.0)
            candidate = start + fraction * vector
            distance = float(np.linalg.norm(position - candidate))
            if distance < best_distance:
                best_distance, best_s, best_point, best_segment = (
                    distance,
                    self.cumulative[i] + fraction * length,
                    candidate,
                    i,
                )
        return float(best_s), np.asarray(best_point), best_segment

    def at(self, distance: float) -> FloatArray:
        distance = float(np.clip(distance, 0.0, self.total_length))
        index = int(
            np.clip(
                np.searchsorted(self.cumulative, distance, side="right") - 1,
                0,
                len(self.segment) - 1,
            )
        )
        fraction = (distance - self.cumulative[index]) / max(self.lengths[index], 1e-12)
        return self.points[index] + fraction * self.segment[index]
