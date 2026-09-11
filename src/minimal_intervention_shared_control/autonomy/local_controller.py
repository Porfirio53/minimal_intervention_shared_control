from __future__ import annotations

import numpy as np

from ..types import Control, FloatArray
from .reference_path import PolylinePath


class LocalController:
    def __init__(
        self,
        path: PolylinePath,
        lookahead: float = 0.55,
        speed: float = 0.55,
        k_heading: float = 2.4,
    ):
        self.path, self.lookahead, self.speed, self.k_heading = (
            path,
            lookahead,
            speed,
            k_heading,
        )

    def __call__(self, state: FloatArray) -> Control:
        state = np.asarray(state, dtype=float)
        progress, _, segment = self.path.project(state[:2])
        target = self.path.at(progress + self.lookahead)
        desired = np.arctan2(*(target - state[:2])[::-1])
        error = (desired - state[2] + np.pi) % (2 * np.pi) - np.pi
        segment_heading = np.arctan2(
            self.path.segment[segment, 1], self.path.segment[segment, 0]
        )
        terminal = np.linalg.norm(self.path.points[-1] - state[:2]) < self.lookahead
        speed = (
            min(self.speed, 0.8 * np.linalg.norm(self.path.points[-1] - state[:2]))
            if terminal
            else self.speed
        )
        return Control(
            float(max(0.0, speed * np.cos(error))),
            float(
                np.clip(
                    self.k_heading * error
                    + 0.2
                    * ((segment_heading - state[2] + np.pi) % (2 * np.pi) - np.pi),
                    -1.8,
                    1.8,
                )
            ),
        )
