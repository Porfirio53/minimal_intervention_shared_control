from __future__ import annotations

import numpy as np


def human_filter_authority(horizon: int) -> np.ndarray:
    return np.zeros(horizon, dtype=float)
