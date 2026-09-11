from __future__ import annotations

import numpy as np


def project_position_rows(sensitivity: np.ndarray, normals: np.ndarray) -> np.ndarray:
    sensitivity, normals = np.asarray(sensitivity), np.asarray(normals)
    if sensitivity.ndim != 3 or sensitivity.shape[2] != 2:
        raise ValueError("sensitivity must have shape (horizon, horizon, 2)")
    if normals.shape != (sensitivity.shape[0], 2):
        raise ValueError("normals must have shape (horizon, 2)")
    return np.einsum("tjd,td->tj", sensitivity, normals)
