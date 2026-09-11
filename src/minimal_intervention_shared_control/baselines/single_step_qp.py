from __future__ import annotations

import numpy as np


def single_step_constraints(
    g: np.ndarray,
    b: np.ndarray,
    constraints_per_step: int,
    labels: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if constraints_per_step < 1:
        raise ValueError("constraints_per_step must be positive")
    matrix, lower = np.asarray(g), np.asarray(b)
    if labels is None:
        return matrix[:constraints_per_step], lower[:constraints_per_step]
    if len(labels) != len(matrix):
        raise ValueError("labels must match the constraint rows")
    first_chance_rows = {
        index
        for index in np.flatnonzero(np.asarray(labels) == "chance")[
            :constraints_per_step
        ]
    }
    selected = np.array(
        [
            label != "chance" or index in first_chance_rows
            for index, label in enumerate(labels)
        ]
    )
    return matrix[selected], lower[selected]
