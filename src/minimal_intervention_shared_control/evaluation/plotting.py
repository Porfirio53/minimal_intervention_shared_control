from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_trajectory(records: list[dict[str, object]], output: str | Path) -> None:
    """Create a compact trajectory/authority diagnostic plot (matplotlib extra)."""
    import matplotlib.pyplot as plt

    states = np.array([row["state"] for row in records], dtype=float)
    times = np.array([row["time"] for row in records], dtype=float)
    alpha = np.array([row["alpha"] for row in records], dtype=float)
    correction = np.linalg.norm(
        np.array([row["filtered"] for row in records])
        - np.array([row["nominal"] for row in records]),
        axis=1,
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(states[:, 0], states[:, 1])
    axes[0].set_aspect("equal")
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Robot trajectory")
    axes[1].plot(times, alpha, label="autonomous authority")
    axes[1].plot(times, correction, label="filter correction")
    axes[1].set(xlabel="time [s]", title="Intervention")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
