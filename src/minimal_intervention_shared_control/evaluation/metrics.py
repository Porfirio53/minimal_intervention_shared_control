from __future__ import annotations

import numpy as np


def trajectory_metrics(
    records: list[dict[str, object]],
    dt: float,
    goal: np.ndarray,
    success_radius: float = 0.3,
) -> dict[str, float | bool]:
    if not records:
        raise ValueError("records cannot be empty")
    states = np.array([row["state"] for row in records], dtype=float)
    human = np.array([row["human"] for row in records], dtype=float)
    nominal = np.array([row["nominal"] for row in records], dtype=float)
    filtered = np.array([row["filtered"] for row in records], dtype=float)
    alpha = np.array([row["alpha"] for row in records], dtype=float)
    duration = max(dt * len(records), dt)
    scales = np.array([1.2, 1.8])
    normalized_nominal = np.linalg.norm((nominal - human) / scales, axis=1)
    normalized_filter = np.linalg.norm((filtered - nominal) / scales, axis=1)
    clearances = np.array([row["clearance"] for row in records], dtype=float)
    metrics: dict[str, float | bool] = {
        "collision": bool(np.any(clearances <= 0.0)),
        "success": bool(np.linalg.norm(states[-1, :2] - goal[:2]) <= success_radius),
        "completion_time": float(duration),
        "minimum_clearance": float(clearances.min()),
        "intervention_budget": float(dt * np.sum(alpha) / duration),
        "nominal_modification": float(dt * np.sum(normalized_nominal) / duration),
        "filter_modification": float(dt * np.sum(normalized_filter) / duration),
        "authority_total_variation": float(np.sum(np.abs(np.diff(alpha)))),
        "filter_trigger_rate": float(
            np.mean([bool(row["filter_triggered"]) for row in records])
        ),
        "qp_fallback_rate": float(
            np.mean([bool(row.get("qp_fallback", False)) for row in records])
        ),
        "upper_infeasible_rate": float(
            np.mean([bool(row.get("upper_infeasible", False)) for row in records])
        ),
        "secondary_fallback_rate": float(
            np.mean([bool(row.get("secondary_fallback", False)) for row in records])
        ),
        "filter_infeasible_rate": float(
            np.mean([bool(row.get("filter_infeasible", False)) for row in records])
        ),
        "mean_human_information_age": float(
            np.mean(
                [
                    float(row["tau_h"])
                    for row in records
                    if np.isfinite(float(row["tau_h"]))
                ]
            )
        ),
    }
    h_minus = [
        float(row["h_minus"])
        for row in records
        if row.get("h_minus") is not None and np.isfinite(float(row["h_minus"]))
    ]
    if h_minus:
        metrics["initial_h_minus"] = h_minus[0]
        metrics["minimum_h_minus"] = min(h_minus)
    prediction_margins = [
        float(row["predicted_minimum_margin"])
        for row in records
        if row.get("predicted_minimum_margin") is not None
        and np.isfinite(float(row["predicted_minimum_margin"]))
    ]
    if prediction_margins:
        metrics["minimum_predicted_margin"] = min(prediction_margins)
    return metrics
