import numpy as np

from minimal_intervention_shared_control.evaluation.metrics import trajectory_metrics


def _record(time: float, clearance: float, alpha: float, filtered_v: float):
    return {
        "time": time,
        "state": [1.0, 0.0, 0.0],
        "human": [0.2, 0.0],
        "nominal": [0.2, 0.0],
        "filtered": [filtered_v, 0.0],
        "alpha": alpha,
        "clearance": clearance,
        "filter_triggered": filtered_v != 0.2,
        "filter_infeasible": False,
        "tau_h": 0.0,
    }


def test_collision_prevents_task_success_and_event_times_are_reported() -> None:
    records = [
        _record(0.1, 0.2, 0.0, 0.2),
        _record(0.2, -0.1, 0.3, 0.1),
    ]
    metrics = trajectory_metrics(records, 0.1, np.array([1.0, 0.0, 0.0]))
    assert metrics["collision"]
    assert not metrics["success"]
    assert metrics["first_authority_time"] == 0.2
    assert metrics["first_filter_time"] == 0.2
