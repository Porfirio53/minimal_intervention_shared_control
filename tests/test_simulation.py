import math

import numpy as np
import pytest

from minimal_intervention_shared_control.sim2d.safety_experiment import (
    replay_safety_filter,
)
from minimal_intervention_shared_control.sim2d.scenarios import SCENARIO_NAMES
from minimal_intervention_shared_control.sim2d.simulation import (
    METHODS,
    SimulationConfig,
    run_simulation,
)


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_run_end_to_end(method: str) -> None:
    result = run_simulation(
        "crossing",
        SimulationConfig(duration=0.3, method=method, network_condition="N0", seed=4),
    )
    assert result.records
    assert result.metrics["completion_time"] > 0
    assert math.isfinite(float(result.metrics["minimum_clearance"]))
    assert 0 <= float(result.metrics["intervention_budget"]) <= 1


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_all_scenarios_run(scenario: str) -> None:
    result = run_simulation(
        scenario, SimulationConfig(duration=0.2, method="ours", seed=1)
    )
    assert result.scenario == scenario
    assert all(float(row["tau_h"]) >= 0 for row in result.records)
    assert all(all(float(age) >= 0 for age in row["tau_s"]) for row in result.records)


def test_paired_methods_receive_identical_network_age_samples() -> None:
    common = {"duration": 0.3, "network_condition": "N1", "seed": 11}
    human_filter = run_simulation(
        "crossing", SimulationConfig(method="human_filter", **common)
    )
    ours = run_simulation("crossing", SimulationConfig(method="ours", **common))
    assert [row["tau_h"] for row in human_filter.records] == [
        row["tau_h"] for row in ours.records
    ]


def test_simulation_reports_explicit_qp_and_risk_diagnostics() -> None:
    result = run_simulation(
        "crossing",
        SimulationConfig(duration=0.1, method="ours", network_condition="N0", seed=3),
    )
    row = result.records[0]
    assert {
        "stage1_status",
        "stage2_status",
        "filter_status",
        "filter_infeasible",
        "h_minus",
        "predicted_minimum_margin",
        "mean_conflict",
        "mean_intent_uncertainty",
        "human_prediction_source",
        "obstacle_prediction_source",
        "scenario_obstacle_source",
        "upper_updated",
    } <= row.keys()


def test_simulation_uses_calibrated_models_after_causal_history_is_available() -> None:
    result = run_simulation(
        "crossing",
        SimulationConfig(duration=0.5, method="ours", network_condition="N0", seed=3),
    )
    assert result.records[-1]["human_prediction_source"] == "scand-human-ar"
    assert result.records[-1]["obstacle_prediction_source"] == "thor-calibrated-cv"
    assert result.records[-1]["scenario_obstacle_source"]
    assert float(result.records[-1]["mean_intent_uncertainty"]) > 0


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_core_simulation_outputs_are_finite_and_physically_bounded(
    scenario: str,
) -> None:
    result = run_simulation(
        scenario, SimulationConfig(duration=0.2, method="ours", seed=9)
    )
    for row in result.records:
        for key in ("state", "human", "autonomous", "nominal", "filtered"):
            assert np.all(np.isfinite(np.asarray(row[key], dtype=float)))
        assert math.isfinite(float(row["clearance"]))
        assert 0.0 <= float(row["alpha"]) <= 1.0
        nominal = np.asarray(row["nominal"], dtype=float)
        filtered = np.asarray(row["filtered"], dtype=float)
        assert 0.0 <= nominal[0] <= 1.2
        assert -1.8 <= nominal[1] <= 1.8
        assert 0.0 <= filtered[0] <= 1.2 + 2e-6
        assert -1.8 - 2e-6 <= filtered[1] <= 1.8 + 2e-6


def test_safety_isolation_replays_identical_nominal_commands() -> None:
    source = run_simulation(
        "network_anomaly", SimulationConfig(duration=0.3, method="human_filter", seed=2)
    )
    unfiltered = replay_safety_filter(
        "network_anomaly", source.records, "none", tau_s=0.0
    )
    robust = replay_safety_filter(
        "network_anomaly", source.records, "robust", tau_s=0.2
    )
    assert [row["nominal"] for row in unfiltered.records] == [
        row["nominal"] for row in robust.records
    ]
    assert all(row["filtered"] == row["nominal"] for row in unfiltered.records)


def test_network_anomaly_uses_accelerating_obstacle_ground_truth() -> None:
    result = run_simulation(
        "network_anomaly", SimulationConfig(duration=0.7, method="ours", seed=2)
    )
    assert all(
        row["scenario_obstacle_source"] == "synthetic-accelerating"
        for row in result.records
    )
