import inspect

import numpy as np
import pytest

from minimal_intervention_shared_control.authority.constraints import (
    build_authority_constraints,
)
from minimal_intervention_shared_control.authority.lexicographic_qp import (
    LexicographicAuthority,
)
from minimal_intervention_shared_control.authority.linearization import (
    affine_state_prediction,
)
from minimal_intervention_shared_control.safety.observations import (
    observations_from_current_estimates,
)
from minimal_intervention_shared_control.safety.robust_cbf_qp import (
    RelativeStateObservation,
    RobustCBFFilter,
)
from minimal_intervention_shared_control.types import Control


def _observation(
    relative_position: tuple[float, float], age: float = 0.0
) -> RelativeStateObservation:
    return RelativeStateObservation(
        source_relative_position=np.array(relative_position),
        source_relative_velocity=np.array([0.2, -0.1]),
        obstacle_velocity=np.zeros(2),
        obstacle_radius=0.2,
        obstacle_speed_bound=0.0,
        age=age,
        position_covariance=np.diag([0.04, 0.09]),
        position_velocity_covariance=np.array([[0.01, 0.0], [0.0, 0.02]]),
        velocity_position_covariance=np.array([[0.01, 0.0], [0.0, 0.02]]),
        velocity_covariance=np.diag([0.16, 0.25]),
    )


def test_lexicographic_zero_authority_when_zero_is_feasible() -> None:
    result = LexicographicAuthority(tolerance=0.0).solve(
        np.eye(2), np.array([-0.2, -0.4])
    )
    assert not result.used_fallback
    assert result.minimum_budget == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(result.alpha, 0.0, atol=1e-6)


def test_second_stage_cannot_exceed_stage_one_budget_bound() -> None:
    result = LexicographicAuthority(tolerance=0.01).solve(
        np.array([[1.0, 1.0]]),
        np.array([0.8]),
        human=np.zeros((2, 2)),
        autonomous=np.ones((2, 2)),
    )
    assert not result.used_fallback
    assert result.minimum_budget == pytest.approx(0.4, abs=1e-6)
    assert result.budget_bound == pytest.approx(0.41, abs=1e-6)
    assert result.intervention_budget <= result.budget_bound + 1e-6


def test_secondary_task_objective_uses_available_budget_without_breaking_it() -> None:
    result = LexicographicAuthority(
        tolerance=1.0, smooth_weight=0.0, input_weight=0.0
    ).solve(
        np.empty((0, 1)),
        np.empty(0),
        task_offset=np.zeros((1, 2)),
        task_sensitivity=np.array([[[1.0, 0.0]]]),
        task_positions=np.array([[1.0, 0.0]]),
        task_weight=1.0,
    )
    assert not result.used_fallback
    assert result.alpha[0] > 0.9
    assert result.intervention_budget <= result.budget_bound + 1e-6


def test_lexicographic_qp_reports_infeasible_problem() -> None:
    result = LexicographicAuthority().solve(np.zeros((1, 2)), np.ones(1))
    assert result.used_fallback
    assert "infeasible" in result.stage1_status
    np.testing.assert_allclose(result.alpha, 0.0)


def test_authority_constraint_builder_enforces_hard_control_limits() -> None:
    human = np.array([[0.2, -0.5], [0.2, -0.5]])
    autonomous = np.array([[0.8, 0.5], [0.8, 0.5]])
    constraints = build_authority_constraints(
        human,
        autonomous,
        np.empty((0, 2)),
        np.empty(0),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]),
    )
    assert set(constraints.labels) == {"control_lower", "control_upper"}
    assert np.all(constraints.matrix @ np.array([0.5, 0.5]) >= constraints.lower)


def test_full_configured_authority_set_has_no_implicit_slack() -> None:
    human = np.array([[0.2, 0.0], [0.2, 0.0]])
    autonomous = np.array([[0.8, 0.5], [0.8, 0.5]])
    reference_alpha = np.array([0.2, 0.2])
    affine = affine_state_prediction(
        np.zeros(3), human, autonomous, reference_alpha, 0.1
    )
    previous_control = human[0] + reference_alpha[0] * (autonomous[0] - human[0])
    constraints = build_authority_constraints(
        human,
        autonomous,
        np.empty((0, 2)),
        np.empty(0),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]),
        previous_control=previous_control,
        control_rate_limit=np.zeros(2),
        previous_alpha=0.2,
        alpha_rate_limit=0.0,
        first_interval=0.02,
        prediction_interval=0.1,
        wheelbase=0.4,
        wheel_speed_limit=1.0,
        affine_prediction=affine,
        corridors=((np.array([[1.0, 0.0, 0.0]]), np.array([10.0])),) * 2,
        state_trust_radius=np.ones(3),
        control_trust_radius=np.ones(2),
    )
    assert {
        "control_rate_lower",
        "control_rate_upper",
        "alpha_rate_lower",
        "alpha_rate_upper",
        "wheel_lower",
        "wheel_upper",
        "corridor",
        "state_trust_lower",
        "state_trust_upper",
        "control_trust_lower",
        "control_trust_upper",
    } <= set(constraints.labels)
    assert constraints.matrix.shape[1] == len(reference_alpha)
    assert np.all(constraints.matrix @ reference_alpha >= constraints.lower - 1e-12)
    assert np.any(constraints.matrix @ np.array([0.3, 0.3]) < constraints.lower - 1e-12)


def test_execution_filter_contract_excludes_human_uncertainty() -> None:
    parameters = inspect.signature(RobustCBFFilter.filter).parameters
    assert not any("human" in name or "sigma_h" in name for name in parameters)
    assert "observations" in parameters


def test_relative_covariance_uses_all_paper_equation_25_blocks() -> None:
    observation = _observation((-5.0, 0.0), age=0.5)
    expected = (
        observation.position_covariance
        + 0.5
        * (
            observation.position_velocity_covariance
            + observation.velocity_position_covariance
        )
        + 0.25 * observation.velocity_covariance
    )
    np.testing.assert_allclose(observation.covariance_at(), expected)
    np.testing.assert_allclose(
        observation.mean_at(), np.array([-4.9, -0.05]), atol=1e-12
    )


def test_tau_s_is_per_obstacle_and_independent_of_human_age() -> None:
    observations = observations_from_current_estimates(
        np.zeros(3),
        Control(0.0, 0.0),
        [
            (np.array([2.0, 0.0]), 0.2, np.zeros(2)),
            (np.array([3.0, 0.0]), 0.2, np.zeros(2)),
        ],
        [0.02, 0.17],
        lookahead=0.18,
        position_std=0.01,
        velocity_std=0.02,
    )
    tau_h = 0.8
    assert [observation.age for observation in observations] == [0.02, 0.17]
    assert all(observation.age != tau_h for observation in observations)


def test_filter_slows_for_close_but_feasible_obstacle() -> None:
    safety = RobustCBFFilter(robot_radius=0.32, clearance=0.08)
    observations = observations_from_current_estimates(
        np.zeros(3),
        Control(0.0, 0.0),
        [(np.array([1.25, 0.0]), 0.2, np.zeros(2))],
        0.0,
        lookahead=0.18,
        position_std=0.0,
        velocity_std=0.0,
    )
    result = safety.filter(np.zeros(3), Control(0.8, 0.0), observations)
    assert result.triggered
    assert not result.infeasible
    assert 0.0 <= result.control.v < 0.8


def test_filter_preserves_safe_command() -> None:
    safety = RobustCBFFilter()
    nominal = Control(0.4, 0.2)
    observations = observations_from_current_estimates(
        np.zeros(3),
        Control(0.0, 0.0),
        [(np.array([5.0, 0.0]), 0.2, np.zeros(2))],
        0.0,
        lookahead=0.18,
        position_std=0.0,
        velocity_std=0.0,
    )
    result = safety.filter(np.zeros(3), nominal, observations)
    np.testing.assert_allclose(result.control.as_array(), nominal.as_array(), atol=1e-5)


def test_filter_reports_geometrically_undefined_problem_as_infeasible() -> None:
    safety = RobustCBFFilter(lookahead=0.18)
    result = safety.filter(np.zeros(3), Control(0.4, 0.0), [_observation((0.0, 0.0))])
    assert result.infeasible
    assert "infeasible" in result.status
    np.testing.assert_allclose(result.control.as_array(), 0.0)
