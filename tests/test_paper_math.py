import numpy as np
import pytest

from minimal_intervention_shared_control.authority.linearization import (
    affine_state_prediction,
)
from minimal_intervention_shared_control.authority.sensitivity import (
    project_position_rows,
)
from minimal_intervention_shared_control.risk.chance_constraint import (
    chance_margin,
    relative_position_covariance,
)
from minimal_intervention_shared_control.risk.conflict import (
    trajectory_conflict_and_uncertainty,
)
from minimal_intervention_shared_control.types import GaussianTrajectory


def test_conflict_and_intent_uncertainty_match_hand_calculation() -> None:
    human = GaussianTrajectory(
        mean=np.array([[2.0, 4.0], [1.0, 2.0]]),
        covariance=np.array([np.diag([4.0, 9.0]), np.diag([1.0, 4.0])]),
    )
    autonomous = np.array([[1.0, 2.0], [1.0, 2.0]])
    conflict, uncertainty, mean_conflict, mean_uncertainty = (
        trajectory_conflict_and_uncertainty(
            human, autonomous, np.array([1.0, 2.0]), np.array([0.25, 0.75])
        )
    )
    np.testing.assert_allclose(conflict, [2.0, 0.0])
    np.testing.assert_allclose(uncertainty, [6.25, 2.0])
    assert mean_conflict == pytest.approx(0.5)
    assert mean_uncertainty == pytest.approx(3.0625)


def test_chance_margin_uses_standard_deviation_not_variance() -> None:
    margin = chance_margin(
        np.array([5.0, 0.0]),
        np.zeros(2),
        np.diag([4.0, 9.0]),
        0.0,
        0.0,
        0.0,
        2.0,
        normal=np.array([1.0, 0.0]),
    )
    assert margin == pytest.approx(1.0)


def test_relative_covariance_requires_explicit_independence() -> None:
    with pytest.raises(ValueError, match="independence"):
        relative_position_covariance(np.eye(2), np.eye(2))
    np.testing.assert_allclose(
        relative_position_covariance(
            np.eye(2), 2.0 * np.eye(2), assume_independent=True
        ),
        3.0 * np.eye(2),
    )


def test_fixed_linearization_produces_affine_positions_in_alpha() -> None:
    human = np.array([[0.4, 0.0], [0.4, 0.2], [0.3, -0.1]])
    autonomous = np.array([[0.7, 0.3], [0.2, -0.2], [0.5, 0.4]])
    prediction = affine_state_prediction(
        np.array([0.0, 0.0, 0.2]),
        human,
        autonomous,
        np.array([0.4, 0.4, 0.4]),
        0.1,
    )
    alpha_a = np.array([0.1, 0.7, 0.2])
    alpha_b = np.array([0.9, 0.2, 0.8])
    midpoint = 0.5 * (alpha_a + alpha_b)
    np.testing.assert_allclose(
        prediction.positions(midpoint),
        0.5 * (prediction.positions(alpha_a) + prediction.positions(alpha_b)),
        atol=1e-12,
    )


def test_sensitivity_sign_reports_helpful_and_harmful_authority() -> None:
    sensitivity = np.array(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[-3.0, 0.0], [1.0, 0.0]],
        ]
    )
    projected = project_position_rows(sensitivity, np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert projected[0, 0] > 0
    assert projected[1, 0] < 0
