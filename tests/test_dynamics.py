import numpy as np

from minimal_intervention_shared_control.dynamics.diff_drive import (
    rollout,
    rollout_gaussian_controls,
    step,
)


def test_straight_step() -> None:
    actual = step(np.array([1.0, 2.0, np.pi / 2]), np.array([0.5, 0.0]), 2.0)
    np.testing.assert_allclose(actual, [1.0, 3.0, np.pi / 2], atol=1e-10)


def test_euler_turning_step_and_rollout_shape() -> None:
    actual = step(np.zeros(3), np.array([1.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(actual, [np.pi / 2, 0.0, np.pi / 2], atol=1e-10)
    states = rollout(np.zeros(3), np.ones((4, 2)), 0.1)
    assert states.shape == (5, 3)


def test_joint_control_covariance_propagates_to_position_sigma_h() -> None:
    trajectory = rollout_gaussian_controls(
        np.zeros(3),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.eye(4),
        0.1,
    )
    np.testing.assert_allclose(trajectory.mean, [[0.1, 0.0], [0.2, 0.0]])
    np.testing.assert_allclose(trajectory.covariance[0], [[0.01, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        trajectory.covariance[1], [[0.02, 0.0], [0.0, 0.0001]], atol=1e-12
    )
