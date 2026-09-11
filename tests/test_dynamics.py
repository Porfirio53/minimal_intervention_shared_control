import numpy as np

from minimal_intervention_shared_control.dynamics.diff_drive import rollout, step


def test_straight_step() -> None:
    actual = step(np.array([1.0, 2.0, np.pi / 2]), np.array([0.5, 0.0]), 2.0)
    np.testing.assert_allclose(actual, [1.0, 3.0, np.pi / 2], atol=1e-10)


def test_euler_turning_step_and_rollout_shape() -> None:
    actual = step(np.zeros(3), np.array([1.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(actual, [np.pi / 2, 0.0, np.pi / 2], atol=1e-10)
    states = rollout(np.zeros(3), np.ones((4, 2)), 0.1)
    assert states.shape == (5, 3)
