import numpy as np

from minimal_intervention_shared_control.evaluation.prediction_metrics import (
    gaussian_trajectory_metrics,
)
from minimal_intervention_shared_control.predictors.calibration import (
    CovarianceCalibrator,
)
from minimal_intervention_shared_control.predictors.human_ar import HumanAR
from minimal_intervention_shared_control.predictors.obstacle_cv import (
    ConstantVelocityPredictor,
)


def test_human_ar_fits_constant_command() -> None:
    controls = np.tile([0.5, -0.2], (30, 1))
    model = HumanAR(order=3).fit(controls)
    prediction = model.predict(controls[-3:], horizon=5)
    np.testing.assert_allclose(prediction.mean, np.tile([0.5, -0.2], (5, 1)), atol=1e-5)
    assert np.all(np.linalg.eigvalsh(prediction.covariance) > 0)


def test_human_ar_fit_sequences_never_builds_cross_run_windows() -> None:
    first = np.tile([0.5, 0.1], (4, 1))
    second = np.tile([-0.5, -0.1], (5, 1))
    model = HumanAR(order=2).fit_sequences([first, second])
    assert model.training_samples_ == (len(first) - 2) + (len(second) - 2)


def test_human_ar_propagates_linear_model_covariance() -> None:
    model = HumanAR(order=1)
    model.coef_ = np.array([[0.5, 0.0], [0.0, 0.5], [0.0, 0.0]])
    model.residual_cov_ = np.eye(2)
    prediction = model.predict(np.zeros((1, 2)), horizon=2)
    np.testing.assert_allclose(prediction.covariance, [np.eye(2), np.eye(2) * 1.25])


def test_constant_velocity_prediction() -> None:
    model = ConstantVelocityPredictor(process_std=0.1)
    velocity = model.fit_velocity(np.array([[0.0, 0.0], [0.2, 0.4]]), dt=0.2)
    trajectory = model.predict(np.zeros(2), velocity, horizon=2, dt=0.5)
    np.testing.assert_allclose(trajectory.mean, [[0.5, 1.0], [1.0, 2.0]])
    assert trajectory.covariance[1, 0, 0] > trajectory.covariance[0, 0, 0]


def test_calibration_only_inflates() -> None:
    errors = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
    covariance = np.tile(np.eye(2)[None, :, :], (3, 1, 1))
    calibrator = CovarianceCalibrator(target_coverage=0.8, kappa=1.0)
    assert calibrator.fit(errors, covariance) > 1.0
    assert np.all(calibrator.transform(covariance) >= covariance)


def test_gaussian_prediction_metrics() -> None:
    truth = np.zeros((2, 3, 2))
    mean = np.zeros_like(truth)
    covariance = np.tile(np.eye(2), (2, 3, 1, 1))
    metrics = gaussian_trajectory_metrics(truth, mean, covariance)
    assert metrics["ade"] == metrics["fde"] == 0.0
    assert metrics["coverage_95"] == 1.0
    assert np.isfinite(metrics["nll"])
