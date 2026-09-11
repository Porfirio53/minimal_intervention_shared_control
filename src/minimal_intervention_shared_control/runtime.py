from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .authority.constraints import AuthorityConstraintSet, build_authority_constraints
from .authority.lexicographic_qp import (
    AuthorityResult,
    LexicographicAuthority,
    budget_weights,
)
from .authority.linearization import AffineStatePrediction, affine_state_prediction
from .baselines.delay_agreement import delay_agreement_authority
from .baselines.human_filter_only import human_filter_authority
from .baselines.single_step_qp import single_step_constraints
from .baselines.weighted_sum import weighted_sum_authority
from .config import load_default, project_root
from .dynamics.diff_drive import rollout_gaussian_controls
from .network.timestamp_buffer import CommandEnvelope, TimestampBuffer
from .predictors.human_ar import HumanAR
from .predictors.obstacle_cv import ConstantVelocityPredictor
from .risk.chance_constraint import chance_margin, relative_position_covariance
from .risk.conflict import trajectory_conflict_and_uncertainty
from .risk.prediction_tube import (
    PredictionRiskTube,
    blend_controls,
    trajectory_positions,
)
from .safety.robust_cbf_qp import (
    FilterResult,
    RelativeStateObservation,
    RobustCBFFilter,
)
from .types import Control, FloatArray, GaussianTrajectory

METHODS = ("delay_agreement", "single_step", "weighted_sum", "human_filter", "ours")


@dataclass(frozen=True)
class RuntimeConfig:
    dt: float = 0.02
    shared_control_period: float = 0.05
    prediction_dt: float = 0.1
    horizon: int = 15
    method: str = "ours"

    def __post_init__(self) -> None:
        periods = np.array(
            [self.dt, self.shared_control_period, self.prediction_dt], dtype=float
        )
        if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
            raise ValueError("runtime periods must be positive")
        if self.horizon < 1 or self.method not in METHODS:
            raise ValueError("invalid runtime horizon or method")


@dataclass(frozen=True)
class ObstacleEstimate:
    position: FloatArray
    velocity: FloatArray
    radius: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        if (
            position.shape != (2,)
            or velocity.shape != (2,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))
            or not np.isfinite(self.radius)
            or self.radius < 0
        ):
            raise ValueError("invalid obstacle estimate")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class StepInput:
    now: float
    state: FloatArray
    human_commands: tuple[CommandEnvelope[Control], ...]
    default_human: Control
    autonomous_controls: FloatArray
    obstacles: tuple[ObstacleEstimate, ...]
    safety_observations: tuple[RelativeStateObservation, ...]


@dataclass(frozen=True)
class StepOutput:
    human_control: Control
    human_information_age: float
    nominal_control: Control
    filtered_control: Control
    alpha: FloatArray
    allocation: AuthorityResult
    filter_result: FilterResult
    risk_tube: PredictionRiskTube
    upper_updated: bool
    upper_solve_time: float
    human_prediction_source: str
    obstacle_prediction_source: str


@dataclass(frozen=True)
class PredictionAssembly:
    constraints: AuthorityConstraintSet
    affine_prediction: AffineStatePrediction
    risk_tube: PredictionRiskTube


def _human_candidate_controls(
    control: Control,
    horizon: int,
    model: HumanAR | None,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if model is not None and len(history) >= model.order:
        if horizon == 1:
            return control.as_array()[None, :], np.zeros((2, 2)), "scand-human-ar"
        prediction = model.predict(history[-model.order :], horizon - 1)
        mean = np.vstack([control.as_array(), prediction.mean])
        joint = np.zeros((2 * horizon, 2 * horizon))
        joint[2:, 2:] = model.joint_prediction_covariance(horizon - 1)
        return mean, joint, "scand-human-ar"
    return (
        np.repeat(control.as_array()[None, :], horizon, axis=0),
        None,
        "constant-hold-fallback",
    )


def _build_prediction_problem(
    state: FloatArray,
    robot_radius: float,
    obstacles: tuple[ObstacleEstimate, ...],
    human_controls: np.ndarray,
    autonomous_controls: np.ndarray,
    prediction_dt: float,
    risk: dict[str, float | bool],
    vehicle: dict[str, object],
    reference_alpha: np.ndarray,
    human_intent: GaussianTrajectory | None,
    obstacle_predictor: ConstantVelocityPredictor,
) -> PredictionAssembly:
    horizon = len(human_controls)
    affine = affine_state_prediction(
        state,
        human_controls,
        autonomous_controls,
        reference_alpha,
        prediction_dt,
    )
    if human_intent is None:
        human_positions = trajectory_positions(state, human_controls, prediction_dt)
        human_covariance = np.zeros((horizon, 2, 2))
        human_intent = GaussianTrajectory(human_positions, human_covariance)
    else:
        human_positions = human_intent.mean
        human_covariance = human_intent.covariance
    autonomous_positions = trajectory_positions(
        state, autonomous_controls, prediction_dt
    )
    conflict, uncertainty, _, _ = trajectory_conflict_and_uncertainty(
        human_intent, autonomous_positions, np.ones(2)
    )

    execution_covariance = np.repeat(
        (np.eye(2) * float(risk["execution_position_std"]) ** 2)[None, :, :],
        horizon,
        axis=0,
    )
    execution = GaussianTrajectory(
        affine.positions(reference_alpha), execution_covariance
    )
    obstacle_forecasts = tuple(
        obstacle_predictor.predict(
            obstacle.position,
            obstacle.velocity,
            horizon,
            prediction_dt,
            initial_std=None,
        )
        for obstacle in obstacles
    )
    obstacle_count = len(obstacle_forecasts)
    human_margins = np.empty((horizon, obstacle_count))
    autonomous_margins = np.empty((horizon, obstacle_count))
    execution_margins = np.empty((horizon, obstacle_count))
    chance_rows: list[np.ndarray] = []
    chance_lower: list[float] = []
    assume_independent = bool(risk["assume_execution_obstacle_independence"])

    for time_index in range(horizon):
        for obstacle_index, (obstacle, forecast) in enumerate(
            zip(obstacles, obstacle_forecasts, strict=True)
        ):
            obstacle_mean = forecast.mean[time_index]
            obstacle_covariance = forecast.covariance[time_index]
            delta = execution.mean[time_index] - obstacle_mean
            distance = float(np.linalg.norm(delta))
            radius = robot_radius + obstacle.radius + float(risk["clearance"])
            if distance <= 1e-12:
                chance_rows.append(np.zeros(horizon))
                chance_lower.append(1.0)
                human_margins[time_index, obstacle_index] = -radius
                autonomous_margins[time_index, obstacle_index] = -radius
                execution_margins[time_index, obstacle_index] = -radius
                continue
            normal = delta / distance
            execution_relative_covariance = relative_position_covariance(
                execution_covariance[time_index],
                obstacle_covariance,
                assume_independent=assume_independent,
            )
            human_relative_covariance = relative_position_covariance(
                human_covariance[time_index],
                obstacle_covariance,
                assume_independent=assume_independent,
            )
            autonomous_relative_covariance = relative_position_covariance(
                np.zeros((2, 2)),
                obstacle_covariance,
                assume_independent=assume_independent,
            )
            common = (
                robot_radius,
                obstacle.radius,
                float(risk["clearance"]),
                float(risk["chance_kappa"]),
            )
            human_margins[time_index, obstacle_index] = chance_margin(
                human_positions[time_index],
                obstacle_mean,
                human_relative_covariance,
                *common,
                normal=normal,
            )
            autonomous_margins[time_index, obstacle_index] = chance_margin(
                autonomous_positions[time_index],
                obstacle_mean,
                autonomous_relative_covariance,
                *common,
                normal=normal,
            )
            execution_margins[time_index, obstacle_index] = chance_margin(
                execution.mean[time_index],
                obstacle_mean,
                execution_relative_covariance,
                *common,
                normal=normal,
            )
            base_margin = chance_margin(
                affine.offset[time_index, :2],
                obstacle_mean,
                execution_relative_covariance,
                *common,
                normal=normal,
            )
            chance_rows.append(
                np.einsum("jd,d->j", affine.sensitivity[time_index, :, :2], normal)
            )
            chance_lower.append(-base_margin)

    limits = vehicle["control_limits"]
    if not isinstance(limits, dict):
        raise TypeError("control_limits must be a mapping")
    constraints = build_authority_constraints(
        human_controls,
        autonomous_controls,
        np.asarray(chance_rows).reshape(-1, horizon),
        np.asarray(chance_lower),
        np.array([limits["v_min"], limits["omega_min"]], dtype=float),
        np.array([limits["v_max"], limits["omega_max"]], dtype=float),
    )
    tube = PredictionRiskTube(
        human_intent=human_intent,
        autonomous_positions=autonomous_positions,
        execution=execution,
        obstacles=obstacle_forecasts,
        conflict=conflict,
        intent_uncertainty=uncertainty,
        human_margins=human_margins,
        autonomous_margins=autonomous_margins,
        execution_margins=execution_margins,
    )
    return PredictionAssembly(constraints, affine, tube)


def _fixed_authority_result(alpha: np.ndarray, status: str) -> AuthorityResult:
    weights = budget_weights(len(alpha))
    return AuthorityResult(
        alpha=alpha,
        status=status,
        intervention_budget=float(weights @ alpha),
        objective=0.0,
    )


def _allocate(
    method: str,
    human: np.ndarray,
    autonomous: np.ndarray,
    tau_h: float,
    constraints: AuthorityConstraintSet,
    obstacle_count: int,
    allocator: LexicographicAuthority,
    step_durations: np.ndarray,
    previous_alpha: float,
) -> AuthorityResult:
    horizon = len(human)
    if method == "human_filter":
        return _fixed_authority_result(human_filter_authority(horizon), "not-run")
    if method == "delay_agreement":
        return _fixed_authority_result(
            delay_agreement_authority(human[0], autonomous[0], tau_h, horizon),
            "not-run",
        )
    matrix, lower = constraints.matrix, constraints.lower
    if method == "single_step":
        matrix, lower = single_step_constraints(
            matrix, lower, max(1, obstacle_count), constraints.labels
        )
        return allocator.solve(
            matrix,
            lower,
            human,
            autonomous,
            step_durations=step_durations,
            previous_alpha=previous_alpha,
        )
    if method == "weighted_sum":
        return weighted_sum_authority(
            matrix,
            lower,
            human,
            autonomous,
            step_durations=step_durations,
            previous_alpha=previous_alpha,
            control_scales=allocator.control_scales,
        )
    return allocator.solve(
        matrix,
        lower,
        human,
        autonomous,
        step_durations=step_durations,
        previous_alpha=previous_alpha,
    )


class SharedControlRuntime:
    """Stateful, simulator-independent entry point for WSL and Windows controllers."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        vehicle: dict[str, object] | None = None,
        risk: dict[str, float | bool] | None = None,
        human_model_path: str | Path | None = None,
        obstacle_model_path: str | Path | None = None,
    ):
        self.config = config or RuntimeConfig()
        self.vehicle = vehicle or load_default("vehicle")
        self.risk = risk or load_default("risk")
        human_path = (
            Path(human_model_path)
            if human_model_path is not None
            else project_root() / "artifacts" / "e0" / "scand_human_ar.json"
        )
        obstacle_path = (
            Path(obstacle_model_path)
            if obstacle_model_path is not None
            else project_root() / "artifacts" / "e0" / "thor_metrics.json"
        )
        self.human_model = HumanAR.load(human_path) if human_path.exists() else None
        if (
            self.human_model is not None
            and self.human_model.sample_interval is not None
            and not np.isclose(
                self.human_model.sample_interval, self.config.prediction_dt
            )
        ):
            raise ValueError("HumanAR sample interval must match prediction_dt")
        if obstacle_path.exists():
            self.obstacle_predictor = ConstantVelocityPredictor.from_e0_metrics(
                obstacle_path
            )
            self.obstacle_prediction_source = "thor-calibrated-cv"
        else:
            self.obstacle_predictor = ConstantVelocityPredictor(
                process_std=float(self.risk["obstacle_process_std"]),
                initial_std=float(self.risk["obstacle_position_std"]),
            )
            self.obstacle_prediction_source = "configured-cv"
        limits = self.vehicle["control_limits"]
        if not isinstance(limits, dict):
            raise TypeError("control_limits must be a mapping")
        control_scales = np.array(
            [
                float(limits["v_max"]),
                max(abs(float(limits["omega_min"])), abs(float(limits["omega_max"]))),
            ]
        )
        self.allocator = LexicographicAuthority(
            tolerance=float(self.risk["lexicographic_tolerance"]),
            control_scales=control_scales,
        )
        self.safety = RobustCBFFilter(
            dt=self.config.dt,
            robot_radius=float(self.vehicle["robot_radius"]),
            clearance=float(self.risk["clearance"]),
            gamma=float(self.risk["cbf_gamma"]),
            lookahead=float(self.risk["cbf_lookahead"]),
            acceleration_bound=float(self.risk["relative_acceleration_bound"]),
            clock_error_bound=float(self.risk["clock_error_bound"]),
            model_error_bound=float(self.risk["model_error_bound"]),
            actuation_error_bound=float(self.risk["actuation_error_bound"]),
            confidence_beta=float(self.risk["state_confidence_beta"]),
            limits=limits,
        )
        self.buffer: TimestampBuffer[Control] = TimestampBuffer()
        self.authority = np.zeros(self.config.horizon)
        self.linearization_reference = np.ones(self.config.horizon)
        self.allocation = _fixed_authority_result(self.authority, "not-run")
        self.next_shared_update: float | None = None
        self.human_history: list[np.ndarray] = []
        self.next_history_sample: float | None = None
        self.last_tube: PredictionRiskTube | None = None
        self.human_prediction_source = "constant-hold-fallback"
        self.last_step_time: float | None = None

    def _history_interval(self) -> float:
        return (
            self.human_model.sample_interval
            if self.human_model is not None
            and self.human_model.sample_interval is not None
            else self.config.prediction_dt
        )

    def _sample_until(
        self, boundary: float, control: Control, *, include_boundary: bool
    ) -> None:
        assert self.next_history_sample is not None
        tolerance = 1e-9 if include_boundary else -1e-9
        while self.next_history_sample <= boundary + tolerance:
            self.human_history.append(control.as_array())
            self.next_history_sample += self._history_interval()

    def _ingest_human_commands(
        self,
        now: float,
        commands: tuple[CommandEnvelope[Control], ...],
        default: Control,
    ) -> tuple[Control, float]:
        if self.buffer.latest() is None:
            for envelope in sorted(commands, key=lambda item: item.received_stamp):
                if envelope.received_stamp > now + 1e-9:
                    raise ValueError(
                        "command receipt time is outside this runtime step"
                    )
                self.buffer.push(envelope)
            if self.buffer.latest() is None:
                self.buffer.push(CommandEnvelope(default, now, now, now))
            self.next_history_sample = now
            held = self.buffer.latest()
            assert held is not None
            self._sample_until(now, held.command, include_boundary=True)
            return held.command, held.information_age(now)
        assert self.next_history_sample is not None
        held = self.buffer.latest()
        assert held is not None
        held_control = held.command
        for envelope in sorted(commands, key=lambda item: item.received_stamp):
            if envelope.received_stamp > now + 1e-9 or (
                self.last_step_time is not None
                and envelope.received_stamp < self.last_step_time - 1e-9
            ):
                raise ValueError("command receipt time is outside this runtime step")
            self._sample_until(
                envelope.received_stamp, held_control, include_boundary=False
            )
            if self.buffer.push(envelope):
                held_control = envelope.command
        self._sample_until(now, held_control, include_boundary=True)
        control, age = self.buffer.get(now, default)
        assert control is not None
        return control, age

    def step(self, value: StepInput) -> StepOutput:
        """Compute nominal and filtered commands without depending on a simulator API."""
        state = np.asarray(value.state, dtype=float)
        autonomous = np.asarray(value.autonomous_controls, dtype=float)
        if (
            not np.isfinite(value.now)
            or value.now < 0
            or state.shape != (3,)
            or not np.all(np.isfinite(state))
        ):
            raise ValueError("step time/state are invalid")
        if self.last_step_time is not None and value.now < self.last_step_time - 1e-9:
            raise ValueError("runtime step time must be monotonic")
        if autonomous.shape != (self.config.horizon, 2) or not np.all(
            np.isfinite(autonomous)
        ):
            raise ValueError("autonomous controls must match runtime horizon")
        if len(value.obstacles) != len(value.safety_observations):
            raise ValueError(
                "upper obstacle estimates and safety observations must match"
            )
        human_control, tau_h = self._ingest_human_commands(
            value.now, value.human_commands, value.default_human
        )
        if not np.all(np.isfinite(human_control.as_array())):
            raise ValueError("human control must be finite")

        if self.next_shared_update is None:
            self.next_shared_update = value.now
        update = value.now + 1e-12 >= self.next_shared_update
        solve_time = 0.0
        if update:
            human, joint_covariance, self.human_prediction_source = (
                _human_candidate_controls(
                    human_control,
                    self.config.horizon,
                    self.human_model,
                    np.asarray(self.human_history),
                )
            )
            human_intent = (
                rollout_gaussian_controls(
                    state, human, joint_covariance, self.config.prediction_dt
                )
                if joint_covariance is not None
                else None
            )
            prediction = _build_prediction_problem(
                state,
                float(self.vehicle["robot_radius"]),
                value.obstacles,
                human,
                autonomous,
                self.config.prediction_dt,
                self.risk,
                self.vehicle,
                self.linearization_reference,
                human_intent,
                self.obstacle_predictor,
            )
            started = time.perf_counter()
            self.allocation = _allocate(
                self.config.method,
                human,
                autonomous,
                tau_h,
                prediction.constraints,
                len(value.obstacles),
                self.allocator,
                np.full(self.config.horizon, self.config.prediction_dt),
                float(self.authority[0]),
            )
            solve_time = time.perf_counter() - started
            self.authority = self.allocation.alpha
            self.last_tube = prediction.risk_tube
            if not self.allocation.used_fallback:
                self.linearization_reference = np.r_[
                    self.authority[1:], self.authority[-1]
                ]
            while self.next_shared_update <= value.now + 1e-12:
                self.next_shared_update += self.config.shared_control_period

        assert self.last_tube is not None
        nominal = Control.from_array(
            blend_controls(
                human_control.as_array()[None, :],
                autonomous[0][None, :],
                self.authority[0],
            )[0]
        )
        filtered = self.safety.filter(state, nominal, list(value.safety_observations))
        self.last_step_time = value.now
        return StepOutput(
            human_control=human_control,
            human_information_age=float(tau_h),
            nominal_control=nominal,
            filtered_control=filtered.control,
            alpha=self.authority.copy(),
            allocation=self.allocation,
            filter_result=filtered,
            risk_tube=self.last_tube,
            upper_updated=update,
            upper_solve_time=solve_time,
            human_prediction_source=self.human_prediction_source,
            obstacle_prediction_source=self.obstacle_prediction_source,
        )
