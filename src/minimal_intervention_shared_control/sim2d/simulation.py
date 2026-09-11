from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..authority.constraints import AuthorityConstraintSet, build_authority_constraints
from ..authority.lexicographic_qp import (
    AuthorityResult,
    LexicographicAuthority,
    budget_weights,
)
from ..authority.linearization import AffineStatePrediction, affine_state_prediction
from ..autonomy.local_controller import LocalController
from ..baselines.delay_agreement import delay_agreement_authority
from ..baselines.human_filter_only import human_filter_authority
from ..baselines.single_step_qp import single_step_constraints
from ..baselines.weighted_sum import weighted_sum_authority
from ..config import load_default
from ..dynamics.diff_drive import step
from ..evaluation.metrics import trajectory_metrics
from ..network.delay_channel import BidirectionalNetwork, DelayModel
from ..network.timestamp_buffer import CommandEnvelope, TimestampBuffer
from ..predictors.obstacle_cv import ConstantVelocityPredictor
from ..risk.chance_constraint import chance_margin, relative_position_covariance
from ..risk.conflict import trajectory_conflict_and_uncertainty
from ..risk.prediction_tube import (
    PredictionRiskTube,
    blend_controls,
    trajectory_positions,
)
from ..safety.observations import observations_from_current_estimates
from ..safety.robust_cbf_qp import RobustCBFFilter
from ..types import Control, GaussianTrajectory
from .scenarios import Scenario, make_scenario
from .world import World

METHODS = ("delay_agreement", "single_step", "weighted_sum", "human_filter", "ours")


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 10.0
    dt: float = 0.02
    shared_control_period: float = 0.05
    prediction_dt: float = 0.1
    horizon: int = 15
    network_condition: str | None = None
    method: str = "ours"
    seed: int = 0


@dataclass(frozen=True)
class SimulationResult:
    scenario: str
    method: str
    seed: int
    records: list[dict[str, object]]
    metrics: dict[str, float | bool]


@dataclass(frozen=True)
class PredictionAssembly:
    constraints: AuthorityConstraintSet
    affine_prediction: AffineStatePrediction
    risk_tube: PredictionRiskTube


def _delay_models(condition_name: str) -> tuple[DelayModel, DelayModel]:
    conditions = load_default("network")["conditions"]
    if condition_name not in conditions:
        raise ValueError(f"unknown network condition: {condition_name}")
    values = conditions[condition_name]
    common = {
        "jitter_std": values["jitter_std"],
        "rho": values["rho"],
        "max_delay": values["max_delay"],
        "loss": values["loss"],
    }
    return DelayModel(mean=values["downlink_mean"], **common), DelayModel(
        mean=values["uplink_mean"], **common
    )


def _autonomous_candidate_controls(
    controller: LocalController,
    state: np.ndarray,
    horizon: int,
    dt: float,
) -> np.ndarray:
    values = np.empty((horizon, 2), dtype=float)
    predicted = np.asarray(state, dtype=float).copy()
    for index in range(horizon):
        values[index] = controller(predicted).as_array()
        predicted = step(predicted, values[index], dt)
    return values


def _human_candidate_controls(control: Control, horizon: int) -> np.ndarray:
    """Causal synthetic predictor used when no trained human model is available."""
    return np.repeat(control.as_array()[None, :], horizon, axis=0)


def _build_prediction_problem(
    world: World,
    human_controls: np.ndarray,
    autonomous_controls: np.ndarray,
    prediction_dt: float,
    risk: dict[str, float | bool],
    vehicle: dict[str, object],
    reference_alpha: np.ndarray,
) -> PredictionAssembly:
    horizon = len(human_controls)
    affine = affine_state_prediction(
        world.state,
        human_controls,
        autonomous_controls,
        reference_alpha,
        prediction_dt,
    )
    human_positions = trajectory_positions(world.state, human_controls, prediction_dt)
    autonomous_positions = trajectory_positions(
        world.state, autonomous_controls, prediction_dt
    )
    # The synthetic constant-hold predictor is deterministic. A calibrated learned model may
    # replace this zero Sigma_H without changing execution or safety-filter uncertainty.
    human_covariance = np.zeros((horizon, 2, 2))
    human_intent = GaussianTrajectory(human_positions, human_covariance)
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
    predictor = ConstantVelocityPredictor(
        process_std=float(risk["obstacle_process_std"])
    )
    obstacle_forecasts = tuple(
        predictor.predict(
            obstacle.position,
            obstacle.velocity,
            horizon,
            prediction_dt,
            initial_std=float(risk["obstacle_position_std"]),
        )
        for obstacle in world.obstacles
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
            zip(world.obstacles, obstacle_forecasts)
        ):
            obstacle_mean = forecast.mean[time_index]
            obstacle_covariance = forecast.covariance[time_index]
            delta = execution.mean[time_index] - obstacle_mean
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                # No separating half-space can be defined at a coincident reference point.
                chance_rows.append(np.zeros(horizon))
                chance_lower.append(1.0)
                radius = world.robot_radius + obstacle.radius + float(risk["clearance"])
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
                world.robot_radius,
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
    if method == "ours":
        return allocator.solve(
            matrix,
            lower,
            human,
            autonomous,
            step_durations=step_durations,
            previous_alpha=previous_alpha,
        )
    raise ValueError(f"unknown method: {method}")


def run_simulation(
    scenario: str | Scenario, config: SimulationConfig | None = None
) -> SimulationResult:
    config = config or SimulationConfig()
    if config.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    scene = make_scenario(scenario) if isinstance(scenario, str) else scenario
    vehicle, risk = load_default("vehicle"), load_default("risk")
    condition = config.network_condition or scene.network_condition
    downlink, uplink = _delay_models(condition)
    network = BidirectionalNetwork(downlink, uplink, seed=config.seed)
    human_controller = LocalController(scene.human_path)
    autonomous_controller = LocalController(scene.autonomous_path)
    world = World(
        scene.initial_state.copy(),
        [spec.instantiate() for spec in scene.obstacles],
        float(vehicle["robot_radius"]),
    )
    buffer: TimestampBuffer[Control] = TimestampBuffer()
    initial_human = human_controller(world.state)
    buffer.push(CommandEnvelope(initial_human, 0.0, 0.0, 0.0))
    control_scales = np.array(
        [
            float(vehicle["control_limits"]["v_max"]),
            max(
                abs(float(vehicle["control_limits"]["omega_min"])),
                abs(float(vehicle["control_limits"]["omega_max"])),
            ),
        ]
    )
    allocator = LexicographicAuthority(
        tolerance=float(risk["lexicographic_tolerance"]),
        control_scales=control_scales,
    )
    safety = RobustCBFFilter(
        dt=config.dt,
        robot_radius=float(vehicle["robot_radius"]),
        clearance=float(risk["clearance"]),
        gamma=float(risk["cbf_gamma"]),
        lookahead=float(risk["cbf_lookahead"]),
        acceleration_bound=float(risk["relative_acceleration_bound"]),
        clock_error_bound=float(risk["clock_error_bound"]),
        model_error_bound=float(risk["model_error_bound"]),
        actuation_error_bound=float(risk["actuation_error_bound"]),
        confidence_beta=float(risk["state_confidence_beta"]),
        limits=vehicle["control_limits"],
    )
    records: list[dict[str, object]] = []
    authority = np.zeros(config.horizon)
    # The paper does not prescribe startup when no previous feasible solution exists.
    linearization_reference = np.ones(config.horizon)
    allocation = _fixed_authority_result(authority, "not-run")
    previous_filtered = Control(0.0, 0.0)
    next_shared_update = 0.0
    last_tube: PredictionRiskTube | None = None
    steps = int(np.ceil(config.duration / config.dt))
    for index in range(steps):
        now = index * config.dt
        network.send_state(world.state.copy(), now)
        for state_packet in network.receive_states(now):
            perceived_state = np.asarray(state_packet.payload, dtype=float)
            remote_command = human_controller(perceived_state)
            # The synthetic remote has zero response time; sent_at is g_m and source_stamp is s_m.
            network.send_command(
                remote_command, now, source_stamp=state_packet.source_stamp
            )
        for command_packet in network.receive_commands(now):
            buffer.push(
                CommandEnvelope(
                    command_packet.payload,
                    command_packet.source_stamp,
                    command_packet.sent_at,
                    now,
                )
            )
        human_control, tau_h = buffer.get(now, initial_human)
        assert human_control is not None
        autonomous_control = autonomous_controller(world.state)
        solve_time = 0.0
        if now + 1e-12 >= next_shared_update:
            human_sequence = _human_candidate_controls(human_control, config.horizon)
            autonomous_sequence = _autonomous_candidate_controls(
                autonomous_controller,
                world.state,
                config.horizon,
                config.prediction_dt,
            )
            prediction = _build_prediction_problem(
                world,
                human_sequence,
                autonomous_sequence,
                config.prediction_dt,
                risk,
                vehicle,
                linearization_reference,
            )
            solve_started = time.perf_counter()
            allocation = _allocate(
                config.method,
                human_sequence,
                autonomous_sequence,
                tau_h,
                prediction.constraints,
                len(world.obstacles),
                allocator,
                np.full(config.horizon, config.prediction_dt),
                float(authority[0]),
            )
            solve_time = time.perf_counter() - solve_started
            authority = allocation.alpha
            last_tube = prediction.risk_tube
            if not allocation.used_fallback:
                linearization_reference = np.r_[authority[1:], authority[-1]]
            next_shared_update += config.shared_control_period

        nominal = Control.from_array(
            blend_controls(
                human_control.as_array()[None, :],
                autonomous_control.as_array()[None, :],
                authority[0],
            )[0]
        )
        obstacle_states = [
            (obstacle.position.copy(), obstacle.radius, obstacle.velocity.copy())
            for obstacle in world.obstacles
        ]
        observations = observations_from_current_estimates(
            world.state,
            previous_filtered,
            obstacle_states,
            [scene.sensor_age] * len(obstacle_states),
            lookahead=float(risk["cbf_lookahead"]),
            position_std=float(risk["state_position_std"]),
            velocity_std=float(risk["state_velocity_std"]),
        )
        filtered = safety.filter(world.state, nominal, observations)
        previous_filtered = filtered.control
        world.state = step(world.state, filtered.control.as_array(), config.dt)
        world.advance_obstacles(config.dt)
        clearance = world.minimum_clearance()
        min_prediction_margin = (
            float(np.min(last_tube.execution_margins))
            if last_tube is not None and last_tube.execution_margins.size
            else float("inf")
        )
        mean_conflict = (
            float(np.mean(last_tube.conflict)) if last_tube is not None else 0.0
        )
        mean_uncertainty = (
            float(np.mean(last_tube.intent_uncertainty))
            if last_tube is not None
            else 0.0
        )
        records.append(
            {
                "time": now + config.dt,
                "state": world.state.tolist(),
                "human": human_control.as_array().tolist(),
                "autonomous": autonomous_control.as_array().tolist(),
                "nominal": nominal.as_array().tolist(),
                "filtered": filtered.control.as_array().tolist(),
                "alpha": float(authority[0]),
                "tau_h": float(tau_h),
                "tau_s": [observation.age for observation in observations],
                "clearance": clearance,
                "filter_triggered": filtered.triggered,
                "filter_infeasible": filtered.infeasible,
                "filter_status": filtered.status,
                "h_minus": filtered.minimum_h_minus,
                "qp_fallback": allocation.used_fallback,
                "upper_infeasible": allocation.used_fallback,
                "secondary_fallback": allocation.secondary_fallback,
                "qp_status": allocation.status,
                "stage1_status": allocation.stage1_status,
                "stage2_status": allocation.stage2_status,
                "minimum_budget": allocation.minimum_budget,
                "budget_bound": allocation.budget_bound,
                "predicted_minimum_margin": min_prediction_margin,
                "mean_conflict": mean_conflict,
                "mean_intent_uncertainty": mean_uncertainty,
                "solve_time": solve_time,
            }
        )
        if np.linalg.norm(world.state[:2] - scene.goal[:2]) <= 0.28:
            break
    metrics = trajectory_metrics(records, config.dt, scene.goal)
    solve_times = np.array(
        [float(row["solve_time"]) for row in records if float(row["solve_time"]) > 0]
    )
    for percentile in (50, 95, 99):
        metrics[f"solve_time_p{percentile}"] = (
            float(np.percentile(solve_times, percentile)) if len(solve_times) else 0.0
        )
    return SimulationResult(scene.name, config.method, config.seed, records, metrics)
