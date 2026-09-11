from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..autonomy.local_controller import LocalController
from ..config import load_default
from ..dynamics.diff_drive import step
from ..evaluation.metrics import trajectory_metrics
from ..network.delay_channel import BidirectionalNetwork, DelayModel
from ..network.timestamp_buffer import CommandEnvelope
from ..runtime import (
    METHODS,
    ObstacleEstimate,
    RuntimeConfig,
    SharedControlRuntime,
    StepInput,
)
from ..safety.observations import observations_from_current_estimates
from ..types import Control
from .scenarios import Scenario, make_scenario
from .world import World


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
    runtime = SharedControlRuntime(
        RuntimeConfig(
            dt=config.dt,
            shared_control_period=config.shared_control_period,
            prediction_dt=config.prediction_dt,
            horizon=config.horizon,
            method=config.method,
        ),
        vehicle=vehicle,
        risk=risk,
    )
    initial_human = human_controller(world.state)
    previous_filtered = Control(0.0, 0.0)
    records: list[dict[str, object]] = []
    steps = int(np.ceil(config.duration / config.dt))

    for index in range(steps):
        now = index * config.dt
        network.send_state(world.state.copy(), now)
        for state_packet in network.receive_states(now):
            perceived_state = np.asarray(state_packet.payload, dtype=float)
            remote_command = human_controller(perceived_state)
            network.send_command(
                remote_command, now, source_stamp=state_packet.source_stamp
            )
        received_commands = tuple(
            CommandEnvelope(
                packet.payload,
                packet.source_stamp,
                packet.sent_at,
                now,
            )
            for packet in network.receive_commands(now)
        )
        autonomous_sequence = _autonomous_candidate_controls(
            autonomous_controller,
            world.state,
            config.horizon,
            config.prediction_dt,
        )
        obstacle_estimates = tuple(
            ObstacleEstimate(
                obstacle.position.copy(), obstacle.velocity.copy(), obstacle.radius
            )
            for obstacle in world.obstacles
        )
        observations = tuple(
            observations_from_current_estimates(
                world.state,
                previous_filtered,
                [
                    (
                        obstacle.position,
                        obstacle.radius,
                        obstacle.velocity,
                    )
                    for obstacle in obstacle_estimates
                ],
                [scene.sensor_age] * len(obstacle_estimates),
                lookahead=float(risk["cbf_lookahead"]),
                position_std=float(risk["state_position_std"]),
                velocity_std=float(risk["state_velocity_std"]),
            )
        )
        output = runtime.step(
            StepInput(
                now=now,
                state=world.state.copy(),
                human_commands=received_commands,
                default_human=initial_human,
                autonomous_controls=autonomous_sequence,
                obstacles=obstacle_estimates,
                safety_observations=observations,
            )
        )
        filtered = output.filter_result
        allocation = output.allocation
        previous_filtered = filtered.control
        world.state = step(world.state, filtered.control.as_array(), config.dt)
        world.advance_obstacles(config.dt)
        tube = output.risk_tube
        records.append(
            {
                "time": now + config.dt,
                "state": world.state.tolist(),
                "human": output.human_control.as_array().tolist(),
                "autonomous": autonomous_sequence[0].tolist(),
                "nominal": output.nominal_control.as_array().tolist(),
                "filtered": filtered.control.as_array().tolist(),
                "alpha": float(output.alpha[0]),
                "tau_h": output.human_information_age,
                "tau_s": [observation.age for observation in observations],
                "clearance": world.minimum_clearance(),
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
                "predicted_minimum_margin": (
                    float(np.min(tube.execution_margins))
                    if tube.execution_margins.size
                    else float("inf")
                ),
                "mean_conflict": float(np.mean(tube.conflict)),
                "mean_intent_uncertainty": float(np.mean(tube.intent_uncertainty)),
                "human_prediction_source": output.human_prediction_source,
                "obstacle_prediction_source": output.obstacle_prediction_source,
                "scenario_obstacle_source": scene.obstacle_source,
                "solve_time": output.upper_solve_time,
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
