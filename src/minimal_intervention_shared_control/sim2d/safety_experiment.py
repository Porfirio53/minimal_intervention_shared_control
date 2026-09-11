from __future__ import annotations

import numpy as np

from ..config import load_default
from ..dynamics.diff_drive import step
from ..evaluation.metrics import trajectory_metrics
from ..safety.observations import observations_from_current_estimates
from ..safety.robust_cbf_qp import RobustCBFFilter
from ..types import Control
from .scenarios import Scenario, make_scenario
from .simulation import SimulationResult
from .world import World

FILTER_MODES = ("none", "standard", "robust")


def replay_safety_filter(
    scenario: str | Scenario,
    nominal_records: list[dict[str, object]],
    mode: str,
    tau_s: float,
    dt: float = 0.02,
) -> SimulationResult:
    """Replay one fixed nominal-command log through an execution filter.

    Nominal commands are never regenerated from the replayed state, keeping E2 controlled.
    """
    if mode not in FILTER_MODES:
        raise ValueError(f"mode must be one of {FILTER_MODES}")
    if tau_s < 0:
        raise ValueError("tau_s must be non-negative")
    scene = make_scenario(scenario) if isinstance(scenario, str) else scenario
    vehicle, risk = load_default("vehicle"), load_default("risk")
    world = World(
        scene.initial_state.copy(),
        [spec.instantiate() for spec in scene.obstacles],
        float(vehicle["robot_radius"]),
    )
    robust = mode == "robust"
    safety = RobustCBFFilter(
        dt=dt,
        robot_radius=float(vehicle["robot_radius"]),
        clearance=float(risk["clearance"]),
        gamma=float(risk["cbf_gamma"]),
        lookahead=float(risk["cbf_lookahead"]),
        acceleration_bound=float(risk["relative_acceleration_bound"])
        if robust
        else 0.0,
        clock_error_bound=float(risk["clock_error_bound"]) if robust else 0.0,
        model_error_bound=float(risk["model_error_bound"]) if robust else 0.0,
        actuation_error_bound=float(risk["actuation_error_bound"]) if robust else 0.0,
        confidence_beta=float(risk["state_confidence_beta"]),
        limits=vehicle["control_limits"],
    )
    replayed: list[dict[str, object]] = []
    previous_filtered = Control(0.0, 0.0)
    for source in nominal_records:
        nominal = Control.from_array(np.asarray(source["nominal"], dtype=float))
        obstacles = [
            (obstacle.position.copy(), obstacle.radius, obstacle.velocity.copy())
            for obstacle in world.obstacles
        ]
        observation_age = tau_s if robust else 0.0
        observations = observations_from_current_estimates(
            world.state,
            previous_filtered,
            obstacles,
            [observation_age] * len(obstacles),
            lookahead=float(risk["cbf_lookahead"]),
            position_std=float(risk["state_position_std"]) if robust else 0.0,
            velocity_std=float(risk["state_velocity_std"]) if robust else 0.0,
        )
        if mode == "none":
            filtered, triggered, infeasible, status, h_minus = (
                nominal,
                False,
                False,
                "not-run",
                None,
            )
        else:
            result = safety.filter(world.state, nominal, observations)
            filtered, triggered, infeasible, status, h_minus = (
                result.control,
                result.triggered,
                result.infeasible,
                result.status,
                result.minimum_h_minus,
            )
        previous_filtered = filtered
        world.state = step(world.state, filtered.as_array(), dt)
        world.advance_obstacles(dt)
        replayed.append(
            {
                **source,
                "state": world.state.tolist(),
                "filtered": filtered.as_array().tolist(),
                "tau_s": [observation.age for observation in observations],
                "clearance": world.minimum_clearance(),
                "filter_triggered": triggered,
                "filter_infeasible": infeasible,
                "filter_status": status,
                "h_minus": h_minus,
            }
        )
    metrics = trajectory_metrics(replayed, dt, scene.goal)
    return SimulationResult(scene.name, f"safety_{mode}", 0, replayed, metrics)
