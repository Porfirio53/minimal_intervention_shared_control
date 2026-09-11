from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from ..types import Control, FloatArray


@dataclass(frozen=True)
class RelativeStateObservation:
    """Execution-side relative state and uncertainty at one source timestamp."""

    source_relative_position: FloatArray
    source_relative_velocity: FloatArray
    obstacle_velocity: FloatArray
    obstacle_radius: float
    obstacle_speed_bound: float
    age: float
    position_covariance: FloatArray
    position_velocity_covariance: FloatArray
    velocity_position_covariance: FloatArray
    velocity_covariance: FloatArray

    def __post_init__(self) -> None:
        vectors = (
            self.source_relative_position,
            self.source_relative_velocity,
            self.obstacle_velocity,
        )
        matrices = (
            self.position_covariance,
            self.position_velocity_covariance,
            self.velocity_position_covariance,
            self.velocity_covariance,
        )
        if any(np.asarray(value).shape != (2,) for value in vectors):
            raise ValueError("relative-state vectors must have shape (2,)")
        if any(np.asarray(value).shape != (2, 2) for value in matrices):
            raise ValueError("relative-state covariance blocks must have shape (2, 2)")
        if self.obstacle_radius < 0 or self.obstacle_speed_bound < 0 or self.age < 0:
            raise ValueError("radius, speed bound, and age must be non-negative")
        for name in (
            "source_relative_position",
            "source_relative_velocity",
            "obstacle_velocity",
            "position_covariance",
            "position_velocity_covariance",
            "velocity_position_covariance",
            "velocity_covariance",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        joint_covariance = np.block(
            [
                [self.position_covariance, self.position_velocity_covariance],
                [self.velocity_position_covariance, self.velocity_covariance],
            ]
        )
        if not np.allclose(joint_covariance, joint_covariance.T, atol=1e-10):
            raise ValueError("relative position/velocity covariance must be symmetric")
        if float(np.linalg.eigvalsh(joint_covariance).min()) < -1e-10:
            raise ValueError(
                "relative position/velocity covariance must be positive semidefinite"
            )

    def mean_at(self, age: float | None = None) -> FloatArray:
        propagation_age = self.age if age is None else float(age)
        if propagation_age < 0:
            raise ValueError("age must be non-negative")
        return (
            self.source_relative_position
            + propagation_age * self.source_relative_velocity
        )

    def covariance_at(self, age: float | None = None) -> FloatArray:
        """Propagate covariance according to paper equation (25)."""
        propagation_age = self.age if age is None else float(age)
        if propagation_age < 0:
            raise ValueError("age must be non-negative")
        covariance = (
            self.position_covariance
            + propagation_age
            * (self.position_velocity_covariance + self.velocity_position_covariance)
            + propagation_age**2 * self.velocity_covariance
        )
        return 0.5 * (covariance + covariance.T)


@dataclass(frozen=True)
class FilterResult:
    control: Control
    status: str
    modified_norm: float
    triggered: bool
    infeasible: bool
    minimum_h_minus: float


class RobustCBFFilter:
    """Hard execution-side QP implementing paper equations (25)-(37).

    The interface accepts only execution-side relative observations and error bounds. Human
    intent means and covariances are intentionally absent from this safety contract.
    """

    def __init__(
        self,
        dt: float = 0.02,
        robot_radius: float = 0.32,
        clearance: float = 0.08,
        gamma: float = 0.2,
        lookahead: float = 0.18,
        acceleration_bound: float = 0.6,
        clock_error_bound: float = 0.005,
        model_error_bound: float = 0.01,
        actuation_error_bound: float = 0.01,
        confidence_beta: float = 2.448,
        limits: dict[str, float] | None = None,
    ):
        if dt <= 0 or lookahead <= 0 or not 0 < gamma <= 1:
            raise ValueError(
                "dt/lookahead must be positive and gamma must be in (0, 1]"
            )
        error_parameters = (
            acceleration_bound,
            clock_error_bound,
            model_error_bound,
            actuation_error_bound,
        )
        if any(value < 0 for value in error_parameters) or confidence_beta < 0:
            raise ValueError("execution error parameters must be non-negative")
        self.dt = dt
        self.robot_radius = robot_radius
        self.clearance = clearance
        self.gamma = gamma
        self.lookahead = lookahead
        self.acceleration_bound = acceleration_bound
        self.fixed_error_bound = (
            clock_error_bound + model_error_bound + actuation_error_bound
        )
        self.confidence_beta = confidence_beta
        self.limits = limits or {
            "v_min": 0.0,
            "v_max": 1.2,
            "omega_min": -1.8,
            "omega_max": 1.8,
        }

    def _deterministic_radius(self, age: float) -> float:
        return 0.5 * self.acceleration_bound * age**2 + self.fixed_error_bound

    def _directional_support(
        self, observation: RelativeStateObservation, normal: np.ndarray, age: float
    ) -> float:
        covariance = observation.covariance_at(age)
        variance = max(0.0, float(normal @ covariance @ normal))
        return self._deterministic_radius(age) + self.confidence_beta * np.sqrt(
            variance
        )

    def _radial_bound(self, observation: RelativeStateObservation, age: float) -> float:
        covariance = observation.covariance_at(age)
        eigenvalue = max(0.0, float(np.linalg.eigvalsh(covariance).max()))
        return self._deterministic_radius(age) + self.confidence_beta * np.sqrt(
            eigenvalue
        )

    def filter(
        self,
        state: FloatArray,
        nominal: Control,
        observations: list[RelativeStateObservation],
    ) -> FilterResult:
        """Minimally modify the nominal command subject to every hard robust constraint."""
        state_array = np.asarray(state, dtype=float)
        if state_array.shape != (3,):
            raise ValueError("state must have shape (3,)")
        heading = np.array([np.cos(state_array[2]), np.sin(state_array[2])])
        lateral = np.array([-np.sin(state_array[2]), np.cos(state_array[2])])
        input_map = np.column_stack([heading, self.lookahead * lateral])
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        h_minus_values: list[float] = []
        for observation in observations:
            relative_position = observation.mean_at()
            distance = float(np.linalg.norm(relative_position))
            relative_speed_bound = (
                self.limits["v_max"]
                + self.lookahead
                * max(abs(self.limits["omega_min"]), abs(self.limits["omega_max"]))
                + observation.obstacle_speed_bound
            )
            robust_radius = (
                self.robot_radius
                + self.lookahead
                + observation.obstacle_radius
                + self.clearance
                + relative_speed_bound * self.dt
            )
            rho_radial = self._radial_bound(observation, observation.age)
            h_minus_values.append(distance - rho_radial - robust_radius)
            if distance <= 1e-12:
                fallback = Control(0.0, 0.0)
                return FilterResult(
                    fallback,
                    "infeasible: undefined separation normal",
                    float(np.linalg.norm(nominal.as_array())),
                    True,
                    True,
                    min(h_minus_values),
                )
            normal = relative_position / distance
            h_upper = distance + rho_radial - robust_radius
            next_age = observation.age + self.dt
            rho_next = self._directional_support(observation, normal, next_age)
            fixed_next = relative_position - self.dt * observation.obstacle_velocity
            rows.append(self.dt * (normal @ input_map))
            rhs.append(
                float(
                    (1.0 - self.gamma) * h_upper
                    - normal @ fixed_next
                    + rho_next
                    + robust_radius
                )
            )
        minimum_h_minus = min(h_minus_values, default=float("inf"))
        if not rows:
            return FilterResult(nominal, "solved", 0.0, False, False, minimum_h_minus)
        q_nominal = nominal.as_array()
        p = sparse.eye(2, format="csc")
        q = -q_nominal
        a = sparse.vstack(
            [sparse.csc_matrix(rows), sparse.eye(2, format="csc")], format="csc"
        )
        lower = np.r_[rhs, [self.limits["v_min"], self.limits["omega_min"]]]
        upper = np.r_[
            np.full(len(rows), np.inf),
            [self.limits["v_max"], self.limits["omega_max"]],
        ]
        solver = osqp.OSQP()
        solver.setup(
            P=p,
            q=q,
            A=a,
            l=lower,
            u=upper,
            verbose=False,
            polishing=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=10000,
        )
        result = solver.solve(raise_error=False)
        status = str(result.info.status).lower()
        solution_valid = result.x is not None and "solved" in status
        if solution_valid:
            evaluated = np.asarray(a @ result.x[:2]).reshape(-1)
            solution_valid = bool(
                np.all(evaluated >= lower - 2e-6) and np.all(evaluated <= upper + 2e-6)
            )
            if not solution_valid:
                status = f"{status}; constraint residual"
        if not solution_valid or result.x is None:
            # A stop is an explicit platform fallback, not a safety certificate.
            fallback = Control(0.0, 0.0)
            return FilterResult(
                fallback,
                status,
                float(np.linalg.norm(q_nominal)),
                True,
                True,
                minimum_h_minus,
            )
        control = Control.from_array(np.asarray(result.x[:2]))
        modified = float(np.linalg.norm(control.as_array() - q_nominal))
        return FilterResult(
            control,
            status,
            modified,
            modified > 1e-5,
            False,
            minimum_h_minus,
        )
