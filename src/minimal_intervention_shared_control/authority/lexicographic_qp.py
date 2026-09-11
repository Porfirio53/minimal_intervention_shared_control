from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse


@dataclass(frozen=True)
class AuthorityResult:
    alpha: np.ndarray
    status: str
    intervention_budget: float
    objective: float
    used_fallback: bool = False
    minimum_budget: float | None = None
    budget_bound: float | None = None
    stage1_status: str = "not-run"
    stage2_status: str = "not-run"
    secondary_fallback: bool = False


def budget_weights(
    horizon: int, step_durations: np.ndarray | None = None
) -> np.ndarray:
    durations = (
        np.ones(horizon, dtype=float)
        if step_durations is None
        else np.asarray(step_durations, dtype=float)
    )
    if durations.shape != (horizon,) or np.any(durations <= 0):
        raise ValueError("step durations must be positive and match the horizon")
    return durations / np.sum(durations)


def secondary_objective_matrices(
    human: np.ndarray | None,
    autonomous: np.ndarray | None,
    horizon: int,
    *,
    previous_alpha: float = 0.0,
    smooth_weight: float = 0.2,
    input_weight: float = 1.0,
    authority_weight: float = 1e-6,
    control_scales: np.ndarray | None = None,
    task_offset: np.ndarray | None = None,
    task_sensitivity: np.ndarray | None = None,
    task_positions: np.ndarray | None = None,
    task_weight: float = 0.0,
    position_scales: np.ndarray | None = None,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Build the implemented subset of paper equation (19)."""
    difference_matrix = sparse.eye(horizon, format="lil")
    if horizon > 1:
        difference_matrix[np.arange(1, horizon), np.arange(horizon - 1)] = -1.0
    difference_matrix = difference_matrix.tocsc()
    difference_target = np.zeros(horizon)
    difference_target[0] = previous_alpha
    scales = (
        np.array([1.0, 1.0])
        if control_scales is None
        else np.asarray(control_scales, dtype=float)
    )
    if scales.shape != (2,) or np.any(scales <= 0):
        raise ValueError("control_scales must contain two positive reference values")
    modification_cost = np.zeros(horizon)
    if human is not None or autonomous is not None:
        if human is None or autonomous is None:
            raise ValueError("human and autonomous controls must be supplied together")
        human_array = np.asarray(human, dtype=float)
        autonomous_array = np.asarray(autonomous, dtype=float)
        if human_array.shape != (horizon, 2) or autonomous_array.shape != (
            horizon,
            2,
        ):
            raise ValueError("candidate controls must have shape (horizon, 2)")
        modification_cost = np.sum(
            ((autonomous_array - human_array) / scales) ** 2, axis=1
        )
    quadratic = smooth_weight * (
        difference_matrix.T @ difference_matrix
    ) + sparse.diags(input_weight * modification_cost + authority_weight, format="csc")
    linear = -smooth_weight * (difference_matrix.T @ difference_target)
    if task_weight < 0:
        raise ValueError("task weight must be non-negative")
    task_values = (task_offset, task_sensitivity, task_positions)
    if task_weight > 0 or any(value is not None for value in task_values):
        if any(value is None for value in task_values):
            raise ValueError(
                "task objective requires offset, sensitivity, and positions"
            )
        offset = np.asarray(task_offset, dtype=float)
        sensitivity = np.asarray(task_sensitivity, dtype=float)
        target = np.asarray(task_positions, dtype=float)
        position_scale = (
            np.ones(2)
            if position_scales is None
            else np.asarray(position_scales, dtype=float)
        )
        if (
            offset.shape != (horizon, 2)
            or sensitivity.shape != (horizon, horizon, 2)
            or target.shape != (horizon, 2)
            or position_scale.shape != (2,)
            or np.any(position_scale <= 0)
        ):
            raise ValueError("task objective arrays have incompatible shapes")
        task_matrix = np.empty((2 * horizon, horizon))
        for index in range(horizon):
            task_matrix[2 * index : 2 * index + 2] = (
                sensitivity[index].T / position_scale[:, None]
            )
        task_residual = ((offset - target) / position_scale).reshape(-1)
        quadratic = (
            quadratic + task_weight * sparse.csc_matrix(task_matrix.T @ task_matrix)
        ).tocsc()
        linear = linear + task_weight * (task_matrix.T @ task_residual)
    return quadratic, np.asarray(linear).reshape(-1)


class LexicographicAuthority:
    """Paper equations (20)-(21) over a preassembled hard feasible set."""

    def __init__(
        self,
        tolerance: float = 0.0,
        smooth_weight: float = 0.2,
        input_weight: float = 1.0,
        authority_weight: float = 1e-6,
        control_scales: np.ndarray | None = None,
    ):
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance
        self.smooth_weight = smooth_weight
        self.input_weight = input_weight
        self.authority_weight = authority_weight
        self.control_scales = control_scales

    @staticmethod
    def _satisfies(
        value: np.ndarray,
        matrix: sparse.csc_matrix,
        lower: np.ndarray,
        upper: np.ndarray,
        tolerance: float = 2e-6,
    ) -> bool:
        evaluated = np.asarray(matrix @ value).reshape(-1)
        return bool(
            np.all(evaluated >= lower - tolerance)
            and np.all(evaluated <= upper + tolerance)
        )

    @staticmethod
    def _solve(
        p: sparse.csc_matrix,
        q: np.ndarray,
        a: sparse.csc_matrix,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray | None, str, float]:
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
            max_iter=20000,
        )
        result = solver.solve(raise_error=False)
        value = float(result.info.obj_val) if result.x is not None else float("nan")
        return (
            None if result.x is None else np.asarray(result.x),
            str(result.info.status).lower(),
            value,
        )

    def solve(
        self,
        g: np.ndarray,
        b: np.ndarray,
        human: np.ndarray | None = None,
        autonomous: np.ndarray | None = None,
        *,
        step_durations: np.ndarray | None = None,
        previous_alpha: float = 0.0,
        task_offset: np.ndarray | None = None,
        task_sensitivity: np.ndarray | None = None,
        task_positions: np.ndarray | None = None,
        task_weight: float = 0.0,
    ) -> AuthorityResult:
        g, b = np.asarray(g, dtype=float), np.asarray(b, dtype=float)
        if g.ndim != 2 or b.shape != (len(g),):
            raise ValueError("g must be (constraints, horizon), b must match")
        horizon = g.shape[1]
        weights = budget_weights(horizon, step_durations)
        identity = sparse.eye(horizon, format="csc")
        constraints = sparse.vstack([sparse.csc_matrix(g), identity], format="csc")
        lower = np.r_[b, np.zeros(horizon)]
        upper = np.r_[np.full(len(g), np.inf), np.ones(horizon)]

        first, stage1_status, _ = self._solve(
            sparse.csc_matrix((horizon, horizon)),
            weights,
            constraints,
            lower,
            upper,
        )
        first_valid = first is not None and "solved" in stage1_status
        if first_valid:
            first = np.clip(first, 0.0, 1.0)
            first_valid = self._satisfies(first, constraints, lower, upper)
            if not first_valid:
                stage1_status = f"{stage1_status}; constraint residual"
        if not first_valid or first is None:
            return AuthorityResult(
                np.zeros(horizon),
                f"{stage1_status}; fallback",
                0.0,
                0.0,
                True,
                None,
                None,
                stage1_status,
            )
        minimum_budget = float(weights @ first)
        budget_bound = minimum_budget + self.tolerance

        p2, q2 = secondary_objective_matrices(
            human,
            autonomous,
            horizon,
            previous_alpha=previous_alpha,
            smooth_weight=self.smooth_weight,
            input_weight=self.input_weight,
            authority_weight=self.authority_weight,
            control_scales=self.control_scales,
            task_offset=task_offset,
            task_sensitivity=task_sensitivity,
            task_positions=task_positions,
            task_weight=task_weight,
        )
        budget_row = sparse.csc_matrix(weights.reshape(1, -1))
        constraints2 = sparse.vstack(
            [sparse.csc_matrix(g), budget_row, identity], format="csc"
        )
        lower2 = np.r_[b, -np.inf, np.zeros(horizon)]
        upper2 = np.r_[np.full(len(g), np.inf), budget_bound, np.ones(horizon)]
        second, stage2_status, objective = self._solve(
            p2, q2, constraints2, lower2, upper2
        )
        second_valid = second is not None and "solved" in stage2_status
        if second_valid:
            second = np.clip(second, 0.0, 1.0)
            second_valid = self._satisfies(second, constraints2, lower2, upper2)
            if not second_valid:
                stage2_status = f"{stage2_status}; constraint residual"
        secondary_fallback = not second_valid
        alpha = first if secondary_fallback else second
        assert alpha is not None
        return AuthorityResult(
            alpha=alpha,
            status=stage1_status if secondary_fallback else stage2_status,
            intervention_budget=float(weights @ alpha),
            objective=objective,
            used_fallback=False,
            minimum_budget=minimum_budget,
            budget_bound=budget_bound,
            stage1_status=stage1_status,
            stage2_status=stage2_status,
            secondary_fallback=secondary_fallback,
        )
