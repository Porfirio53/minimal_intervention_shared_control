from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse

from ..authority.lexicographic_qp import (
    AuthorityResult,
    budget_weights,
    secondary_objective_matrices,
)


def weighted_sum_authority(
    g: np.ndarray,
    b: np.ndarray,
    human: np.ndarray | None = None,
    autonomous: np.ndarray | None = None,
    *,
    intervention_weight: float = 2.0,
    smooth_weight: float = 0.2,
    input_weight: float = 1.0,
    step_durations: np.ndarray | None = None,
    previous_alpha: float = 0.0,
    control_scales: np.ndarray | None = None,
) -> AuthorityResult:
    """One-stage w_alpha I_alpha + J_secondary baseline from the design."""
    g, b = np.asarray(g, dtype=float), np.asarray(b, dtype=float)
    if g.ndim != 2 or b.shape != (len(g),):
        raise ValueError("g must be (constraints, horizon), b must match")
    horizon = g.shape[1]
    weights = budget_weights(horizon, step_durations)
    p, secondary_q = secondary_objective_matrices(
        human,
        autonomous,
        horizon,
        previous_alpha=previous_alpha,
        smooth_weight=smooth_weight,
        input_weight=input_weight,
        control_scales=control_scales,
    )
    q = secondary_q + intervention_weight * weights
    identity = sparse.eye(horizon, format="csc")
    constraints = sparse.vstack([sparse.csc_matrix(g), identity], format="csc")
    lower = np.r_[b, np.zeros(horizon)]
    upper = np.r_[np.full(len(g), np.inf), np.ones(horizon)]
    solver = osqp.OSQP()
    solver.setup(
        P=p,
        q=q,
        A=constraints,
        l=lower,
        u=upper,
        verbose=False,
        polishing=False,
        eps_abs=1e-7,
        eps_rel=1e-7,
        max_iter=20000,
    )
    result = solver.solve(raise_error=False)
    status = str(result.info.status).lower()
    if result.x is None or "solved" not in status:
        return AuthorityResult(
            np.zeros(horizon),
            f"{status}; fallback",
            0.0,
            0.0,
            True,
            stage1_status=status,
        )
    alpha = np.clip(np.asarray(result.x), 0.0, 1.0)
    return AuthorityResult(
        alpha,
        status,
        float(weights @ alpha),
        float(result.info.obj_val),
        False,
        stage1_status=status,
    )
