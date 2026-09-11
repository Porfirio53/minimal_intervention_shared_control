from __future__ import annotations

import numpy as np

from ..risk.conflict import direction_cosine


def delay_agreement_authority(
    human: np.ndarray,
    autonomous: np.ndarray,
    delay: float,
    horizon: int,
    delay_gain: float = 0.8,
) -> np.ndarray:
    agreement = max(direction_cosine(human, autonomous), 0.0)
    authority = agreement ** max(0.0, 1.0 - delay_gain * max(0.0, delay))
    # The paper's alpha denotes autonomous authority, while agreement denotes human authority.
    return np.full(horizon, float(np.clip(1.0 - authority, 0.0, 1.0)))
