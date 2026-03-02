from __future__ import annotations

import numpy as np


def action_to_q_des(
    action: np.ndarray,
    q_nominal: np.ndarray,
    action_scale_rad: float,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    q_des = q_nominal + a * action_scale_rad
    return np.clip(q_des, q_low, q_high)
