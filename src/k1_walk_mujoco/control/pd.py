from __future__ import annotations

import numpy as np


def compute_pd_torque(
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    effort_limit: np.ndarray,
    qd_des: np.ndarray | None = None,
    tau_ff: np.ndarray | None = None,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    q_des = np.asarray(q_des, dtype=np.float64)
    kp = np.asarray(kp, dtype=np.float64)
    kd = np.asarray(kd, dtype=np.float64)

    if qd_des is None:
        qd_des = np.zeros_like(q)
    if tau_ff is None:
        tau_ff = np.zeros_like(q)

    tau = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
    return np.clip(tau, -effort_limit, effort_limit)
