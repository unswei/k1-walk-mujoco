from __future__ import annotations

import numpy as np


def wrap_angle_rad(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def world_to_body_xy(
    delta_xy_world: np.ndarray,
    yaw_world_rad: float,
) -> np.ndarray:
    dxy = np.asarray(delta_xy_world, dtype=np.float64).reshape(2)
    c = float(np.cos(yaw_world_rad))
    s = float(np.sin(yaw_world_rad))
    rot_t = np.array([[c, s], [-s, c]], dtype=np.float64)
    return rot_t @ dxy
