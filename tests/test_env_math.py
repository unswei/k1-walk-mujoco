from __future__ import annotations

import numpy as np

from k1_walk_mujoco.envs.math_utils import quat_wxyz_to_yaw, world_to_body_xy, wrap_angle_rad


def test_wrap_angle_range() -> None:
    assert np.isclose(wrap_angle_rad(0.0), 0.0)
    assert np.isclose(wrap_angle_rad(np.pi), -np.pi)
    assert np.isclose(wrap_angle_rad(-3.0 * np.pi), -np.pi)


def test_quat_yaw_identity_and_half_turn() -> None:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert np.isclose(quat_wxyz_to_yaw(q_identity), 0.0)

    q_half_turn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    yaw = quat_wxyz_to_yaw(q_half_turn)
    assert np.isclose(abs(yaw), np.pi)


def test_world_to_body_xy_rotation() -> None:
    dxy_world = np.array([1.0, 0.0], dtype=np.float64)
    dxy_body = world_to_body_xy(dxy_world, yaw_world_rad=np.pi / 2.0)
    assert np.allclose(dxy_body, np.array([0.0, -1.0]), atol=1e-6)
