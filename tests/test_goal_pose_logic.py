from __future__ import annotations

import numpy as np

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


def test_goal_pose_success_hold_steps() -> None:
    env = K1WalkEnv(
        cfg_overrides={
            "task": {
                "mode": "goal_pose",
                "goal": {
                    "success_pos_tol_m": 100.0,
                    "success_yaw_tol_deg": 180.0,
                    "goal_reach_hold_steps": 3,
                },
            },
            "termination": {"terminate_on_success_train": False},
        }
    )
    env.reset(seed=123)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    success_flags: list[bool] = []
    for _ in range(3):
        _obs, _rew, term, trunc, info = env.step(action)
        success_flags.append(bool(info["is_success"]))
        assert not term
        assert not trunc

    assert success_flags == [False, False, True]
    env.close()


def test_goal_pose_reward_terms_finite() -> None:
    env = K1WalkEnv(cfg_overrides={"task": {"mode": "goal_pose"}})
    env.reset(seed=321)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _obs, rew, _term, _trunc, info = env.step(action)
    assert np.isfinite(rew)
    for key in (
        "r_progress",
        "r_goal_bonus",
        "r_goal_yaw",
        "distance_to_goal",
        "yaw_error_rad",
        "time_to_goal_s",
    ):
        assert key in info
        assert np.isfinite(float(info[key])) or key == "time_to_goal_s"
    env.close()
