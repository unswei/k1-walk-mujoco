from __future__ import annotations

import numpy as np
import pytest

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


@pytest.mark.parametrize("task_mode", ["velocity", "command_tracking", "goal_pose"])
def test_env_mode_smoke_no_nans(task_mode: str) -> None:
    env = K1WalkEnv(cfg_overrides={"task": {"mode": task_mode}})
    obs, _ = env.reset(seed=42)
    assert np.isfinite(obs).all()

    steps = 0
    while steps < 200:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(rew)
        assert "termination_reason" in info
        if term or trunc:
            break
        steps += 1

    env.close()
