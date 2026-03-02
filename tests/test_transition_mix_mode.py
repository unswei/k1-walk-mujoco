from __future__ import annotations

import numpy as np

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


def _stub_env(
    *,
    mode: str,
    enabled: bool,
    previous_mode: str,
    previous_fraction: float,
    override: str | None = None,
) -> K1WalkEnv:
    env = object.__new__(K1WalkEnv)
    env._task_mode = mode
    env._episode_task_mode = mode
    env._task_mode_override = override
    env.cfg = {
        "task": {
            "transition_mix": {
                "enabled": enabled,
                "previous_mode": previous_mode,
                "previous_fraction": previous_fraction,
            }
        }
    }
    env.np_random = np.random.default_rng(123)
    return env


def test_transition_mix_switches_to_previous_mode() -> None:
    env = _stub_env(
        mode="goal_pose",
        enabled=True,
        previous_mode="command_tracking",
        previous_fraction=1.0,
    )
    K1WalkEnv._resolve_episode_task_mode(env)
    assert env.task_mode == "command_tracking"


def test_transition_mix_respects_explicit_reset_override() -> None:
    env = _stub_env(
        mode="goal_pose",
        enabled=True,
        previous_mode="command_tracking",
        previous_fraction=1.0,
        override="velocity",
    )
    K1WalkEnv._resolve_episode_task_mode(env)
    assert env.task_mode == "velocity"
