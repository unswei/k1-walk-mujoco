#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from k1_walk_mujoco.assets.verify import ensure_k1_assets_present
from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo passive viewer")
    args = parser.parse_args()

    try:
        ensure_k1_assets_present()
    except FileNotFoundError as e:
        print(e)
        print("python scripts/fetch_assets.py")
        return 1

    env = K1WalkEnv(render_mode="human" if args.render else None)
    obs, info = env.reset(seed=123)
    del info
    rewards = []
    termination = None

    viewer = None
    if args.render:
        try:
            import mujoco.viewer as mj_viewer

            viewer = mj_viewer.launch_passive(env.backend.model, env.backend.data)
        except Exception as e:  # pragma: no cover
            print(f"Viewer unavailable: {e}")
            viewer = None

    for _ in range(200):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)
        rewards.append(rew)
        if viewer is not None:
            viewer.sync()
        if term or trunc:
            termination = info.get("termination_reason", "unknown")
            break

    if viewer is not None:
        viewer.close()

    print(f"obs shape: {obs.shape}")
    print(f"action shape: {env.action_space.shape}")
    print(
        "reward summary: "
        f"mean={float(np.mean(rewards)):.4f}, "
        f"min={float(np.min(rewards)):.4f}, "
        f"max={float(np.max(rewards)):.4f}, "
        f"steps={len(rewards)}"
    )
    print(f"termination reason: {termination or 'none'}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
