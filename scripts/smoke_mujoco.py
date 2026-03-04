#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from k1_walk_mujoco.assets.verify import ensure_k1_assets_present
from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


def _run_mujoco_smoke(*, render: bool, steps: int) -> int:
    env = K1WalkEnv(render_mode="human" if render else None)
    obs, info = env.reset(seed=123)
    del info
    rewards = []
    termination = None

    viewer = None
    if render:
        try:
            import mujoco.viewer as mj_viewer

            viewer = mj_viewer.launch_passive(env.backend.model, env.backend.data)
        except Exception as e:  # pragma: no cover
            print(f"Viewer unavailable: {e}")
            viewer = None

    for _ in range(steps):
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

    print("backend: mujoco")
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


def _run_warp_smoke(*, mjcf_path: Path, steps: int) -> int:
    try:
        import mujoco_warp as mjw
    except ImportError as exc:
        print("MuJoCo Warp backend requested but mujoco-warp is not installed.")
        print("Install with: uv pip install -e '.[warp,dev]'")
        print(f"Import error: {exc}")
        return 2

    model_cpu = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data_cpu = mujoco.MjData(model_cpu)
    model_warp = mjw.put_model(model_cpu)
    data_warp = mjw.put_data(model_cpu, data_cpu, nworld=1)

    for _ in range(steps):
        mjw.step(model_warp, data_warp)

    mjw.get_data_into(data_cpu, model_cpu, data_warp)
    obs = np.concatenate([data_cpu.qpos.copy(), data_cpu.qvel.copy()])

    print("backend: warp")
    print(f"obs shape: {obs.shape}")
    print(f"action shape: ({model_cpu.nu},)")
    print(f"qpos norm: {float(np.linalg.norm(data_cpu.qpos)):.4f}")
    print(f"qvel norm: {float(np.linalg.norm(data_cpu.qvel)):.4f}")
    print(f"steps: {steps}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("mujoco", "warp"),
        default="mujoco",
        help="Simulation backend for smoke run",
    )
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo passive viewer")
    parser.add_argument("--steps", type=int, default=200, help="Number of policy/sim steps")
    args = parser.parse_args()

    try:
        mjcf_path = ensure_k1_assets_present()
    except FileNotFoundError as e:
        print(e)
        print("python scripts/fetch_assets.py")
        return 1

    if args.backend == "warp":
        if args.render:
            print("Rendering is only supported with --backend mujoco.")
            return 2
        return _run_warp_smoke(mjcf_path=mjcf_path, steps=args.steps)

    return _run_mujoco_smoke(render=args.render, steps=args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
