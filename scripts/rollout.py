#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time

import mujoco
import numpy as np
import torch

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv
from k1_walk_mujoco.rl.cleanrl.ppo_train import ActorCritic
from k1_walk_mujoco.rl.cleanrl.utils import select_device


def _render_relaunch_cmd() -> str:
    return " ".join(shlex.quote(part) for part in ["mjpython", *sys.argv])


def _libpython_link_cmd() -> str:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if "VIRTUAL_ENV" in os.environ:
        venv_dir = Path(os.environ["VIRTUAL_ENV"]).resolve()
    else:
        mjpython_path = shutil.which("mjpython")
        if mjpython_path is not None:
            venv_dir = Path(mjpython_path).resolve().parent.parent
        else:
            venv_dir = Path(sys.prefix).resolve()
    source = Path(sys.base_prefix) / "lib" / f"libpython{py_ver}.dylib"
    target = venv_dir / f"libpython{py_ver}.dylib"
    return f"ln -sfn {shlex.quote(str(source))} {shlex.quote(str(target))}"


class VideoRecorder:
    def __init__(self, path: Path, fps: int, width: int = 640, height: int = 480) -> None:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "--record requires imageio. Install with `pip install imageio imageio-ffmpeg`."
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = imageio.get_writer(path, fps=fps, codec="libx264")
        self._renderer = None
        self._width = width
        self._height = height

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, width=self._width, height=self._height)
        self._renderer.update_scene(data)
        frame = self._renderer.render()
        self._writer.append_data(frame)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._writer.close()


def _load_agent(ckpt_path: Path, env: K1WalkEnv, device: torch.device) -> ActorCritic:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "agent" not in payload:
        raise RuntimeError(
            f"Checkpoint format not recognised for {ckpt_path}. Expected dict with `agent` key."
        )

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = ActorCritic(obs_dim=obs_dim, action_dim=act_dim).to(device)
    agent.load_state_dict(payload["agent"])
    agent.eval()
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout random policy or checkpoint policy in K1 env.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. If omitted, use random actions.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Base episode seed.")
    parser.add_argument("--render", action="store_true", help="Open interactive MuJoCo viewer.")
    parser.add_argument("--record", type=str, default=None, help="Optional output MP4 path for offscreen recording.")
    parser.add_argument("--deterministic", action="store_true", help="Use mean action for checkpoint playback.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device for policy inference.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/env_k1_walk.yaml",
        help="Environment config path.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard step limit per episode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.render and platform.system() == "Darwin" and "MJPYTHON_BIN" not in os.environ:
        print("macOS rendering requires `mjpython` instead of `python`.")
        print("Re-run with:")
        print(_render_relaunch_cmd())
        print("")
        print("If `mjpython` fails with `Library not loaded: @rpath/libpythonX.Y.dylib`, run:")
        print(_libpython_link_cmd())
        return 2

    env = K1WalkEnv(env_config_path=Path(args.env_config), render_mode="human" if args.render else None)
    device = select_device(args.device)

    agent: ActorCritic | None = None
    if args.ckpt is not None:
        agent = _load_agent(Path(args.ckpt), env, device)

    viewer = None
    if args.render:
        try:
            import mujoco.viewer as mj_viewer

            viewer = mj_viewer.launch_passive(env.backend.model, env.backend.data)
        except Exception as exc:  # pragma: no cover
            print(f"Render requested but viewer is unavailable: {exc}")
            if platform.system() == "Darwin":
                print("")
                print("Re-run with:")
                print(_render_relaunch_cmd())
                print("")
                print("If `mjpython` fails with `Library not loaded: @rpath/libpythonX.Y.dylib`, run:")
                print(_libpython_link_cmd())
            viewer = None
        if viewer is None:
            env.close()
            return 1

    recorder = None
    if args.record is not None:
        fps = max(1, int(round(1.0 / env.policy_dt)))
        recorder = VideoRecorder(path=Path(args.record), fps=fps)

    episode_returns: list[float] = []
    max_steps = args.max_steps if args.max_steps is not None and args.max_steps > 0 else None

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            if recorder is not None:
                recorder.capture(env.backend.model, env.backend.data)

            done = False
            trunc = False
            steps = 0
            ep_return = 0.0
            info: dict[str, object] = {"termination_reason": "none"}

            while not (done or trunc):
                if max_steps is not None and steps >= max_steps:
                    info = {"termination_reason": "max_steps"}
                    break

                step_start = time.perf_counter()
                if agent is None:
                    action = env.action_space.sample()
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action_t = agent.act(obs_t, deterministic=args.deterministic)
                    action = action_t.squeeze(0).cpu().numpy()
                    action = np.clip(action, env.action_space.low, env.action_space.high)

                obs, reward, done, trunc, info = env.step(action)
                ep_return += float(reward)
                steps += 1

                if recorder is not None:
                    recorder.capture(env.backend.model, env.backend.data)
                if viewer is not None:
                    viewer.sync()
                    elapsed = time.perf_counter() - step_start
                    remaining = env.policy_dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            termination_reason = str(info.get("termination_reason", "unknown"))
            episode_returns.append(ep_return)
            print(
                f"episode={ep} return={ep_return:.4f} steps={steps} "
                f"termination_reason={termination_reason}"
            )

    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        env.close()

    if episode_returns:
        print(
            "summary: "
            f"episodes={len(episode_returns)} "
            f"mean_return={float(np.mean(episode_returns)):.4f} "
            f"min_return={float(np.min(episode_returns)):.4f} "
            f"max_return={float(np.max(episode_returns)):.4f}"
        )
    if args.record is not None:
        print(f"recorded_video={Path(args.record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
