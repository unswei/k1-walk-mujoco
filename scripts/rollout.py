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
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

from k1_walk_mujoco.controllers.param_gait_15 import ParamGait15
from k1_walk_mujoco.control.pd import compute_pd_torque
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
    def __init__(
        self,
        path: Path,
        fps: int,
        width: int = 640,
        height: int = 480,
        camera_mode: str = "free",
        track_body: str | None = None,
        fixed_camera_id: int | None = None,
    ) -> None:
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
        self._camera_mode = camera_mode
        self._track_body = track_body
        self._fixed_camera_id = fixed_camera_id
        self._camera = None

    def capture(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, width=self._width, height=self._height)
            self._camera = self._build_camera(model=model)
        self._renderer.update_scene(data, camera=self._camera)
        frame = self._renderer.render()
        self._writer.append_data(frame)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._writer.close()

    def _build_camera(self, model: mujoco.MjModel):
        if self._camera_mode == "free":
            if self._fixed_camera_id is None:
                return -1
            if self._fixed_camera_id < 0 or self._fixed_camera_id >= model.ncam:
                raise ValueError(
                    f"--camera-id must be in [0, {model.ncam - 1}] for this model; got {self._fixed_camera_id}"
                )
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = int(self._fixed_camera_id)
            return cam

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        body_name = self._track_body or "base"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            fallback_names = ("base_link", "torso", "pelvis", "base")
            for candidate in fallback_names:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
                if body_id >= 0:
                    break
        if body_id < 0:
            raise ValueError(
                f"Unable to resolve tracking body `{body_name}`. "
                "Pass --track-body with a valid MuJoCo body name."
            )
        cam.trackbodyid = int(body_id)
        cam.distance = 2.5
        cam.elevation = -15.0
        return cam


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


def _load_gait_controller(path: Path, env: K1WalkEnv) -> ParamGait15:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping in gait config {path}, got {type(cfg)!r}")

    controller_name = str(cfg.get("controller", "param_gait_15"))
    if controller_name != "param_gait_15":
        raise ValueError(f"Unsupported gait controller in {path}: {controller_name}")

    return ParamGait15.from_config(cfg, controlled_joints=env.j.names)


def _parse_fixed_command(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--fixed-command must be 'vx,vy,yaw_rate'")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _build_reset_options(args: argparse.Namespace, fixed_command: list[float] | None) -> dict[str, object]:
    reset_options: dict[str, object] = {}
    if args.task_mode is not None:
        reset_options["task_mode"] = args.task_mode
    if args.goal_x is not None and args.goal_y is not None:
        reset_options["goal_xy"] = [args.goal_x, args.goal_y]
    if args.goal_yaw_deg is not None:
        reset_options["goal_yaw_deg"] = args.goal_yaw_deg
    if fixed_command is not None:
        reset_options["command"] = fixed_command
    return reset_options


def _compute_tilt(quat_wxyz: np.ndarray) -> float:
    qw = float(np.clip(abs(quat_wxyz[0]), 0.0, 1.0))
    return 2.0 * np.arccos(qw)


def _run_policy_episode(
    *,
    env: K1WalkEnv,
    args: argparse.Namespace,
    agent: ActorCritic | None,
    device: torch.device,
    viewer: Any,
    recorder: VideoRecorder | None,
    obs: np.ndarray,
    max_steps: int | None,
) -> dict[str, float | int | str | bool]:
    done = False
    trunc = False
    steps = 0
    ep_return = 0.0
    info: dict[str, object] = {"termination_reason": "none"}

    while not (done or trunc):
        if max_steps is not None and steps >= max_steps:
            info["termination_reason"] = "max_steps"
            break

        step_start = time.perf_counter()
        if agent is None:
            if args.policy == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
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
    final_distance = float(info.get("distance_to_goal", float("nan")))
    final_yaw_err_deg = float(np.rad2deg(abs(float(info.get("yaw_error_rad", float("nan"))))))
    is_success = bool(info.get("is_success", False))

    return {
        "return": ep_return,
        "steps": steps,
        "termination_reason": termination_reason,
        "distance_to_goal": final_distance,
        "final_yaw_err_deg": final_yaw_err_deg,
        "success": is_success,
    }


def _run_gait_episode(
    *,
    env: K1WalkEnv,
    controller: ParamGait15,
    viewer: Any,
    recorder: VideoRecorder | None,
    max_steps: int | None,
) -> dict[str, float | int | str | bool]:
    controller.reset()

    term_cfg = env.cfg["termination"]
    min_base_height = float(term_cfg["min_base_height_m"])
    max_tilt = float(term_cfg["max_tilt_rad"])

    steps = 0
    termination_reason = "none"
    vx_hist: list[float] = []
    vy_abs_hist: list[float] = []
    yaw_abs_hist: list[float] = []
    tau_abs_hist: list[float] = []
    limit_hits = 0
    t = 0.0

    while True:
        if max_steps is not None and steps >= max_steps:
            termination_reason = "max_steps"
            break

        step_start = time.perf_counter()
        s = env.backend.get_state()
        q_des = controller.step(t=t, dt=env.policy_dt)
        q_des = np.clip(q_des, env.j.q_low, env.j.q_high)
        tau = compute_pd_torque(
            q=s.joint_qpos,
            qd=s.joint_qvel,
            q_des=q_des,
            kp=env.j.kp,
            kd=env.j.kd,
            effort_limit=env.j.effort,
            qd_des=np.zeros_like(env.j.q_nominal),
            tau_ff=np.zeros_like(env.j.q_nominal),
        )
        env.backend.step(tau=tau, n_substeps=env.decimation)

        s_next = env.backend.get_state()
        vx_hist.append(float(s_next.base_lin_vel[0]))
        vy_abs_hist.append(abs(float(s_next.base_lin_vel[1])))
        yaw_abs_hist.append(abs(float(s_next.base_ang_vel[2])))
        tau_abs_hist.append(float(np.mean(np.abs(tau))))

        is_low = np.isclose(q_des, env.j.q_low, atol=1e-4)
        is_high = np.isclose(q_des, env.j.q_high, atol=1e-4)
        limit_hits += int(np.count_nonzero(is_low | is_high))

        steps += 1
        t += env.policy_dt

        if recorder is not None:
            recorder.capture(env.backend.model, env.backend.data)
        if viewer is not None:
            viewer.sync()
            elapsed = time.perf_counter() - step_start
            remaining = env.policy_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

        tilt = _compute_tilt(s_next.base_quat)
        if s_next.base_pos[2] < min_base_height:
            termination_reason = "base_height"
            break
        if tilt > max_tilt:
            termination_reason = "tilt"
            break
        if steps >= env.max_steps:
            termination_reason = "time_limit"
            break

    mean_vx = float(np.mean(vx_hist)) if vx_hist else 0.0
    mean_abs_vy = float(np.mean(vy_abs_hist)) if vy_abs_hist else 0.0
    mean_abs_yaw = float(np.mean(yaw_abs_hist)) if yaw_abs_hist else 0.0
    mean_abs_tau = float(np.mean(tau_abs_hist)) if tau_abs_hist else 0.0

    return {
        "steps": steps,
        "termination_reason": termination_reason,
        "mean_vx": mean_vx,
        "mean_abs_vy": mean_abs_vy,
        "mean_abs_yaw_rate": mean_abs_yaw,
        "mean_abs_tau": mean_abs_tau,
        "joint_limit_hits": limit_hits,
        "success": termination_reason == "time_limit",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout policy or classical controller in K1 env.")
    parser.add_argument(
        "--controller",
        type=str,
        default="policy",
        choices=["policy", "param_gait_15"],
        help="Controller source. `policy` uses random/zero/ckpt, `param_gait_15` uses gait targets.",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. If omitted, use random/zero.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Base episode seed.")
    parser.add_argument("--render", action="store_true", help="Open interactive MuJoCo viewer.")
    parser.add_argument("--record", type=str, default=None, help="Optional output MP4 path for offscreen recording.")
    parser.add_argument("--deterministic", action="store_true", help="Use mean action for checkpoint playback.")
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "zero"],
        help="Action source when --controller=policy and --ckpt is omitted.",
    )
    parser.add_argument(
        "--gait-config",
        type=str,
        default="configs/gait_param_15.yaml",
        help="Gait config path for --controller=param_gait_15.",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default=None,
        choices=["velocity", "command_tracking", "goal_pose"],
        help="Optional task mode override for env reset.",
    )
    parser.add_argument("--goal-x", type=float, default=None, help="Goal X position in world frame.")
    parser.add_argument("--goal-y", type=float, default=None, help="Goal Y position in world frame.")
    parser.add_argument("--goal-yaw-deg", type=float, default=None, help="Goal yaw (degrees, world frame).")
    parser.add_argument(
        "--fixed-command",
        type=str,
        default=None,
        help="Optional fixed command as 'vx,vy,yaw_rate'.",
    )
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
    parser.add_argument(
        "--record-camera",
        type=str,
        default="free",
        choices=["free", "track"],
        help="Recording camera mode: free camera or body-tracking camera.",
    )
    parser.add_argument(
        "--track-body",
        type=str,
        default="base",
        help="Body name to track when --record-camera=track.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        help="Optional fixed camera id for recording (overrides free camera).",
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

    if args.controller == "param_gait_15" and args.ckpt is not None:
        raise ValueError("--ckpt cannot be used with --controller=param_gait_15")

    env = K1WalkEnv(env_config_path=Path(args.env_config), render_mode="human" if args.render else None)
    device = select_device(args.device)

    agent: ActorCritic | None = None
    gait_controller: ParamGait15 | None = None

    if args.controller == "policy":
        if args.ckpt is not None:
            agent = _load_agent(Path(args.ckpt), env, device)
    else:
        gait_controller = _load_gait_controller(Path(args.gait_config), env)

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
        recorder = VideoRecorder(
            path=Path(args.record),
            fps=fps,
            camera_mode=args.record_camera,
            track_body=args.track_body,
            fixed_camera_id=args.camera_id,
        )

    episode_returns: list[float] = []
    episode_vx: list[float] = []
    max_steps = args.max_steps if args.max_steps is not None and args.max_steps > 0 else None

    try:
        fixed_command = _parse_fixed_command(args.fixed_command)

        for ep in range(args.episodes):
            reset_options = _build_reset_options(args, fixed_command)
            obs, _ = env.reset(seed=args.seed + ep, options=reset_options if reset_options else None)
            if recorder is not None:
                recorder.capture(env.backend.model, env.backend.data)

            if args.controller == "policy":
                result = _run_policy_episode(
                    env=env,
                    args=args,
                    agent=agent,
                    device=device,
                    viewer=viewer,
                    recorder=recorder,
                    obs=obs,
                    max_steps=max_steps,
                )
                ep_return = float(result["return"])
                episode_returns.append(ep_return)
                print(
                    f"episode={ep} return={ep_return:.4f} steps={int(result['steps'])} "
                    f"termination_reason={result['termination_reason']} success={bool(result['success'])} "
                    f"distance_to_goal={float(result['distance_to_goal']):.3f} "
                    f"final_yaw_err_deg={float(result['final_yaw_err_deg']):.2f}"
                )
            else:
                assert gait_controller is not None
                result = _run_gait_episode(
                    env=env,
                    controller=gait_controller,
                    viewer=viewer,
                    recorder=recorder,
                    max_steps=max_steps,
                )
                episode_vx.append(float(result["mean_vx"]))
                print(
                    f"episode={ep} controller=param_gait_15 steps={int(result['steps'])} "
                    f"termination_reason={result['termination_reason']} success={bool(result['success'])} "
                    f"mean_vx={float(result['mean_vx']):.3f} "
                    f"mean_abs_vy={float(result['mean_abs_vy']):.3f} "
                    f"mean_abs_yaw_rate={float(result['mean_abs_yaw_rate']):.3f} "
                    f"mean_abs_tau={float(result['mean_abs_tau']):.3f} "
                    f"joint_limit_hits={int(result['joint_limit_hits'])}"
                )

    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        env.close()

    if args.controller == "policy" and episode_returns:
        print(
            "summary: "
            f"episodes={len(episode_returns)} "
            f"mean_return={float(np.mean(episode_returns)):.4f} "
            f"min_return={float(np.min(episode_returns)):.4f} "
            f"max_return={float(np.max(episode_returns)):.4f}"
        )

    if args.controller == "param_gait_15" and episode_vx:
        print(
            "summary: "
            f"episodes={len(episode_vx)} "
            f"mean_forward_velocity={float(np.mean(episode_vx)):.4f} "
            f"min_forward_velocity={float(np.min(episode_vx)):.4f} "
            f"max_forward_velocity={float(np.max(episode_vx)):.4f}"
        )

    if args.record is not None:
        print(f"recorded_video={Path(args.record)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
