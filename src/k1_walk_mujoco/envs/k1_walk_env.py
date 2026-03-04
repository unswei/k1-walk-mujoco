from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from k1_walk_mujoco.assets.paths import REPO_ROOT
from k1_walk_mujoco.assets.verify import ensure_k1_assets_present
from k1_walk_mujoco.control.action_mapping import action_to_q_des
from k1_walk_mujoco.control.pd import compute_pd_torque
from k1_walk_mujoco.envs.math_utils import quat_wxyz_to_yaw, world_to_body_xy, wrap_angle_rad
from k1_walk_mujoco.robot.k1_spec import ACTION_SCALE_RAD, controlled_joint_arrays
from k1_walk_mujoco.sim.mujoco_backend import MujocoBackend


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


DEFAULT_ENV_CFG: dict[str, Any] = {
    "seed": 123,
    "frequencies": {
        "policy_dt": 0.02,
        "physics_dt": 0.002,
        "decimation": 10,
    },
    "env": {
        "episode_seconds": 20.0,
        "reset_noise_qpos_std": 0.01,
        "reset_noise_qvel_std": 0.05,
        "reset_settle_steps": 20,
        "v_x_target": 0.5,
    },
    "task": {
        "mode": "velocity",
        "transition_mix": {
            "enabled": False,
            "previous_mode": "velocity",
            "previous_fraction": 0.0,
        },
        "goal": {
            "success_pos_tol_m": 0.20,
            "success_yaw_tol_deg": 12.0,
            "max_goal_radius_m": 2.5,
            "min_goal_radius_m": 0.5,
            "goal_reach_hold_steps": 10,
            "min_progress_for_bonus_m": 0.05,
        },
    },
    "commands": {
        "resample_seconds_range": [2.0, 4.0],
        "vx_range_mps": [0.2, 0.8],
        "vy_range_mps": [-0.3, 0.3],
        "yaw_rate_range_rps": [-0.8, 0.8],
    },
    "randomization": {
        "enabled": False,
        "friction_range": [1.0, 1.0],
        "mass_scale_range": [1.0, 1.0],
        "motor_strength_scale_range": [1.0, 1.0],
        "obs_noise_std": 0.0,
        "action_latency_steps": 0,
        "push": {
            "enabled": False,
            "interval_seconds_range": [5.0, 8.0],
            "impulse_xy_range": [0.0, 0.0],
        },
    },
    "termination": {
        "min_base_height_m": 0.45,
        "max_tilt_rad": 0.9,
        "terminate_on_success_train": False,
        "terminate_on_success_eval": True,
    },
    "reward": {
        "velocity_sigma": 0.5,
        "command_sigma_vx": 0.5,
        "command_sigma_vy": 0.4,
        "command_sigma_yaw": 0.8,
        "goal_yaw_sigma_rad": 0.35,
        "w_forward_velocity": 2.0,
        "w_command_vx": 2.0,
        "w_command_vy": 0.6,
        "w_command_yaw_rate": 0.6,
        "w_upright": 1.0,
        "w_joint_velocity_penalty": 0.02,
        "w_torque_penalty": 0.01,
        "w_progress": 8.0,
        "w_goal_bonus": 8.0,
        "w_goal_yaw": 0.8,
        "w_action_smooth": 0.05,
        "w_alive": 0.05,
    },
}


class K1WalkEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", None], "render_fps": 50}

    def __init__(
        self,
        env_config_path: Path | None = None,
        render_mode: str | None = None,
        cfg_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        env_cfg_path = env_config_path or (REPO_ROOT / "configs" / "env_k1_walk.yaml")
        loaded_cfg: dict[str, Any] = {}
        if env_cfg_path.exists():
            with env_cfg_path.open("r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
            if isinstance(raw_cfg, dict):
                loaded_cfg = raw_cfg
        self.cfg = _deep_update(DEFAULT_ENV_CFG, loaded_cfg)
        if cfg_overrides is not None:
            self.cfg = _deep_update(self.cfg, cfg_overrides)

        mjcf_path = ensure_k1_assets_present()
        self.j = controlled_joint_arrays()
        self.backend = MujocoBackend(mjcf_path, self.j.names)

        freqs = self.cfg["frequencies"]
        self.policy_dt = float(freqs["policy_dt"])
        self.physics_dt = float(freqs["physics_dt"])
        self.decimation = int(freqs["decimation"])
        self.backend.model.opt.timestep = self.physics_dt

        self.max_steps = int(self.cfg["env"]["episode_seconds"] / self.policy_dt)
        self.v_x_target = float(self.cfg["env"]["v_x_target"])
        self._qpos_noise_std = float(self.cfg["env"]["reset_noise_qpos_std"])
        self._qvel_noise_std = float(self.cfg["env"]["reset_noise_qvel_std"])
        self._reset_settle_steps = int(self.cfg["env"].get("reset_settle_steps", 20))
        self._task_mode = str(self.cfg["task"]["mode"])
        self._episode_task_mode = self._task_mode

        n = len(self.j.names)
        obs_dim = 10 + 3 * n + 3 + 4
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._prev_action = np.zeros(n, dtype=np.float64)
        self._last_tau = np.zeros(n, dtype=np.float64)
        self._step_count = 0
        self._termination_reason = "none"

        self._current_command = np.zeros(3, dtype=np.float64)
        self._next_command_resample_step = 0

        self._goal_xy_world = np.zeros(2, dtype=np.float64)
        self._goal_yaw_world_rad = 0.0
        self._goal_reach_streak = 0
        self._prev_goal_distance_m = np.nan
        self._cumulative_progress_m = 0.0
        self._time_to_goal_s = np.nan
        self._is_success = False
        self._yaw_error_rad = 0.0

        self._eval_mode = False
        self._terminate_on_success_override: bool | None = None
        self._fixed_reset_command: np.ndarray | None = None
        self._fixed_reset_goal_xy: np.ndarray | None = None
        self._fixed_reset_goal_yaw_rad: float | None = None
        self._task_mode_override: str | None = None

        self._default_geom_friction = self.backend.model.geom_friction.copy()
        self._default_body_mass = self.backend.model.body_mass.copy()
        self._motor_strength_scale = 1.0
        self._obs_noise_std = 0.0
        self._action_latency_steps = 0
        self._action_latency_queue: deque[np.ndarray] = deque()
        self._next_push_step = int(1e12)
        self._current_action_smooth_penalty = 0.0

    @property
    def task_mode(self) -> str:
        return self._episode_task_mode

    def _resolve_episode_task_mode(self) -> None:
        if self._task_mode_override is not None:
            self._episode_task_mode = self._task_mode_override
            return

        mode = self._task_mode
        mix_cfg = self.cfg.get("task", {}).get("transition_mix", {})
        if isinstance(mix_cfg, dict) and bool(mix_cfg.get("enabled", False)):
            previous_mode = str(mix_cfg.get("previous_mode", mode))
            mix_prob = float(mix_cfg.get("previous_fraction", 0.0))
            mix_prob = float(np.clip(mix_prob, 0.0, 1.0))
            if (
                previous_mode in {"velocity", "command_tracking", "goal_pose"}
                and previous_mode != mode
                and float(self.np_random.random()) < mix_prob
            ):
                mode = previous_mode
        self._episode_task_mode = mode

    def _compute_tilt(self, quat_wxyz: np.ndarray) -> float:
        qw = float(np.clip(abs(quat_wxyz[0]), 0.0, 1.0))
        return 2.0 * np.arccos(qw)

    def _current_base_yaw_rad(self) -> float:
        s = self.backend.get_state()
        return quat_wxyz_to_yaw(s.base_quat)

    def _get_goal_features(self) -> np.ndarray:
        if self.task_mode != "goal_pose":
            return np.zeros(4, dtype=np.float64)
        s = self.backend.get_state()
        delta_xy_world = self._goal_xy_world - s.base_pos[:2]
        base_yaw = quat_wxyz_to_yaw(s.base_quat)
        delta_xy_body = world_to_body_xy(delta_xy_world, base_yaw)
        yaw_err = wrap_angle_rad(self._goal_yaw_world_rad - base_yaw)
        self._yaw_error_rad = yaw_err
        return np.array(
            [delta_xy_body[0], delta_xy_body[1], np.sin(yaw_err), np.cos(yaw_err)],
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        s = self.backend.get_state()
        goal_features = self._get_goal_features()
        obs = np.concatenate(
            [
                s.base_quat,
                s.base_ang_vel,
                s.base_lin_vel,
                s.joint_qpos,
                s.joint_qvel,
                self._prev_action,
                self._current_command,
                goal_features,
            ]
        )
        if self._obs_noise_std > 0.0:
            obs = obs + self.np_random.normal(0.0, self._obs_noise_std, size=obs.shape)
        return obs.astype(np.float32)

    def _draw_uniform(self, lo_hi: list[float] | tuple[float, float]) -> float:
        lo = float(lo_hi[0])
        hi = float(lo_hi[1])
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            return lo
        return float(self.np_random.uniform(lo, hi))

    def _draw_int(self, value: int | list[int] | tuple[int, int]) -> int:
        if isinstance(value, int):
            return max(0, value)
        lo = int(value[0])
        hi = int(value[1])
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            return max(0, lo)
        return int(self.np_random.integers(lo, hi + 1))

    def _sample_command(self) -> np.ndarray:
        if self._fixed_reset_command is not None:
            return self._fixed_reset_command.copy()

        cmd_cfg = self.cfg["commands"]
        if self.task_mode == "velocity":
            return np.array([self.v_x_target, 0.0, 0.0], dtype=np.float64)
        if self.task_mode == "command_tracking":
            return np.array(
                [
                    self._draw_uniform(cmd_cfg["vx_range_mps"]),
                    self._draw_uniform(cmd_cfg["vy_range_mps"]),
                    self._draw_uniform(cmd_cfg["yaw_rate_range_rps"]),
                ],
                dtype=np.float64,
            )
        return np.zeros(3, dtype=np.float64)

    def _resample_command_horizon_steps(self) -> int:
        sec_lo, sec_hi = self.cfg["commands"]["resample_seconds_range"]
        resample_sec = self._draw_uniform([float(sec_lo), float(sec_hi)])
        return max(1, int(round(resample_sec / self.policy_dt)))

    def _sample_goal(self) -> tuple[np.ndarray, float]:
        if self._fixed_reset_goal_xy is not None and self._fixed_reset_goal_yaw_rad is not None:
            return self._fixed_reset_goal_xy.copy(), float(self._fixed_reset_goal_yaw_rad)

        task_goal_cfg = self.cfg["task"]["goal"]
        s = self.backend.get_state()
        base_xy = s.base_pos[:2]
        radius = self._draw_uniform(
            [task_goal_cfg["min_goal_radius_m"], task_goal_cfg["max_goal_radius_m"]]
        )
        heading = self._draw_uniform([-np.pi, np.pi])
        goal_xy = base_xy + radius * np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        goal_yaw = self._draw_uniform([-np.pi, np.pi])
        return goal_xy, goal_yaw

    def _parse_reset_options(self, options: dict[str, Any] | None) -> None:
        self._eval_mode = False
        self._terminate_on_success_override = None
        self._fixed_reset_command = None
        self._fixed_reset_goal_xy = None
        self._fixed_reset_goal_yaw_rad = None
        self._task_mode_override = None
        if options is None:
            return

        if "eval_mode" in options:
            self._eval_mode = bool(options["eval_mode"])
        if "terminate_on_success" in options:
            self._terminate_on_success_override = bool(options["terminate_on_success"])
        if "task_mode" in options:
            self._task_mode_override = str(options["task_mode"])

        if "command" in options and options["command"] is not None:
            cmd_raw = options["command"]
            if isinstance(cmd_raw, dict):
                cmd = np.array(
                    [cmd_raw.get("vx", 0.0), cmd_raw.get("vy", 0.0), cmd_raw.get("yaw_rate", 0.0)],
                    dtype=np.float64,
                )
            else:
                cmd = np.asarray(cmd_raw, dtype=np.float64).reshape(3)
            self._fixed_reset_command = cmd

        if "goal_xy" in options and options["goal_xy"] is not None:
            self._fixed_reset_goal_xy = np.asarray(options["goal_xy"], dtype=np.float64).reshape(2)
        if "goal_yaw_rad" in options and options["goal_yaw_rad"] is not None:
            self._fixed_reset_goal_yaw_rad = float(options["goal_yaw_rad"])
        if "goal_yaw_deg" in options and options["goal_yaw_deg"] is not None:
            self._fixed_reset_goal_yaw_rad = np.deg2rad(float(options["goal_yaw_deg"]))

    def _apply_randomization(self) -> None:
        rand_cfg = self.cfg["randomization"]
        enabled = bool(rand_cfg.get("enabled", False))

        self.backend.model.geom_friction[:] = self._default_geom_friction
        self.backend.model.body_mass[:] = self._default_body_mass
        self._motor_strength_scale = 1.0
        self._obs_noise_std = 0.0
        self._action_latency_steps = 0
        self._next_push_step = int(1e12)

        if not enabled:
            mujoco.mj_forward(self.backend.model, self.backend.data)
            return

        friction_scale = self._draw_uniform(rand_cfg["friction_range"])
        self.backend.model.geom_friction[:, 0] = self._default_geom_friction[:, 0] * friction_scale

        mass_scale = self._draw_uniform(rand_cfg["mass_scale_range"])
        self.backend.model.body_mass[:] = self._default_body_mass * mass_scale

        self._motor_strength_scale = self._draw_uniform(rand_cfg["motor_strength_scale_range"])
        self._obs_noise_std = float(rand_cfg.get("obs_noise_std", 0.0))
        self._action_latency_steps = self._draw_int(rand_cfg.get("action_latency_steps", 0))

        push_cfg = rand_cfg.get("push", {})
        if bool(push_cfg.get("enabled", False)):
            interval_sec = self._draw_uniform(push_cfg["interval_seconds_range"])
            self._next_push_step = max(1, int(round(interval_sec / self.policy_dt)))

        mujoco.mj_forward(self.backend.model, self.backend.data)

    def _maybe_apply_push(self) -> None:
        if self._step_count < self._next_push_step:
            return
        push_cfg = self.cfg["randomization"].get("push", {})
        impulse_mag = self._draw_uniform(push_cfg.get("impulse_xy_range", [0.0, 0.0]))
        if impulse_mag > 0.0:
            direction = self._draw_uniform([-np.pi, np.pi])
            self.backend.data.qvel[0] += impulse_mag * np.cos(direction)
            self.backend.data.qvel[1] += impulse_mag * np.sin(direction)
            mujoco.mj_forward(self.backend.model, self.backend.data)
        interval_sec = self._draw_uniform(push_cfg.get("interval_seconds_range", [5.0, 8.0]))
        self._next_push_step += max(1, int(round(interval_sec / self.policy_dt)))

    def _command_terms(self, s: Any) -> tuple[float, float, float]:
        rew_cfg = self.cfg["reward"]
        vx_cmd, vy_cmd, yaw_cmd = self._current_command
        vx_err = float(s.base_lin_vel[0] - vx_cmd)
        vy_err = float(s.base_lin_vel[1] - vy_cmd)
        yaw_err = float(s.base_ang_vel[2] - yaw_cmd)

        sig_vx = float(rew_cfg["command_sigma_vx"])
        sig_vy = float(rew_cfg["command_sigma_vy"])
        sig_yaw = float(rew_cfg["command_sigma_yaw"])

        r_vx = float(np.exp(-(vx_err * vx_err) / (sig_vx * sig_vx)))
        r_vy = float(np.exp(-(vy_err * vy_err) / (sig_vy * sig_vy)))
        r_yaw = float(np.exp(-(yaw_err * yaw_err) / (sig_yaw * sig_yaw)))
        return r_vx, r_vy, r_yaw

    def _goal_terms(self, s: Any) -> tuple[float, float, float, bool]:
        task_goal_cfg = self.cfg["task"]["goal"]
        rew_cfg = self.cfg["reward"]
        delta_xy = self._goal_xy_world - s.base_pos[:2]
        distance = float(np.linalg.norm(delta_xy))
        progress = 0.0
        if np.isfinite(self._prev_goal_distance_m):
            progress = float(self._prev_goal_distance_m - distance)
        self._prev_goal_distance_m = distance
        self._cumulative_progress_m += max(progress, 0.0)

        base_yaw = quat_wxyz_to_yaw(s.base_quat)
        self._yaw_error_rad = wrap_angle_rad(self._goal_yaw_world_rad - base_yaw)
        sig_yaw = float(rew_cfg["goal_yaw_sigma_rad"])
        r_goal_yaw = float(
            np.exp(-(self._yaw_error_rad * self._yaw_error_rad) / (sig_yaw * sig_yaw))
        )

        success_pos = distance <= float(task_goal_cfg["success_pos_tol_m"])
        success_yaw = abs(self._yaw_error_rad) <= np.deg2rad(float(task_goal_cfg["success_yaw_tol_deg"]))
        if success_pos and success_yaw:
            self._goal_reach_streak += 1
        else:
            self._goal_reach_streak = 0

        hold_steps = int(task_goal_cfg["goal_reach_hold_steps"])
        self._is_success = self._goal_reach_streak >= hold_steps
        if self._is_success and not np.isfinite(self._time_to_goal_s):
            self._time_to_goal_s = self._step_count * self.policy_dt

        min_progress = float(task_goal_cfg["min_progress_for_bonus_m"])
        goal_bonus = 1.0 if (self._is_success and self._cumulative_progress_m >= min_progress) else 0.0
        return progress, goal_bonus, r_goal_yaw, self._is_success

    def _terminate_on_success_enabled(self) -> bool:
        if self._terminate_on_success_override is not None:
            return self._terminate_on_success_override
        term_cfg = self.cfg["termination"]
        if self._eval_mode:
            return bool(term_cfg.get("terminate_on_success_eval", True))
        return bool(term_cfg.get("terminate_on_success_train", False))

    def _reward_and_done(self) -> tuple[float, bool, dict[str, float | str | bool]]:
        s = self.backend.get_state()
        rew_cfg = self.cfg["reward"]
        term_cfg = self.cfg["termination"]

        r_cmd_vx, r_cmd_vy, r_cmd_yaw = self._command_terms(s)
        r_forward = r_cmd_vx

        tilt = self._compute_tilt(s.base_quat)
        r_upright = -tilt * tilt
        r_joint_vel = -float(np.mean(s.joint_qvel * s.joint_qvel))
        tau_norm = self._last_tau / np.maximum(self.j.effort, 1e-6)
        r_torque = -float(np.mean(tau_norm * tau_norm))
        r_alive = 1.0

        progress = 0.0
        goal_bonus = 0.0
        r_goal_yaw = 0.0
        is_success = False
        if self.task_mode == "goal_pose":
            progress, goal_bonus, r_goal_yaw, is_success = self._goal_terms(s)

        if self.task_mode == "velocity":
            reward = (
                float(rew_cfg["w_forward_velocity"]) * r_forward
                + float(rew_cfg["w_upright"]) * r_upright
                + float(rew_cfg["w_joint_velocity_penalty"]) * r_joint_vel
                + float(rew_cfg["w_torque_penalty"]) * r_torque
                + float(rew_cfg["w_alive"]) * r_alive
            )
        elif self.task_mode == "command_tracking":
            reward = (
                float(rew_cfg["w_command_vx"]) * r_cmd_vx
                + float(rew_cfg["w_command_vy"]) * r_cmd_vy
                + float(rew_cfg["w_command_yaw_rate"]) * r_cmd_yaw
                + float(rew_cfg["w_upright"]) * r_upright
                + float(rew_cfg["w_joint_velocity_penalty"]) * r_joint_vel
                + float(rew_cfg["w_torque_penalty"]) * r_torque
                + float(rew_cfg["w_alive"]) * r_alive
            )
        else:
            reward = (
                float(rew_cfg["w_progress"]) * progress
                + float(rew_cfg["w_goal_bonus"]) * goal_bonus
                + float(rew_cfg["w_goal_yaw"]) * r_goal_yaw
                + float(rew_cfg["w_upright"]) * r_upright
                + float(rew_cfg["w_torque_penalty"]) * r_torque
                + float(rew_cfg["w_action_smooth"]) * self._current_action_smooth_penalty
                + float(rew_cfg["w_alive"]) * r_alive
            )

        done = False
        self._termination_reason = "none"
        if s.base_pos[2] < float(term_cfg["min_base_height_m"]):
            done = True
            self._termination_reason = "base_height"
        elif tilt > float(term_cfg["max_tilt_rad"]):
            done = True
            self._termination_reason = "tilt"
        elif not np.isfinite(reward):
            done = True
            self._termination_reason = "non_finite_reward"
        elif is_success and self._terminate_on_success_enabled():
            done = True
            self._termination_reason = "success"

        distance_to_goal = float(np.linalg.norm(self._goal_xy_world - s.base_pos[:2]))
        info: dict[str, float | str | bool] = {
            "r_forward": r_forward,
            "r_cmd_vx": r_cmd_vx,
            "r_cmd_vy": r_cmd_vy,
            "r_cmd_yaw_rate": r_cmd_yaw,
            "r_upright": r_upright,
            "r_joint_vel": r_joint_vel,
            "r_torque": r_torque,
            "r_progress": progress,
            "r_goal_bonus": goal_bonus,
            "r_goal_yaw": r_goal_yaw,
            "r_action_smooth": self._current_action_smooth_penalty,
            "r_alive": r_alive,
            "distance_to_goal": distance_to_goal,
            "yaw_error_rad": float(self._yaw_error_rad),
            "is_success": bool(is_success),
            "time_to_goal_s": float(self._time_to_goal_s),
            "command_vx": float(self._current_command[0]),
            "command_vy": float(self._current_command[1]),
            "command_yaw_rate": float(self._current_command[2]),
            "base_vx": float(s.base_lin_vel[0]),
            "base_vy": float(s.base_lin_vel[1]),
            "base_yaw_rate": float(s.base_ang_vel[2]),
            "termination_reason": self._termination_reason,
            "task_mode": self.task_mode,
        }
        return float(reward), done, info

    def _settle_after_reset(self) -> None:
        if self._reset_settle_steps <= 0:
            return

        q_des = self.j.q_nominal
        zeros = np.zeros_like(self.j.q_nominal)
        for _ in range(self._reset_settle_steps):
            s = self.backend.get_state()
            tau = compute_pd_torque(
                q=s.joint_qpos,
                qd=s.joint_qvel,
                q_des=q_des,
                kp=self.j.kp,
                kd=self.j.kd,
                effort_limit=self.j.effort,
                qd_des=zeros,
                tau_ff=zeros,
            )
            tau = tau * self._motor_strength_scale
            self.backend.step(tau=tau, n_substeps=self.decimation)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._parse_reset_options(options)
        self._resolve_episode_task_mode()

        self._prev_action[:] = 0.0
        self._last_tau[:] = 0.0
        self._step_count = 0
        self._termination_reason = "none"
        self._goal_reach_streak = 0
        self._prev_goal_distance_m = np.nan
        self._cumulative_progress_m = 0.0
        self._time_to_goal_s = np.nan
        self._is_success = False
        self._yaw_error_rad = 0.0
        self._current_action_smooth_penalty = 0.0

        self._apply_randomization()
        self.backend.reset(
            rng=self.np_random,
            qpos_noise_std=self._qpos_noise_std,
            qvel_noise_std=self._qvel_noise_std,
        )
        self._settle_after_reset()

        self._current_command = self._sample_command()
        self._next_command_resample_step = self._resample_command_horizon_steps()

        s = self.backend.get_state()
        if self.task_mode == "goal_pose":
            self._goal_xy_world, self._goal_yaw_world_rad = self._sample_goal()
            self._prev_goal_distance_m = float(np.linalg.norm(self._goal_xy_world - s.base_pos[:2]))
        else:
            self._goal_xy_world = s.base_pos[:2].copy()
            self._goal_yaw_world_rad = quat_wxyz_to_yaw(s.base_quat)
            self._prev_goal_distance_m = 0.0

        self._action_latency_queue = deque(
            [np.zeros_like(self._prev_action) for _ in range(self._action_latency_steps + 1)],
            maxlen=self._action_latency_steps + 1,
        )

        obs = self._get_obs()
        info = {
            "termination_reason": "none",
            "distance_to_goal": float(self._prev_goal_distance_m),
            "yaw_error_rad": 0.0,
            "is_success": False,
            "time_to_goal_s": float(self._time_to_goal_s),
            "command_vx": float(self._current_command[0]),
            "command_vy": float(self._current_command[1]),
            "command_yaw_rate": float(self._current_command[2]),
            "task_mode": self.task_mode,
        }
        return obs, info

    def step(self, action: np.ndarray):
        action_in = np.asarray(action, dtype=np.float64)
        clipped_action = np.clip(action_in, -1.0, 1.0)

        self._action_latency_queue.append(clipped_action.copy())
        delayed_action = self._action_latency_queue.popleft()
        self._current_action_smooth_penalty = -float(np.mean((delayed_action - self._prev_action) ** 2))

        q_des = action_to_q_des(
            action=delayed_action,
            q_nominal=self.j.q_nominal,
            action_scale_rad=ACTION_SCALE_RAD,
            q_low=self.j.q_low,
            q_high=self.j.q_high,
        )

        s = self.backend.get_state()
        tau = compute_pd_torque(
            q=s.joint_qpos,
            qd=s.joint_qvel,
            q_des=q_des,
            kp=self.j.kp,
            kd=self.j.kd,
            effort_limit=self.j.effort,
            qd_des=np.zeros_like(self.j.q_nominal),
            tau_ff=np.zeros_like(self.j.q_nominal),
        )
        tau = tau * self._motor_strength_scale

        self.backend.step(tau=tau, n_substeps=self.decimation)
        self._last_tau = tau
        self._prev_action = delayed_action
        self._step_count += 1

        if self.task_mode == "command_tracking" and self._step_count >= self._next_command_resample_step:
            self._current_command = self._sample_command()
            self._next_command_resample_step += self._resample_command_horizon_steps()

        self._maybe_apply_push()

        obs = self._get_obs()
        reward, terminated, info = self._reward_and_done()
        truncated = self._step_count >= self.max_steps
        if truncated and not terminated:
            info["termination_reason"] = "time_limit"

        if not np.isfinite(obs).all():
            terminated = True
            info["termination_reason"] = "non_finite_obs"

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
