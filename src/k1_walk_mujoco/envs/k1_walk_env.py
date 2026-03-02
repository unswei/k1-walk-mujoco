from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from k1_walk_mujoco.assets.paths import REPO_ROOT
from k1_walk_mujoco.assets.verify import ensure_k1_assets_present
from k1_walk_mujoco.control.action_mapping import action_to_q_des
from k1_walk_mujoco.control.pd import compute_pd_torque
from k1_walk_mujoco.robot.k1_spec import ACTION_SCALE_RAD, controlled_joint_arrays
from k1_walk_mujoco.sim.mujoco_backend import MujocoBackend


class K1WalkEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", None], "render_fps": 50}

    def __init__(
        self,
        env_config_path: Path | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        env_cfg_path = env_config_path or (REPO_ROOT / "configs" / "env_k1_walk.yaml")
        with env_cfg_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

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

        n = len(self.j.names)
        obs_dim = 10 + 3 * n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._prev_action = np.zeros(n, dtype=np.float64)
        self._last_tau = np.zeros(n, dtype=np.float64)
        self._step_count = 0
        self._termination_reason = "none"

    def _compute_tilt(self, quat_wxyz: np.ndarray) -> float:
        qw = float(np.clip(abs(quat_wxyz[0]), 0.0, 1.0))
        return 2.0 * np.arccos(qw)

    def _get_obs(self) -> np.ndarray:
        s = self.backend.get_state()
        obs = np.concatenate(
            [
                s.base_quat,
                s.base_ang_vel,
                s.base_lin_vel,
                s.joint_qpos,
                s.joint_qvel,
                self._prev_action,
            ]
        )
        return obs.astype(np.float32)

    def _reward_and_done(self) -> tuple[float, bool, dict[str, float | str]]:
        s = self.backend.get_state()
        rew_cfg = self.cfg["reward"]
        term_cfg = self.cfg["termination"]

        vel_sigma = float(rew_cfg["velocity_sigma"])
        vel_err = float(s.base_lin_vel[0] - self.v_x_target)
        r_forward = float(np.exp(-(vel_err * vel_err) / (vel_sigma * vel_sigma)))

        tilt = self._compute_tilt(s.base_quat)
        r_upright = -tilt * tilt

        r_joint_vel = -float(np.mean(s.joint_qvel * s.joint_qvel))
        tau_norm = self._last_tau / np.maximum(self.j.effort, 1e-6)
        r_torque = -float(np.mean(tau_norm * tau_norm))

        reward = (
            float(rew_cfg["w_forward_velocity"]) * r_forward
            + float(rew_cfg["w_upright"]) * r_upright
            + float(rew_cfg["w_joint_velocity_penalty"]) * r_joint_vel
            + float(rew_cfg["w_torque_penalty"]) * r_torque
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

        info: dict[str, float | str] = {
            "r_forward": r_forward,
            "r_upright": r_upright,
            "r_joint_vel": r_joint_vel,
            "r_torque": r_torque,
            "termination_reason": self._termination_reason,
        }
        return reward, done, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self._prev_action[:] = 0.0
        self._last_tau[:] = 0.0
        self._step_count = 0

        self.backend.reset(
            rng=self.np_random,
            qpos_noise_std=self._qpos_noise_std,
            qvel_noise_std=self._qvel_noise_std,
        )
        obs = self._get_obs()
        return obs, {"termination_reason": "none"}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        q_des = action_to_q_des(
            action=action,
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

        self.backend.step(tau=tau, n_substeps=self.decimation)
        self._last_tau = tau
        self._prev_action = np.clip(action, -1.0, 1.0)
        self._step_count += 1

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
