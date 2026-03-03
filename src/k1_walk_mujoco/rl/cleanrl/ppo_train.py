from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv
from k1_walk_mujoco.rl.cleanrl.eval_harness import append_eval_jsonl, evaluate_suite, load_eval_suites
from k1_walk_mujoco.rl.cleanrl.utils import (
    build_run_name,
    ensure_dir,
    resolve_num_envs,
    select_device,
)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise TypeError(f"Expected str or list[str], got {type(value)!r}")


def _layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


@dataclass
class PPOTrainConfig:
    seed: int = 1
    milestone: str = "m3"
    env_config_path: Path = Path("configs/env_k1_walk.yaml")
    env_overrides: dict[str, Any] = field(default_factory=dict)
    init_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    resume_training_state: bool = False
    device: str = "auto"
    total_timesteps: int = 1_000_000
    num_envs: int | str | None = "auto"
    num_steps: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    update_epochs: int = 10
    num_minibatches: int = 32
    anneal_lr: bool = True
    norm_adv: bool = True
    clip_vloss: bool = True
    eval_every_updates: int = 10
    eval_episodes: int = 1
    save_every_updates: int = 0
    eval_suite_path: Path | None = Path("configs/eval_suites_goal_pose.yaml")
    eval_nominal_suite: str = "easy"
    eval_stress_suite: str = "stress"
    eval_extra_suites: list[str] = field(default_factory=list)
    print_every_updates: int = 10
    nan_watchdog: bool = True
    run_dir: Path = Path("runs/cleanrl_ppo")
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PPOTrainConfig:
        default = cls()
        env_cfg = raw.get("env_config_path", raw.get("env_config", default.env_config_path))
        target_kl = raw.get("target_kl", default.target_kl)
        suite_path_raw = raw.get("eval_suite_path", default.eval_suite_path)
        suite_path = None if suite_path_raw in (None, "", "null") else Path(suite_path_raw)
        init_ckpt_raw = raw.get("init_checkpoint", default.init_checkpoint)
        init_ckpt = None if init_ckpt_raw in (None, "", "null") else Path(init_ckpt_raw)
        resume_ckpt_raw = raw.get("resume_checkpoint", default.resume_checkpoint)
        resume_ckpt = None if resume_ckpt_raw in (None, "", "null") else Path(resume_ckpt_raw)
        env_overrides = raw.get("env_overrides", default.env_overrides)
        if env_overrides is None:
            env_overrides = {}
        if not isinstance(env_overrides, dict):
            raise TypeError(f"env_overrides must be dict, got {type(env_overrides)!r}")
        return cls(
            seed=int(raw.get("seed", default.seed)),
            milestone=str(raw.get("milestone", default.milestone)),
            env_config_path=Path(env_cfg),
            env_overrides=env_overrides,
            init_checkpoint=init_ckpt,
            resume_checkpoint=resume_ckpt,
            resume_training_state=_as_bool(
                raw.get("resume_training_state"),
                default.resume_training_state,
            ),
            device=str(raw.get("device", default.device)),
            total_timesteps=int(raw.get("total_timesteps", default.total_timesteps)),
            num_envs=raw.get("num_envs", default.num_envs),
            num_steps=int(raw.get("num_steps", default.num_steps)),
            learning_rate=float(raw.get("learning_rate", default.learning_rate)),
            gamma=float(raw.get("gamma", default.gamma)),
            gae_lambda=float(raw.get("gae_lambda", default.gae_lambda)),
            clip_coef=float(raw.get("clip_coef", default.clip_coef)),
            vf_coef=float(raw.get("vf_coef", default.vf_coef)),
            ent_coef=float(raw.get("ent_coef", default.ent_coef)),
            max_grad_norm=float(raw.get("max_grad_norm", default.max_grad_norm)),
            target_kl=None if target_kl in (None, "", "null") else float(target_kl),
            update_epochs=int(raw.get("update_epochs", default.update_epochs)),
            num_minibatches=int(raw.get("num_minibatches", default.num_minibatches)),
            anneal_lr=_as_bool(raw.get("anneal_lr"), default.anneal_lr),
            norm_adv=_as_bool(raw.get("norm_adv"), default.norm_adv),
            clip_vloss=_as_bool(raw.get("clip_vloss"), default.clip_vloss),
            eval_every_updates=int(raw.get("eval_every_updates", default.eval_every_updates)),
            eval_episodes=int(raw.get("eval_episodes", default.eval_episodes)),
            save_every_updates=int(raw.get("save_every_updates", default.save_every_updates)),
            eval_suite_path=suite_path,
            eval_nominal_suite=str(raw.get("eval_nominal_suite", default.eval_nominal_suite)),
            eval_stress_suite=str(raw.get("eval_stress_suite", default.eval_stress_suite)),
            eval_extra_suites=_as_list_str(raw.get("eval_extra_suites", default.eval_extra_suites)),
            print_every_updates=int(raw.get("print_every_updates", default.print_every_updates)),
            nan_watchdog=_as_bool(raw.get("nan_watchdog"), default.nan_watchdog),
            run_dir=Path(raw.get("run_dir", default.run_dir)),
            tensorboard=_as_bool(raw.get("tensorboard"), default.tensorboard),
            wandb=_as_bool(raw.get("wandb"), default.wandb),
            wandb_project=raw.get("wandb_project", default.wandb_project),
        )


def _make_env(env_config_path: Path, seed: int, env_overrides: dict[str, Any]):
    def thunk() -> K1WalkEnv:
        env = K1WalkEnv(env_config_path=env_config_path, cfg_overrides=env_overrides)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def get_dist(self, x: torch.Tensor) -> Normal:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def act(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        dist = self.get_dist(x)
        if deterministic:
            return dist.mean
        return dist.sample()

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(x)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.get_value(x)
        return action, log_prob, entropy, value


def _extract_info_values(infos: dict[str, Any], key: str, num_envs: int) -> np.ndarray | None:
    if key not in infos:
        return None

    values = np.asarray(infos[key], dtype=object)
    if values.shape == ():
        return np.asarray([values.item()] * num_envs, dtype=object)

    mask_key = f"_{key}"
    if mask_key not in infos:
        return values

    mask = np.asarray(infos[mask_key], dtype=bool)
    out = np.empty(num_envs, dtype=object)
    out[:] = None
    limit = min(num_envs, values.shape[0], mask.shape[0])
    for i in range(limit):
        if bool(mask[i]):
            out[i] = values[i]
    return out


def _extract_info_numeric(infos: dict[str, Any], key: str, num_envs: int) -> np.ndarray | None:
    raw = _extract_info_values(infos=infos, key=key, num_envs=num_envs)
    if raw is None:
        return None

    out = np.full(raw.shape[0], np.nan, dtype=np.float32)
    for i, value in enumerate(raw):
        if value is None:
            continue
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _save_checkpoint(
    path: Path,
    agent: ActorCritic,
    optimizer: optim.Optimizer,
    cfg: PPOTrainConfig,
    run_name: str,
    update: int,
    global_step: int,
    best_nominal_success: float,
    best_stress_success: float,
) -> None:
    payload = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "run_name": run_name,
        "update": update,
        "global_step": global_step,
        "best_nominal_success": best_nominal_success,
        "best_stress_success": best_stress_success,
    }
    torch.save(payload, path)


def _load_checkpoint_payload(path: Path, device: torch.device, *, label: str) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} checkpoint must be a dict payload: {path}")
    if "agent" not in payload:
        raise RuntimeError(f"{label} checkpoint is missing required key `agent`: {path}")
    return payload


def _build_agent_for_env(env: K1WalkEnv, device: torch.device) -> ActorCritic:
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)


def evaluate_checkpoint(
    *,
    ckpt_path: Path,
    cfg: PPOTrainConfig,
    device: torch.device,
    suite_name: str | None = None,
) -> dict[str, Any]:
    if cfg.eval_suite_path is None:
        raise ValueError("eval_suite_path must be set for eval-only mode.")
    suites = load_eval_suites(cfg.eval_suite_path)
    if suite_name is not None and suite_name not in suites:
        raise KeyError(f"Unknown eval suite: {suite_name}. Available: {sorted(suites)}")

    env_probe = K1WalkEnv(env_config_path=cfg.env_config_path, cfg_overrides=cfg.env_overrides)
    agent = _build_agent_for_env(env_probe, device)
    env_probe.close()

    payload = _load_checkpoint_payload(ckpt_path, device, label="Eval")
    agent.load_state_dict(payload["agent"])
    agent.eval()

    target_suites = [suite_name] if suite_name is not None else sorted(suites.keys())
    results = {
        sname: evaluate_suite(
            agent=agent,
            device=device,
            env_config_path=cfg.env_config_path,
            env_overrides=cfg.env_overrides,
            suite_name=sname,
            scenarios=suites[sname],
        )
        for sname in target_suites
    }
    return {"checkpoint": str(ckpt_path), "suites": results}


def _evaluate_milestone_suites(
    *,
    agent: ActorCritic,
    cfg: PPOTrainConfig,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    if cfg.eval_suite_path is None:
        return {}
    suite_path = cfg.eval_suite_path
    if not suite_path.exists():
        return {}

    suites = load_eval_suites(suite_path)
    selected = [cfg.eval_nominal_suite, cfg.eval_stress_suite, *cfg.eval_extra_suites]
    ordered_unique = []
    for name in selected:
        if name not in ordered_unique:
            ordered_unique.append(name)

    results: dict[str, dict[str, Any]] = {}
    for suite_name in ordered_unique:
        if suite_name not in suites:
            continue
        results[suite_name] = evaluate_suite(
            agent=agent,
            device=device,
            env_config_path=cfg.env_config_path,
            env_overrides=cfg.env_overrides,
            suite_name=suite_name,
            scenarios=suites[suite_name],
        )
    return results


def train_ppo(
    raw_config: dict[str, Any] | PPOTrainConfig,
    *,
    run_name: str | None = None,
    seed_override: int | None = None,
    device_override: str | None = None,
    num_envs_override: int | None = None,
    total_timesteps_override: int | None = None,
    learning_rate_override: float | None = None,
    wandb_override: bool | None = None,
    milestone_override: str | None = None,
    eval_suite_override: str | None = None,
    print_every_updates_override: int | None = None,
    eval_every_updates_override: int | None = None,
    num_steps_override: int | None = None,
    resume_checkpoint_override: str | Path | None = None,
    resume_training_state_override: bool | None = None,
) -> dict[str, Any]:
    cfg = raw_config if isinstance(raw_config, PPOTrainConfig) else PPOTrainConfig.from_dict(raw_config)
    if seed_override is not None:
        cfg.seed = int(seed_override)
    if device_override is not None:
        cfg.device = device_override
    if num_envs_override is not None:
        cfg.num_envs = int(num_envs_override)
    if total_timesteps_override is not None:
        cfg.total_timesteps = int(total_timesteps_override)
    explicit_lr_override = learning_rate_override is not None
    if learning_rate_override is not None:
        cfg.learning_rate = float(learning_rate_override)
    if wandb_override is not None:
        cfg.wandb = bool(wandb_override)
    if milestone_override is not None:
        cfg.milestone = str(milestone_override)
    if eval_suite_override is not None:
        cfg.eval_nominal_suite = str(eval_suite_override)
    if print_every_updates_override is not None:
        cfg.print_every_updates = int(print_every_updates_override)
    if eval_every_updates_override is not None:
        cfg.eval_every_updates = int(eval_every_updates_override)
    if num_steps_override is not None:
        cfg.num_steps = int(num_steps_override)
    if resume_checkpoint_override is not None:
        cfg.resume_checkpoint = Path(resume_checkpoint_override)
    if resume_training_state_override is not None:
        cfg.resume_training_state = bool(resume_training_state_override)

    if cfg.init_checkpoint is not None and cfg.resume_checkpoint is not None:
        raise ValueError("Only one of init_checkpoint or resume_checkpoint can be set.")
    if cfg.resume_training_state and cfg.resume_checkpoint is None:
        raise ValueError("resume_training_state=True requires resume_checkpoint.")

    device = select_device(cfg.device)
    cfg.num_envs = resolve_num_envs(cfg.num_envs, device.type)
    if cfg.num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {cfg.num_envs}")

    resume_payload: dict[str, Any] | None = None
    if cfg.resume_checkpoint is not None:
        resume_payload = _load_checkpoint_payload(
            cfg.resume_checkpoint,
            device,
            label="Resume",
        )

    if run_name is None and cfg.resume_training_state and resume_payload is not None:
        saved_run_name = resume_payload.get("run_name")
        if isinstance(saved_run_name, str) and saved_run_name:
            run_name = saved_run_name

    run_name = run_name or build_run_name(prefix=f"k1_{cfg.milestone}_ppo", seed=cfg.seed)
    run_dir = ensure_dir(Path(cfg.run_dir))
    out_dir = ensure_dir(run_dir / run_name)
    tb_dir = ensure_dir(out_dir / "tb")
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    eval_dir = ensure_dir(out_dir / "eval")
    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt = ckpt_dir / "best.pt"
    best_nominal_ckpt = ckpt_dir / "best_nominal.pt"
    best_stress_ckpt = ckpt_dir / "best_stress.pt"
    eval_jsonl = eval_dir / "metrics.jsonl"

    writer: SummaryWriter | None = SummaryWriter(log_dir=str(tb_dir)) if cfg.tensorboard else None

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wandb logging requested, but `wandb` is not installed. "
                "Install with: pip install -e '.[wandb]'"
            ) from exc
        wandb_run = wandb.init(
            project=cfg.wandb_project or "k1-walk-mujoco",
            name=run_name,
            config=asdict(cfg),
            dir=str(out_dir),
            reinit=True,
        )

    envs: gym.vector.VectorEnv | None = None
    try:
        env_fns = [
            _make_env(cfg.env_config_path, cfg.seed + i, env_overrides=cfg.env_overrides)
            for i in range(cfg.num_envs)
        ]
        envs = gym.vector.SyncVectorEnv(env_fns) if cfg.num_envs == 1 else gym.vector.AsyncVectorEnv(env_fns)

        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape
        if obs_shape is None or action_shape is None:
            raise RuntimeError("Expected fixed-shape Box observation and action spaces.")

        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))
        agent = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
        if cfg.resume_checkpoint is not None and resume_payload is not None:
            agent.load_state_dict(resume_payload["agent"])
        elif cfg.init_checkpoint is not None:
            payload = torch.load(cfg.init_checkpoint, map_location=device, weights_only=False)
            if isinstance(payload, dict) and "agent" in payload:
                agent.load_state_dict(payload["agent"])
            elif isinstance(payload, dict):
                agent.load_state_dict(payload)
            else:
                raise RuntimeError(
                    f"Unsupported init checkpoint format: {cfg.init_checkpoint}"
                )
        optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

        start_update = 1
        global_step = 0
        best_nominal_success = -float("inf")
        best_stress_success = -float("inf")
        if cfg.resume_training_state:
            if resume_payload is None:
                raise RuntimeError("resume_payload missing while resume_training_state is enabled.")
            if "optimizer" not in resume_payload:
                raise RuntimeError(
                    f"Resume checkpoint is missing optimizer state: {cfg.resume_checkpoint}"
                )
            optimizer.load_state_dict(resume_payload["optimizer"])
            start_update = int(resume_payload.get("update", 0)) + 1
            global_step = int(resume_payload.get("global_step", 0))
            best_nominal_success = float(
                resume_payload.get("best_nominal_success", best_nominal_success)
            )
            best_stress_success = float(
                resume_payload.get("best_stress_success", best_stress_success)
            )
            if explicit_lr_override:
                for group in optimizer.param_groups:
                    group["lr"] = cfg.learning_rate

        action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
        action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=device)

        batch_size = cfg.num_envs * cfg.num_steps
        updates_to_run = max(1, math.ceil(cfg.total_timesteps / batch_size))
        end_update = start_update + updates_to_run - 1
        minibatch_size = max(1, batch_size // cfg.num_minibatches)

        obs = torch.zeros((cfg.num_steps, cfg.num_envs, *obs_shape), device=device)
        actions = torch.zeros((cfg.num_steps, cfg.num_envs, *action_shape), device=device)
        logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        rewards = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        dones = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        values = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

        next_obs_np, _ = envs.reset(seed=[cfg.seed + i for i in range(cfg.num_envs)])
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        next_done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

        start_time = time.time()
        start_global_step_for_sps = global_step
        recent_returns: deque[float] = deque(maxlen=100)
        running_returns = np.zeros(cfg.num_envs, dtype=np.float32)
        running_lengths = np.zeros(cfg.num_envs, dtype=np.int64)

        reward_keys = (
            "r_forward",
            "r_cmd_vx",
            "r_cmd_vy",
            "r_cmd_yaw_rate",
            "r_upright",
            "r_joint_vel",
            "r_torque",
            "r_progress",
            "r_goal_bonus",
            "r_goal_yaw",
            "r_action_smooth",
            "r_alive",
        )

        for local_update in range(1, updates_to_run + 1):
            update = start_update + local_update - 1
            if cfg.anneal_lr:
                frac = 1.0 - (local_update - 1.0) / updates_to_run
                optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            reward_term_sums = {k: 0.0 for k in reward_keys}
            reward_term_counts = {k: 0 for k in reward_keys}
            termination_counts: dict[str, int] = {}

            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value

                clipped_action = torch.clamp(action, action_low, action_high).cpu().numpy()
                next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(clipped_action)
                done_np = np.logical_or(term_np, trunc_np)

                if cfg.nan_watchdog and (
                    not np.isfinite(next_obs_np).all() or not np.isfinite(reward_np).all()
                ):
                    raise RuntimeError("NaN/Inf detected in env rollout data.")

                rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(done_np.astype(np.float32), dtype=torch.float32, device=device)

                running_returns += reward_np.astype(np.float32)
                running_lengths += 1
                done_indices = np.flatnonzero(done_np)
                reasons = _extract_info_values(infos, key="termination_reason", num_envs=cfg.num_envs)
                for done_idx in done_indices:
                    ep_return = float(running_returns[done_idx])
                    ep_length = int(running_lengths[done_idx])
                    recent_returns.append(ep_return)
                    if writer is not None:
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    running_returns[done_idx] = 0.0
                    running_lengths[done_idx] = 0
                    if reasons is not None:
                        reason = reasons[done_idx]
                        if reason is not None:
                            reason_str = str(reason)
                            termination_counts[reason_str] = termination_counts.get(reason_str, 0) + 1

                for key in reward_keys:
                    values_np = _extract_info_numeric(infos, key=key, num_envs=cfg.num_envs)
                    if values_np is None:
                        continue
                    finite_vals = values_np[np.isfinite(values_np)]
                    if finite_vals.size == 0:
                        continue
                    reward_term_sums[key] += float(np.mean(finite_vals))
                    reward_term_counts[key] += 1

            with torch.no_grad():
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = rewards[t] + cfg.gamma * next_values * next_non_terminal - values[t]
                    lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, *obs_shape))
            b_actions = actions.reshape((-1, *action_shape))
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(batch_size)
            clipfracs: list[float] = []
            approx_kl = torch.zeros((), device=device)
            pg_loss = torch.zeros((), device=device)
            v_loss = torch.zeros((), device=device)
            entropy_loss = torch.zeros((), device=device)

            for _epoch in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clipfracs.append(
                            float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().cpu().item())
                        )

                    mb_advantages = b_advantages[mb_inds]
                    if cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - cfg.clip_coef,
                        1.0 + cfg.clip_coef,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -cfg.clip_coef,
                            cfg.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                    if cfg.nan_watchdog and not torch.isfinite(loss):
                        raise RuntimeError("NaN/Inf detected in policy optimization loss.")

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = float("nan") if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / var_y)

            sps = int((global_step - start_global_step_for_sps) / max(1e-9, (time.time() - start_time)))
            metrics: dict[str, float] = {
                "charts/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "charts/sps": float(sps),
                "losses/policy_loss": float(pg_loss.detach().cpu().item()),
                "losses/value_loss": float(v_loss.detach().cpu().item()),
                "losses/entropy": float(entropy_loss.detach().cpu().item()),
                "losses/approx_kl": float(approx_kl.detach().cpu().item()),
                "losses/clipfrac": float(np.mean(clipfracs) if clipfracs else 0.0),
                "losses/explained_variance": explained_var,
            }
            if recent_returns:
                metrics["charts/episodic_return_mean100"] = float(np.mean(recent_returns))

            for key in reward_keys:
                count = reward_term_counts[key]
                if count > 0:
                    metrics[f"rewards/{key}"] = reward_term_sums[key] / count

            for reason, count in termination_counts.items():
                metrics[f"termination/{reason}"] = float(count)

            suite_results: dict[str, dict[str, Any]] = {}
            should_eval = (update % cfg.eval_every_updates == 0) or (local_update == updates_to_run)
            if should_eval:
                suite_results = _evaluate_milestone_suites(agent=agent, cfg=cfg, device=device)
                for suite_name, result in suite_results.items():
                    prefix = f"eval/{suite_name}"
                    metrics[f"{prefix}/success_rate"] = float(result["success_rate"])
                    metrics[f"{prefix}/fall_rate"] = float(result["fall_rate"])
                    metrics[f"{prefix}/median_final_pos_error_m"] = float(
                        result["median_final_pos_error_m"]
                    )
                    metrics[f"{prefix}/median_final_yaw_error_deg"] = float(
                        result["median_final_yaw_error_deg"]
                    )
                    metrics[f"{prefix}/median_time_to_goal_s"] = float(
                        result["median_time_to_goal_s"]
                    )
                    metrics[f"{prefix}/median_command_vx_rmse_mps"] = float(
                        result.get("median_command_vx_rmse_mps", float("nan"))
                    )
                    metrics[f"{prefix}/median_command_vy_rmse_mps"] = float(
                        result.get("median_command_vy_rmse_mps", float("nan"))
                    )
                    metrics[f"{prefix}/median_command_yaw_rate_rmse_rps"] = float(
                        result.get("median_command_yaw_rate_rmse_rps", float("nan"))
                    )
                    metrics[f"{prefix}/command_tracking_success_rate"] = float(
                        result.get("command_tracking_success_rate", 0.0)
                    )

                append_eval_jsonl(
                    eval_jsonl,
                    {
                        "update": update,
                        "global_step": global_step,
                        "milestone": cfg.milestone,
                        "suites": suite_results,
                    },
                )

                _save_checkpoint(
                    path=latest_ckpt,
                    agent=agent,
                    optimizer=optimizer,
                    cfg=cfg,
                    run_name=run_name,
                    update=update,
                    global_step=global_step,
                    best_nominal_success=best_nominal_success,
                    best_stress_success=best_stress_success,
                )

                nominal_key = cfg.eval_nominal_suite
                stress_key = cfg.eval_stress_suite
                nominal_success = (
                    float(suite_results[nominal_key]["success_rate"])
                    if nominal_key in suite_results
                    else float("nan")
                )
                stress_success = (
                    float(suite_results[stress_key]["success_rate"])
                    if stress_key in suite_results
                    else float("nan")
                )

                if np.isfinite(nominal_success) and nominal_success > best_nominal_success:
                    best_nominal_success = nominal_success
                    _save_checkpoint(
                        path=best_nominal_ckpt,
                        agent=agent,
                        optimizer=optimizer,
                        cfg=cfg,
                        run_name=run_name,
                        update=update,
                        global_step=global_step,
                        best_nominal_success=best_nominal_success,
                        best_stress_success=best_stress_success,
                    )
                    _save_checkpoint(
                        path=best_ckpt,
                        agent=agent,
                        optimizer=optimizer,
                        cfg=cfg,
                        run_name=run_name,
                        update=update,
                        global_step=global_step,
                        best_nominal_success=best_nominal_success,
                        best_stress_success=best_stress_success,
                    )

                if np.isfinite(stress_success) and stress_success > best_stress_success:
                    best_stress_success = stress_success
                    _save_checkpoint(
                        path=best_stress_ckpt,
                        agent=agent,
                        optimizer=optimizer,
                        cfg=cfg,
                        run_name=run_name,
                        update=update,
                        global_step=global_step,
                        best_nominal_success=best_nominal_success,
                        best_stress_success=best_stress_success,
                    )

            if cfg.save_every_updates > 0 and (update % cfg.save_every_updates == 0):
                periodic_ckpt = ckpt_dir / f"update_{update:06d}.pt"
                _save_checkpoint(
                    path=periodic_ckpt,
                    agent=agent,
                    optimizer=optimizer,
                    cfg=cfg,
                    run_name=run_name,
                    update=update,
                    global_step=global_step,
                    best_nominal_success=best_nominal_success,
                    best_stress_success=best_stress_success,
                )

            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, global_step)
                writer.flush()

            if wandb_run is not None:
                wandb_run.log(metrics, step=global_step)

            if cfg.print_every_updates > 0 and (
                local_update == 1
                or update % cfg.print_every_updates == 0
                or local_update == updates_to_run
            ):
                nominal_val = metrics.get(f"eval/{cfg.eval_nominal_suite}/success_rate", float("nan"))
                stress_val = metrics.get(f"eval/{cfg.eval_stress_suite}/success_rate", float("nan"))
                print(
                    f"[{cfg.milestone}] update={update} (+{local_update}/{updates_to_run}) "
                    f"step={global_step} "
                    f"sps={sps} mean100={metrics.get('charts/episodic_return_mean100', float('nan')):.3f} "
                    f"nominal_success={nominal_val:.3f} stress_success={stress_val:.3f}"
                )

        _save_checkpoint(
            path=latest_ckpt,
            agent=agent,
            optimizer=optimizer,
            cfg=cfg,
            run_name=run_name,
            update=end_update,
            global_step=global_step,
            best_nominal_success=best_nominal_success,
            best_stress_success=best_stress_success,
        )

    finally:
        if envs is not None:
            envs.close()
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()

    return {
        "run_dir": str(out_dir),
        "device": device.type,
        "num_envs": cfg.num_envs,
        "start_update": start_update,
        "end_update": end_update,
        "start_global_step": start_global_step_for_sps,
        "end_global_step": global_step,
        "initialized_from": str(cfg.init_checkpoint) if cfg.init_checkpoint is not None else None,
        "resumed_from": str(cfg.resume_checkpoint) if cfg.resume_checkpoint is not None else None,
        "resume_training_state": bool(cfg.resume_training_state),
        "latest_ckpt": str(latest_ckpt),
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else str(latest_ckpt),
        "best_nominal_ckpt": str(best_nominal_ckpt)
        if best_nominal_ckpt.exists()
        else str(latest_ckpt),
        "best_stress_ckpt": str(best_stress_ckpt) if best_stress_ckpt.exists() else str(latest_ckpt),
        "eval_jsonl": str(eval_jsonl),
    }
