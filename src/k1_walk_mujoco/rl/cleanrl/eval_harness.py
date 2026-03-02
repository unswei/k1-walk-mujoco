from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class EpisodeEvalResult:
    success: bool
    fall: bool
    final_pos_error_m: float
    final_yaw_error_deg: float
    time_to_goal_s: float
    steps: int
    termination_reason: str


def load_eval_suites(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(raw)!r}")

    suites: dict[str, list[dict[str, Any]]] = {}
    for suite_name, scenarios in raw.items():
        if not isinstance(scenarios, list):
            raise TypeError(f"Expected list for suite {suite_name}, got {type(scenarios)!r}")
        validated: list[dict[str, Any]] = []
        for sc in scenarios:
            if not isinstance(sc, dict):
                raise TypeError(f"Expected scenario dict in suite {suite_name}, got {type(sc)!r}")
            validated.append(sc)
        suites[str(suite_name)] = validated
    return suites


def _run_scenario(
    agent: Any,
    device: torch.device,
    env_config_path: Path,
    env_overrides: dict[str, Any],
    scenario: dict[str, Any],
) -> EpisodeEvalResult:
    scenario_rand = scenario.get("randomization")
    if isinstance(scenario_rand, dict):
        scoped_overrides = _deep_update(env_overrides, {"randomization": scenario_rand})
    else:
        scoped_overrides = env_overrides

    env = K1WalkEnv(env_config_path=env_config_path, cfg_overrides=scoped_overrides)
    try:
        reset_options: dict[str, Any] = {
            "eval_mode": True,
            "terminate_on_success": True,
        }
        if "task_mode" in scenario:
            reset_options["task_mode"] = str(scenario["task_mode"])
        if "goal_xy" in scenario:
            reset_options["goal_xy"] = scenario["goal_xy"]
        if "goal_yaw_deg" in scenario:
            reset_options["goal_yaw_deg"] = float(scenario["goal_yaw_deg"])
        if "command" in scenario:
            reset_options["command"] = scenario["command"]

        seed = int(scenario.get("seed", 0))
        max_steps = int(scenario.get("max_steps", env.max_steps))

        obs, info = env.reset(seed=seed, options=reset_options)
        done = False
        trunc = False
        steps = 0
        while not (done or trunc) and steps < max_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t = agent.act(obs_t, deterministic=True)
            action = action_t.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, _reward, done, trunc, info = env.step(action)
            steps += 1

        final_pos_error_m = float(info.get("distance_to_goal", np.nan))
        final_yaw_error_deg = float(np.rad2deg(abs(float(info.get("yaw_error_rad", np.nan)))))
        time_to_goal_s = float(info.get("time_to_goal_s", np.nan))
        if not np.isfinite(time_to_goal_s):
            time_to_goal_s = steps * env.policy_dt

        reason = str(info.get("termination_reason", "unknown"))
        fall = reason in {"base_height", "tilt", "non_finite_obs", "non_finite_reward"}
        success = bool(info.get("is_success", False))
        return EpisodeEvalResult(
            success=success,
            fall=fall,
            final_pos_error_m=final_pos_error_m,
            final_yaw_error_deg=final_yaw_error_deg,
            time_to_goal_s=time_to_goal_s,
            steps=steps,
            termination_reason=reason,
        )
    finally:
        env.close()


def evaluate_suite(
    agent: Any,
    device: torch.device,
    env_config_path: Path,
    env_overrides: dict[str, Any],
    suite_name: str,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_results = [
        _run_scenario(
            agent=agent,
            device=device,
            env_config_path=env_config_path,
            env_overrides=env_overrides,
            scenario=scenario,
        )
        for scenario in scenarios
    ]
    success = np.asarray([r.success for r in episode_results], dtype=bool)
    fall = np.asarray([r.fall for r in episode_results], dtype=bool)
    pos = np.asarray([r.final_pos_error_m for r in episode_results], dtype=np.float64)
    yaw = np.asarray([r.final_yaw_error_deg for r in episode_results], dtype=np.float64)
    time_to_goal = np.asarray([r.time_to_goal_s for r in episode_results], dtype=np.float64)

    return {
        "suite": suite_name,
        "episodes": int(len(episode_results)),
        "success_rate": float(np.mean(success)) if success.size else 0.0,
        "fall_rate": float(np.mean(fall)) if fall.size else 0.0,
        "median_final_pos_error_m": float(np.nanmedian(pos)) if pos.size else float("nan"),
        "median_final_yaw_error_deg": float(np.nanmedian(yaw)) if yaw.size else float("nan"),
        "median_time_to_goal_s": float(np.nanmedian(time_to_goal)) if time_to_goal.size else float("nan"),
        "termination_reasons": [r.termination_reason for r in episode_results],
        "per_episode": [
            {
                "success": r.success,
                "fall": r.fall,
                "final_pos_error_m": r.final_pos_error_m,
                "final_yaw_error_deg": r.final_yaw_error_deg,
                "time_to_goal_s": r.time_to_goal_s,
                "steps": r.steps,
                "termination_reason": r.termination_reason,
            }
            for r in episode_results
        ],
    }


def append_eval_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
