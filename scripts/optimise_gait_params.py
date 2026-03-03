#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

from k1_walk_mujoco.controllers.param_gait_15 import (
    PARAMETER_NAMES,
    ParamGait15,
    clamp_params,
    params_dict_to_vector,
    params_vector_to_dict,
)
from k1_walk_mujoco.control.pd import compute_pd_torque
from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


_WORKER_STATE: "_WorkerState | None" = None


@dataclass
class _WorkerState:
    env: K1WalkEnv
    rollout_seconds: float
    objective: dict[str, float]
    filter_alpha: float
    base_seed: int
    bounds: dict[str, tuple[float, float]]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(loaded)!r}")
    return loaded


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _compute_tilt(quat_wxyz: np.ndarray) -> float:
    qw = float(np.clip(abs(quat_wxyz[0]), 0.0, 1.0))
    return 2.0 * np.arccos(qw)


def _init_worker(
    env_config_path: str,
    rollout_seconds: float,
    objective: dict[str, float],
    filter_alpha: float,
    base_seed: int,
    bounds: dict[str, tuple[float, float]],
) -> None:
    global _WORKER_STATE
    env = K1WalkEnv(env_config_path=Path(env_config_path), render_mode=None)
    _WORKER_STATE = _WorkerState(
        env=env,
        rollout_seconds=float(rollout_seconds),
        objective={k: float(v) for k, v in objective.items()},
        filter_alpha=float(filter_alpha),
        base_seed=int(base_seed),
        bounds={k: (float(v[0]), float(v[1])) for k, v in bounds.items()},
    )


def _evaluate_candidate(task: tuple[int, int, int, list[float]]) -> dict[str, Any]:
    global _WORKER_STATE
    if _WORKER_STATE is None:
        raise RuntimeError("Worker state was not initialized")

    candidate_id, round_idx, episodes_per_candidate, vector_list = task
    env = _WORKER_STATE.env
    objective = _WORKER_STATE.objective

    params = params_vector_to_dict(vector_list)
    controller = ParamGait15(
        params=clamp_params(params, bounds=_WORKER_STATE.bounds),
        controlled_joints=env.j.names,
        filter_alpha=_WORKER_STATE.filter_alpha,
    )

    max_steps = max(1, min(env.max_steps, int(round(_WORKER_STATE.rollout_seconds / env.policy_dt))))
    term_cfg = env.cfg["termination"]
    min_base_height = float(term_cfg["min_base_height_m"])
    max_tilt = float(term_cfg["max_tilt_rad"])

    total_steps = 0
    total_vx = 0.0
    total_abs_vy = 0.0
    total_abs_yaw = 0.0
    total_abs_tau = 0.0
    total_limit_hits = 0
    falls = 0
    early_termination_sum = 0.0

    seed_base = _WORKER_STATE.base_seed + 10_000 * round_idx + 97 * candidate_id

    for ep in range(int(episodes_per_candidate)):
        env.reset(seed=seed_base + ep)
        controller.reset()
        t = 0.0

        steps = 0
        fell = False
        while steps < max_steps:
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
            total_vx += float(s_next.base_lin_vel[0])
            total_abs_vy += abs(float(s_next.base_lin_vel[1]))
            total_abs_yaw += abs(float(s_next.base_ang_vel[2]))
            total_abs_tau += float(np.mean(np.abs(tau)))

            is_low = np.isclose(q_des, env.j.q_low, atol=1e-4)
            is_high = np.isclose(q_des, env.j.q_high, atol=1e-4)
            total_limit_hits += int(np.count_nonzero(is_low | is_high))

            steps += 1
            total_steps += 1
            t += env.policy_dt

            tilt = _compute_tilt(s_next.base_quat)
            if s_next.base_pos[2] < min_base_height or tilt > max_tilt:
                fell = True
                break

        if fell:
            falls += 1
            early_termination_sum += float(max_steps - steps) / float(max_steps)

    denom_steps = max(1, total_steps)
    joints = max(1, len(env.j.names))
    mean_forward_velocity = total_vx / denom_steps
    mean_abs_lateral_velocity = total_abs_vy / denom_steps
    mean_abs_yaw_rate = total_abs_yaw / denom_steps
    mean_abs_torque = total_abs_tau / denom_steps
    joint_limit_hit_rate = total_limit_hits / float(denom_steps * joints)

    fall_penalty = (falls / max(1, int(episodes_per_candidate))) + early_termination_sum / max(
        1, int(episodes_per_candidate)
    )

    score = (
        float(objective["w_v"]) * mean_forward_velocity
        - float(objective["w_fall"]) * fall_penalty
        - float(objective["w_lat"]) * mean_abs_lateral_velocity
        - float(objective["w_yaw"]) * mean_abs_yaw_rate
        - float(objective["w_tau"]) * mean_abs_torque
        - float(objective["w_limits"]) * joint_limit_hit_rate
    )

    return {
        "candidate_id": int(candidate_id),
        "round": int(round_idx),
        "episodes": int(episodes_per_candidate),
        "score": float(score),
        "fall_penalty": float(fall_penalty),
        "falls": int(falls),
        "mean_forward_velocity": float(mean_forward_velocity),
        "mean_abs_lateral_velocity": float(mean_abs_lateral_velocity),
        "mean_abs_yaw_rate": float(mean_abs_yaw_rate),
        "mean_abs_torque": float(mean_abs_torque),
        "joint_limit_hit_rate": float(joint_limit_hit_rate),
        "total_steps": int(total_steps),
        "params": {k: float(v) for k, v in zip(PARAMETER_NAMES, vector_list, strict=True)},
    }


def _sample_population(
    *,
    rng: np.random.Generator,
    size: int,
    lo: np.ndarray,
    hi: np.ndarray,
    seed_vector: np.ndarray,
) -> list[np.ndarray]:
    out: list[np.ndarray] = [seed_vector.copy()]
    if size <= 1:
        return out
    samples = rng.uniform(lo, hi, size=(size - 1, lo.shape[0]))
    for row in samples:
        out.append(row.astype(np.float64))
    return out


def _candidate_sort_key(result: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(result["score"]),
        float(result["mean_forward_velocity"]),
        -float(result["fall_penalty"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise 15-parameter gait for K1 with parallel search.")
    parser.add_argument("--config", type=str, default="configs/optimise_gait.yaml")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--record-best-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)

    defaults = {
        "seed": 123,
        "num_workers": 4,
        "env": {
            "config_path": "configs/env_k1_walk.yaml",
            "rollout_seconds": 6.0,
            "episodes_per_candidate": 1,
        },
        "search": {
            "strategy": "random_successive_halving",
            "rounds": 3,
            "population_per_round": 64,
            "keep_top_k": 16,
        },
        "objective": {
            "w_v": 3.0,
            "w_fall": 6.0,
            "w_lat": 1.0,
            "w_yaw": 0.8,
            "w_tau": 0.25,
            "w_limits": 0.5,
        },
        "io": {
            "out_dir": "runs/gait_optim",
            "run_prefix": "param_gait_15",
            "gait_seed_config": "configs/gait_param_15.yaml",
            "record_best_video": False,
        },
    }
    cfg = _deep_update(defaults, cfg)

    if str(cfg["search"].get("strategy", "")).lower() != "random_successive_halving":
        raise ValueError("Only search.strategy=random_successive_halving is supported in v0")

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg["num_workers"])
    num_workers = max(1, num_workers)

    env_config_path = Path(str(cfg["env"]["config_path"]))
    rollout_seconds = float(cfg["env"]["rollout_seconds"])
    base_episodes = max(1, int(cfg["env"].get("episodes_per_candidate", 1)))

    rounds = max(1, int(cfg["search"]["rounds"]))
    population_per_round = max(1, int(cfg["search"]["population_per_round"]))
    keep_top_k = max(1, int(cfg["search"]["keep_top_k"]))

    objective = {k: float(v) for k, v in cfg["objective"].items()}

    gait_seed_cfg_path = Path(str(cfg["io"]["gait_seed_config"]))
    gait_seed_cfg = _load_yaml(gait_seed_cfg_path)
    seed_controller = ParamGait15.from_config(gait_seed_cfg)
    bounds = seed_controller.bounds
    filter_alpha = float(seed_controller.filter_alpha)

    lo = np.array([bounds[name][0] for name in PARAMETER_NAMES], dtype=np.float64)
    hi = np.array([bounds[name][1] for name in PARAMETER_NAMES], dtype=np.float64)
    seed_vector = params_dict_to_vector(seed_controller.params)

    out_root = Path(str(cfg["io"]["out_dir"]))
    run_prefix = str(cfg["io"]["run_prefix"])
    run_dir = out_root / f"{run_prefix}_{_timestamp_utc()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = run_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else None
    if writer is None:
        print("TensorBoard disabled: torch.utils.tensorboard.SummaryWriter not available.")

    results_path = run_dir / "candidates.jsonl"
    summary_path = run_dir / "summary.json"
    best_yaml_path = run_dir / "best_params.yaml"

    main_env = K1WalkEnv(env_config_path=env_config_path, render_mode=None)
    max_rollout_steps = max(1, min(main_env.max_steps, int(round(rollout_seconds / main_env.policy_dt))))
    main_env.close()

    population = _sample_population(
        rng=rng,
        size=population_per_round,
        lo=lo,
        hi=hi,
        seed_vector=seed_vector,
    )

    candidates: list[dict[str, Any]] = [
        {"candidate_id": i, "vector": population[i]} for i in range(len(population))
    ]

    all_results: list[dict[str, Any]] = []
    final_round_results: list[dict[str, Any]] = []
    global_step = 0

    ctx = mp.get_context("spawn")
    with (
        results_path.open("w", encoding="utf-8") as results_file,
        ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                str(env_config_path),
                rollout_seconds,
                objective,
                filter_alpha,
                seed,
                bounds,
            ),
        ) as pool,
    ):
        for round_idx in range(rounds):
            episodes_this_round = base_episodes * (2**round_idx)
            tasks = [
                (
                    int(cand["candidate_id"]),
                    int(round_idx),
                    int(episodes_this_round),
                    [float(x) for x in np.asarray(cand["vector"], dtype=np.float64)],
                )
                for cand in candidates
            ]

            round_results = list(pool.imap_unordered(_evaluate_candidate, tasks))
            round_results.sort(key=_candidate_sort_key, reverse=True)

            for result in round_results:
                result["episodes"] = int(episodes_this_round)
                all_results.append(result)
                results_file.write(json.dumps(result) + "\n")
                results_file.flush()

                if writer is not None:
                    writer.add_scalar("candidate/score", float(result["score"]), global_step)
                    writer.add_scalar(
                        "candidate/mean_forward_velocity",
                        float(result["mean_forward_velocity"]),
                        global_step,
                    )
                    writer.add_scalar("candidate/fall_penalty", float(result["fall_penalty"]), global_step)
                    writer.add_scalar(
                        "candidate/mean_abs_lateral_velocity",
                        float(result["mean_abs_lateral_velocity"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "candidate/mean_abs_yaw_rate",
                        float(result["mean_abs_yaw_rate"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "candidate/mean_abs_torque",
                        float(result["mean_abs_torque"]),
                        global_step,
                    )
                    writer.add_scalar(
                        "candidate/joint_limit_hit_rate",
                        float(result["joint_limit_hit_rate"]),
                        global_step,
                    )
                global_step += 1

            best_round = round_results[0]
            final_round_results = round_results
            print(
                f"round={round_idx} episodes={episodes_this_round} candidates={len(round_results)} "
                f"best_id={best_round['candidate_id']} best_score={best_round['score']:.4f} "
                f"best_vx={best_round['mean_forward_velocity']:.4f} falls={best_round['falls']}"
            )

            if writer is not None:
                writer.add_scalar("round/best_score", float(best_round["score"]), round_idx)
                writer.add_scalar(
                    "round/best_forward_velocity",
                    float(best_round["mean_forward_velocity"]),
                    round_idx,
                )
                writer.add_scalar("round/best_fall_penalty", float(best_round["fall_penalty"]), round_idx)

            if round_idx < rounds - 1:
                keep_n = max(1, min(keep_top_k, len(round_results) // 2))
                candidates = [
                    {
                        "candidate_id": int(result["candidate_id"]),
                        "vector": params_dict_to_vector(result["params"]),
                    }
                    for result in round_results[:keep_n]
                ]

    if writer is not None:
        writer.close()

    if not all_results:
        raise RuntimeError("No candidate results were produced.")
    if not final_round_results:
        raise RuntimeError("Final round produced no results.")

    best = final_round_results[0]

    best_cfg = {
        "controller": "param_gait_15",
        "filter_alpha": float(filter_alpha),
        "params": {name: float(best["params"][name]) for name in PARAMETER_NAMES},
    }
    with best_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "num_workers": num_workers,
        "rounds": rounds,
        "population_per_round": population_per_round,
        "best": best,
        "max_rollout_steps": max_rollout_steps,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    record_best_video = bool(cfg["io"].get("record_best_video", False)) or bool(args.record_best_video)
    video_path = run_dir / "best_rollout.mp4"
    if record_best_video:
        cmd = [
            sys.executable,
            "scripts/rollout.py",
            "--controller",
            "param_gait_15",
            "--gait-config",
            str(best_yaml_path),
            "--env-config",
            str(env_config_path),
            "--episodes",
            "1",
            "--max-steps",
            str(max_rollout_steps),
            "--record",
            str(video_path),
        ]
        subprocess.run(cmd, check=False)

    print(f"best_score={best['score']:.6f}")
    print(f"best_forward_velocity={best['mean_forward_velocity']:.6f}")
    print(f"best_params_yaml={best_yaml_path}")
    print(f"summary_json={summary_path}")
    if record_best_video:
        print(f"best_video={video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
