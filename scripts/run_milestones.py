#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from k1_walk_mujoco.rl.cleanrl.milestone_gates import evaluate_milestone_gates, load_milestone_gates
from k1_walk_mujoco.rl.cleanrl.ppo_train import PPOTrainConfig, evaluate_checkpoint, train_ppo
from k1_walk_mujoco.rl.cleanrl.utils import load_yaml_config

MILESTONE_ORDER = ["m0", "m1", "m2", "m3", "m4", "m5"]


def _parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")
    return seeds


def _load_last_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval metrics file: {path}")
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    if last is None:
        raise ValueError(f"No JSON entries in {path}")
    return last


def _deterministic_eval_ok(
    *,
    ckpt_path: Path,
    cfg_dict: dict[str, Any],
    suite_name: str,
    tol: float,
) -> bool:
    cfg_obj = PPOTrainConfig.from_dict(cfg_dict)
    eval1 = evaluate_checkpoint(
        ckpt_path=ckpt_path,
        cfg=cfg_obj,
        device=torch.device("cpu"),
        suite_name=suite_name,
    )
    eval2 = evaluate_checkpoint(
        ckpt_path=ckpt_path,
        cfg=cfg_obj,
        device=torch.device("cpu"),
        suite_name=suite_name,
    )
    s1 = eval1["suites"][suite_name]
    s2 = eval2["suites"][suite_name]
    keys = [
        "success_rate",
        "fall_rate",
        "median_final_pos_error_m",
        "median_final_yaw_error_deg",
        "median_time_to_goal_s",
        "median_command_vx_rmse_mps",
        "median_command_vy_rmse_mps",
        "median_command_yaw_rate_rmse_rps",
        "command_tracking_success_rate",
    ]
    for key in keys:
        v1 = float(s1.get(key, float("nan")))
        v2 = float(s2.get(key, float("nan")))
        if np.isfinite(v1) != np.isfinite(v2):
            return False
        if np.isfinite(v1) and abs(v1 - v2) > tol:
            return False
    return True


def _milestone_span(start: str, end: str | None, auto_progress: bool) -> list[str]:
    if start not in MILESTONE_ORDER:
        raise ValueError(f"Unknown start milestone: {start}")
    if not auto_progress:
        return [start]
    end_m = end or MILESTONE_ORDER[-1]
    if end_m not in MILESTONE_ORDER:
        raise ValueError(f"Unknown end milestone: {end_m}")
    i = MILESTONE_ORDER.index(start)
    j = MILESTONE_ORDER.index(end_m)
    if j < i:
        raise ValueError("--until-milestone must not be before --milestone")
    return MILESTONE_ORDER[i : j + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3-seed milestone training with gate enforcement and optional auto-progression."
    )
    parser.add_argument(
        "--milestone",
        type=str,
        required=True,
        choices=MILESTONE_ORDER,
        help="Start milestone to run.",
    )
    parser.add_argument(
        "--auto-progress",
        action="store_true",
        help="Advance to next milestones automatically when gates pass.",
    )
    parser.add_argument(
        "--until-milestone",
        type=str,
        default=None,
        choices=MILESTONE_ORDER,
        help="Last milestone for --auto-progress (default m5).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-separated seeds, e.g. 1,2,3",
    )
    parser.add_argument(
        "--config-template",
        type=str,
        default="configs/train_ppo_{milestone}.yaml",
        help="Config template with {milestone}.",
    )
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--print-every-updates", type=int, default=None)
    parser.add_argument("--run-prefix", type=str, default="pipeline")
    parser.add_argument("--run-dir", type=str, default="runs/cleanrl_ppo")
    parser.add_argument(
        "--gates-config",
        type=str,
        default="configs/milestone_gates.yaml",
    )
    parser.add_argument("--skip-determinism-check", action="store_true")
    parser.add_argument("--determinism-tol", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = _parse_seed_list(args.seeds)
    milestones = _milestone_span(args.milestone, args.until_milestone, args.auto_progress)
    gates = load_milestone_gates(Path(args.gates_config))
    reports_dir = Path(args.run_dir) / "milestone_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    init_ckpt_by_seed: dict[int, str] = {}
    pipeline_reports: list[dict[str, Any]] = []

    for milestone in milestones:
        run_summaries: list[dict[str, Any]] = []
        best_ckpts_for_next: dict[int, str] = {}
        cfg_path = Path(args.config_template.format(milestone=milestone))
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing milestone config: {cfg_path}")

        for seed in seeds:
            cfg = load_yaml_config(cfg_path)
            cfg["run_dir"] = args.run_dir
            if seed in init_ckpt_by_seed:
                cfg["init_checkpoint"] = init_ckpt_by_seed[seed]

            run_name = f"{args.run_prefix}_{milestone}_s{seed}"
            result = train_ppo(
                cfg,
                run_name=run_name,
                seed_override=seed,
                device_override=args.device,
                num_envs_override=args.num_envs,
                total_timesteps_override=args.total_timesteps,
                print_every_updates_override=args.print_every_updates,
            )
            eval_json_path = Path(result["eval_jsonl"])
            last_eval = _load_last_eval(eval_json_path)

            deterministic_ok = True
            if not args.skip_determinism_check:
                suite_name = str(cfg.get("eval_nominal_suite", "easy"))
                deterministic_ok = _deterministic_eval_ok(
                    ckpt_path=Path(result["latest_ckpt"]),
                    cfg_dict=cfg,
                    suite_name=suite_name,
                    tol=float(args.determinism_tol),
                )

            run_summary = {
                "milestone": milestone,
                "seed": seed,
                "run_dir": result["run_dir"],
                "latest_checkpoint": result["latest_ckpt"],
                "best_nominal_checkpoint": result["best_nominal_ckpt"],
                "latest_checkpoint_exists": Path(result["latest_ckpt"]).exists(),
                "eval_json_exists": eval_json_path.exists(),
                "deterministic_eval_ok": deterministic_ok,
                "suites": last_eval.get("suites", {}),
            }
            run_summaries.append(run_summary)
            best_ckpts_for_next[seed] = str(result["best_nominal_ckpt"])

        gate_result = evaluate_milestone_gates(
            milestone=milestone,
            run_summaries=run_summaries,
            gate_config=gates,
        )
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "milestone": milestone,
            "seeds": seeds,
            "run_summaries": run_summaries,
            "gate_result": gate_result,
        }
        report_path = reports_dir / f"{milestone}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(
            f"[{milestone}] passed={gate_result['passed']} "
            f"report={report_path}"
        )
        pipeline_reports.append(
            {
                "milestone": milestone,
                "passed": bool(gate_result["passed"]),
                "report_path": str(report_path),
            }
        )

        if not gate_result["passed"]:
            print(json.dumps({"pipeline_reports": pipeline_reports}, indent=2))
            return 2

        init_ckpt_by_seed = best_ckpts_for_next

    print(json.dumps({"pipeline_reports": pipeline_reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
