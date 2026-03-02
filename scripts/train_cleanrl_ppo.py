#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from k1_walk_mujoco.rl.cleanrl.ppo_train import PPOTrainConfig, evaluate_checkpoint, train_ppo
from k1_walk_mujoco.rl.cleanrl.utils import load_yaml_config, select_device


def _resolve_config_path(config: str | None, milestone: str | None) -> Path:
    if config is not None:
        return Path(config)
    if milestone is not None:
        return Path(f"configs/train_ppo_{milestone}.yaml")
    return Path("configs/train_ppo_cleanrl.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on K1 MuJoCo using a milestone-aware loop.")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML.")
    parser.add_argument(
        "--milestone",
        type=str,
        default=None,
        choices=["m0", "m1", "m2", "m3", "m4", "m5"],
        help="Use milestone config (defaults to configs/train_ppo_<milestone>.yaml).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Override training device from config.",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of vector envs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional timestep override.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires `wandb` extra).",
    )
    parser.add_argument(
        "--eval-suite",
        type=str,
        default=None,
        help="Override nominal eval suite name (e.g. easy/medium/hard/stress).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run deterministic evaluation only (requires --ckpt).",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for --eval-only.")
    parser.add_argument(
        "--print-every-updates",
        type=int,
        default=None,
        help="Terminal progress print frequency in updates.",
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default=None,
        help="Optional checkpoint to initialize policy weights from.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = _resolve_config_path(args.config, args.milestone)
    cfg_dict = load_yaml_config(cfg_path)
    if args.init_ckpt is not None:
        cfg_dict["init_checkpoint"] = args.init_ckpt

    if args.eval_only:
        if args.ckpt is None:
            raise ValueError("--eval-only requires --ckpt <path>.")
        cfg = PPOTrainConfig.from_dict(cfg_dict)
        if args.milestone is not None:
            cfg.milestone = args.milestone
        if args.seed is not None:
            cfg.seed = args.seed
        if args.device is not None:
            cfg.device = args.device
        if args.eval_suite is not None:
            cfg.eval_nominal_suite = args.eval_suite
        result = evaluate_checkpoint(
            ckpt_path=Path(args.ckpt),
            cfg=cfg,
            device=select_device(cfg.device),
            suite_name=args.eval_suite,
        )
        print(json.dumps(result, indent=2))
        return 0

    result = train_ppo(
        cfg_dict,
        run_name=args.run_name,
        seed_override=args.seed,
        device_override=args.device,
        num_envs_override=args.num_envs,
        total_timesteps_override=args.total_timesteps,
        wandb_override=True if args.wandb else None,
        milestone_override=args.milestone,
        eval_suite_override=args.eval_suite,
        print_every_updates_override=args.print_every_updates,
    )

    print(f"Run dir: {result['run_dir']}")
    print(f"Device: {result['device']}")
    print(f"Num envs: {result['num_envs']}")
    print(f"Latest checkpoint: {result['latest_ckpt']}")
    print(f"Best checkpoint: {result['best_ckpt']}")
    print(f"Best nominal checkpoint: {result['best_nominal_ckpt']}")
    print(f"Best stress checkpoint: {result['best_stress_ckpt']}")
    print(f"Eval JSONL: {result['eval_jsonl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
