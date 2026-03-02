#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from k1_walk_mujoco.rl.cleanrl.ppo_train import train_ppo
from k1_walk_mujoco.rl.cleanrl.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on K1 MuJoCo using a CleanRL-style loop.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_ppo_cleanrl.yaml",
        help="Path to training config YAML.",
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
        help="Optional short-run override for smoke tests.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires `wandb` extra).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml_config(cfg_path)

    result = train_ppo(
        cfg,
        run_name=args.run_name,
        seed_override=args.seed,
        device_override=args.device,
        num_envs_override=args.num_envs,
        total_timesteps_override=args.total_timesteps,
        wandb_override=True if args.wandb else None,
    )

    print(f"Run dir: {result['run_dir']}")
    print(f"Device: {result['device']}")
    print(f"Num envs: {result['num_envs']}")
    print(f"Latest checkpoint: {result['latest_ckpt']}")
    print(f"Best checkpoint: {result['best_ckpt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
