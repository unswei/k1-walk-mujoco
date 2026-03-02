#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--policy", choices=["zero", "random"], default="zero")
    args = parser.parse_args()

    env = K1WalkEnv()
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rollouts_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=123 + ep)
            done = False
            trunc = False
            t = 0
            while t < args.max_steps and not (done or trunc):
                if args.policy == "zero":
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action = env.action_space.sample()
                next_obs, reward, done, trunc, info = env.step(action)
                row = {
                    "episode": ep,
                    "t": t,
                    "obs": obs.tolist(),
                    "action": np.asarray(action).tolist(),
                    "reward": float(reward),
                    "next_obs": next_obs.tolist(),
                    "terminated": bool(done),
                    "truncated": bool(trunc),
                    "termination_reason": info.get("termination_reason", "none"),
                }
                f.write(json.dumps(row) + "\n")
                obs = next_obs
                t += 1

    print(f"Wrote rollouts to {out_path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
