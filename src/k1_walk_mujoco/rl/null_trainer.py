from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from k1_walk_mujoco.logging.episode_logger import EpisodeLogger
from k1_walk_mujoco.rl.api import Trainer


class NullTrainer(Trainer):
    def train(self, env: Any, cfg: dict[str, Any]) -> Path:
        episodes = int(cfg.get("episodes", 5))
        rng = np.random.default_rng(int(cfg.get("seed", 123)))
        out = Path(cfg.get("output", "runs/null_trainer_episodes.jsonl"))
        logger = EpisodeLogger(out)

        for ep in range(episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000_000)))
            del obs
            done = False
            trunc = False
            total_rew = 0.0
            steps = 0

            while not (done or trunc):
                action = env.action_space.sample()
                _, rew, done, trunc, _ = env.step(action)
                total_rew += rew
                steps += 1

            logger.log_episode(ep, total_rew, steps)

        ckpt = Path("runs") / f"null_trainer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("null-trainer checkpoint placeholder\n", encoding="utf-8")
        return ckpt

    def evaluate(self, env: Any, ckpt: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        del ckpt
        eval_steps = int(cfg.get("eval_steps", 200))
        obs, _ = env.reset(seed=int(cfg.get("seed", 123)))
        del obs
        total_rew = 0.0
        done = False
        trunc = False
        steps = 0

        while steps < eval_steps and not (done or trunc):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, rew, done, trunc, _ = env.step(action)
            total_rew += rew
            steps += 1

        return {"steps": steps, "return": total_rew, "done": done, "truncated": trunc}
