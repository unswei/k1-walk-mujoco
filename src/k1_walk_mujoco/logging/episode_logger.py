from __future__ import annotations

import json
from pathlib import Path


class EpisodeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_episode(self, episode_idx: int, total_reward: float, length: int) -> None:
        row = {
            "episode": episode_idx,
            "return": float(total_reward),
            "length": int(length),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
