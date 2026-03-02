from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Trainer(ABC):
    @abstractmethod
    def train(self, env: Any, cfg: dict[str, Any]) -> Path:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, env: Any, ckpt: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class TrainerRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[Trainer]] = {}

    def register(self, name: str, trainer_cls: type[Trainer]) -> None:
        if name in self._registry:
            raise KeyError(f"Trainer already registered: {name}")
        self._registry[name] = trainer_cls

    def create(self, name: str, **kwargs: Any) -> Trainer:
        if name not in self._registry:
            raise KeyError(f"Unknown trainer: {name}")
        return self._registry[name](**kwargs)
