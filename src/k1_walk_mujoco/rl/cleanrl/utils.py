from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
from typing import Any

import torch
import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(loaded)!r}")
    return loaded


def _is_cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _is_mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def select_device(requested: str) -> torch.device:
    choice = requested.strip().lower()
    if choice == "auto":
        if _is_cuda_available():
            return torch.device("cuda")
        if _is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda":
        if not _is_cuda_available():
            raise RuntimeError("Requested device=cuda, but CUDA is not available.")
        return torch.device("cuda")
    if choice == "mps":
        if not _is_mps_available():
            raise RuntimeError("Requested device=mps, but MPS is not available.")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested!r}")


def default_num_envs(device_type: str, system_name: str | None = None) -> int:
    system = (system_name or platform.system()).lower()
    if device_type == "cuda":
        return 16
    if device_type == "mps":
        return 8
    if system == "darwin":
        return 4
    return 8


def resolve_num_envs(requested: int | str | None, device_type: str) -> int:
    if requested is None:
        return default_num_envs(device_type=device_type)
    if isinstance(requested, str):
        if requested.strip().lower() == "auto":
            return default_num_envs(device_type=device_type)
        value = int(requested)
    else:
        value = int(requested)
    if value <= 0:
        return default_num_envs(device_type=device_type)
    return value


def build_run_name(prefix: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_s{seed}_{ts}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
