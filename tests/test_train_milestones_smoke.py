from __future__ import annotations

from pathlib import Path

import torch
import yaml

from k1_walk_mujoco.rl.cleanrl.ppo_train import PPOTrainConfig, evaluate_checkpoint, train_ppo


MILESTONE_CONFIGS = [
    "configs/train_ppo_m0.yaml",
    "configs/train_ppo_m1.yaml",
    "configs/train_ppo_m2.yaml",
    "configs/train_ppo_m3.yaml",
    "configs/train_ppo_m4.yaml",
    "configs/train_ppo_m5.yaml",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def test_train_smoke_each_milestone(tmp_path: Path) -> None:
    for idx, cfg_rel in enumerate(MILESTONE_CONFIGS):
        cfg = _load_yaml(Path(cfg_rel))
        cfg["run_dir"] = str(tmp_path / "runs")
        cfg["eval_suite_path"] = None
        result = train_ppo(
            cfg,
            run_name=f"smoke_{idx}",
            total_timesteps_override=64,
            num_envs_override=1,
            device_override="cpu",
            print_every_updates_override=0,
        )
        assert Path(result["latest_ckpt"]).exists()


def test_eval_suite_deterministic_for_fixed_checkpoint(tmp_path: Path) -> None:
    cfg = _load_yaml(Path("configs/train_ppo_m3.yaml"))
    cfg["run_dir"] = str(tmp_path / "runs")
    result = train_ppo(
        cfg,
        run_name="eval_determinism_smoke",
        total_timesteps_override=64,
        num_envs_override=1,
        device_override="cpu",
        print_every_updates_override=0,
    )
    ckpt = Path(result["latest_ckpt"])
    cfg_obj = PPOTrainConfig.from_dict(cfg)
    ev1 = evaluate_checkpoint(ckpt_path=ckpt, cfg=cfg_obj, device=torch.device("cpu"))
    ev2 = evaluate_checkpoint(ckpt_path=ckpt, cfg=cfg_obj, device=torch.device("cpu"))

    s1 = ev1["suites"][cfg_obj.eval_nominal_suite]["success_rate"]
    s2 = ev2["suites"][cfg_obj.eval_nominal_suite]["success_rate"]
    assert s1 == s2
