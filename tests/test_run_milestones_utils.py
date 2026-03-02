from __future__ import annotations

import importlib.util
from pathlib import Path


_RUN_MILESTONES_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_milestones.py"
_SPEC = importlib.util.spec_from_file_location("run_milestones_module", _RUN_MILESTONES_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_milestone_span = _MODULE._milestone_span
_set_transition_mix = _MODULE._set_transition_mix
_task_mode_from_config = _MODULE._task_mode_from_config


def test_task_mode_from_config_reads_env_override() -> None:
    cfg = {
        "env_overrides": {
            "task": {
                "mode": "goal_pose",
            }
        }
    }
    assert _task_mode_from_config(cfg) == "goal_pose"


def test_set_transition_mix_preserves_existing_keys() -> None:
    cfg = {
        "env_overrides": {
            "task": {"mode": "goal_pose"},
            "reward": {"w_progress": 10.0},
        }
    }
    updated = _set_transition_mix(
        cfg=cfg,
        previous_mode="command_tracking",
        previous_fraction=0.2,
        enabled=True,
    )
    assert updated["env_overrides"]["task"]["mode"] == "goal_pose"
    assert updated["env_overrides"]["reward"]["w_progress"] == 10.0
    assert updated["env_overrides"]["task"]["transition_mix"] == {
        "enabled": True,
        "previous_mode": "command_tracking",
        "previous_fraction": 0.2,
    }


def test_milestone_span_respects_auto_progress_flag() -> None:
    assert _milestone_span("m2", "m4", auto_progress=True) == ["m2", "m3", "m4"]
    assert _milestone_span("m2", "m4", auto_progress=False) == ["m2"]
