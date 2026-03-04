from __future__ import annotations

from pathlib import Path

from k1_walk_mujoco.rl.cleanrl.eval_harness import load_eval_suites


def test_holdout_suite_exists_and_has_scenarios() -> None:
    suites = load_eval_suites(Path("configs/eval_suites_goal_pose_holdout.yaml"))
    assert "holdout" in suites
    scenarios = suites["holdout"]
    assert len(scenarios) >= 3
    assert all(isinstance(s, dict) for s in scenarios)
