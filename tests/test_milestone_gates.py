from __future__ import annotations

from pathlib import Path

from k1_walk_mujoco.rl.cleanrl.milestone_gates import (
    evaluate_milestone_gates,
    load_milestone_gates,
)


def _run_summary(
    *,
    suite_metrics: dict[str, dict[str, float]],
    deterministic_ok: bool = True,
) -> dict:
    return {
        "latest_checkpoint_exists": True,
        "eval_json_exists": True,
        "deterministic_eval_ok": deterministic_ok,
        "suites": suite_metrics,
    }


def test_m0_gate_passes_with_all_artifacts() -> None:
    gates = load_milestone_gates(Path("configs/milestone_gates.yaml"))
    runs = [_run_summary(suite_metrics={}) for _ in range(3)]
    result = evaluate_milestone_gates(milestone="m0", run_summaries=runs, gate_config=gates)
    assert result["passed"]


def test_m1_gate_fails_on_rmse() -> None:
    gates = load_milestone_gates(Path("configs/milestone_gates.yaml"))
    runs = [
        _run_summary(
            suite_metrics={
                "m1_nominal": {
                    "fall_rate": 0.05,
                    "median_command_vx_rmse_mps": 0.20,
                }
            }
        ),
        _run_summary(
            suite_metrics={
                "m1_nominal": {
                    "fall_rate": 0.08,
                    "median_command_vx_rmse_mps": 0.24,
                }
            }
        ),
        _run_summary(
            suite_metrics={
                "m1_nominal": {
                    "fall_rate": 0.09,
                    "median_command_vx_rmse_mps": 0.30,
                }
            }
        ),
    ]
    result = evaluate_milestone_gates(milestone="m1", run_summaries=runs, gate_config=gates)
    assert not result["passed"]


def test_m3_gate_passes_with_thresholds() -> None:
    gates = load_milestone_gates(Path("configs/milestone_gates.yaml"))
    runs = [
        _run_summary(
            suite_metrics={
                "easy": {
                    "success_rate": 0.85,
                    "median_final_pos_error_m": 0.18,
                    "median_final_yaw_error_deg": 11.0,
                }
            }
        ),
        _run_summary(
            suite_metrics={
                "easy": {
                    "success_rate": 0.84,
                    "median_final_pos_error_m": 0.19,
                    "median_final_yaw_error_deg": 10.0,
                }
            }
        ),
        _run_summary(
            suite_metrics={
                "easy": {
                    "success_rate": 0.83,
                    "median_final_pos_error_m": 0.20,
                    "median_final_yaw_error_deg": 12.0,
                }
            }
        ),
    ]
    result = evaluate_milestone_gates(milestone="m3", run_summaries=runs, gate_config=gates)
    assert result["passed"]
