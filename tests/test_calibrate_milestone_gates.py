from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_milestone_gates.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_milestone_gates_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_aggregate = _MODULE._aggregate
_suggest_threshold = _MODULE._suggest_threshold
_maybe_clamp_threshold = _MODULE._maybe_clamp_threshold
_load_reports = _MODULE._load_reports
_collect_inputs = _MODULE._collect_inputs


def test_suggest_threshold_directionality() -> None:
    assert _suggest_threshold(op=">=", actual=0.8, rel_margin=0.05, abs_margin=0.0) < 0.8
    assert _suggest_threshold(op="<=", actual=0.2, rel_margin=0.05, abs_margin=0.0) > 0.2


def test_aggregate_variants() -> None:
    vals = [1.0, 2.0, 3.0]
    assert _aggregate(vals, "median") == 2.0
    assert _aggregate(vals, "mean") == 2.0
    assert _aggregate(vals, "worst_min") == 1.0
    assert _aggregate(vals, "worst_max") == 3.0


def test_rate_thresholds_are_clamped() -> None:
    assert _maybe_clamp_threshold("success_rate", 1.2) == 1.0
    assert _maybe_clamp_threshold("fall_rate", -0.1) == 0.0
    assert _maybe_clamp_threshold("median_time_to_goal_s", 4.2) == 4.2


def test_load_reports_keeps_latest_per_seed(tmp_path: Path) -> None:
    old_report = {
        "milestone": "m3",
        "run_summaries": [
            {"seed": 1, "suites": {"easy": {"success_rate": 0.5}}},
        ],
    }
    new_report = {
        "milestone": "m3",
        "run_summaries": [
            {"seed": 1, "suites": {"easy": {"success_rate": 0.7}}},
        ],
    }
    p1 = tmp_path / "old.json"
    p2 = tmp_path / "new.json"
    p1.write_text(json.dumps(old_report), encoding="utf-8")
    p2.write_text(json.dumps(new_report), encoding="utf-8")
    p2.touch()

    by_key = _load_reports([p1, p2])
    grouped = _collect_inputs(runs_by_key=by_key)
    assert grouped["m3"][0]["suites"]["easy"]["success_rate"] == 0.7
