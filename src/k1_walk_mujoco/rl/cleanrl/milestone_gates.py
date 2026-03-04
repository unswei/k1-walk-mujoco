from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_milestone_gates(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(raw)!r}")
    return raw


def _compare(op: str, actual: float, threshold: float) -> bool:
    if op == "<":
        return actual < threshold
    if op == "<=":
        return actual <= threshold
    if op == ">":
        return actual > threshold
    if op == ">=":
        return actual >= threshold
    if op == "==":
        return actual == threshold
    raise ValueError(f"Unsupported operator: {op}")


def _aggregate(values: np.ndarray, kind: str) -> float:
    if values.size == 0:
        return float("nan")
    if kind == "median":
        return float(np.nanmedian(values))
    if kind == "mean":
        return float(np.nanmean(values))
    if kind == "worst_min":
        return float(np.nanmin(values))
    if kind == "worst_max":
        return float(np.nanmax(values))
    raise ValueError(f"Unsupported aggregation: {kind}")


def evaluate_milestone_gates(
    *,
    milestone: str,
    run_summaries: list[dict[str, Any]],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    if milestone not in gate_config:
        raise KeyError(f"No gate config for milestone {milestone!r}")
    checks_cfg = gate_config[milestone].get("checks", [])
    if not isinstance(checks_cfg, list):
        raise TypeError(f"Gate checks for {milestone} must be list")

    check_results: list[dict[str, Any]] = []
    all_passed = True
    for check in checks_cfg:
        check_id = str(check["id"])
        kind = str(check["kind"])

        if kind == "all_true":
            field = str(check["field"])
            vals = [bool(run.get(field, False)) for run in run_summaries]
            passed = all(vals)
            details = {
                "id": check_id,
                "kind": kind,
                "field": field,
                "values": vals,
                "passed": passed,
            }
        elif kind == "suite_threshold":
            suite = str(check["suite"])
            metric = str(check["metric"])
            aggregation = str(check["aggregation"])
            op = str(check["op"])
            threshold = float(check["threshold"])

            vals: list[float] = []
            missing = False
            for run in run_summaries:
                suites = run.get("suites", {})
                if suite not in suites or metric not in suites[suite]:
                    missing = True
                    break
                vals.append(float(suites[suite][metric]))

            if missing or not vals:
                passed = False
                actual = float("nan")
            else:
                arr = np.asarray(vals, dtype=np.float64)
                actual = _aggregate(arr, aggregation)
                passed = np.isfinite(actual) and _compare(op, actual, threshold)

            details = {
                "id": check_id,
                "kind": kind,
                "suite": suite,
                "metric": metric,
                "aggregation": aggregation,
                "op": op,
                "threshold": threshold,
                "values": vals,
                "actual": actual,
                "passed": passed,
            }
        else:
            raise ValueError(f"Unsupported gate kind: {kind}")

        check_results.append(details)
        all_passed = all_passed and bool(details["passed"])

    return {
        "milestone": milestone,
        "num_runs": len(run_summaries),
        "passed": bool(all_passed),
        "checks": check_results,
    }
