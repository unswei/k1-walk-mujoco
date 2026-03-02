#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from k1_walk_mujoco.rl.cleanrl.milestone_gates import load_milestone_gates


def _aggregate(values: list[float], kind: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if kind == "median":
        return float(np.nanmedian(arr))
    if kind == "mean":
        return float(np.nanmean(arr))
    if kind == "worst_min":
        return float(np.nanmin(arr))
    if kind == "worst_max":
        return float(np.nanmax(arr))
    raise ValueError(f"Unsupported aggregation: {kind}")


def _suggest_threshold(
    *,
    op: str,
    actual: float,
    rel_margin: float,
    abs_margin: float,
) -> float:
    span = max(abs(actual), 1.0) * rel_margin + abs_margin
    if op in {">", ">="}:
        return actual - span
    if op in {"<", "<="}:
        return actual + span
    if op == "==":
        return actual
    raise ValueError(f"Unsupported operator: {op}")


def _maybe_clamp_threshold(metric: str, threshold: float) -> float:
    metric_l = metric.lower()
    if metric_l.endswith("_rate") or "success_rate" in metric_l or "fall_rate" in metric_l:
        return float(np.clip(threshold, 0.0, 1.0))
    return float(threshold)


def _load_reports(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    ordered = sorted(paths, key=lambda p: p.stat().st_mtime)
    for path in ordered:
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        milestone = str(report["milestone"])
        run_summaries = report.get("run_summaries", [])
        if not isinstance(run_summaries, list):
            continue
        for run in run_summaries:
            if not isinstance(run, dict):
                continue
            seed = int(run.get("seed", -1))
            if seed < 0:
                continue
            latest[(milestone, seed)] = run
    return latest


def _collect_inputs(
    *,
    runs_by_key: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (milestone, _seed), run in sorted(runs_by_key.items()):
        out[milestone].append(run)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate milestone gate thresholds from milestone report JSON files."
    )
    p.add_argument(
        "--reports-glob",
        type=str,
        required=True,
        help="Glob for milestone report JSON files, e.g. runs/cleanrl_ppo/milestone_reports/*.json",
    )
    p.add_argument(
        "--gates-config",
        type=str,
        default="configs/milestone_gates.yaml",
    )
    p.add_argument(
        "--rel-margin",
        type=float,
        default=0.05,
        help="Relative slack around observed aggregate value.",
    )
    p.add_argument(
        "--abs-margin",
        type=float,
        default=1e-6,
        help="Absolute slack around observed aggregate value.",
    )
    p.add_argument(
        "--output-gates",
        type=str,
        default=None,
        help="Optional path for calibrated gates YAML in JSON-compatible formatting.",
    )
    p.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional path for calibration diagnostic report JSON.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = sorted(Path().glob(args.reports_glob))
    if not report_paths:
        raise FileNotFoundError(f"No report files matched: {args.reports_glob}")

    gates = load_milestone_gates(Path(args.gates_config))
    runs_by_key = _load_reports(report_paths)
    runs_by_milestone = _collect_inputs(runs_by_key=runs_by_key)
    calibrated = copy.deepcopy(gates)

    diagnostics: dict[str, Any] = {
        "reports_glob": args.reports_glob,
        "num_report_files": len(report_paths),
        "num_seed_runs": len(runs_by_key),
        "milestones": {},
    }

    for milestone, cfg in gates.items():
        checks = cfg.get("checks", [])
        runs = runs_by_milestone.get(milestone, [])
        mdiag: dict[str, Any] = {
            "num_runs": len(runs),
            "checks": [],
        }

        for idx, check in enumerate(checks):
            if check.get("kind") != "suite_threshold":
                mdiag["checks"].append(
                    {
                        "id": check.get("id"),
                        "kind": check.get("kind"),
                        "status": "unchanged_non_threshold",
                    }
                )
                continue

            suite = str(check["suite"])
            metric = str(check["metric"])
            aggregation = str(check["aggregation"])
            op = str(check["op"])
            old_threshold = float(check["threshold"])

            values: list[float] = []
            missing = False
            for run in runs:
                suites = run.get("suites", {})
                if suite not in suites or metric not in suites[suite]:
                    missing = True
                    break
                values.append(float(suites[suite][metric]))

            if missing or not values:
                mdiag["checks"].append(
                    {
                        "id": check.get("id"),
                        "kind": "suite_threshold",
                        "status": "missing_inputs",
                        "suite": suite,
                        "metric": metric,
                        "values": values,
                        "old_threshold": old_threshold,
                    }
                )
                continue

            actual = _aggregate(values, aggregation)
            if not np.isfinite(actual):
                mdiag["checks"].append(
                    {
                        "id": check.get("id"),
                        "kind": "suite_threshold",
                        "status": "non_finite_actual",
                        "suite": suite,
                        "metric": metric,
                        "values": values,
                        "actual": actual,
                        "old_threshold": old_threshold,
                    }
                )
                continue

            new_threshold = _suggest_threshold(
                op=op,
                actual=float(actual),
                rel_margin=float(args.rel_margin),
                abs_margin=float(args.abs_margin),
            )
            new_threshold = _maybe_clamp_threshold(metric=metric, threshold=float(new_threshold))
            calibrated[milestone]["checks"][idx]["threshold"] = float(new_threshold)

            mdiag["checks"].append(
                {
                    "id": check.get("id"),
                    "kind": "suite_threshold",
                    "status": "updated",
                    "suite": suite,
                    "metric": metric,
                    "aggregation": aggregation,
                    "op": op,
                    "values": values,
                    "actual": float(actual),
                    "old_threshold": old_threshold,
                    "new_threshold": float(new_threshold),
                }
            )

        diagnostics["milestones"][milestone] = mdiag

    if args.output_report is not None:
        out_report = Path(args.output_report)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    if args.output_gates is not None:
        out_gates = Path(args.output_gates)
        out_gates.parent.mkdir(parents=True, exist_ok=True)
        # JSON syntax is valid YAML and keeps deterministic formatting for this tool.
        out_gates.write_text(json.dumps(calibrated, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
