#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_last_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval metrics file: {path}")
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    if last is None:
        raise ValueError(f"No JSON records in {path}")
    return last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate milestone success metrics across multiple seed runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories under runs/cleanrl_ppo (or absolute paths).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="easy",
        help="Suite name to aggregate (easy/medium/hard/stress).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = args.suite

    values: list[float] = []
    per_run: list[tuple[str, float]] = []
    for run in args.runs:
        run_path = Path(run)
        metrics_file = run_path / "eval" / "metrics.jsonl"
        last = _load_last_eval(metrics_file)
        suites = last.get("suites", {})
        if suite not in suites:
            raise KeyError(f"Suite '{suite}' not found in {metrics_file}")
        val = float(suites[suite]["success_rate"])
        values.append(val)
        per_run.append((str(run_path), val))

    arr = np.asarray(values, dtype=np.float64)
    worst_idx = int(np.argmin(arr))
    report = {
        "suite": suite,
        "num_runs": int(arr.size),
        "success_rate_mean": float(np.mean(arr)),
        "success_rate_median": float(np.median(arr)),
        "success_rate_worst": float(np.min(arr)),
        "worst_run": per_run[worst_idx][0],
        "per_run": [{"run": run, "success_rate": val} for run, val in per_run],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
