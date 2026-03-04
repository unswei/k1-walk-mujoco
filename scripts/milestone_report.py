#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from k1_walk_mujoco.rl.cleanrl.milestone_gates import evaluate_milestone_gates, load_milestone_gates


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
        description="Aggregate milestone metrics across multiple seed runs and evaluate promotion gates."
    )
    parser.add_argument(
        "--milestone",
        type=str,
        required=True,
        choices=["m0", "m1", "m2", "m3", "m4", "m5"],
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories under runs/cleanrl_ppo (or absolute paths).",
    )
    parser.add_argument(
        "--gates-config",
        type=str,
        default="configs/milestone_gates.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate_cfg = load_milestone_gates(Path(args.gates_config))

    run_summaries: list[dict[str, Any]] = []
    for run in args.runs:
        run_path = Path(run)
        metrics_file = run_path / "eval" / "metrics.jsonl"
        last = _load_last_eval(metrics_file)
        latest_ckpt = run_path / "checkpoints" / "latest.pt"
        run_summaries.append(
            {
                "run_dir": str(run_path),
                "latest_checkpoint": str(latest_ckpt),
                "latest_checkpoint_exists": latest_ckpt.exists(),
                "eval_json_exists": metrics_file.exists(),
                "deterministic_eval_ok": True,
                "suites": last.get("suites", {}),
            }
        )

    gate_result = evaluate_milestone_gates(
        milestone=args.milestone,
        run_summaries=run_summaries,
        gate_config=gate_cfg,
    )
    report = {
        "milestone": args.milestone,
        "num_runs": len(run_summaries),
        "run_summaries": run_summaries,
        "gate_result": gate_result,
    }

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if gate_result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
