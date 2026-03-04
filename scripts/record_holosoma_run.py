#!/usr/bin/env python3
"""Structured experiment records and status parsing for Holosoma MJWarp runs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(ts: str) -> Optional[dt.datetime]:
    try:
        if ts.endswith("Z"):
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.datetime.fromisoformat(ts)
    except Exception:
        return None


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def git_info(repo_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not repo_dir:
        return None, None
    repo = Path(repo_dir)
    if not repo.exists():
        return None, None

    def run_git(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)
            return out.decode("utf-8", "ignore").strip() or None
        except Exception:
            return None

    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_git(["rev-parse", "HEAD"])
    return branch, commit


def _last_float(pattern: str, text: str) -> Optional[float]:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _last_int(pattern: str, text: str) -> Optional[int]:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def parse_training_log(log_path: Path, tail_bytes: int = 2_000_000) -> Dict[str, Any]:
    if not log_path.exists():
        return {}

    try:
        raw = log_path.read_bytes()
    except Exception:
        return {}

    if len(raw) > tail_bytes:
        raw = raw[-tail_bytes:]
    text = raw.decode("utf-8", "ignore")

    progress: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}

    iter_matches = re.findall(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", text)
    if iter_matches:
        cur, total = iter_matches[-1]
        progress["learning_iteration"] = int(cur)
        progress["learning_iteration_total"] = int(total)

    total_timesteps = _last_int(r"Total timesteps:\s*([0-9]+)", text)
    if total_timesteps is not None:
        progress["total_timesteps"] = total_timesteps

    iteration_time_s = _last_float(r"Iteration time:\s*([0-9.]+)s", text)
    if iteration_time_s is not None:
        progress["iteration_time_s"] = iteration_time_s

    total_time_s = _last_float(r"Total time:\s*([0-9.]+)s", text)
    if total_time_s is not None:
        progress["total_time_s"] = total_time_s

    eta_s = _last_float(r"ETA:\s*([0-9.]+)s", text)
    if eta_s is not None:
        now = utc_now()
        eta_finish = now + dt.timedelta(seconds=eta_s)
        progress["eta_s"] = eta_s
        progress["estimated_finish_utc"] = iso_z(eta_finish)
        progress["estimated_finish_local"] = eta_finish.astimezone().replace(microsecond=0).isoformat()

    steps_per_s = _last_float(r"Computation:\s*([0-9.]+)\s*steps/s", text)
    if steps_per_s is not None:
        metrics["computation_steps_per_s"] = steps_per_s

    mean_reward = _last_float(r"Mean reward:\s*([-0-9.]+)", text)
    if mean_reward is not None:
        metrics["mean_reward"] = mean_reward

    mean_episode_length = _last_float(r"Mean episode length:\s*([-0-9.]+)", text)
    if mean_episode_length is not None:
        metrics["mean_episode_length"] = mean_episode_length

    if not progress and not metrics:
        return {}

    out: Dict[str, Any] = {}
    if progress:
        out["progress"] = progress
    if metrics:
        out["metrics"] = metrics
    return out


def start_record(args: argparse.Namespace) -> int:
    output = Path(args.output)
    now = utc_now()
    if args.start_utc:
        parsed_start = parse_iso(args.start_utc)
        if parsed_start is None:
            print(f"Invalid --start-utc value: {args.start_utc}", file=sys.stderr)
            return 2
        now = parsed_start.astimezone(dt.timezone.utc)
    branch, commit = git_info(args.repo_dir)

    settings: Dict[str, Any] = {}
    if args.project:
        settings["project"] = args.project
    if args.experiment:
        settings["experiment"] = args.experiment
    if args.simulator:
        settings["simulator"] = args.simulator
    if args.mode:
        settings["mode"] = args.mode
    if args.num_envs is not None:
        settings["training_num_envs"] = args.num_envs
    if args.iterations is not None:
        settings["num_learning_iterations"] = args.iterations

    paths: Dict[str, Any] = {}
    if args.log_path:
        paths["log_file"] = args.log_path
    if args.repo_dir:
        paths["repo_dir"] = args.repo_dir

    environment: Dict[str, Any] = {"host": args.host or socket.gethostname()}
    if branch:
        environment["git_branch"] = branch
    if commit:
        environment["git_commit"] = commit

    record: Dict[str, Any] = {
        "record_type": "run",
        "date": now.date().isoformat(),
        "run_name": args.run_name,
        "status": "running",
        "timing": {"start_utc": iso_z(now)},
    }

    if args.seed is not None:
        record["seed"] = args.seed
    if settings:
        record["settings"] = settings
    if args.command:
        record["command"] = args.command
    if paths:
        record["paths"] = paths
    if environment:
        record["environment"] = environment
    if args.notes:
        record["notes"] = args.notes

    parsed = parse_training_log(Path(args.log_path)) if args.log_path else {}
    if parsed.get("progress"):
        record["progress"] = parsed["progress"]
    if parsed.get("metrics"):
        record["metrics"] = parsed["metrics"]

    write_json(output, record)
    print(output)
    return 0


def finish_record(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if not output.exists():
        print(f"Record does not exist: {output}", file=sys.stderr)
        return 2

    record = read_json(output)
    now = utc_now()

    exit_code = args.exit_code
    if args.status:
        status = args.status
    else:
        if exit_code is None:
            status = "completed"
        else:
            status = "completed" if exit_code == 0 else "failed"

    record["status"] = status
    if exit_code is not None:
        record["exit_code"] = exit_code

    timing = record.setdefault("timing", {})
    timing["end_utc"] = iso_z(now)

    start_utc = timing.get("start_utc")
    start_ts = parse_iso(str(start_utc)) if start_utc else None
    if start_ts is not None:
        timing["duration_s"] = int(round((now - start_ts).total_seconds()))

    log_path_str = args.log_path
    if not log_path_str:
        log_path_str = record.get("paths", {}).get("log_file")

    if log_path_str:
        parsed = parse_training_log(Path(log_path_str))
        if parsed.get("progress"):
            record["progress"] = parsed["progress"]
        if parsed.get("metrics"):
            existing = record.get("metrics", {})
            existing.update(parsed["metrics"])
            record["metrics"] = existing

    if args.notes:
        if record.get("notes"):
            record["notes"] = f"{record['notes']} {args.notes}".strip()
        else:
            record["notes"] = args.notes

    write_json(output, record)
    print(output)
    return 0


def status_record(args: argparse.Namespace) -> int:
    parsed = parse_training_log(Path(args.log_path))
    if args.json:
        print(json.dumps(parsed, indent=2, sort_keys=True))
        return 0

    progress = parsed.get("progress", {})
    metrics = parsed.get("metrics", {})

    print(f"log_file: {args.log_path}")

    cur = progress.get("learning_iteration")
    total = progress.get("learning_iteration_total")
    if cur is not None and total:
        pct = 100.0 * float(cur) / float(total)
        print(f"iteration: {cur}/{total} ({pct:.2f}%)")

    if "total_timesteps" in progress:
        print(f"total_timesteps: {progress['total_timesteps']}")

    if "computation_steps_per_s" in metrics:
        print(f"throughput_steps_per_s: {metrics['computation_steps_per_s']:.1f}")

    if "iteration_time_s" in progress:
        print(f"iteration_time_s: {progress['iteration_time_s']:.2f}")

    if "total_time_s" in progress:
        print(f"elapsed_s: {progress['total_time_s']:.1f}")

    if "eta_s" in progress:
        eta_s = float(progress["eta_s"])
        print(f"eta_s: {eta_s:.1f}")
        print(f"eta_h: {eta_s/3600.0:.2f}")
        print(f"estimated_finish_utc: {progress.get('estimated_finish_utc', 'unknown')}")
        print(f"estimated_finish_local: {progress.get('estimated_finish_local', 'unknown')}")

    if "mean_reward" in metrics:
        print(f"mean_reward: {metrics['mean_reward']:.4f}")

    if "mean_episode_length" in metrics:
        print(f"mean_episode_length: {metrics['mean_episode_length']:.4f}")

    if not progress and not metrics:
        print("status: no training stats found yet")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Create a run record with running status")
    p_start.add_argument("--output", required=True)
    p_start.add_argument("--run-name", required=True)
    p_start.add_argument("--project")
    p_start.add_argument("--experiment")
    p_start.add_argument("--simulator", default="mjwarp")
    p_start.add_argument("--mode")
    p_start.add_argument("--seed", type=int)
    p_start.add_argument("--num-envs", type=int)
    p_start.add_argument("--iterations", type=int)
    p_start.add_argument("--repo-dir")
    p_start.add_argument("--host")
    p_start.add_argument("--command")
    p_start.add_argument("--log-path")
    p_start.add_argument("--notes")
    p_start.add_argument("--start-utc", help="Override start time (ISO 8601, e.g. 2026-03-04T11:12:00Z)")
    p_start.set_defaults(func=start_record)

    p_finish = sub.add_parser("finish", help="Finalise a run record with completion status")
    p_finish.add_argument("--output", required=True)
    p_finish.add_argument("--status", choices=["completed", "failed", "stopped"], default=None)
    p_finish.add_argument("--exit-code", type=int, default=None)
    p_finish.add_argument("--log-path")
    p_finish.add_argument("--notes")
    p_finish.set_defaults(func=finish_record)

    p_status = sub.add_parser("status", help="Parse a training log and print live status/ETA")
    p_status.add_argument("--log-path", required=True)
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=status_record)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
