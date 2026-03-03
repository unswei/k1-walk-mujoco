#!/usr/bin/env python3
"""Generate a markdown summary from old-experiments/experiments/logs JSON files."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_entries(logs_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(logs_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            data["_path"] = str(path)
            entries.append(data)
    return entries


def parse_iso(dt_str: str) -> Optional[dt.datetime]:
    try:
        if dt_str.endswith("Z"):
            return dt.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.datetime.fromisoformat(dt_str)
    except Exception:
        return None


def entry_date(entry: Dict[str, Any]) -> str:
    if "date" in entry and entry["date"]:
        return str(entry["date"])
    timing = entry.get("timing", {})
    for key in ("start_utc", "start_time_utc"):
        if key in timing:
            parsed = parse_iso(str(timing[key]))
            if parsed:
                return parsed.date().isoformat()
    for key in ("start_utc", "start_time_utc"):
        if key in entry:
            parsed = parse_iso(str(entry[key]))
            if parsed:
                return parsed.date().isoformat()
    return "unknown"


def record_type(entry: Dict[str, Any]) -> str:
    return str(entry.get("record_type", "run"))


def fmt_float(val: float) -> str:
    s = f"{val:.4f}"
    return s.rstrip("0").rstrip(".")


def fmt_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return fmt_float(val)
    return str(val)


def fmt_duration_s(seconds: Any) -> str:
    try:
        total = int(round(float(seconds)))
    except Exception:
        return ""
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def fmt_metrics(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return ""
    ordered_keys = [
        "suite",
        "fall_rate",
        "vx_rmse",
        "median_command_vx_rmse_mps",
        "command_tracking_success_rate",
        "success_rate",
    ]
    parts: List[str] = []
    seen = set()
    for key in ordered_keys:
        if key in metrics:
            seen.add(key)
            if key == "suite":
                parts.append(fmt_value(metrics[key]))
            else:
                parts.append(f"{key}={fmt_value(metrics[key])}")
    for key in sorted(k for k in metrics.keys() if k not in seen):
        parts.append(f"{key}={fmt_value(metrics[key])}")
    return ", ".join(parts)


def fmt_command(cmd: Optional[str]) -> str:
    if not cmd:
        return ""
    return f"`{cmd}`"


def get_timesteps(entry: Dict[str, Any]) -> str:
    settings = entry.get("settings", {})
    val = settings.get("total_timesteps", entry.get("total_timesteps"))
    if val is None:
        return ""
    try:
        return f"{int(val):,}"
    except Exception:
        return str(val)


def get_note_text(entry: Dict[str, Any]) -> str:
    if "notes" in entry and entry["notes"]:
        return str(entry["notes"])
    return ""


def get_campaign_notes(entry: Dict[str, Any]) -> str:
    camp = entry.get("campaign", {})
    return str(camp.get("notes", "")) if camp else ""


def group_by_date(entries: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    bucket: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        d = entry_date(e)
        bucket.setdefault(d, []).append(e)
    def date_key(d: str) -> Tuple[int, str]:
        if d == "unknown":
            return (1, d)
        return (0, d)
    return [(d, bucket[d]) for d in sorted(bucket.keys(), key=date_key)]


def render_summary_table(runs: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("| Date | Run | Status | Milestone | Seed | Timesteps | Key metrics | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for e in runs:
        d = entry_date(e)
        run = fmt_value(e.get("run_name", ""))
        status = fmt_value(e.get("status", ""))
        milestone = fmt_value(e.get("milestone", ""))
        seed = fmt_value(e.get("seed", ""))
        timesteps = get_timesteps(e)
        metrics = fmt_metrics(e.get("metrics", {}))
        notes = get_note_text(e)
        line = f"| {d} | {run} | {status} | {milestone} | {seed} | {timesteps} | {metrics} | {notes} |"
        lines.append(line)
    return "\n".join(lines)


def render_date_section(date: str, items: List[Dict[str, Any]]) -> str:
    notes = [e for e in items if record_type(e) == "note"]
    runs = [e for e in items if record_type(e) != "note"]
    lines: List[str] = []
    lines.append(f"## {date}")
    if notes:
        lines.append("Notes:")
        for n in notes:
            title = str(n.get("title", ""))
            body = n.get("body", "")
            if isinstance(body, list):
                body_text = "; ".join(str(x) for x in body)
            else:
                body_text = str(body)
            label = f"{title}: " if title else ""
            lines.append(f"- {label}{body_text}")
    if runs:
        lines.append("Runs:")
        lines.append("| Run | Status | Start (UTC) | Duration | Command | Init/Resume | Metrics | Notes |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for e in runs:
            run = fmt_value(e.get("run_name", ""))
            status = fmt_value(e.get("status", ""))
            timing = e.get("timing", {})
            start = fmt_value(timing.get("start_utc", ""))
            duration = fmt_duration_s(timing.get("duration_s", ""))
            cmd = fmt_command(e.get("command"))
            init = ""
            if "init" in e and e["init"]:
                init = f"init={e['init']}"
            resume = e.get("resume", {})
            if resume:
                resume_ckpt = resume.get("ckpt")
                resume_state = resume.get("resume_training_state")
                if resume_ckpt:
                    extra = f"resume={resume_ckpt}"
                    if resume_state is not None:
                        extra = f"{extra}, state={fmt_value(resume_state)}"
                    init = extra if not init else f"{init}; {extra}"
            metrics = fmt_metrics(e.get("metrics", {}))
            notes = get_note_text(e)
            camp_notes = get_campaign_notes(e)
            if camp_notes and camp_notes not in notes:
                notes = f"{notes} {camp_notes}".strip()
            line = f"| {run} | {status} | {start} | {duration} | {cmd} | {init} | {metrics} | {notes} |"
            lines.append(line)
    return "\n".join(lines)


def generate_markdown(entries: List[Dict[str, Any]]) -> str:
    runs = [e for e in entries if record_type(e) != "note"]
    grouped = group_by_date(entries)

    lines: List[str] = []
    lines.append("# Experiment Log (Generated)")
    lines.append("")
    lines.append(
        "This file is generated by scripts/generate_experiment_summary.py "
        "from old-experiments/experiments/logs/*.json."
    )
    lines.append("Do not edit by hand. Add or update JSON logs and re-run the script.")
    lines.append("")
    lines.append("## Summary")
    lines.append(render_summary_table(runs))
    lines.append("")
    for date, items in grouped:
        lines.append(render_date_section(date, items))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        default="old-experiments/experiments/logs",
        help="Directory containing JSON logs",
    )
    parser.add_argument(
        "--output",
        default="old-experiments/experiments.md",
        help="Output markdown file",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output = Path(args.output)

    entries = load_entries(logs_dir)
    content = generate_markdown(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
