# Experiments

Experiment runs are logged as one JSON file per run under `experiments/logs/YYYY-MM-DD/<branch>/`.
The summary table in `docs/experiment_log.md` is generated from these files.

Legacy narrative log: `docs/experiment_log_legacy.md` (kept for historical context).

## Adding a new run

1. Create a JSON file in the log directory for the run date and branch.
2. Use a filename like `YYYYMMDDThhmmssZ_<shortsha>_<run_name>.json`.
3. Re-generate the summary: `python scripts/generate_experiment_summary.py`.

## Minimal schema

```
{
  "schema_version": 1,
  "record_type": "run",
  "run_name": "example_run",
  "date": "2026-03-03",
  "branch": "feature/cleanrl-ppo-training-pipeline",
  "milestone": "m1",
  "seed": 1,
  "settings": {
    "total_timesteps": 1000000,
    "num_envs": 32,
    "device": "cuda"
  },
  "command": "train_cleanrl_ppo.py --milestone m1 ...",
  "init": "scratch",
  "status": "completed",
  "timing": {
    "start_utc": "2026-03-03T00:00:00Z",
    "duration_s": 3600
  },
  "metrics": {
    "suite": "m1_nominal",
    "fall_rate": 0.0,
    "vx_rmse": 0.12
  }
}
```
