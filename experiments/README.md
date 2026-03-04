# Experiments Workspace

This directory is the canonical home for experiment records in this repo.

## What goes where

- `experiments/logs/`
  - Structured JSON/JSONL run records and notes.
  - Intended to be committed to Git (small, reviewable metadata).
- `experiments/experiments.md`
  - Generated summary from `experiments/logs/`.
  - Rebuild with:
    - `python scripts/generate_experiment_summary.py`
- `experiments/artifacts/`
  - Pulled raw artifacts (remote campaign logs, eval dumps, copied checkpoint manifests).
  - Not committed to Git.
- `experiments/videos/`
  - Local review videos for runs/checkpoints.
  - Not committed to Git.

## Policy

- Checkpoints and TensorBoard event files stay in `runs/` (already gitignored).
- Prefer committing only:
  - experiment metadata JSON records,
  - generated experiment summary markdown,
  - scripts/docs changes.
- Do not commit large binaries (`.pt`, `.mp4`, event files, raw dumps).

## Remote workflow

1. Run experiments on remote host under `runs/cleanrl_ppo/...`.
   - For Holosoma MJWarp launches, the launcher writes JSON records under:
     - `experiments/logs/<date>/feature/holosoma-mjwarp/`
2. Pull selected artifacts with:
   - `scripts/sync_remote_experiment.sh`
3. Write or update structured JSON records in `experiments/logs/...`.
4. Regenerate summary:
   - `python scripts/generate_experiment_summary.py`
