# Experiment Workflow

This is the standard workflow for remote PPO experiments.

## 1. Capacity probe

- Verify remote machine setup (`python`, CUDA visibility, MuJoCo headless, `nvidia-smi`).
- Run short throughput probe to pick `num_envs` and process layout.

## 2. Training campaign

- Launch named campaign with deterministic run naming.
- Keep one run per seed and fixed GPU mapping for reproducibility.
- Prefer warm-start/resume explicitly (`--init-ckpt` or `--resume-ckpt`).

## 3. Pull and register artifacts

- Sync remote artifacts into `experiments/artifacts/<date>/<run_prefix>/`.
- Keep model checkpoints in `runs/` (local or remote), not in Git.
- Record structured experiment metadata in `experiments/logs/...`.

## 4. Summarize and gate

- Regenerate summary markdown:
  - `python scripts/generate_experiment_summary.py`
- Decide promotion/failure based on milestone gates.

## 5. Video review

- Generate rollout videos from selected checkpoints.
- Use camera tracking where useful:
  - `--record-camera track --track-body base`

## Git retention policy

Commit:
- code/config changes,
- structured metadata logs,
- generated summary markdown.

Do not commit:
- checkpoints (`.pt`),
- TensorBoard event files,
- raw remote dumps,
- videos (`.mp4`).
