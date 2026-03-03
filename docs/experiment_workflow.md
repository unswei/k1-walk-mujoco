# Experiment Workflow

This is the standard workflow for remote PPO experiments.

## 1. Capacity probe

- Verify remote machine setup (`python`, CUDA visibility, MuJoCo headless, `nvidia-smi`).
- Run short throughput probe to pick `num_envs` and process layout.

## 2. Training campaign

- Launch named campaign with deterministic run naming.
- Keep one run per seed and fixed GPU mapping for reproducibility.
- Prefer warm-start/resume explicitly (`--init-ckpt` or `--resume-ckpt`).
- For launch safety, require checkpoint existence before starting workers.
- Record launch manifest with seed/GPU/checkpoint mapping.

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

## M2 continuation recipe (candidate-B)

Two-stage continuation from the strongest M2 checkpoint family:

1. Stage 1 (easier command distribution, mild randomization):
   - `configs/train_ppo_m2_candidate_b_stage1.yaml`
   - launch with:
     - `scripts/launch_m2_candidate_b.sh --stage stage1 --load-mode init --eval-every-updates 20 --dry-run`
     - `scripts/launch_m2_candidate_b.sh --stage stage1 --load-mode init --eval-every-updates 20`
2. Stage 2 (wider command distribution toward full M2):
   - `configs/train_ppo_m2_candidate_b_stage2.yaml`
   - launch with:
     - `scripts/launch_m2_candidate_b.sh --stage stage2 --stage1-prefix <stage1_run_prefix> --load-mode init`

Notes:
- Stage 1 defaults to per-seed init from `update_000300.pt` of candidate-A.
- Stage 2 defaults to per-seed init from stage1 `latest.pt`.
- Both stages keep fresh optimizer state via `--init-ckpt` (no resume state).
- The launcher writes `runs/cleanrl_ppo/campaign_logs/<run_prefix>_launch_manifest.tsv`.

## GPU utilization tuning checklist

- Start from `--jobs-per-gpu 2 --num-envs 32` on a remote Linux GPU box.
- Tune `--eval-every-updates` upward (for example `20 -> 40`) to reduce eval overhead when training is stable.
- Keep CPU thread caps explicit via `--cpu-threads` to avoid oversubscription.
- Use `--stagger-seconds 5` if startup contention causes uneven GPU utilization.

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
