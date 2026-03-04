# k1-walk-mujoco

Train and evaluate humanoid locomotion stacks centred on MuJoCo, with both in-repo CleanRL work and external Holosoma MJWarp runs.

## What this repo covers

| Track | Purpose | Main entry points |
| --- | --- | --- |
| Core K1 MuJoCo env | Stable simulation/task layer for Booster K1 walking | `scripts/fetch_assets.py`, `scripts/smoke_mujoco.py` |
| CleanRL milestone pipeline | In-repo PPO training, gating, experiment operations | `scripts/train_cleanrl_ppo.py`, `scripts/run_milestones.py` |
| Holosoma MJWarp experiments | External Holosoma locomotion training workflow on Linux GPUs | `scripts/setup_holosoma_mjwarp_uv.sh`, `scripts/run_holosoma_mjwarp_training.sh` |

This repository uses PD position targets (policy outputs joint targets/deltas, PD converts to torques), with MuJoCo physics and Gymnasium interfaces.

## Quick navigation

- [Core Quickstart](#core-quickstart)
- [CleanRL PPO](#cleanrl-ppo)
- [Experiment Ops (Canonical)](#experiment-ops-canonical)
- [Holosoma MJWarp (experimental)](#holosoma-mjwarp-experimental)
- [Design choices](#design-choices)

## Core Quickstart

1. Create a virtual environment (uv preferred):

```bash
uv venv
source .venv/bin/activate
```

2. Install package and dev tools:

```bash
uv pip install -e '.[dev]'
```

Install MuJoCo Warp support (optional):

```bash
uv pip install -e '.[dev,warp]'
```

3. Fetch pinned Booster assets:

```bash
python scripts/fetch_assets.py
```

4. Run a smoke test:

```bash
python scripts/smoke_mujoco.py
```

Run the same smoke script against MuJoCo Warp:

```bash
python scripts/smoke_mujoco.py --backend warp
```

## CleanRL PPO

Install CleanRL dependencies:

```bash
uv pip install -e '.[cleanrl,dev]'
```

Enable W&B optionally:

```bash
uv pip install -e '.[cleanrl,dev,wandb]'
```

### Milestone configs

Milestone training configs are provided in:

- `configs/train_ppo_m0.yaml`
- `configs/train_ppo_m1.yaml`
- `configs/train_ppo_m2.yaml`
- `configs/train_ppo_m3.yaml`
- `configs/train_ppo_m4.yaml`
- `configs/train_ppo_m5.yaml`

Default env/task schema lives in `configs/env_k1_walk.yaml`.

### Training commands

Run headless training for a milestone:

```bash
python scripts/train_cleanrl_ppo.py --milestone m3
```

Run a short smoke job:

```bash
python scripts/train_cleanrl_ppo.py --milestone m0 --total-timesteps 512 --num-envs 1 --device cpu --run-name smoke
```

Train from an explicit config:

```bash
python scripts/train_cleanrl_ppo.py --config configs/train_ppo_cleanrl.yaml
```

Warm-start from a previous checkpoint:

```bash
python scripts/train_cleanrl_ppo.py --milestone m4 --init-ckpt runs/cleanrl_ppo/<m3_run>/checkpoints/best_nominal.pt
```

Evaluate a checkpoint on fixed eval suites:

```bash
python scripts/train_cleanrl_ppo.py --milestone m3 --eval-only --ckpt runs/cleanrl_ppo/<run>/checkpoints/latest.pt --eval-suite easy
```

Run enforced 3-seed milestone gating:

```bash
python scripts/run_milestones.py --milestone m3 --seeds 1,2,3
```

Auto-progress milestones when gates pass:

```bash
python scripts/run_milestones.py --milestone m0 --auto-progress --until-milestone m5 --seeds 1,2,3
```

`run_milestones.py` now applies transition anti-forgetting by default on milestone handoff:
- first 20% of the new milestone runs with mixed previous-task episodes (`--transition-mix-fraction 0.2`)
- previous-task sampling probability defaults to `0.35` (`--transition-mix-prob 0.35`)

Disable this behavior if needed:

```bash
python scripts/run_milestones.py --milestone m2 --auto-progress --disable-transition-mix
```

Hidden holdout evaluation is also enabled by default and runs against:
- suite file: `configs/eval_suites_goal_pose_holdout.yaml`
- suite name: `holdout`

Disable holdout evaluation:

```bash
python scripts/run_milestones.py --milestone m3 --disable-holdout-eval
```

Aggregate 3-seed milestone results and evaluate promotion gates:

```bash
python scripts/milestone_report.py --milestone m3 --runs runs/cleanrl_ppo/<run_seed1> runs/cleanrl_ppo/<run_seed2> runs/cleanrl_ppo/<run_seed3>
```

Rollout random policy:

```bash
python scripts/rollout.py --episodes 1
```

Rollout parameterised gait baseline (no learned policy):

```bash
python scripts/rollout.py --controller param_gait_15 --gait-config configs/gait_param_15.yaml --episodes 1
```

Rollout in goal-pose mode with explicit target:

```bash
python scripts/rollout.py --episodes 1 --task-mode goal_pose --goal-x 1.0 --goal-y 0.2 --goal-yaw-deg 30 --policy zero
```

Rollout checkpoint with rendering:

```bash
mjpython scripts/rollout.py --ckpt runs/cleanrl_ppo/<run>/checkpoints/best_nominal.pt --render --episodes 1 --deterministic --task-mode goal_pose
```

On macOS, MuJoCo viewer rendering requires `mjpython` (not plain `python`).

If `mjpython` fails with `Library not loaded: @rpath/libpythonX.Y.dylib`, create a venv-local symlink to the base Python shared library:

```bash
PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VENV_DIR="$(python -c 'import pathlib, sys; print(pathlib.Path(sys.executable).resolve().parent.parent)')"
BASE_LIBPY="$(python -c 'import pathlib, sys; print(pathlib.Path(sys.base_prefix) / "lib" / f"libpython{sys.version_info.major}.{sys.version_info.minor}.dylib")')"
ln -sfn "$BASE_LIBPY" "$VENV_DIR/libpython${PY_VER}.dylib"
```

Then verify:

```bash
mjpython -c "import mujoco; print('mjpython ok')"
```

Headless video recording:

```bash
python scripts/rollout.py --ckpt runs/cleanrl_ppo/<run>/checkpoints/best_nominal.pt --record runs/cleanrl_ppo/<run>/rollout.mp4 --episodes 1 --deterministic --task-mode goal_pose
```

Headless video with body-tracking camera:

```bash
python scripts/rollout.py --ckpt runs/cleanrl_ppo/<run>/checkpoints/best_nominal.pt --record runs/cleanrl_ppo/<run>/rollout_track.mp4 --record-camera track --track-body base --episodes 1 --deterministic
```

Optimise the 15-parameter gait with parallel random-search successive halving:

```bash
python scripts/optimise_gait_params.py --config configs/optimise_gait.yaml
```

Artifacts are written under `runs/gait_optim/<run_name>/` including:
- `best_params.yaml`
- `summary.json`
- `candidates.jsonl`
- `tb/` TensorBoard logs

Device selection for training:

- `auto` prefers `cuda`, then `mps`, then `cpu`.
- Recommended defaults: macOS uses smaller vectorization (`4-8` envs), Linux CUDA uses larger (`16+` envs).

### Eval outputs and checkpoints

Each run writes:

- TensorBoard logs: `runs/cleanrl_ppo/<run>/tb`
- Eval JSONL: `runs/cleanrl_ppo/<run>/eval/metrics.jsonl`
- Checkpoints:
  - `latest.pt`
  - `best.pt` (best nominal)
  - `best_nominal.pt`
  - `best_stress.pt`

Run TensorBoard:

```bash
tensorboard --logdir runs/cleanrl_ppo --port 6006
```

### Experiment Ops (Canonical)

Use `experiments/` as the canonical experiment record:

- Structured logs: `experiments/logs/` (commit to Git)
- Generated summary: `experiments/experiments.md` (commit to Git)
- Raw pulled artifacts: `experiments/artifacts/` (do not commit)
- Review videos: `experiments/videos/` (do not commit)

Generate summary from structured logs:

```bash
python scripts/generate_experiment_summary.py
```

Sync a remote run prefix into local artifacts:

```bash
scripts/sync_remote_experiment.sh --remote deploy@<remotehost> --prefix <run_prefix>
```

Staged M2 continuation launch (candidate-B):

```bash
scripts/launch_m2_candidate_b.sh --stage stage1
scripts/launch_m2_candidate_b.sh --stage stage2 --stage1-prefix <stage1_run_prefix>
```

Warm-start provenance and GPU-utilization knobs:

```bash
scripts/launch_m2_candidate_b.sh --stage stage1 --dry-run
scripts/launch_m2_candidate_b.sh --stage stage1 --load-mode init --eval-every-updates 20 --num-envs 32 --jobs-per-gpu 2
```

The launcher writes `runs/cleanrl_ppo/campaign_logs/<run_prefix>_launch_manifest.tsv` with
seed/GPU/checkpoint mapping and checkpoint existence checks.

Detailed conventions live in `experiments/README.md`.
Full process steps live in `docs/experiment_workflow.md`.

### Multi-GPU recipe (remote Linux box)

Run two independent seeds, one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cleanrl_ppo.py --milestone m3 --seed 1 --num-envs 16 --run-name k1_m3_s1_gpu0
CUDA_VISIBLE_DEVICES=1 python scripts/train_cleanrl_ppo.py --milestone m3 --seed 2 --num-envs 16 --run-name k1_m3_s2_gpu1
```

Suggested remote workflow:

```bash
ssh <user>@<remotehost>
tmux new -s k1_ppo_gpu0
# run GPU0 command, then detach
tmux new -s k1_ppo_gpu1
# run GPU1 command, then detach
```

### Holosoma MJWarp (experimental)

For paper-style Holosoma locomotion training with `simulator:mjwarp` on 2x 2080 Ti GPUs:

```bash
scripts/setup_holosoma_mjwarp_uv.sh
scripts/smoke_holosoma_mjwarp.sh
scripts/run_holosoma_mjwarp_training.sh --mode torchrun --total-envs 512
```

If you are on an older NVIDIA driver and hit Warp graph-capture errors, re-run setup with `--apply-patch`.

Live status and ETA from a running log:

```bash
python scripts/record_holosoma_run.py status --log-path ~/work/warp-holosoma/holosoma/logs/<run>.log
```

Generate local MP4 videos from a trained checkpoint (CPU-safe path, does not consume training GPUs):

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl \
python ~/work/warp-holosoma/holosoma/src/holosoma/holosoma/eval_agent.py \
  --checkpoint ~/work/warp-holosoma/holosoma/logs/warp-holosoma/<run>/model_0004000.pt \
  simulator:mujoco \
  --training.headless True \
  --training.num-envs 1 \
  --training.max-eval-steps 120 \
  --simulator.config.sim.max-episode-length-s 0.5 \
  --randomization.ignore-unsupported True \
  --logger.video.enabled True \
  --logger.headless-recording True \
  --logger.video.interval 1 \
  --logger.video.output-format mp4 \
  --logger.video.upload-to-wandb False \
  --logger.video.save-dir ~/work/warp-holosoma/holosoma/logs/video_smoke
```

Note: with `simulator:mujoco`, use `--randomization.ignore-unsupported True` for MJWarp-trained configs.

The launcher now writes structured run metadata to:
- `experiments/logs/<date>/feature/holosoma-mjwarp/*.json`

If you use `--detach`, finalise the record after the process exits:

```bash
python scripts/record_holosoma_run.py finish --output <record.json> --log-path <run.log> --exit-code 0
```

Detailed workflow:

- `docs/holosoma_mjwarp_2080ti.md`

## Design choices

- PD position targets instead of direct torque actions for early stability and easier RL-library portability.
- Legs-only control (12 DoF) in v0; arms/head are held at nominal pose.
- RL plug-in architecture under `src/k1_walk_mujoco/rl/` keeps core simulation and environment independent from specific training frameworks.

## Milestone checks

A clean clone should run:

```bash
python scripts/fetch_assets.py
python scripts/smoke_mujoco.py
```
