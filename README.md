# k1-walk-mujoco

Train a Booster K1 walking policy in MuJoCo from scratch using a stable core environment and thin RL plug-in adapters.

This repository uses PD position targets (policy outputs joint targets/deltas, PD converts to torques), with MuJoCo physics and Gymnasium environment interfaces.

## Quickstart

1. Create a virtual environment (uv preferred):

```bash
uv venv
source .venv/bin/activate
```

2. Install package and dev tools:

```bash
uv pip install -e '.[dev]'
```

3. Fetch pinned Booster assets:

```bash
python scripts/fetch_assets.py
```

4. Run a smoke test:

```bash
python scripts/smoke_mujoco.py
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
