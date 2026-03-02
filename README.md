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

Run headless training with TensorBoard logs:

```bash
python scripts/train_cleanrl_ppo.py --config configs/train_ppo_cleanrl.yaml
```

Run a short smoke training job:

```bash
python scripts/train_cleanrl_ppo.py --config configs/train_ppo_cleanrl.yaml --total-timesteps 512 --num-envs 1 --device cpu --run-name smoke
```

Rollout random policy:

```bash
python scripts/rollout.py --episodes 1
```

Rollout checkpoint with rendering:

```bash
mjpython scripts/rollout.py --ckpt runs/cleanrl_ppo/<run>/checkpoints/best.pt --render --episodes 1 --deterministic
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
python scripts/rollout.py --ckpt runs/cleanrl_ppo/<run>/checkpoints/best.pt --record runs/cleanrl_ppo/<run>/rollout.mp4 --episodes 1 --deterministic
```

Device selection for training:

- `auto` prefers `cuda`, then `mps`, then `cpu`.
- Recommended defaults: macOS uses smaller vectorization (`4-8` envs), Linux CUDA uses larger (`16+` envs).

### Multi-GPU recipe (remote Linux box)

Run two independent seeds, one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cleanrl_ppo.py --config configs/train_ppo_cleanrl.yaml --seed 1 --num-envs 16 --run-name k1_ppo_s1_gpu0
CUDA_VISIBLE_DEVICES=1 python scripts/train_cleanrl_ppo.py --config configs/train_ppo_cleanrl.yaml --seed 2 --num-envs 16 --run-name k1_ppo_s2_gpu1
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
