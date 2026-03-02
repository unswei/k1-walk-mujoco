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
