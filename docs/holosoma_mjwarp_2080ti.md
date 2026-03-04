# Holosoma MJWarp Training (2x RTX 2080 Ti)

This runbook sets up and launches Holosoma locomotion training (`exp:g1-29dof-fast-sac`) with `simulator:mjwarp` on Linux hosts with 2x 11 GB GPUs.

Validated on `magrathea-1` on 4 March 2026 with:
- NVIDIA driver `570.211.01`
- CUDA reported by `nvidia-smi`: `12.8`
- No local Holosoma source patch required.

## 1) Setup (uv-first, no conda)

Run:

```bash
scripts/setup_holosoma_mjwarp_uv.sh
```

Default target paths:
- workspace: `~/work/warp-holosoma`
- Holosoma repo: `~/work/warp-holosoma/holosoma`
- mujoco_warp clone: `~/.holosoma_deps/mujoco_warp` (checked out to commit `09ec1da`)

The setup script now defaults to **no patch**. If you must support an older driver stack (for example `535.x`) and hit CUDA graph-capture errors, re-run setup with:

```bash
scripts/setup_holosoma_mjwarp_uv.sh --apply-patch
```

Patch file:
- [`scripts/patches/holosoma_warp_backend_capture_fallback.patch`](/Users/z3550628/Code/2026/k1-walk-mujoco/scripts/patches/holosoma_warp_backend_capture_fallback.patch)

## 2) Smoke test

Run both a single-GPU and a 2-GPU torchrun smoke:

```bash
scripts/smoke_holosoma_mjwarp.sh
```

Defaults:
- single GPU: `128` envs, `3` learning iterations
- torchrun (2 GPUs): `512` total envs, `3` learning iterations

Override values if needed:

```bash
scripts/smoke_holosoma_mjwarp.sh \
  --iterations 5 \
  --single-envs 256 \
  --total-envs 512
```

## 3) Launch training

Unified launcher:
- [`scripts/run_holosoma_mjwarp_training.sh`](/Users/z3550628/Code/2026/k1-walk-mujoco/scripts/run_holosoma_mjwarp_training.sh)

Examples:

```bash
# Preferred: one distributed run across 2 GPUs
scripts/run_holosoma_mjwarp_training.sh \
  --mode torchrun \
  --total-envs 512 \
  --run-name g1_unitree_mjwarp_2gpu

# Single-GPU baseline
scripts/run_holosoma_mjwarp_training.sh \
  --mode single \
  --single-envs 256 \
  --run-name g1_unitree_mjwarp_1gpu

# Fallback: two independent jobs, one per GPU
scripts/run_holosoma_mjwarp_training.sh \
  --mode dual-single \
  --single-envs 256 \
  --run-name g1_unitree_mjwarp_dual
```

Detached launch (returns immediately):

```bash
scripts/run_holosoma_mjwarp_training.sh \
  --mode torchrun \
  --total-envs 512 \
  --run-name g1_unitree_mjwarp_2gpu \
  --detach
```

## 4) Live ETA and progress

Use the status parser against the run log:

```bash
python scripts/record_holosoma_run.py status \
  --log-path ~/work/warp-holosoma/holosoma/logs/<run>.log
```

It reports:
- learning iteration progress
- elapsed time and ETA
- estimated finish time (UTC and local)
- latest mean reward / episode length (if present in log)

## 5) Structured experiment records

By default, the launcher writes run metadata to:
- `experiments/logs/<date>/feature/holosoma-mjwarp/*.json`

For foreground launches, records are finalised automatically with completion status and exit code.
For detached launches, records stay `running`; finalise after the process exits:

```bash
python scripts/record_holosoma_run.py finish \
  --output experiments/logs/<date>/feature/holosoma-mjwarp/<record>.json \
  --log-path ~/work/warp-holosoma/holosoma/logs/<run>.log \
  --exit-code 0
```

Regenerate the consolidated summary after adding records:

```bash
python scripts/generate_experiment_summary.py
```

## 6) Generate evaluation videos from checkpoints

To generate local MP4 videos without touching active training GPUs, run evaluation on CPU with `simulator:mujoco`:

```bash
cd ~/work/warp-holosoma/holosoma
source .venv/bin/activate

CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl \
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint logs/warp-holosoma/<run>/model_0004000.pt \
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
  --logger.video.save-dir logs/video_smoke
```

Expected result:
- `logs/video_smoke/episode_*.mp4` files are created.

Notes:
- `--simulator.config.sim.max-episode-length-s 0.5` forces short episodes so videos are flushed quickly.
- `--randomization.ignore-unsupported True` is required when evaluating MJWarp-trained configs with plain MuJoCo.
- You may still see MuJoCo EGL teardown warnings at process exit; these did not block MP4 generation in validation runs.

## 7) Recommended env-count ramp on 11 GB GPUs

Increase only after a stable run:
1. `512` total envs (`256`/GPU)
2. `768` total envs (`384`/GPU)
3. `1024` total envs (`512`/GPU)

If OOM or instability appears, lower `training.num-envs` first.
