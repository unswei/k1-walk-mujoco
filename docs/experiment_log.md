# Experiment Log (Plain English)

This file is the running experiment diary for CleanRL PPO training in this repo.
It records:
- what we ran,
- where the run started from,
- how long it ran,
- and what result we got.

## Environment

- Remote host: `magrathea-1` (`deploy`)
- Repo path on host: `/home/deploy/work/k1-walk-mujoco`
- Branch: `feature/cleanrl-ppo-training-pipeline`
- GPUs: 2x RTX 2080 Ti (11 GB each)

## 2026-03-02: Throughput Probes

Goal: estimate realistic wall-clock speed before long campaigns.

### Probe A

- Run name: `probe_m1_gpu0`
- Command summary: `train_cleanrl_ppo.py --milestone m1 --total-timesteps 200000 --num-envs 16 --device cuda --seed 1`
- Started from: scratch (no init checkpoint)
- Runtime: `2m 57.92s` (captured with `/usr/bin/time`)
- Result (last eval):
  - suite: `m1_nominal`
  - `fall_rate=1.0`
  - `median_command_vx_rmse_mps=0.3708`
  - all terminations were `base_height`

### Probe B

- Run name: `probe_m1_gpu1_n32`
- Command summary: `train_cleanrl_ppo.py --milestone m1 --total-timesteps 200000 --num-envs 32 --device cuda --seed 2`
- Started from: scratch (no init checkpoint)
- Runtime: `2m 04.86s` (captured with `/usr/bin/time`)
- Result (last eval):
  - suite: `m1_nominal`
  - `fall_rate=1.0`
  - `median_command_vx_rmse_mps=0.3566`
  - all terminations were `base_height`

## 2026-03-02: Calibration Sweep Attempt (Stopped)

Goal: calibrate milestone gates from real runs.

Configuration used:
- `run_milestones.py --milestone m1 --auto-progress --until-milestone m5`
- seeds launched: 1, 2, 3
- `--total-timesteps 1000000` per milestone
- `--num-envs 32` (seed 3 later run with 16 during overlap)
- gates opened temporarily for data collection (`checks: []`)

### What happened

- `m1` finished for seeds 1 and 2, seed 3 partial.
- `m2` started for seeds 1 and 2 (partial).
- Run quality was not usable for gate calibration:
  - `m1`: all observed seeds had `fall_rate=1.0`, `success_rate=0.0`
  - `m2` partial: same collapse (`fall_rate=1.0`, `success_rate=0.0`)
- Sweep was intentionally stopped early to avoid calibrating gates to bad behavior.

### Key run snapshots

- `calib_t1m_m1_s1`: `fall_rate=1.0`, `vx_rmse=0.4317`
- `calib_t1m_m1_s2`: `fall_rate=1.0`, `vx_rmse=0.3504`
- `calib_t1m_m1_s3` (partial): `fall_rate=1.0`, `vx_rmse=0.4255`
- `calib_t1m_m2_s1` (partial): `fall_rate=1.0`, `command_tracking_success_rate=0.0`
- `calib_t1m_m2_s2` (partial): `fall_rate=1.0`, `command_tracking_success_rate=0.0`

## 2026-03-02: M1 Curriculum Easing (Config Change)

Reason: M1 looked too hard / misaligned in early setup.

Changes made before next campaign:
- Added stage-aligned eval suite `m1_nominal_stage1` (`vx=0.25/0.35/0.45`) in `configs/eval_suites_goal_pose.yaml`
- Updated `configs/train_ppo_m1.yaml`:
  - eval suite switched to `m1_nominal_stage1`
  - `v_x_target: 0.35` (was 0.5)
  - lower reset noise (`qpos 0.005`, `qvel 0.02`)
  - modest stability reward tweaks (`w_upright`, `w_alive`, torque penalty)
- Commit on branch: `071d661`

## 2026-03-02: Main 4-Seed M1 Campaign (Completed)

Goal: run proper M1 training with easier curriculum and collect reliable seed spread.

Execution model:
- GPU0 queue: seed 1, then seed 3
- GPU1 queue: seed 2, then seed 4
- Command summary per seed:
  - `train_cleanrl_ppo.py --milestone m1 --total-timesteps 10000000 --num-envs 32 --device cuda`
- Started from: scratch (no init checkpoint for these 4 runs)

Global wall-clock:
- Start: `2026-03-02T11:38:26Z`
- End: `2026-03-02T14:37:35Z`
- Total: `2h 59m 09s`

Per-seed runtime and result:

1. `m1_easier_10m_s1_g0`
- runtime: `1h 29m 28s`
- final suite: `m1_nominal_stage1`
- result: `fall_rate=0.0`, `vx_rmse=0.1066`
- termination reasons: all `time_limit`

2. `m1_easier_10m_s2_g1`
- runtime: `1h 29m 29s`
- final suite: `m1_nominal_stage1`
- result: `fall_rate=0.0`, `vx_rmse=0.1060`
- termination reasons: all `time_limit`

3. `m1_easier_10m_s3_g0`
- runtime: `1h 29m 41s`
- final suite: `m1_nominal_stage1`
- result: `fall_rate=0.3333`, `vx_rmse=0.1842`
- termination reasons: 2x `time_limit`, 1x `base_height`

4. `m1_easier_10m_s4_g1`
- runtime: `1h 29m 35s`
- final suite: `m1_nominal_stage1`
- result: `fall_rate=0.3333`, `vx_rmse=0.1502`
- termination reasons: 2x `time_limit`, 1x `base_height`

4-seed aggregate after this campaign:
- mean `fall_rate=0.1667`
- worst-seed `fall_rate=0.3333`
- median seed-level `vx_rmse=0.1284`
- worst-seed `vx_rmse=0.1842`

Notes:
- `success_rate` stayed `0.0` for M1 velocity mode (expected because this is not goal-pose success).
- `command_tracking_success_rate` stayed `0.0` because yaw-rate RMSE remained above threshold even when walking looked stable.

## 2026-03-03: Continuation Campaign (+2M for Seeds 3 and 4, Completed)

Goal: recover the remaining instability on weaker seeds without restarting.

Runs:
- `m1_easier_10m_s3_g0_plus2m` on GPU0
- `m1_easier_10m_s4_g1_plus2m` on GPU1

Start point (explicit warm-start):
- seed 3 init checkpoint: `runs/cleanrl_ppo/m1_easier_10m_s3_g0/checkpoints/best_nominal.pt`
- seed 4 init checkpoint: `runs/cleanrl_ppo/m1_easier_10m_s4_g1/checkpoints/best_nominal.pt`

Command summary:
- `train_cleanrl_ppo.py --milestone m1 --total-timesteps 2000000 --num-envs 32 --device cuda --init-ckpt ...`

Timing:
- both started at host local time `Tue Mar 3 08:22:30 AEDT 2026`
- seed 3 finished at approx `08:40:53 AEDT` (`~18m 23s`)
- seed 4 finished at approx `08:41:03 AEDT` (`~18m 33s`)

Final result (last eval):

1. `m1_easier_10m_s3_g0_plus2m`
- final suite: `m1_nominal_stage1`
- result: `fall_rate=0.6667`, `vx_rmse=0.3104`
- termination reasons: 2x `base_height`, 1x `time_limit`

2. `m1_easier_10m_s4_g1_plus2m`
- final suite: `m1_nominal_stage1`
- result: `fall_rate=0.6667`, `vx_rmse=0.3158`
- termination reasons: 2x `base_height`, 1x `time_limit`

Outcome:
- this continuation regressed both seeds versus their original 10M checkpoints.
- keep `m1_easier_10m_s3_g0` and `m1_easier_10m_s4_g1` as the better artifacts; do not promote the `+2M` checkpoints.

Before/after comparison for 4-seed aggregate:
- original set (`s1,s2,s3,s4` from 10M): `fall_mean=0.1667`, `fall_worst=0.3333`, `vx_median=0.1284`
- with continued seeds (`s1,s2,s3_plus2m,s4_plus2m`): `fall_mean=0.3333`, `fall_worst=0.6667`, `vx_median=0.2085`

## 2026-03-03: True-Resume + Lower-LR 10M Continuation (In Progress)

Goal: continue M1 from real optimizer/training state (not weights-only warm start) and reduce LR for stability.

Why this run exists:
- prior `+2M` continuation regressed and was launched with weights-only init (`--init-ckpt`).
- pipeline now supports true resume (`--resume-ckpt --resume-training-state`).

Execution model:
- GPU0 queue: seed 1, then seed 3
- GPU1 queue: seed 2, then seed 4
- each seed runs `+10M` additional timesteps at lower LR (`1e-4`)
- command summary per seed:
  - `train_cleanrl_ppo.py --milestone m1 --total-timesteps 10000000 --num-envs 32 --device cuda --learning-rate 1e-4 --resume-ckpt ... --resume-training-state`

Start points (true resume checkpoints):
- seed 1: `runs/cleanrl_ppo/m1_easier_10m_s1_g0/checkpoints/latest.pt`
- seed 2: `runs/cleanrl_ppo/m1_easier_10m_s2_g1/checkpoints/latest.pt`
- seed 3: `runs/cleanrl_ppo/m1_easier_10m_s3_g0/checkpoints/latest.pt`
- seed 4: `runs/cleanrl_ppo/m1_easier_10m_s4_g1/checkpoints/latest.pt`

Run names:
- `m1_resume10m_lr1e4_s1_g0`
- `m1_resume10m_lr1e4_s2_g1`
- `m1_resume10m_lr1e4_s3_g0`
- `m1_resume10m_lr1e4_s4_g1`

Timing:
- launch timestamp: `2026-03-02T22:58:47Z` (host local `Tue Mar 3 09:58:47 AEDT 2026`)
- status at log entry: running

Early sanity signal (first progress line):
- seed 1: `update=612 (+1/611), step=10027008`
- seed 2: `update=612 (+1/611), step=10027008`
- this confirms true resume behavior (update/global step continue from prior 10M run instead of restarting at 0).

## Logging Conventions Going Forward

For every future campaign entry, always capture:
- run names
- exact command used
- start point (`scratch` or exact `init_checkpoint` path)
- start/end time in UTC and host local time
- wall-clock duration
- final eval metrics (suite name, `fall_rate`, `vx_rmse`, key termination reasons)
- short interpretation and next action
