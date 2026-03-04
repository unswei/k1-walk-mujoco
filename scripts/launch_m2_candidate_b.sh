#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

STAGE="stage1"
CONFIG=""
RUN_PREFIX=""
TOTAL_TIMESTEPS=6000000
NUM_ENVS=32
NUM_STEPS=""
EVAL_EVERY_UPDATES=""
PRINT_EVERY_UPDATES=20
RUN_DIR="runs/cleanrl_ppo"
LOG_DIR="$RUN_DIR/campaign_logs"
INIT_CKPT_TEMPLATE=""
STAGE1_PREFIX="m2_candidate_b_stage1_6m"
DRY_RUN=0
LOAD_MODE="init"
STRICT_CKPT_CHECK=1
CPU_THREADS=1
STAGGER_SECONDS=0
SEEDS_CSV="1,3,2,4"
JOBS_PER_GPU=2

usage() {
  cat <<USAGE
Usage: scripts/launch_m2_candidate_b.sh [options]

Options:
  --stage stage1|stage1b|stage2 Stage selector (default: $STAGE)
  --config PATH                 Training config override
  --run-prefix NAME             Run prefix override
  --total-timesteps N           Timesteps per seed (default: $TOTAL_TIMESTEPS)
  --num-envs N                  Env count per job (default: $NUM_ENVS)
  --num-steps N                 Rollout horizon override (passes --num-steps)
  --eval-every-updates N        Eval cadence override (passes --eval-every-updates)
  --print-every-updates N       Progress print interval (default: $PRINT_EVERY_UPDATES)
  --run-dir PATH                Base run dir (default: $RUN_DIR)
  --python-bin PATH             Python interpreter (default: auto-detect)
  --load-mode init|resume       init: load weights only, resume: load full state (default: $LOAD_MODE)
  --init-ckpt-template TPL      Checkpoint template with {seed} {gpu} {slot}
  --stage1-prefix NAME          Stage1 prefix used to build stage2 init template
  --no-strict-ckpt-check        Allow missing checkpoints (default: strict)
  --cpu-threads N               OMP/MKL/OPENBLAS threads per job (default: $CPU_THREADS)
  --stagger-seconds N           Sleep between launches to reduce startup spikes
  --seeds CSV                   Seed list in launch order (default: $SEEDS_CSV)
  --jobs-per-gpu N              Parallel jobs per GPU (default: $JOBS_PER_GPU)
  --dry-run                     Print commands only, do not launch
  -h, --help                    Show this help

Default stage templates:
  stage1: runs/cleanrl_ppo/m2_warm_m1_40m_20260303T110912Z_s{seed}_g{gpu}j{slot}/checkpoints/update_000300.pt
  stage1b: runs/cleanrl_ppo/m2_warm_m1_40m_20260303T110912Z_s{seed}_g{gpu}j{slot}/checkpoints/update_000300.pt
  stage2: runs/cleanrl_ppo/<stage1-prefix>_s{seed}_g{gpu}j{slot}/checkpoints/latest.pt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --total-timesteps)
      TOTAL_TIMESTEPS="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --eval-every-updates)
      EVAL_EVERY_UPDATES="$2"
      shift 2
      ;;
    --print-every-updates)
      PRINT_EVERY_UPDATES="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      LOG_DIR="$RUN_DIR/campaign_logs"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --load-mode)
      LOAD_MODE="$2"
      shift 2
      ;;
    --init-ckpt-template)
      INIT_CKPT_TEMPLATE="$2"
      shift 2
      ;;
    --stage1-prefix)
      STAGE1_PREFIX="$2"
      shift 2
      ;;
    --no-strict-ckpt-check)
      STRICT_CKPT_CHECK=0
      shift
      ;;
    --cpu-threads)
      CPU_THREADS="$2"
      shift 2
      ;;
    --stagger-seconds)
      STAGGER_SECONDS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS_CSV="$2"
      shift 2
      ;;
    --jobs-per-gpu)
      JOBS_PER_GPU="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$STAGE" != "stage1" && "$STAGE" != "stage1b" && "$STAGE" != "stage2" ]]; then
  echo "--stage must be stage1, stage1b, or stage2" >&2
  exit 2
fi
if [[ "$LOAD_MODE" != "init" && "$LOAD_MODE" != "resume" ]]; then
  echo "--load-mode must be init or resume" >&2
  exit 2
fi
if [[ "$JOBS_PER_GPU" -lt 1 ]]; then
  echo "--jobs-per-gpu must be >= 1" >&2
  exit 2
fi
if [[ "$CPU_THREADS" -lt 1 ]]; then
  echo "--cpu-threads must be >= 1" >&2
  exit 2
fi

if [[ -z "$CONFIG" ]]; then
  if [[ "$STAGE" == "stage1" ]]; then
    CONFIG="configs/train_ppo_m2_candidate_b_stage1.yaml"
  elif [[ "$STAGE" == "stage1b" ]]; then
    CONFIG="configs/train_ppo_m2_candidate_b_stage1b.yaml"
  else
    CONFIG="configs/train_ppo_m2_candidate_b_stage2.yaml"
  fi
fi

if [[ -z "$RUN_PREFIX" ]]; then
  if [[ "$STAGE" == "stage1" ]]; then
    RUN_PREFIX="m2_candidate_b_stage1_6m"
  elif [[ "$STAGE" == "stage1b" ]]; then
    RUN_PREFIX="m2_candidate_b_stage1b_8m"
  else
    RUN_PREFIX="m2_candidate_b_stage2_6m"
  fi
fi

if [[ -z "$INIT_CKPT_TEMPLATE" ]]; then
  if [[ "$STAGE" == "stage1" || "$STAGE" == "stage1b" ]]; then
    INIT_CKPT_TEMPLATE="runs/cleanrl_ppo/m2_warm_m1_40m_20260303T110912Z_s{seed}_g{gpu}j{slot}/checkpoints/update_000300.pt"
  else
    INIT_CKPT_TEMPLATE="runs/cleanrl_ppo/${STAGE1_PREFIX}_s{seed}_g{gpu}j{slot}/checkpoints/latest.pt"
  fi
fi

mkdir -p "$LOG_DIR"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "No seeds configured." >&2
  exit 2
fi

TOTAL_JOBS="${#SEEDS[@]}"
GPU_COUNT=2
MAX_JOBS=$((GPU_COUNT * JOBS_PER_GPU))
if [[ "$TOTAL_JOBS" -gt "$MAX_JOBS" ]]; then
  echo "Seed count ($TOTAL_JOBS) exceeds capacity for GPU mapping ($MAX_JOBS)." >&2
  echo "Increase --jobs-per-gpu or reduce --seeds." >&2
  exit 2
fi

echo "Launching M2 candidate-B $STAGE schedule:"
echo "  config=$CONFIG"
echo "  run_prefix=$RUN_PREFIX"
echo "  total_timesteps=$TOTAL_TIMESTEPS"
echo "  num_envs_per_job=$NUM_ENVS"
echo "  jobs_total=$TOTAL_JOBS (${JOBS_PER_GPU} per GPU target)"
echo "  seeds=$SEEDS_CSV"
echo "  load_mode=$LOAD_MODE"
echo "  strict_ckpt_check=$STRICT_CKPT_CHECK"
echo "  cpu_threads_per_job=$CPU_THREADS"
echo "  num_steps_override=${NUM_STEPS:-<config>}"
echo "  eval_every_updates_override=${EVAL_EVERY_UPDATES:-<config>}"
echo "  init_ckpt_template=$INIT_CKPT_TEMPLATE"
echo "  log_dir=$LOG_DIR"
echo "  python_bin=$PYTHON_BIN"
echo

manifest_path="${LOG_DIR}/${RUN_PREFIX}_launch_manifest.tsv"
printf "run_name\tseed\tgpu\tslot\tload_mode\tcheckpoint\tcheckpoint_exists\n" > "$manifest_path"

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu=$(( (idx / JOBS_PER_GPU) % GPU_COUNT ))
  slot=$(( (idx % JOBS_PER_GPU) + 1 ))

  run_name="${RUN_PREFIX}_s${seed}_g${gpu}j${slot}"
  log_path="${LOG_DIR}/${run_name}.log"
  pid_path="${LOG_DIR}/${run_name}.pid"

  ckpt_path="${INIT_CKPT_TEMPLATE//\{seed\}/$seed}"
  ckpt_path="${ckpt_path//\{gpu\}/$gpu}"
  ckpt_path="${ckpt_path//\{slot\}/$slot}"

  ckpt_exists=0
  if [[ -f "$ckpt_path" ]]; then
    ckpt_exists=1
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$seed" "$gpu" "$slot" "$LOAD_MODE" "$ckpt_path" "$ckpt_exists" >> "$manifest_path"

  if [[ "$STRICT_CKPT_CHECK" -eq 1 && "$ckpt_exists" -ne 1 ]]; then
    echo "Checkpoint missing for $run_name: $ckpt_path" >&2
    echo "Abort due to strict checkpoint validation." >&2
    exit 3
  fi

  cmd=("$PYTHON_BIN" scripts/train_cleanrl_ppo.py
    --config "$CONFIG"
    --seed "$seed"
    --device cuda
    --num-envs "$NUM_ENVS"
    --total-timesteps "$TOTAL_TIMESTEPS"
    --print-every-updates "$PRINT_EVERY_UPDATES"
    --run-name "$run_name"
  )

  if [[ -n "$NUM_STEPS" ]]; then
    cmd+=(--num-steps "$NUM_STEPS")
  fi
  if [[ -n "$EVAL_EVERY_UPDATES" ]]; then
    cmd+=(--eval-every-updates "$EVAL_EVERY_UPDATES")
  fi

  if [[ "$LOAD_MODE" == "resume" ]]; then
    cmd+=(--resume-ckpt "$ckpt_path" --resume-training-state)
  else
    cmd+=(--init-ckpt "$ckpt_path")
  fi

  echo "[plan] GPU=$gpu slot=$slot seed=$seed run=$run_name"
  echo "       log=$log_path"
  echo "       checkpoint=$ckpt_path (exists=$ckpt_exists, mode=$LOAD_MODE)"
  echo "       cmd=CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=$CPU_THREADS ${cmd[*]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="$CPU_THREADS"
    export MKL_NUM_THREADS="$CPU_THREADS"
    export OPENBLAS_NUM_THREADS="$CPU_THREADS"
    nohup "${cmd[@]}" >"$log_path" 2>&1 &
    echo $! >"$pid_path"
  )

  if [[ "$STAGGER_SECONDS" -gt 0 ]]; then
    sleep "$STAGGER_SECONDS"
  fi
done

echo
echo "Launch manifest: $manifest_path"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "Dry-run complete."
  exit 0
fi

echo
echo "Launched ${TOTAL_JOBS} jobs. PID files:"
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu=$(( (idx / JOBS_PER_GPU) % GPU_COUNT ))
  slot=$(( (idx % JOBS_PER_GPU) + 1 ))
  run_name="${RUN_PREFIX}_s${seed}_g${gpu}j${slot}"
  pid_path="${LOG_DIR}/${run_name}.pid"
  if [[ -f "$pid_path" ]]; then
    printf "  %s: %s\n" "$run_name" "$(cat "$pid_path")"
  fi
done

echo
echo "Monitor example:"
echo "  tail -f ${LOG_DIR}/${RUN_PREFIX}_s1_g0j1.log"
