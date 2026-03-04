#!/usr/bin/env bash
set -euo pipefail

# Best profile from 2026-03-03 throughput sweep on <remotehost>:
#   jobs_per_gpu=2, num_envs=32 per job (4 concurrent jobs total).

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

CONFIG="configs/train_ppo_m2.yaml"
RUN_PREFIX="m2_highutil_12m"
TOTAL_TIMESTEPS=12000000
NUM_ENVS=32
PRINT_EVERY_UPDATES=20
RUN_DIR="runs/cleanrl_ppo"
LOG_DIR="$RUN_DIR/campaign_logs"
INIT_CKPT_TEMPLATE=""
DRY_RUN=0

usage() {
  cat <<EOF
Usage: scripts/launch_m2_highutil.sh [options]

Options:
  --config PATH                 Training config (default: $CONFIG)
  --run-prefix NAME             Run prefix (default: $RUN_PREFIX)
  --total-timesteps N           Timesteps per seed (default: $TOTAL_TIMESTEPS)
  --num-envs N                  Env count per job (default: $NUM_ENVS)
  --print-every-updates N       Progress print interval (default: $PRINT_EVERY_UPDATES)
  --run-dir PATH                Base run dir (default: $RUN_DIR)
  --python-bin PATH             Python interpreter (default: auto-detect)
  --init-ckpt-template TPL      Optional checkpoint template with {seed} and {gpu}
  --dry-run                     Print commands only, do not launch
  -h, --help                    Show this help

Example:
  scripts/launch_m2_highutil.sh \\
    --run-prefix m2_from_m1_12m \\
    --init-ckpt-template 'runs/cleanrl_ppo/m1_reallyeasy_12m_s{seed}_g{gpu}/checkpoints/latest.pt'
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --init-ckpt-template)
      INIT_CKPT_TEMPLATE="$2"
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

mkdir -p "$LOG_DIR"

# Fixed 4-seed mapping for two-GPU, two-jobs-per-GPU layout.
SEEDS=(1 3 2 4)
GPUS=(0 0 1 1)
SLOTS=(1 2 1 2)

echo "Launching M2 high-utilization schedule:"
echo "  config=$CONFIG"
echo "  run_prefix=$RUN_PREFIX"
echo "  total_timesteps=$TOTAL_TIMESTEPS"
echo "  num_envs_per_job=$NUM_ENVS"
echo "  jobs_total=4 (2 per GPU)"
echo "  log_dir=$LOG_DIR"
echo "  python_bin=$PYTHON_BIN"
echo

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  slot="${SLOTS[$i]}"
  run_name="${RUN_PREFIX}_s${seed}_g${gpu}j${slot}"
  log_path="${LOG_DIR}/${run_name}.log"
  pid_path="${LOG_DIR}/${run_name}.pid"

  cmd=("$PYTHON_BIN" scripts/train_cleanrl_ppo.py
    --config "$CONFIG"
    --seed "$seed"
    --device cuda
    --num-envs "$NUM_ENVS"
    --total-timesteps "$TOTAL_TIMESTEPS"
    --print-every-updates "$PRINT_EVERY_UPDATES"
    --run-name "$run_name"
  )

  if [[ -n "$INIT_CKPT_TEMPLATE" ]]; then
    init_ckpt="${INIT_CKPT_TEMPLATE//\{seed\}/$seed}"
    init_ckpt="${init_ckpt//\{gpu\}/$gpu}"
    cmd+=(--init-ckpt "$init_ckpt")
  fi

  echo "[plan] GPU=$gpu slot=$slot seed=$seed run=$run_name"
  echo "       log=$log_path"
  echo "       cmd=CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ${cmd[*]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    nohup "${cmd[@]}" >"$log_path" 2>&1 &
    echo $! >"$pid_path"
  )
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "Dry-run complete."
  exit 0
fi

echo
echo "Launched 4 jobs. PID files:"
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  slot="${SLOTS[$i]}"
  run_name="${RUN_PREFIX}_s${seed}_g${gpu}j${slot}"
  pid_path="${LOG_DIR}/${run_name}.pid"
  if [[ -f "$pid_path" ]]; then
    printf "  %s: %s\n" "$run_name" "$(cat "$pid_path")"
  fi
done

echo
echo "Monitor example:"
echo "  tail -f ${LOG_DIR}/${RUN_PREFIX}_s1_g0j1.log"
