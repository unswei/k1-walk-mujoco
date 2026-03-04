#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_holosoma_mjwarp_training.sh [options] [-- extra train_agent args]

Modes:
  torchrun      Single distributed run across 2 GPUs (default)
  single        Single-GPU run on GPU 0
  dual-single   Two independent single-GPU runs (GPU 0 and 1)

Options:
  --repo-dir PATH        Holosoma repo path (default: $HOME/work/warp-holosoma/holosoma)
  --mode MODE            torchrun|single|dual-single (default: torchrun)
  --exp NAME             Holosoma experiment subcommand (default: exp:g1-29dof-fast-sac)
  --total-envs INT       Global envs for torchrun mode (default: 512)
  --single-envs INT      Envs per process in single/dual-single mode (default: 256)
  --iterations INT       algo.config.num-learning-iterations (default: 50000)
  --project NAME         training.project value (default: warp-holosoma)
  --run-name NAME        training.name base (default: g1_mjwarp)
  --seed INT             Base seed (default: 42)
  --logger NAME          logger config name (default: disabled)
  --log-dir PATH         Directory for launcher logs/pids (default: <repo>/logs)
  --detach               Launch and return immediately (records stay "running")
  --record-dir PATH      Root for experiment JSON logs (default: <this repo>/experiments/logs)
  --record-track PATH    Sub-path under record dir (default: feature/holosoma-mjwarp)
  --no-record            Disable structured experiment recording
  --dry-run              Print commands only
  -h, --help             Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_DIR="${HOME}/work/warp-holosoma/holosoma"
MODE="torchrun"
EXP_NAME="exp:g1-29dof-fast-sac"
TOTAL_ENVS=512
SINGLE_ENVS=256
ITERATIONS=50000
PROJECT_NAME="warp-holosoma"
RUN_NAME="g1_mjwarp"
SEED=42
LOGGER_NAME="disabled"
LOG_DIR=""
DETACH=0
RECORD_ENABLED=1
RECORD_DIR="${PROJECT_ROOT}/experiments/logs"
RECORD_TRACK="feature/holosoma-mjwarp"
DRY_RUN=0

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --exp)
      EXP_NAME="$2"
      shift 2
      ;;
    --total-envs)
      TOTAL_ENVS="$2"
      shift 2
      ;;
    --single-envs)
      SINGLE_ENVS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --project)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --logger)
      LOGGER_NAME="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --record-dir)
      RECORD_DIR="$2"
      shift 2
      ;;
    --record-track)
      RECORD_TRACK="$2"
      shift 2
      ;;
    --no-record)
      RECORD_ENABLED=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${REPO_DIR}/.venv" ]]; then
  echo "Holosoma venv not found at ${REPO_DIR}/.venv" >&2
  echo "Run scripts/setup_holosoma_mjwarp_uv.sh first." >&2
  exit 1
fi

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${REPO_DIR}/logs"
fi

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DATE_UTC="$(date -u +%F)"
HOST_NAME="$(hostname -s 2>/dev/null || hostname)"

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"
source .venv/bin/activate

cmd_to_string() {
  local out=""
  printf -v out '%q ' "$@"
  printf '%s' "${out% }"
}

record_file_path() {
  local safe_name="$1"
  printf '%s/%s/%s/%s_%s.json' "${RECORD_DIR}" "${RUN_DATE_UTC}" "${RECORD_TRACK}" "${TIMESTAMP_UTC}" "${safe_name}"
}

start_record() {
  local record_file="$1"
  local run_name="$2"
  local mode_name="$3"
  local num_envs="$4"
  local seed_val="$5"
  local cmd_string="$6"
  local log_file="$7"

  [[ "${RECORD_ENABLED}" -eq 1 ]] || return 0

  mkdir -p "$(dirname "${record_file}")"
  python "${SCRIPT_DIR}/record_holosoma_run.py" start \
    --output "${record_file}" \
    --run-name "${run_name}" \
    --project "${PROJECT_NAME}" \
    --experiment "${EXP_NAME}" \
    --simulator "mjwarp" \
    --mode "${mode_name}" \
    --seed "${seed_val}" \
    --num-envs "${num_envs}" \
    --iterations "${ITERATIONS}" \
    --repo-dir "${REPO_DIR}" \
    --host "${HOST_NAME}" \
    --command "${cmd_string}" \
    --log-path "${log_file}" >/dev/null
}

finish_record() {
  local record_file="$1"
  local status="$2"
  local exit_code="$3"
  local log_file="$4"

  [[ "${RECORD_ENABLED}" -eq 1 ]] || return 0
  [[ -n "${record_file}" ]] || return 0

  python "${SCRIPT_DIR}/record_holosoma_run.py" finish \
    --output "${record_file}" \
    --status "${status}" \
    --exit-code "${exit_code}" \
    --log-path "${log_file}" >/dev/null
}

JOB_PIDS=()
JOB_NAMES=()
JOB_LOGS=()
JOB_RECORDS=()

launch_job() {
  local run_name="$1"
  local mode_name="$2"
  local num_envs="$3"
  local seed_val="$4"
  shift 4

  if [[ "${1:-}" != "--" ]]; then
    echo "Internal error: launch_job missing -- delimiter" >&2
    exit 2
  fi
  shift

  local cmd=("$@")
  local safe_run="${run_name//[^A-Za-z0-9_.-]/_}"
  local log_file="${LOG_DIR}/${safe_run}.${TIMESTAMP_UTC}.log"
  local pid_file="${LOG_DIR}/${safe_run}.${TIMESTAMP_UTC}.pid"
  local cmd_string
  cmd_string="$(cmd_to_string "${cmd[@]}")"
  local record_file=""

  if [[ "${RECORD_ENABLED}" -eq 1 ]]; then
    record_file="$(record_file_path "${safe_run}")"
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${run_name}"
    echo "  cmd: ${cmd_string}"
    echo "  log: ${log_file}"
    if [[ "${RECORD_ENABLED}" -eq 1 ]]; then
      echo "  record: ${record_file}"
    fi
    return 0
  fi

  if [[ "${RECORD_ENABLED}" -eq 1 ]]; then
    start_record "${record_file}" "${run_name}" "${mode_name}" "${num_envs}" "${seed_val}" "${cmd_string}" "${log_file}"
  fi

  if [[ "${DETACH}" -eq 1 ]]; then
    nohup "${cmd[@]}" >"${log_file}" 2>&1 &
    local pid=$!
    echo "${pid}" >"${pid_file}"
    echo "Detached run ${run_name}: pid=${pid}"
    echo "  log: ${log_file}"
    if [[ "${RECORD_ENABLED}" -eq 1 ]]; then
      echo "  record: ${record_file}"
      echo "  finalise: python ${SCRIPT_DIR}/record_holosoma_run.py finish --output ${record_file} --log-path ${log_file} --exit-code <code>"
    fi
    return 0
  fi

  "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_file}"
  JOB_PIDS+=("${pid}")
  JOB_NAMES+=("${run_name}")
  JOB_LOGS+=("${log_file}")
  JOB_RECORDS+=("${record_file}")

  echo "Started run ${run_name}: pid=${pid}"
  echo "  log: ${log_file}"
  if [[ "${RECORD_ENABLED}" -eq 1 ]]; then
    echo "  record: ${record_file}"
  fi
}

case "${MODE}" in
  single)
    launch_job "${RUN_NAME}_single_gpu0" "single" "${SINGLE_ENVS}" "${SEED}" -- \
      env CUDA_VISIBLE_DEVICES=0 \
      python src/holosoma/holosoma/train_agent.py \
      "${EXP_NAME}" simulator:mjwarp "logger:${LOGGER_NAME}" \
      --training.project "${PROJECT_NAME}" \
      --training.headless True \
      --training.num-envs "${SINGLE_ENVS}" \
      --algo.config.num-learning-iterations "${ITERATIONS}" \
      --training.name "${RUN_NAME}_single_gpu0" \
      --training.seed "${SEED}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ;;
  torchrun)
    launch_job "${RUN_NAME}_ddp2" "torchrun" "${TOTAL_ENVS}" "${SEED}" -- \
      env CUDA_VISIBLE_DEVICES=0,1 \
      torchrun --standalone --nproc_per_node=2 \
      src/holosoma/holosoma/train_agent.py \
      "${EXP_NAME}" simulator:mjwarp "logger:${LOGGER_NAME}" \
      --training.project "${PROJECT_NAME}" \
      --training.headless True \
      --training.num-envs "${TOTAL_ENVS}" \
      --algo.config.num-learning-iterations "${ITERATIONS}" \
      --training.name "${RUN_NAME}_ddp2" \
      --training.seed "${SEED}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ;;
  dual-single)
    launch_job "${RUN_NAME}_gpu0" "dual-single" "${SINGLE_ENVS}" "${SEED}" -- \
      env CUDA_VISIBLE_DEVICES=0 \
      python src/holosoma/holosoma/train_agent.py \
      "${EXP_NAME}" simulator:mjwarp "logger:${LOGGER_NAME}" \
      --training.project "${PROJECT_NAME}" \
      --training.headless True \
      --training.num-envs "${SINGLE_ENVS}" \
      --algo.config.num-learning-iterations "${ITERATIONS}" \
      --training.name "${RUN_NAME}_gpu0" \
      --training.seed "${SEED}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

    launch_job "${RUN_NAME}_gpu1" "dual-single" "${SINGLE_ENVS}" "$((SEED + 1))" -- \
      env CUDA_VISIBLE_DEVICES=1 \
      python src/holosoma/holosoma/train_agent.py \
      "${EXP_NAME}" simulator:mjwarp "logger:${LOGGER_NAME}" \
      --training.project "${PROJECT_NAME}" \
      --training.headless True \
      --training.num-envs "${SINGLE_ENVS}" \
      --algo.config.num-learning-iterations "${ITERATIONS}" \
      --training.name "${RUN_NAME}_gpu1" \
      --training.seed "$((SEED + 1))" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

if [[ "${DRY_RUN}" -eq 1 || "${DETACH}" -eq 1 ]]; then
  exit 0
fi

overall_rc=0
for idx in "${!JOB_PIDS[@]}"; do
  pid="${JOB_PIDS[$idx]}"
  run_name="${JOB_NAMES[$idx]}"
  log_file="${JOB_LOGS[$idx]}"
  record_file="${JOB_RECORDS[$idx]}"

  set +e
  wait "${pid}"
  rc=$?
  set -e

  status="completed"
  if [[ "${rc}" -ne 0 ]]; then
    status="failed"
    if [[ "${overall_rc}" -eq 0 ]]; then
      overall_rc="${rc}"
    fi
  fi

  echo "Run ${run_name} exited with code ${rc}"

  if [[ "${RECORD_ENABLED}" -eq 1 && -n "${record_file}" ]]; then
    if ! finish_record "${record_file}" "${status}" "${rc}" "${log_file}"; then
      echo "Warning: failed to finalise experiment record ${record_file}" >&2
    fi
  fi
done

exit "${overall_rc}"
