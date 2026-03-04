#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: smoke_holosoma_mjwarp.sh [options]

Options:
  --repo-dir PATH      Holosoma repo path (default: $HOME/work/warp-holosoma/holosoma)
  --iterations INT     Learning iterations per smoke run (default: 3)
  --single-envs INT    Envs for single-GPU smoke (default: 128)
  --total-envs INT     Global envs for 2-GPU smoke (default: 512)
  --project NAME       training.project for smoke runs (default: warp-holosoma-smoke)
  --run-prefix NAME    Base run-name prefix (default: g1_mjwarp_smoke)
  --no-record          Disable structured experiment recording
  -h, --help           Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_holosoma_mjwarp_training.sh"

REPO_DIR="${HOME}/work/warp-holosoma/holosoma"
ITERATIONS="${ITERATIONS:-3}"
SINGLE_ENVS="${SINGLE_ENVS:-128}"
TOTAL_ENVS="${TOTAL_ENVS:-512}"
PROJECT_NAME="${PROJECT_NAME:-warp-holosoma-smoke}"
RUN_PREFIX="${RUN_PREFIX:-g1_mjwarp_smoke}"
RECORD_ENABLED=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --single-envs)
      SINGLE_ENVS="$2"
      shift 2
      ;;
    --total-envs)
      TOTAL_ENVS="$2"
      shift 2
      ;;
    --project)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --no-record)
      RECORD_ENABLED=0
      shift
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

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "Launcher script not executable: ${LAUNCHER}" >&2
  exit 1
fi

RECORD_FLAG=()
if [[ "${RECORD_ENABLED}" -eq 0 ]]; then
  RECORD_FLAG+=("--no-record")
fi

echo "[1/2] Single-GPU MJWarp smoke (${SINGLE_ENVS} envs, ${ITERATIONS} iterations)"
"${LAUNCHER}" \
  --repo-dir "${REPO_DIR}" \
  --mode single \
  --single-envs "${SINGLE_ENVS}" \
  --iterations "${ITERATIONS}" \
  --project "${PROJECT_NAME}" \
  --run-name "${RUN_PREFIX}" \
  --seed 42 \
  ${RECORD_FLAG[@]+"${RECORD_FLAG[@]}"} \
  -- \
  --algo.config.logging-interval 1 \
  --algo.config.save-interval 100000

echo "[2/2] Two-GPU MJWarp smoke (${TOTAL_ENVS} total envs, ${ITERATIONS} iterations)"
"${LAUNCHER}" \
  --repo-dir "${REPO_DIR}" \
  --mode torchrun \
  --total-envs "${TOTAL_ENVS}" \
  --iterations "${ITERATIONS}" \
  --project "${PROJECT_NAME}" \
  --run-name "${RUN_PREFIX}" \
  --seed 42 \
  ${RECORD_FLAG[@]+"${RECORD_FLAG[@]}"} \
  -- \
  --algo.config.logging-interval 1 \
  --algo.config.save-interval 100000
