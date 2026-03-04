#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_holosoma_mjwarp_uv.sh [options]

Options:
  --workspace-dir PATH   Workspace root (default: $HOME/work/warp-holosoma)
  --repo-dir PATH        Holosoma repo path (default: <workspace>/holosoma)
  --python BIN           Python interpreter for uv venv (default: python3.11)
  --warp-commit SHA      mujoco_warp commit (default: 09ec1da)
  --apply-patch          Apply compatibility patch (only for older NVIDIA drivers)
  --no-patch             Alias for not applying the patch
  -h, --help             Show this help
EOF
}

WORKSPACE_DIR="${HOME}/work/warp-holosoma"
REPO_DIR=""
PYTHON_BIN="python3.11"
WARP_COMMIT="09ec1da"
APPLY_PATCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-dir)
      WORKSPACE_DIR="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --warp-commit)
      WARP_COMMIT="$2"
      shift 2
      ;;
    --no-patch)
      APPLY_PATCH=0
      shift
      ;;
    --apply-patch)
      APPLY_PATCH=1
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

if [[ -z "${REPO_DIR}" ]]; then
  REPO_DIR="${WORKSPACE_DIR}/holosoma"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_PATH="${SCRIPT_DIR}/patches/holosoma_warp_backend_capture_fallback.patch"
MUJOCO_WARP_DIR="${HOME}/.holosoma_deps/mujoco_warp"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  echo "Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

mkdir -p "${WORKSPACE_DIR}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/amazon-far/holosoma.git "${REPO_DIR}"
else
  echo "Using existing Holosoma clone at ${REPO_DIR}"
fi

cd "${REPO_DIR}"

if [[ ! -d .venv ]]; then
  uv venv .venv --python "${PYTHON_BIN}"
fi

source .venv/bin/activate

uv pip install --upgrade pip
uv pip install "mujoco>=3.0.0" mujoco-python-viewer
uv pip install -e "src/holosoma[unitree,booster]"

mkdir -p "$(dirname "${MUJOCO_WARP_DIR}")"
if [[ ! -d "${MUJOCO_WARP_DIR}/.git" ]]; then
  git clone https://github.com/google-deepmind/mujoco_warp.git "${MUJOCO_WARP_DIR}"
fi
git -C "${MUJOCO_WARP_DIR}" fetch --all --tags
git -C "${MUJOCO_WARP_DIR}" checkout "${WARP_COMMIT}"
uv pip install -e "${MUJOCO_WARP_DIR}[dev,cuda]"

if [[ "${APPLY_PATCH}" -eq 1 ]]; then
  if [[ ! -f "${PATCH_PATH}" ]]; then
    echo "Patch file not found: ${PATCH_PATH}" >&2
    exit 1
  fi
  if git apply --check "${PATCH_PATH}" >/dev/null 2>&1; then
    git apply "${PATCH_PATH}"
    echo "Applied patch: ${PATCH_PATH}"
  elif git apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
    echo "Patch already applied: ${PATCH_PATH}"
  else
    echo "Patch did not apply cleanly. Holosoma source may have changed." >&2
    echo "Patch file: ${PATCH_PATH}" >&2
    exit 1
  fi
fi

python - <<'PY'
import mujoco
import mujoco_warp
import holosoma

print("mujoco", mujoco.__version__)
print("mujoco_warp ok")
print("holosoma ok")
PY

echo "Holosoma MJWarp uv setup complete in ${REPO_DIR}"
