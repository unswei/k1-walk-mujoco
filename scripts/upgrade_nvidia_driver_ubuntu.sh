#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Upgrade NVIDIA driver on Ubuntu with rollback scaffolding.

Default behaviour is a dry-run plan. Use --apply to execute.

Usage:
  scripts/upgrade_nvidia_driver_ubuntu.sh [options]

Options:
  --target-package PKG   Driver meta package to install (default: nvidia-driver-570)
  --min-driver VER       Minimum acceptable version for verification (default: 550.54.14)
  --snapshot-dir PATH    Snapshot/output directory (default: /var/tmp/nvidia-driver-upgrade-<timestamp>)
  --apply                Execute apt install steps (otherwise dry-run only)
  --no-reboot            Do not reboot automatically after install
  --post-reboot-verify   Verification-only mode (run after reboot)
  -h, --help             Show this help

Examples:
  scripts/upgrade_nvidia_driver_ubuntu.sh
  scripts/upgrade_nvidia_driver_ubuntu.sh --apply
  scripts/upgrade_nvidia_driver_ubuntu.sh --apply --target-package nvidia-driver-550
  scripts/upgrade_nvidia_driver_ubuntu.sh --post-reboot-verify
EOF
}

TARGET_PACKAGE="nvidia-driver-570"
MIN_DRIVER="550.54.14"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_DIR="/var/tmp/nvidia-driver-upgrade-${TIMESTAMP}"
APPLY=0
NO_REBOOT=0
POST_REBOOT_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-package)
      TARGET_PACKAGE="$2"
      shift 2
      ;;
    --min-driver)
      MIN_DRIVER="$2"
      shift 2
      ;;
    --snapshot-dir)
      SNAPSHOT_DIR="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --no-reboot)
      NO_REBOOT=1
      shift
      ;;
    --post-reboot-verify)
      POST_REBOOT_VERIFY=1
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

version_ge() {
  local lhs="$1"
  local rhs="$2"
  [[ "$(printf '%s\n' "$rhs" "$lhs" | sort -V | head -n1)" == "$rhs" ]]
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "$APPLY" -eq 1 ]]; then
    echo "+ $*"
    "$@"
  else
    echo "DRY-RUN: $*"
  fi
}

require_cmd apt-cache
require_cmd apt-get
require_cmd ubuntu-drivers

if [[ "$POST_REBOOT_VERIFY" -eq 1 ]]; then
  require_cmd nvidia-smi
  DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
  CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]\+\).*/\1/p' | head -n1)"

  echo "Post-reboot verification:"
  echo "  Driver: ${DRIVER_VERSION}"
  echo "  CUDA (reported): ${CUDA_VERSION:-unknown}"
  echo "  Minimum required driver: ${MIN_DRIVER}"

  if version_ge "$DRIVER_VERSION" "$MIN_DRIVER"; then
    echo "PASS: driver version satisfies minimum."
    exit 0
  fi

  echo "FAIL: driver version is below minimum." >&2
  exit 2
fi

if [[ -r /etc/os-release ]]; then
  # shellcheck source=/dev/null
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "This script currently supports Ubuntu only (detected: ${ID:-unknown})." >&2
    exit 1
  fi
fi

CURRENT_DRIVER_VERSION="unknown"
CUDA_VERSION="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  CURRENT_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
  CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]\+\).*/\1/p' | head -n1)"
fi

CURRENT_DRIVER_MAJOR="$(echo "$CURRENT_DRIVER_VERSION" | awk -F. '/^[0-9]+(\.[0-9]+)*$/ {print $1}')"
INSTALLED_DRIVER_PACKAGES="$(
  dpkg-query -W -f='${Package} ${Status}\n' 'nvidia-driver-*' 2>/dev/null \
    | awk '$4=="installed" && $1 ~ /^nvidia-driver-[0-9]+$/ {print $1}'
)"
CURRENT_META_PACKAGE="none"
if [[ -n "${INSTALLED_DRIVER_PACKAGES}" ]]; then
  if [[ -n "${CURRENT_DRIVER_MAJOR}" ]] && echo "${INSTALLED_DRIVER_PACKAGES}" | grep -qx "nvidia-driver-${CURRENT_DRIVER_MAJOR}"; then
    CURRENT_META_PACKAGE="nvidia-driver-${CURRENT_DRIVER_MAJOR}"
  else
    CURRENT_META_PACKAGE="$(echo "${INSTALLED_DRIVER_PACKAGES}" | sort -V | tail -n1)"
  fi
fi

RECOMMENDED_PACKAGE="$(
  ubuntu-drivers devices 2>/dev/null \
    | awk '/recommended/ && $0 ~ /nvidia-driver-/ {for (i=1;i<=NF;i++) if ($i ~ /^nvidia-driver-/) {print $i; exit}}'
)"
RECOMMENDED_PACKAGE="${RECOMMENDED_PACKAGE:-unknown}"

CANDIDATE_VERSION="$(apt-cache policy "$TARGET_PACKAGE" | awk '/Candidate:/ {print $2}')"
if [[ -z "$CANDIDATE_VERSION" || "$CANDIDATE_VERSION" == "(none)" ]]; then
  echo "Target package not available: ${TARGET_PACKAGE}" >&2
  exit 1
fi

echo "System preflight summary"
echo "  Host: $(hostname)"
echo "  Ubuntu release: ${VERSION_ID:-unknown}"
echo "  Current driver: ${CURRENT_DRIVER_VERSION}"
echo "  Current CUDA: ${CUDA_VERSION}"
echo "  Current driver package: ${CURRENT_META_PACKAGE}"
echo "  ubuntu-drivers recommended: ${RECOMMENDED_PACKAGE}"
echo "  Target package: ${TARGET_PACKAGE} (${CANDIDATE_VERSION})"
echo "  Minimum required: ${MIN_DRIVER}"
echo "  Snapshot dir: ${SNAPSHOT_DIR}"
echo

if [[ "$APPLY" -eq 1 ]]; then
  require_cmd sudo
fi

run_cmd mkdir -p "$SNAPSHOT_DIR"
if [[ "$APPLY" -eq 1 && ! -w "$SNAPSHOT_DIR" ]]; then
  run_cmd sudo chown "$USER":"$USER" "$SNAPSHOT_DIR"
fi
run_cmd bash -lc "ubuntu-drivers devices > '$SNAPSHOT_DIR/ubuntu-drivers-devices.txt'"
run_cmd bash -lc "apt-cache policy '$TARGET_PACKAGE' > '$SNAPSHOT_DIR/apt-policy-${TARGET_PACKAGE}.txt'"
run_cmd bash -lc "dpkg -l 'nvidia-*' > '$SNAPSHOT_DIR/dpkg-nvidia-before.txt' || true"
run_cmd bash -lc "nvidia-smi > '$SNAPSHOT_DIR/nvidia-smi-before.txt' || true"

if [[ "$CURRENT_META_PACKAGE" != "none" ]]; then
  if [[ "$APPLY" -eq 1 ]]; then
    cat > "${SNAPSHOT_DIR}/rollback.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y ${CURRENT_META_PACKAGE}
sudo reboot
EOF
    chmod +x "${SNAPSHOT_DIR}/rollback.sh"
    echo "+ wrote rollback script: ${SNAPSHOT_DIR}/rollback.sh"
  else
    echo "DRY-RUN: write rollback script to ${SNAPSHOT_DIR}/rollback.sh (restores ${CURRENT_META_PACKAGE})"
  fi
fi

run_cmd sudo apt-get update
run_cmd sudo apt-get install -y "linux-headers-$(uname -r)"
run_cmd sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$TARGET_PACKAGE"
run_cmd bash -lc "dpkg -l 'nvidia-*' > '$SNAPSHOT_DIR/dpkg-nvidia-after-install.txt' || true"

cat <<EOF

Install stage complete.

Next steps:
1. Reboot to load the new kernel modules.
2. Run:
   scripts/upgrade_nvidia_driver_ubuntu.sh --post-reboot-verify --min-driver ${MIN_DRIVER}
3. If needed, rollback script:
   ${SNAPSHOT_DIR}/rollback.sh
EOF

if [[ "$APPLY" -eq 1 && "$NO_REBOOT" -eq 0 ]]; then
  echo
  echo "Rebooting now..."
  sudo reboot
fi
