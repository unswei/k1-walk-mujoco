#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE="deploy@<remotehost>"
REMOTE_ROOT="/home/deploy/work/k1-walk-mujoco"
PREFIX=""
OUT_DATE="$(date +%F)"
OUT_BASE="experiments/artifacts"
INCLUDE_VIDEOS=1

usage() {
  cat <<USAGE
Usage: scripts/sync_remote_experiment.sh --prefix RUN_PREFIX [options]

Required:
  --prefix RUN_PREFIX         Run prefix (example: m2_warm_m1_40m_20260303T110912Z)

Options:
  --remote USER@HOST          SSH remote (default: $REMOTE)
  --remote-root PATH          Remote repo root (default: $REMOTE_ROOT)
  --out-date YYYY-MM-DD       Output date folder (default: $OUT_DATE)
  --out-base PATH             Local artifacts base (default: $OUT_BASE)
  --no-videos                 Skip video copy
  -h, --help                  Show this help

Outputs:
  Local target: <out-base>/<out-date>/<prefix>/
  - campaign_logs/* matching prefix
  - run eval metrics for all runs matching prefix
  - optional videos matching prefix
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --out-date)
      OUT_DATE="$2"
      shift 2
      ;;
    --out-base)
      OUT_BASE="$2"
      shift 2
      ;;
    --no-videos)
      INCLUDE_VIDEOS=0
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

if [[ -z "$PREFIX" ]]; then
  echo "--prefix is required" >&2
  usage
  exit 2
fi

DEST="$OUT_BASE/$OUT_DATE/$PREFIX"
mkdir -p "$DEST"

echo "[sync] remote=$REMOTE"
echo "[sync] remote_root=$REMOTE_ROOT"
echo "[sync] prefix=$PREFIX"
echo "[sync] dest=$DEST"

# Pull campaign-level logs in a shell-safe way.
ssh "$REMOTE" "cd '$REMOTE_ROOT' && ls runs/cleanrl_ppo/campaign_logs/${PREFIX}_* 2>/dev/null || true" |
while IFS= read -r remote_path; do
  [[ -z "$remote_path" ]] && continue
  scp "$REMOTE:$REMOTE_ROOT/$remote_path" "$DEST/" || true
done

# Pull per-run eval metrics.
ssh "$REMOTE" "cd '$REMOTE_ROOT' && ls -d runs/cleanrl_ppo/${PREFIX}_* 2>/dev/null || true" |
while IFS= read -r run_dir; do
  [[ -z "$run_dir" ]] && continue
  run_name="$(basename "$run_dir")"
  remote_eval="$REMOTE_ROOT/$run_dir/eval/metrics.jsonl"
  local_eval="$DEST/${run_name}_eval_metrics.jsonl"
  scp "$REMOTE:$remote_eval" "$local_eval" || true
done

if [[ "$INCLUDE_VIDEOS" -eq 1 ]]; then
  ssh "$REMOTE" "cd '$REMOTE_ROOT' && ls runs/cleanrl_ppo/videos/*/${PREFIX}*.mp4 2>/dev/null || true" |
  while IFS= read -r remote_video; do
    [[ -z "$remote_video" ]] && continue
    scp "$REMOTE:$REMOTE_ROOT/$remote_video" "$DEST/" || true
  done
fi

echo "[sync] complete"
