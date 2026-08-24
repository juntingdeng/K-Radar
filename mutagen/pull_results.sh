#!/usr/bin/env bash
# Pull evaluation-result .txt files from the cluster's logs/ down to the local
# repo, preserving the exp_<date>_<time>_.../ folder structure. Only .txt files
# are transferred (checkpoints, images, docx/pptx are skipped). Safe to re-run;
# rsync only copies new/changed files.
#
# Usage:
#   ./pull_results.sh                 # pull ALL experiments' .txt results
#   ./pull_results.sh --log 260222_165721   # pull only exp_260222_165721* results
#   ./pull_results.sh --log=260222_165721   # same (= form also accepted)
#
# The --log value is the exp signature (date_time); any suffix like _RTNH / _1ppv
# is matched automatically, so several matching exp dirs are pulled if present.
set -euo pipefail

REMOTE_HOST="login-gpu.ece.local.cmu.edu"
REMOTE_LOGS="research/3DImage/lidar/K-radar/K-Radar/logs"
LOCAL_LOGS="$(cd "$(dirname "$0")/.." && pwd)/logs"   # repo root = parent of mutagen/

LOG_SIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)   LOG_SIG="${2:-}"; shift 2 ;;
    --log=*) LOG_SIG="${1#*=}"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# rsync filter: keep the directory tree but transfer only .txt files.
txt_only=(--prune-empty-dirs --include='*/' --include='*.txt' --exclude='*')

if [[ -n "$LOG_SIG" ]]; then
  # Resolve the actual exp dir name(s) on the remote (allows _RTNH etc. suffixes).
  matches="$(ssh "$REMOTE_HOST" "cd '$REMOTE_LOGS' && ls -d exp_${LOG_SIG}* 2>/dev/null" || true)"
  if [[ -z "$matches" ]]; then
    echo "No remote experiment matches: exp_${LOG_SIG}*" >&2
    exit 1
  fi
  while IFS= read -r m; do
    [[ -z "$m" ]] && continue
    echo "* pulling $m ..."
    rsync -az "${txt_only[@]}" \
      "${REMOTE_HOST}:${REMOTE_LOGS}/${m}/" "${LOCAL_LOGS}/${m}/"
  done <<< "$matches"
else
  echo "* pulling ALL experiments (.txt only) ..."
  rsync -az "${txt_only[@]}" \
    "${REMOTE_HOST}:${REMOTE_LOGS}/" "${LOCAL_LOGS}/"
fi

echo "Done. Results synced to: ${LOCAL_LOGS}"
