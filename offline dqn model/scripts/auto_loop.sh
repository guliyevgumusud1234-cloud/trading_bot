#!/usr/bin/env bash
set -euo pipefail

INTERVAL_MINUTES=${INTERVAL_MINUTES:-180}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/auto_loop.log"
mkdir -p "$PROJECT_ROOT/logs"

export PYTHONPATH="$PROJECT_ROOT/src"

while true; do
  echo "[auto_loop] $(date -u '+%Y-%m-%d %H:%M:%S') starting cycle" | tee -a "$LOG_FILE"
if "$PROJECT_ROOT/scripts/run_cycle.sh" >>"$LOG_FILE" 2>&1; then
    echo "[auto_loop] $(date -u '+%Y-%m-%d %H:%M:%S') cycle completed" | tee -a "$LOG_FILE"
else
    echo "[auto_loop] $(date -u '+%Y-%m-%d %H:%M:%S') cycle failed" | tee -a "$LOG_FILE"
    python "$PROJECT_ROOT/scripts/notify.py" --message "DQN auto loop failure at $(date -u '+%Y-%m-%d %H:%M:%S')" || true
fi
  sleep $((INTERVAL_MINUTES * 60))
done
