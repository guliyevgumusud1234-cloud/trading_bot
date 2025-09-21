#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

python "$PROJECT_ROOT/scripts/fetch_data.py" --group all
python "$PROJECT_ROOT/scripts/train_loop.py" --group core

if [ "${TRAIN_EXTENDED:-0}" = "1" ]; then
  python "$PROJECT_ROOT/scripts/train_loop.py" --group extended --train-window-days "${EXTENDED_WINDOW_DAYS:-30}" --limit "${EXTENDED_LIMIT:-5}"
fi

latest_checkpoint=$(PROJECT_ROOT="$PROJECT_ROOT" python - <<'PY'
import os
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
runs_dir = project_root / "runs"
latest = None
latest_mtime = -1.0
if runs_dir.exists():
    for path in runs_dir.rglob("model_best.pt"):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest = path
if latest:
    print(latest)
PY
)

if [ -n "${latest_checkpoint:-}" ]; then
  python "$PROJECT_ROOT/scripts/backtest_top_coins.py" --checkpoint "$latest_checkpoint" --coins-config "$PROJECT_ROOT/config/coins.yaml" --data-dir "$PROJECT_ROOT/data"
else
  echo "[run_cycle] No checkpoint found to evaluate"
fi

python "$PROJECT_ROOT/scripts/prune_runs.py" --runs-dir "$PROJECT_ROOT/runs" --keep "${PRUNE_KEEP:-5}" || true
