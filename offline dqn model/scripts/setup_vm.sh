#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PYTHON_VERSION="python3"
VENV_DIR="$PROJECT_ROOT/.venv"

mkdir -p "$LOG_DIR"

if ! command -v $PYTHON_VERSION >/dev/null 2>&1; then
  echo "[setup] python3 not found. install Python 3 before running." >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[setup] installing tmux"
  sudo apt-get update && sudo apt-get install -y tmux
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "[setup] installing wget"
  sudo apt-get update && sudo apt-get install -y wget
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating virtualenv at $VENV_DIR"
  $PYTHON_VERSION -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

cat <<CONF > "$PROJECT_ROOT/config/runtime_env.json"
{
  "project_root": "$PROJECT_ROOT",
  "logs": "$LOG_DIR",
  "venv": "$VENV_DIR"
}
CONF

echo "[setup] completed. activate with: source $VENV_DIR/bin/activate"
