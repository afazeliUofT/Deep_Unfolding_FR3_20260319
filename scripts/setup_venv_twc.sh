#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME="${1:-venv_fr3_twc}"
VENV_PATH="$ROOT/$VENV_NAME"

module purge >/dev/null 2>&1 || true
module load python/3.10 >/dev/null 2>&1 || module load python >/dev/null 2>&1 || true

python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT/requirements/requirements_twc.txt"

export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
python "$ROOT/scripts/probe_env.py" || true
python - <<'PY'
import sys
print('VENV_READY', sys.prefix)
PY
