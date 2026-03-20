#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME="${1:-venv_fr3_twc}"
VENV_PATH="$ROOT/$VENV_NAME"

module purge >/dev/null 2>&1 || true
module load python/3.10 >/dev/null 2>&1 || module load python >/dev/null 2>&1 || true

if [[ -d "$VENV_PATH" ]]; then
    python -m venv --clear "$VENV_PATH"
else
    python -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y deep-unfolding-fr3-twc sionna sionna-no-rt >/dev/null 2>&1 || true
python -m pip install -r "$ROOT/requirements/requirements_twc.txt"

SITE_PKGS="$(python - <<'PYVENV'
import site
paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass
try:
    user = site.getusersitepackages()
    if user:
        paths.append(user)
except Exception:
    pass
for p in paths:
    if p and 'site-packages' in p:
        print(p)
        break
else:
    raise SystemExit('Could not locate site-packages inside the venv')
PYVENV
)"

PTH_FILE="$SITE_PKGS/fr3_repo_paths.pth"
cat > "$PTH_FILE" <<PTH
$ROOT
$ROOT/src
$ROOT/scripts
PTH

echo "WROTE_PTH $PTH_FILE"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
python "$ROOT/scripts/probe_env.py"
python - <<'PYDONE'
import site
import sys
print('VENV_READY', sys.prefix)
print('SITE_PACKAGES', site.getsitepackages()[0])
PYDONE
