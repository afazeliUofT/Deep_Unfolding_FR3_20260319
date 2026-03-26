#!/bin/bash
set -euo pipefail

VARIANT="${1:?usage: run_train_unfolding_job.sh <variant> [venv_name] [config_path]}"
VENV_NAME="${2:-venv_fr3_twc}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${3:-$ROOT/configs/twc_base.yaml}"
QUIET_PY="$ROOT/scripts/run_quiet_python.py"

case "$VARIANT" in
  soft|cognitive) ;;
  *)
    echo "Unsupported variant: $VARIANT" >&2
    exit 2
    ;;
esac

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 3
fi

cd "$ROOT"
mkdir -p logs
module purge >/dev/null 2>&1 || true
module load python/3.10 cuda >/dev/null 2>&1 || module load python cuda >/dev/null 2>&1 || true
source "$ROOT/$VENV_NAME/bin/activate"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
export FR3_REPO_ROOT="$ROOT"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

echo "TRAIN_JOB variant=$VARIANT root=$ROOT venv=$VENV_NAME config=$CONFIG_PATH"
if [[ "${FR3_RUN_PROBE:-0}" == "1" ]]; then
  python -u "$QUIET_PY" "$ROOT/scripts/probe_env.py"
fi
python -u "$QUIET_PY" "$ROOT/scripts/train_unfolding.py" --config "$CONFIG_PATH" --variant "$VARIANT"
