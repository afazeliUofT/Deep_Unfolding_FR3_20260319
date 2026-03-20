#!/bin/bash
set -euo pipefail

VARIANT="${1:?usage: run_train_unfolding_job.sh <soft|cognitive> [venv_name]}"
VENV_NAME="${2:-venv_fr3_twc}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$VARIANT" in
    soft|cognitive) ;;
    *)
        echo "Unsupported variant: $VARIANT" >&2
        exit 2
        ;;
esac

cd "$ROOT"
mkdir -p logs

module purge >/dev/null 2>&1 || true
module load python/3.10 cuda >/dev/null 2>&1 || module load python cuda >/dev/null 2>&1 || true
source "$ROOT/$VENV_NAME/bin/activate"

export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
export FR3_REPO_ROOT="$ROOT"
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "TRAIN_JOB variant=$VARIANT root=$ROOT venv=$VENV_NAME"
python "$ROOT/scripts/probe_env.py"
python -u "$ROOT/scripts/train_unfolding.py" --config "$ROOT/configs/twc_base.yaml" --variant "$VARIANT"
