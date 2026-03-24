#!/bin/bash
#SBATCH --job-name=fr3_twc_eval_final
#SBATCH --account=def-rsadve
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
ROOT=/home/rsadve1/Deep_Unfolding_FR3_20260319
cd "$ROOT"
mkdir -p logs
module purge >/dev/null 2>&1 || true
module load python/3.10 cuda >/dev/null 2>&1 || module load python cuda >/dev/null 2>&1 || true
source "$ROOT/venv_fr3_twc/bin/activate"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
export FR3_REPO_ROOT="$ROOT"
export TF_FORCE_GPU_ALLOW_GROWTH=true
python -u scripts/eval_unfolding.py --config configs/twc_eval_final.yaml
