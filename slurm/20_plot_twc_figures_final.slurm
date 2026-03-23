#!/bin/bash
#SBATCH --job-name=fr3_twc_figures_final
#SBATCH --account=def-rsadve
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
ROOT=/home/rsadve1/Deep_Unfolding_FR3_20260319
cd "$ROOT"
mkdir -p logs
module purge >/dev/null 2>&1 || true
module load python/3.10 >/dev/null 2>&1 || module load python >/dev/null 2>&1 || true
source "$ROOT/venv_fr3_twc/bin/activate"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
export FR3_REPO_ROOT="$ROOT"
python -u scripts/plot_twc_figures.py --output-root results_twc
