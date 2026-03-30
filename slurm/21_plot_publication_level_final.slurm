#!/bin/bash
#SBATCH --job-name=fr3_twc_publication_final
#SBATCH --account=def-rsadve
#SBATCH --time=04:00:00
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
export MPLBACKEND=Agg
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

# Always rebuild the publication folder and the rollout-only convergence input from scratch.
rm -rf results_twc/Publication_Level_Final
rm -rf results_twc/publication_convergence_rollout_*

python -u scripts/run_publication_convergence_rollout.py \
  --config configs/twc_eval_final.yaml \
  --suite-name publication_convergence_rollout \
  --snr-db 15 \
  --rollout-layers 56

python -u scripts/plot_publication_level_final.py --output-root results_twc --out-dir results_twc/Publication_Level_Final
python -u scripts/export_publication_level_final_csvs.py --output-root results_twc --out-dir results_twc/Publication_Level_Final

# Hard checks so a partial publication folder is never mistaken for success.
test -f results_twc/Publication_Level_Final/Fig03_Convergence_Quality.png
test -f results_twc/Publication_Level_Final/Fig03_Convergence_Quality.pdf
test -f results_twc/Publication_Level_Final/Fig06_Selected_FER.png
test -f results_twc/Publication_Level_Final/Fig06_Selected_FER.pdf
test -d results_twc/Publication_Level_Final/csv_data
test -f results_twc/Publication_Level_Final/csv_data/Fig03_Convergence_Quality.csv
test -f results_twc/Publication_Level_Final/csv_data/Fig06_Selected_FER.csv
test -f results_twc/Publication_Level_Final/csv_data/00_Publication_Level_Final_Metadata.csv

echo "PUBLICATION_LEVEL_FINAL_OK"
