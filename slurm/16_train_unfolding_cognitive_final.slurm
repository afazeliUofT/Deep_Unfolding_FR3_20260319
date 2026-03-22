#!/bin/bash
#SBATCH --job-name=fr3_twc_du_cognitive_final
#SBATCH --account=def-rsadve
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

ROOT=/home/rsadve1/Deep_Unfolding_FR3_20260319
cd "$ROOT"
mkdir -p logs

bash "$ROOT/scripts/run_train_unfolding_job.sh" cognitive venv_fr3_twc "$ROOT/configs/twc_train_final.yaml"
