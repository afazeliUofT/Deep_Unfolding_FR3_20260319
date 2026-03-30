#!/bin/bash
set -euo pipefail
ROOT=/home/rsadve1/Deep_Unfolding_FR3_20260319
cd "$ROOT"

rm -rf results_twc/Publication_Level_Final
rm -rf results_twc/publication_convergence_rollout_*
rm -rf results_twc/figures_twc_*

find logs -maxdepth 1 -type f \( \
    -name 'fr3_twc_figures_final-*.out' -o \
    -name 'fr3_twc_publication_final-*.out' \
\) -delete

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

rm -f twc_patch_fix*.zip

git rm -f --ignore-unmatch twc_patch_fix*.zip >/dev/null 2>&1 || true

echo "Cleaned publication-level figure artifacts and rollout-only convergence outputs. Raw eval/scaling/selectivity results and checkpoints were preserved."
