#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/slurm" ]]; then
  echo "Missing slurm directory under $ROOT" >&2
  exit 2
fi

bash "$ROOT/scripts/clean_old_twc_outputs.sh" --purge-checkpoints

jid_soft="$(sbatch --parsable "$ROOT/slurm/15_train_unfolding_soft_final.slurm")"
jid_cognitive="$(sbatch --parsable "$ROOT/slurm/16_train_unfolding_cognitive_final.slurm")"
train_dep="${jid_soft}:${jid_cognitive}"

jid_eval="$(sbatch --parsable --dependency=afterok:${train_dep} "$ROOT/slurm/17_eval_final.slurm")"
jid_scaling="$(sbatch --parsable --dependency=afterok:${train_dep} "$ROOT/slurm/18_run_scaling_final.slurm")"
jid_selectivity="$(sbatch --parsable --dependency=afterok:${train_dep} "$ROOT/slurm/19_run_selectivity_final.slurm")"
post_dep="${jid_eval}:${jid_scaling}:${jid_selectivity}"

jid_figures="$(sbatch --parsable --dependency=afterok:${post_dep} "$ROOT/slurm/20_plot_twc_figures_final.slurm")"

cat <<EOF
SUBMITTED_FROM_SCRATCH
  train_soft_job=${jid_soft}
  train_cognitive_job=${jid_cognitive}
  eval_job=${jid_eval}
  scaling_job=${jid_scaling}
  selectivity_job=${jid_selectivity}
  figures_job=${jid_figures}

This submission order prevents eval/scaling/selectivity from starting before checkpoints exist.
Use bash scripts/submit_twc_publication.sh in normal operation; it will restore checkpoints if possible and otherwise fall back to this full from-scratch chain automatically.
EOF
