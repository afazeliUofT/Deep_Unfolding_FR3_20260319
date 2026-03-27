#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/slurm" ]]; then
  echo "Missing slurm directory under $ROOT" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/venv_fr3_twc/bin/python" ]]; then
    PYTHON_BIN="$ROOT/venv_fr3_twc/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
export FR3_REPO_ROOT="$ROOT"

if "$PYTHON_BIN" "$ROOT/scripts/restore_twc_checkpoints.py" --config configs/twc_eval_final.yaml; then
  bash "$ROOT/scripts/clean_old_twc_outputs.sh"

  jid_eval="$(sbatch --parsable "$ROOT/slurm/17_eval_final.slurm")"
  jid_scaling="$(sbatch --parsable "$ROOT/slurm/18_run_scaling_final.slurm")"
  jid_selectivity="$(sbatch --parsable "$ROOT/slurm/19_run_selectivity_final.slurm")"
  post_dep="${jid_eval}:${jid_scaling}:${jid_selectivity}"
  jid_figures="$(sbatch --parsable --dependency=afterok:${post_dep} "$ROOT/slurm/20_plot_twc_figures_final.slurm")"

  cat <<EOF
SUBMITTED_USING_EXISTING_OR_RESTORED_CHECKPOINTS
  eval_job=${jid_eval}
  scaling_job=${jid_scaling}
  selectivity_job=${jid_selectivity}
  figures_job=${jid_figures}
EOF
else
  echo "Existing checkpoints are not locally recoverable. Submitting the full from-scratch publication pipeline." >&2
  bash "$ROOT/scripts/submit_twc_publication_from_scratch.sh"
fi
