#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESERVE_CHECKPOINTS=1

for arg in "$@"; do
  case "$arg" in
    --all)
      # Backward-compatible alias: keep canonical checkpoints unless purge is requested.
      PRESERVE_CHECKPOINTS=1
      ;;
    --purge-checkpoints|--from-scratch)
      PRESERVE_CHECKPOINTS=0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: bash scripts/clean_old_twc_outputs.sh [--all] [--purge-checkpoints|--from-scratch]" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$ROOT/results_twc" "$ROOT/logs"
mkdir -p "$ROOT/results_twc/checkpoints"
find "$ROOT/results_twc" -mindepth 1 -maxdepth 1 ! -name checkpoints -exec rm -rf {} + 2>/dev/null || true

if [[ "$PRESERVE_CHECKPOINTS" == "0" ]]; then
  rm -rf "$ROOT/results_twc/checkpoints"
  mkdir -p "$ROOT/results_twc/checkpoints"
fi

rm -rf "$ROOT"/results_twc/scaling_*/subruns 2>/dev/null || true
rm -f "$ROOT"/logs/fr3_twc_*.out "$ROOT"/logs/fr3_twc_*.err 2>/dev/null || true
find "$ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$ROOT" -maxdepth 1 -type f -name 'twc_patch_fix*.zip' -delete 2>/dev/null || true

if [[ "$PRESERVE_CHECKPOINTS" == "1" ]]; then
  echo "Cleaned old TWC outputs/logs while preserving results_twc/checkpoints"
else
  echo "Cleaned old TWC outputs/logs and purged results_twc/checkpoints"
fi
