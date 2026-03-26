#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEEP_CHECKPOINTS=1
if [[ "${1:-}" == "--all" ]]; then
  KEEP_CHECKPOINTS=0
fi

mkdir -p "$ROOT/results_twc" "$ROOT/logs"

if [[ "$KEEP_CHECKPOINTS" == "1" ]]; then
  mkdir -p "$ROOT/results_twc/checkpoints"
  find "$ROOT/results_twc" -mindepth 1 -maxdepth 1 ! -name checkpoints -exec rm -rf {} + 2>/dev/null || true
else
  rm -rf "$ROOT/results_twc"
  mkdir -p "$ROOT/results_twc"
fi

rm -rf "$ROOT"/results_twc/scaling_*/subruns 2>/dev/null || true
rm -f "$ROOT"/logs/fr3_twc_*.out "$ROOT"/logs/fr3_twc_*.err 2>/dev/null || true
find "$ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$ROOT" -maxdepth 1 -type f -name 'twc_patch_fix*.zip' -delete 2>/dev/null || true

if [[ "$KEEP_CHECKPOINTS" == "1" ]]; then
  echo "Cleaned old TWC outputs/logs while preserving results_twc/checkpoints"
else
  echo "Cleaned all TWC outputs/logs including checkpoints"
fi
