#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "$ROOT/results_twc"
mkdir -p "$ROOT/results_twc"
mkdir -p "$ROOT/logs"
rm -f "$ROOT"/logs/fr3_twc_*.out "$ROOT"/logs/fr3_twc_*.err 2>/dev/null || true
echo "Cleaned $ROOT/results_twc and old FR3-TWC slurm logs"
