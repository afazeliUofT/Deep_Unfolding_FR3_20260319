#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"
rm -f "$ROOT"/logs/fr3_twc_du_soft-*.out "$ROOT"/logs/fr3_twc_du_cognitive-*.out
rm -f "$ROOT"/logs/probe_train_soft-*.out "$ROOT"/logs/probe_train_cognitive-*.out
printf 'REMOVED_FAILED_TRAIN_LOGS %s\n' "$ROOT/logs"
