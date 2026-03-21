#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf scripts/__pycache__ src/__pycache__ src/fr3_sim/__pycache__ src/fr3_twc/__pycache__
find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
find . -maxdepth 1 -type f -name "fr3_twc_*.zip" -delete
