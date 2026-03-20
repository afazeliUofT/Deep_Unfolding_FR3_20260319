#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "$ROOT/results_twc"
mkdir -p "$ROOT/results_twc"
echo "Cleaned $ROOT/results_twc"
