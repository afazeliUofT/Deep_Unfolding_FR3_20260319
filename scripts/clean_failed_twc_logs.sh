#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -f logs/fr3_twc_du_soft-*.out
rm -f logs/fr3_twc_du_cognitive-*.out
