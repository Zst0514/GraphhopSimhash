#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-cora}" exec "${SCRIPT_DIR}/run_graphbit_predictor_free_flow.sh"
