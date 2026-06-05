#!/usr/bin/env bash
set -euo pipefail

# Sweep the BFPA6 lift ratio for the Cora/LLaMA progressive BFP full-stack path.
# The front-end is fixed to h8_53_T31.  The miss-node back-end uses BFPA4 as the
# base path and lifts the selected ratio to BFPA6.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_progressive_bfp_fullstack.sh"

RATIOS=(${RATIOS:-0.05 0.10 0.15 0.20 0.25})
RUNS="${RUNS:-10}"
DATASET="${DATASET:-cora}"

for ratio in "${RATIOS[@]}"; do
  echo "================================================================"
  echo "[ProgressiveBFP] dataset=${DATASET} runs=${RUNS} BFPA6 lift ratio=${ratio}"
  echo "================================================================"
  DATASET="${DATASET}" \
  RUNS="${RUNS}" \
  REFINE_BIT=6 \
  REFINE_RATIO="${ratio}" \
  BUDGET_PRIORITIES="random degree tser" \
  FORCE="${FORCE:-0}" \
    bash "${RUN_SCRIPT}"
done
