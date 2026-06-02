#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"

DATASET="${DATASET:-cora}"
FRONTEND_ID="${FRONTEND_ID:-h8_53_T31_bfp128}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/graphbfp_predictor_free/${DATASET}_${FRONTEND_ID}}"

export DATASET
export FRONTEND_ID
export OUT_DIR
export PRECISION_DEPTH_REFERENCE_TAG="${PRECISION_DEPTH_REFERENCE_TAG:-W4BFPA8_B128}"
export PRECISION_DEPTH_TAGS="${PRECISION_DEPTH_TAGS:-W4BFPA7_B128 W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128}"
export PRECISION_DEPTH_BITS="${PRECISION_DEPTH_BITS:-7 6 5 4}"

# Keep the shared T31 residual-gate front-end unless the caller overrides it.
export THRESHOLD="${THRESHOLD:-31}"
export HARD_SUPPORT="${HARD_SUPPORT:-5}"
export SOFT_SUPPORT="${SOFT_SUPPORT:-3}"
export RUNS="${RUNS:-3}"
export BUDGET="${BUDGET:-p8heavy}"
export RUN_ALGO="${RUN_ALGO:-1}"
export TRACE_EXPORT="${TRACE_EXPORT:-1}"

echo "[GraphBFP] dataset=${DATASET} out=${OUT_DIR}"
echo "[GraphBFP] reference=${PRECISION_DEPTH_REFERENCE_TAG} tags=${PRECISION_DEPTH_TAGS} bits=${PRECISION_DEPTH_BITS}"
bash "${SCRIPT_DIR}/run_graphbit_predictor_free_flow.sh"
