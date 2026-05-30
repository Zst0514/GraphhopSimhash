#!/usr/bin/env bash
set -euo pipefail

# Run ONNXim component microbenchmarks for Graph-Bit risk buckets.
#
# Each component represents one graph-risk bucket:
#   P8: high-risk, min depth 8
#   P6: middle-risk, min depth 6
#   P5: lower-risk, min depth 4 with stricter tolerance
#   P4: lowest-risk, min depth 4 with looser tolerance
#
# The final full-stack table combines these components with real workload
# ratios exported from residual/reuse + Graph-Bit runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

SEQ_LEN="${SEQ_LEN:-8}"
LOG_LEVEL="${LOG_LEVEL:-info}"
BASELINE_TILE_BATCH="${BASELINE_TILE_BATCH:-16}"
STATIONARY_TILE_BATCHES="${STATIONARY_TILE_BATCHES:-32 64}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/onnxim_graphbit/risk_bucket_components_s${SEQ_LEN}}"

COMMON_ARGS=(
  --graphbit-depth 8
  --graphbit-bound-enable
  --graphbit-bound-mode tile_mean
  --graphbit-activation-layout plane_group
  --graphbit-plane-group-bits 2
  --graphbit-weight-rf-gate
  --graphbit-psum-gate
)

run_case() {
  local name="$1"
  shift
  local ws="${OUT_ROOT}/${name}"
  echo "[GraphBitRiskBucket] ${name} -> ${ws}"
  "${PYTHON}" "${REPO_DIR}/scripts/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --workspace "${ws}" \
    --action all \
    --log-level "${LOG_LEVEL}" \
    "$@"
}

run_bucket() {
  local bucket="$1"
  local min_depth="$2"
  local tolerance="$3"
  shift 3
  run_case "${bucket}_now" \
    "${COMMON_ARGS[@]}" \
    --graphbit-min-depth "${min_depth}" \
    --graphbit-bound-tolerance "${tolerance}"

  for batch in ${STATIONARY_TILE_BATCHES}; do
    run_case "${bucket}_ws_b${batch}" \
      "${COMMON_ARGS[@]}" \
      --graphbit-min-depth "${min_depth}" \
      --graphbit-bound-tolerance "${tolerance}" \
      --graphbit-weight-stationary \
      --graphbit-baseline-weight-tile-batch "${BASELINE_TILE_BATCH}" \
      --graphbit-weight-stationary-tile-batch "${batch}"
  done
}

mkdir -p "${OUT_ROOT}"

run_case "full_p8" --graphbit-depth 8

run_bucket "p8" 8 0.0
run_bucket "p6" 6 0.02
run_bucket "p5" 4 0.04
run_bucket "p4" 4 0.12

echo "[GraphBitRiskBucket] done: ${OUT_ROOT}"
