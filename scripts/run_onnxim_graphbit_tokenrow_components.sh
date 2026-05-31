#!/usr/bin/env bash
set -euo pipefail

# Generate ONNXim component lookups for realistic Transformer token-row GEMMs.
#
# In LLM encoder Linear layers, the GEMM row dimension is:
#   M = node_batch * padded_sequence_length
#
# This script replaces the older tiny-M component lookup with token-row-scale
# components such as M=2048, 4096, 8192.  The output can be passed to
# replay_graphbit_trace_scheduler.py as --components-root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

M_VALUES=(${M_VALUES:-2048})
LOG_LEVEL="${LOG_LEVEL:-info}"
BASELINE_TILE_BATCH="${BASELINE_TILE_BATCH:-16}"
STATIONARY_TILE_BATCHES="${STATIONARY_TILE_BATCHES:-32 64}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/onnxim_graphbit/tokenrow_components}"

COMMON_ARGS=(
  --graphbit-depth 8
  --graphbit-bound-enable
  --graphbit-bound-mode tile_mean
  --graphbit-activation-layout byte_major
  --graphbit-weight-rf-gate
  --graphbit-psum-gate
)

run_case() {
  local m="$1"
  local name="$2"
  shift 2
  local ws="${OUT_ROOT}/m${m}/${name}"
  echo "[GraphBitTokenRows] M=${m} ${name} -> ${ws}"
  "${PYTHON}" "${REPO_DIR}/scripts/onnxim_graphbit_microbench.py" \
    --seq-len "${m}" \
    --workspace "${ws}" \
    --action all \
    --log-level "${LOG_LEVEL}" \
    "$@"
}

run_bucket() {
  local m="$1"
  local bucket="$2"
  local min_depth="$3"
  local tolerance="$4"
  run_case "${m}" "${bucket}_now" \
    "${COMMON_ARGS[@]}" \
    --graphbit-min-depth "${min_depth}" \
    --graphbit-bound-tolerance "${tolerance}"

  for batch in ${STATIONARY_TILE_BATCHES}; do
    run_case "${m}" "${bucket}_ws_b${batch}" \
      "${COMMON_ARGS[@]}" \
      --graphbit-min-depth "${min_depth}" \
      --graphbit-bound-tolerance "${tolerance}" \
      --graphbit-weight-stationary \
      --graphbit-baseline-weight-tile-batch "${BASELINE_TILE_BATCH}" \
      --graphbit-weight-stationary-tile-batch "${batch}"
  done
}

mkdir -p "${OUT_ROOT}"

for m in "${M_VALUES[@]}"; do
  run_case "${m}" "full_p8" --graphbit-depth 8 --graphbit-activation-layout byte_major
  run_bucket "${m}" "p8" 8 0.0
  run_bucket "${m}" "p6" 6 0.02
  run_bucket "${m}" "p5" 4 0.04
  run_bucket "${m}" "p4" 4 0.12
done

echo "[GraphBitTokenRows] done: ${OUT_ROOT}"
