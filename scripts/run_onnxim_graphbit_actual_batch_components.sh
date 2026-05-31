#!/usr/bin/env bash
set -euo pipefail

# Generate ONNXim Graph-Bit component costs with real GEMM M sizes.
#
# This is the stricter alternative to the older ws_b32/ws_b64 proxy.  Instead
# of shrinking weight traffic with a config knob, it runs actual ONNX GEMMs with
# M = 1/2/4/8/16/32/64 so ONNXim sees the true batch shape.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

BATCHES="${BATCHES:-1 2 4 8 16 32 64}"
LOG_LEVEL="${LOG_LEVEL:-info}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/onnxim_graphbit/actual_batch_components}"

COMMON_ARGS=(
  --graphbit-depth 8
  --graphbit-bound-enable
  --graphbit-bound-mode tile_mean
  --graphbit-activation-layout plane_group
  --graphbit-plane-group-bits 2
  --graphbit-weight-rf-gate
  --graphbit-psum-gate
)

run_depth() {
  local batch="$1"
  local depth_name="$2"
  local min_depth="$3"
  local tolerance="$4"
  local ws="${OUT_ROOT}/m${batch}_${depth_name}"
  echo "[GraphBitActualBatch] M=${batch} ${depth_name} -> ${ws}"
  "${PYTHON}" "${REPO_DIR}/scripts/onnxim_graphbit_microbench.py" \
    --seq-len "${batch}" \
    --workspace "${ws}" \
    --action all \
    --log-level "${LOG_LEVEL}" \
    "${COMMON_ARGS[@]}" \
    --graphbit-min-depth "${min_depth}" \
    --graphbit-bound-tolerance "${tolerance}"
}

mkdir -p "${OUT_ROOT}"

for batch in ${BATCHES}; do
  run_depth "${batch}" "p8" 8 0.0
  run_depth "${batch}" "p6" 6 0.02
  run_depth "${batch}" "p5" 4 0.04
  run_depth "${batch}" "p4" 4 0.12
done

echo "[GraphBitActualBatch] done: ${OUT_ROOT}"
