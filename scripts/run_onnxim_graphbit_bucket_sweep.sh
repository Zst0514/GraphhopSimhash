#!/usr/bin/env bash
set -euo pipefail

# Sweep Graph-Bit bucket/micro-batch size and weight-stationary reuse.
#
# seq_len models the real same-risk micro-batch size entering a LLaMA encoder
# GEMM.  stationary_tile_batch models an explicit scheduler/capacity assumption:
# how many same-risk node blocks can reuse one loaded W tile.  The two are kept
# separate so the conservative mainline and sensitivity points are not mixed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

SEQ_LENS="${SEQ_LENS:-8 16 32}"
BASELINE_TILE_BATCH="${BASELINE_TILE_BATCH:-16}"
STATIONARY_TILE_BATCHES="${STATIONARY_TILE_BATCHES:-16 32 64 128}"
LOG_LEVEL="${LOG_LEVEL:-info}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/onnxim_graphbit/bucket_sweep}"

COMMON_GRAPHBIT_ARGS=(
  --graphbit-depth 8
  --graphbit-min-depth 4
  --graphbit-bound-enable
  --graphbit-bound-mode tile_mean
  --graphbit-bound-tolerance 0.04
  --graphbit-activation-layout plane_group
  --graphbit-plane-group-bits 2
  --graphbit-weight-rf-gate
  --graphbit-psum-gate
)

run_case() {
  local seq="$1"
  local name="$2"
  shift 2
  local ws="${OUT_ROOT}/s${seq}/${name}"
  echo "[GraphBitBucketSweep] seq=${seq} case=${name} -> ${ws}"
  "${PYTHON}" "${REPO_DIR}/scripts/onnxim_graphbit_microbench.py" \
    --seq-len "${seq}" \
    --workspace "${ws}" \
    --action all \
    --log-level "${LOG_LEVEL}" \
    "$@"
}

mkdir -p "${OUT_ROOT}"

CASES=("gb_now")
for batch in ${STATIONARY_TILE_BATCHES}; do
  CASES+=("gb_ws_b${batch}")
done

for seq in ${SEQ_LENS}; do
  run_case "${seq}" "full_p8" --graphbit-depth 8

  run_case "${seq}" "gb_now" "${COMMON_GRAPHBIT_ARGS[@]}"

  for batch in ${STATIONARY_TILE_BATCHES}; do
    run_case "${seq}" "gb_ws_b${batch}" \
      "${COMMON_GRAPHBIT_ARGS[@]}" \
      --graphbit-weight-stationary \
      --graphbit-baseline-weight-tile-batch "${BASELINE_TILE_BATCH}" \
      --graphbit-weight-stationary-tile-batch "${batch}"
  done
done

"${PYTHON}" "${REPO_DIR}/scripts/summarize_onnxim_graphbit_bucket_sweep.py" \
  --root "${OUT_ROOT}" \
  --seq-lens ${SEQ_LENS} \
  --cases "${CASES[@]}"

echo "[GraphBitBucketSweep] done: ${OUT_ROOT}"
