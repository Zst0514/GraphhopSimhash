#!/usr/bin/env bash
set -euo pipefail

# Cora-only hardware validation for predictor-free Graph-Bit early stop.
#
# Unlike the old static P8/P6/P4 proxy, every miss-node class starts from P8.
# The graph-risk class only selects min_depth/tolerance; ONNXim's internal
# bound logic decides the actual bit-plane depth.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LEN="${SEQ_LEN:-64}"
OUT_ROOT="${OFA_DIR}/output/onnxim_graphbit"
FORCE_ONNXIM="${FORCE_ONNXIM:-0}"

cd "${OFA_DIR}"

run_bound() {
  local name="$1"
  local min_depth="$2"
  local tolerance="$3"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --workspace "${OUT_ROOT}/microbench_s${SEQ_LEN}_internal_${name}" \
    --graphbit-depth 8 \
    --graphbit-min-depth "${min_depth}" \
    --graphbit-bound-enable \
    --graphbit-bound-tolerance "${tolerance}" \
    --action all \
    --log-level info
}

echo "[GraphBitEarlyStop] seq_len=${SEQ_LEN}"

if [[ "${FORCE_ONNXIM}" == "1" || ! -f "${OUT_ROOT}/microbench_s${SEQ_LEN}/aggregate.json" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --action all \
    --log-level info
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --action summarize
fi

for depth in 8 6 4; do
  if [[ "${FORCE_ONNXIM}" == "1" || ! -f "${OUT_ROOT}/microbench_s${SEQ_LEN}_internal_p${depth}/aggregate.json" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${SEQ_LEN}" \
      --workspace "${OUT_ROOT}/microbench_s${SEQ_LEN}_internal_p${depth}" \
      --graphbit-depth "${depth}" \
      --action all \
      --log-level info
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${SEQ_LEN}" \
      --workspace "${OUT_ROOT}/microbench_s${SEQ_LEN}_internal_p${depth}" \
      --graphbit-depth "${depth}" \
      --action summarize
  fi
done

run_bound "bound_mid_min6_t0p006" 6 0.006
run_bound "bound_mid_min6_t0p02" 6 0.02
run_bound "bound_mid_min6_t0p06" 6 0.06
run_bound "bound_low_min4_t0p02" 4 0.02
run_bound "bound_low_min4_t0p04" 4 0.04
run_bound "bound_low_min4_t0p06" 4 0.06

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_cora_graphbit_earlystop_sweep.py" \
  --seq-len "${SEQ_LEN}"
