#!/usr/bin/env bash
set -euo pipefail

# ONNXim batch-size sweep for weight-stationary amortization.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LENS="${SEQ_LENS:-8 16 32 64 128}"
OUT_ROOT="${OFA_DIR}/output/onnxim_graphbit"

cd "${OFA_DIR}"

WORKSPACES=()
for seq in ${SEQ_LENS}; do
  ws="${OUT_ROOT}/microbench_s${seq}_internal_p8"
  WORKSPACES+=("${ws}")
  if [[ ! -f "${ws}/aggregate.json" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${seq}" \
      --workspace "${ws}" \
      --graphbit-depth 8 \
      --action all \
      --log-level info
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${seq}" \
      --workspace "${ws}" \
      --graphbit-depth 8 \
      --action summarize
  fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_onnxim_batch_amortization.py" \
  --workspaces "${WORKSPACES[@]}"
