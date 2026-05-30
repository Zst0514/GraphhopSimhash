#!/usr/bin/env bash
set -euo pipefail

# Cora h8_54_T40 Graph-Bit memory dataflow summary.
# Requires the Cora predictor-free summary and ONNXim early-stop aggregates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LEN="${SEQ_LEN:-64}"

cd "${OFA_DIR}"

if [[ ! -f "${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40/summary.tsv" ]]; then
  echo "[GraphBitMemoryDataflow] missing Cora summary; run predictor-free flow first." >&2
  exit 1
fi

if [[ ! -f "${OFA_DIR}/output/onnxim_graphbit/microbench_s${SEQ_LEN}_internal_bound_low_min4_t0p04/aggregate.json" ]]; then
  bash "${SCRIPT_DIR}/run_cora_graphbit_earlystop_sweep.sh"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_graphbit_memory_dataflow.py" \
  --seq-len "${SEQ_LEN}"
