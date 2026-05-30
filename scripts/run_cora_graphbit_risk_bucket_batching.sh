#!/usr/bin/env bash
set -euo pipefail

# Cora h8_54_T40 Graph-Bit risk-bucket batching summary.
#
# This script does not retrain the reuse front-end.  It reuses the existing
# residual + Graph-Bit summary and ONNXim microbench aggregates, then compares:
#   1. random mixed micro-batches
#   2. degree-risk bucketed micro-batches
#   3. degree-risk bucketed predictor-free early-stop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LEN="${SEQ_LEN:-64}"

cd "${OFA_DIR}"

SUMMARY="${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40/summary.tsv"
MICRO_ROOT="${OFA_DIR}/output/onnxim_graphbit"

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[GraphBitRiskBucket] missing ${SUMMARY}" >&2
  echo "Run: bash GraphhopSimhash/scripts/run_cora_graphbit_predictor_free_flow.sh" >&2
  exit 1
fi

if [[ ! -f "${MICRO_ROOT}/microbench_s${SEQ_LEN}_internal_bound_low_min4_t0p04/aggregate.json" ]]; then
  echo "[GraphBitRiskBucket] missing ONNXim early-stop aggregate; generating now." >&2
  bash "${SCRIPT_DIR}/run_cora_graphbit_earlystop_sweep.sh"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_graphbit_risk_bucket_batching.py" \
  --seq-len "${SEQ_LEN}" \
  --summary "${SUMMARY}" \
  --microbench-dir "${MICRO_ROOT}"
