#!/usr/bin/env bash
set -euo pipefail

# Run the unified Graph-Bit bit-plane demand-fetch model.
#
# By default this uses the latest Cora/LLaMA learned-gate smoke profile:
#   output/graphbit_predictor_free/cora_h8_53_T30/predictor_free_workload.json
#
# Override examples:
#   WORKLOAD=output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_workload.json \
#     bash scripts/run_graphbit_demand_fetch_model.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

WORKLOAD="${WORKLOAD:-${OFA_DIR}/output/graphbit_predictor_free/cora_h8_53_T30/predictor_free_workload.json}"
MICROBENCH_DIR="${MICROBENCH_DIR:-${OFA_DIR}/output/onnxim_graphbit}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NODE_COUNT="${NODE_COUNT:-2708}"
OUT_DIR="${OUT_DIR:-$(dirname "${WORKLOAD}")/demand_fetch_model}"

cd "${OFA_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/model_graphbit_demand_fetch.py" \
  --workload "${WORKLOAD}" \
  --microbench-dir "${MICROBENCH_DIR}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --node-count "${NODE_COUNT}" \
  --output-dir "${OUT_DIR}"
