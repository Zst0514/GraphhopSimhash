#!/usr/bin/env bash
set -euo pipefail

# Regenerate the Cora Graph-Bit closure table.
#
# This is the quick evidence-chain check:
#   FullP8-miss
#   Degree compute-mask only
#   Degree random-mixed demand-fetch
#   Degree risk-bucket demand-fetch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

cd "${REPO_DIR}"

bash "${SCRIPT_DIR}/run_graphbit_demand_fetch_model.sh"

WORKLOAD="${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_workload.json" \
OUT_DIR="${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40/demand_fetch_model" \
bash "${SCRIPT_DIR}/run_graphbit_demand_fetch_model.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_graphbit_closure_suite.py" \
  --output-dir "${OFA_DIR}/output/graphbit_closure/cora"
