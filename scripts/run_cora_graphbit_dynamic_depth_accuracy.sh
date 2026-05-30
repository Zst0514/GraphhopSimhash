#!/usr/bin/env bash
set -euo pipefail

# Conservative dynamic-depth accuracy check for Cora.
#
# Motivation:
#   predictor-free early stop may stop low-risk nodes around depth 5 rather
#   than the older static P4 floor.  This script reruns the residual+Graph-Bit
#   software path with W4A5 available, so the low-risk bucket can be mapped to
#   P5 instead of P4:
#
#   miss high-risk 20% -> P8
#   miss mid-risk  50% -> P6
#   miss low-risk  30% -> P5
#   miss rest       0% -> P4
#
# This is still a conservative proxy, not a true numerical bit-serial dynamic
# embedding.  It answers: "if early stop lands near P5, does downstream accuracy
# improve compared with the old P4 low bucket?"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LEN="${SEQ_LEN:-64}"

P5_AGG="${OFA_DIR}/output/onnxim_graphbit/microbench_s${SEQ_LEN}_internal_p5/aggregate.json"
if [[ ! -f "${P5_AGG}" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --workspace "${OFA_DIR}/output/onnxim_graphbit/microbench_s${SEQ_LEN}_internal_p5" \
    --graphbit-depth 5 \
    --action all \
    --log-level info
fi

DATASET=cora \
RUNS="${RUNS:-3}" \
THRESHOLD=40 \
BUDGET=dynp5 \
FRONTEND_ID=h8_54_T40_dynp5 \
OUT_DIR="${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40_dynp5" \
HARD_SUPPORT=5 \
SOFT_SUPPORT=4 \
HIGH_RATIO=0.20 \
MID_RATIO=0.50 \
LOW_RATIO=0.30 \
PRECISION_DEPTH_TAGS="W4A6 W4A5 W4A4" \
PRECISION_DEPTH_BITS="6 5 4" \
RUN_ONNXIM=0 \
bash "${SCRIPT_DIR}/run_graphbit_predictor_free_flow.sh"

WORKLOAD="${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40_dynp5/predictor_free_workload.json" \
OUT_DIR="${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40_dynp5/demand_fetch_model" \
bash "${SCRIPT_DIR}/run_graphbit_demand_fetch_model.sh"
