#!/usr/bin/env bash
set -euo pipefail

# Run the component-level Graph-Bit NPU dataflow model.
#
# Default workload:
#   Cora h8_54_T40_dynp5, which contains P8/P6/P5 depth anchors.
#
# The script does not rerun GNN or ONNXim.  It replays an existing workload and
# estimates how byte-major, bit-plane demand fetch, risk-bucket scheduling, and
# weight-stationary reuse change cycles/traffic/energy components.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

WORKLOAD="${WORKLOAD:-${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40_dynp5/predictor_free_workload.json}"
OUT_DIR="${OUT_DIR:-$(dirname "${WORKLOAD}")/npu_dataflow_model}"
NODE_COUNT="${NODE_COUNT:-2708}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PLANE_GROUP_BITS="${PLANE_GROUP_BITS:-2}"
BASELINE_WEIGHT_TILE_BATCH="${BASELINE_WEIGHT_TILE_BATCH:-16}"
# Keep the default conservative: Graph-Bit does not automatically reduce HBM
# weight reads.  A larger value is a sensitivity study for extra
# risk-bucket/weight-stationary batching, not the main claim.
WEIGHT_STATIONARY_TILE_BATCH="${WEIGHT_STATIONARY_TILE_BATCH:-16}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/model_graphbit_npu_dataflow.py" \
  --workload "${WORKLOAD}" \
  --output-dir "${OUT_DIR}" \
  --node-count "${NODE_COUNT}" \
  --batch-size "${BATCH_SIZE}" \
  --plane-group-bits "${PLANE_GROUP_BITS}" \
  --baseline-weight-tile-batch "${BASELINE_WEIGHT_TILE_BATCH}" \
  --weight-stationary-tile-batch "${WEIGHT_STATIONARY_TILE_BATCH}"
