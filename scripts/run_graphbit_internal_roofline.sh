#!/usr/bin/env bash
set -euo pipefail

# Run the analytical NPU-internal Graph-Bit roofline/activity model.
# This does not run GNN accuracy.  It isolates QKV/O and FFN GEMMs with
# M = batch_nodes * padded_sequence_length.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

BATCH_NODES=(${BATCH_NODES:-1 2 4 8 16 32})
SEQ_LENS=(${SEQ_LENS:-128 256 512})
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/graphbit_internal_roofline}"

cd "${OFA_DIR}"

"${PYTHON_BIN}" "${REPO_DIR}/scripts/model_graphbit_internal_roofline.py" \
  --batch-nodes "${BATCH_NODES[@]}" \
  --seq-lens "${SEQ_LENS[@]}" \
  --output-dir "${OUT_ROOT}/default"

echo "[GraphBitInternalRoofline] report:"
echo "  ${OUT_ROOT}/default/graphbit_internal_roofline.txt"
