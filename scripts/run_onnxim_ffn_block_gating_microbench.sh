#!/usr/bin/env bash
set -euo pipefail

# Hardware-only FFN block-gating probe for LLaMA-7B encoder GEMMs.
#
# This changes the FFN intermediate dimension in the ONNXim shape-carrier
# graphs.  It does not claim accuracy; it measures whether reducing FFN blocks
# would actually reduce cycles/traffic/weight reads.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LEN="${SEQ_LEN:-64}"
BASE_INTERMEDIATE="${BASE_INTERMEDIATE:-11008}"
KEEP_RATIOS="${KEEP_RATIOS:-0.75 0.50}"
OUT_ROOT="${OFA_DIR}/output/onnxim_graphbit"

cd "${OFA_DIR}"

BASE_WS="${OUT_ROOT}/microbench_s${SEQ_LEN}_internal_p8"
if [[ ! -f "${BASE_WS}/aggregate.json" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --workspace "${BASE_WS}" \
    --graphbit-depth 8 \
    --action all \
    --log-level info
fi

WORKSPACES=()
for ratio in ${KEEP_RATIOS}; do
  interm="$("${PYTHON_BIN}" - "${BASE_INTERMEDIATE}" "${ratio}" <<'PY'
import sys
base = int(sys.argv[1])
ratio = float(sys.argv[2])
value = int(round(base * ratio))
# Keep dimensions friendly to common channel-block granularities.
block = 128
value = max(block, int(round(value / block)) * block)
print(value)
PY
)"
  tag="$(printf "%s" "${ratio}" | sed 's/\./p/g')"
  ws="${OUT_ROOT}/ffn_block_s${SEQ_LEN}_keep${tag}"
  WORKSPACES+=("${ws}")
  if [[ ! -f "${ws}/aggregate.json" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${SEQ_LEN}" \
      --intermediate "${interm}" \
      --workspace "${ws}" \
      --graphbit-depth 8 \
      --action all \
      --log-level info
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${SEQ_LEN}" \
      --intermediate "${interm}" \
      --workspace "${ws}" \
      --graphbit-depth 8 \
      --action summarize
  fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_onnxim_ffn_block_gating.py" \
  --baseline "${BASE_WS}" \
  --workspaces "${WORKSPACES[@]}"
