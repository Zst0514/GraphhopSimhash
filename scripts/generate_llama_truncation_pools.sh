#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS=(${DATASETS:-cora})
CONFIGS=(${CONFIGS:-W4A8_TRUNC7 W4A8_TRUNC6 W4A8_TRUNC5 W4A8_TRUNC4})
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
AWQ_CALIB_SAMPLES="${AWQ_CALIB_SAMPLES:-128}"
AWQ_SEQLEN="${AWQ_SEQLEN:-512}"
OVERWRITE="${OVERWRITE:-0}"

args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

cd "${OFA_DIR}"
export PYTHONPATH="${OFA_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
  --datasets "${DATASETS[@]}" \
  --llm_name llama2_7b \
  --configs "${CONFIGS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --awq_calib_samples "${AWQ_CALIB_SAMPLES}" \
  --awq_seqlen "${AWQ_SEQLEN}" \
  "${args[@]}"
