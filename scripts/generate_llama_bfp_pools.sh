#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS=(${DATASETS:-cora})
CONFIGS=(${CONFIGS:-W4BFPA8_B128 W4BFPA7_B128 W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128})
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
AWQ_CALIB_SAMPLES="${AWQ_CALIB_SAMPLES:-128}"
AWQ_SEQLEN="${AWQ_SEQLEN:-512}"
OVERWRITE="${OVERWRITE:-0}"

cd "${OFA_DIR}"

args=(
  -m GraphhopSimhash.generate_real_quant_pools
  --datasets "${DATASETS[@]}"
  --llm_name llama2_7b
  --configs "${CONFIGS[@]}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --awq_calib_samples "${AWQ_CALIB_SAMPLES}"
  --awq_seqlen "${AWQ_SEQLEN}"
  --awq_q_group_size 128
)

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

echo "[GraphBFP] datasets=${DATASETS[*]} configs=${CONFIGS[*]} batch=${BATCH_SIZE} max_length=${MAX_LENGTH}"
"${PYTHON_BIN}" "${args[@]}"
