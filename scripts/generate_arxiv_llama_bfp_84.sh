#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASET="${DATASET:-arxiv}"
LLM_NAME="${LLM_NAME:-llama2_7b}"
CONFIGS=(${CONFIGS:-W4BFPA8_B128 W4BFPA4_B128})
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
AWQ_CALIB_SAMPLES="${AWQ_CALIB_SAMPLES:-128}"
AWQ_SEQLEN="${AWQ_SEQLEN:-512}"
AWQ_Q_GROUP_SIZE="${AWQ_Q_GROUP_SIZE:-128}"
OVERWRITE="${OVERWRITE:-0}"

OUT_DIR="${OFA_DIR}/output/graphbfp_pools/${DATASET}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/${DATASET}_${LLM_NAME}_bfp_84_B128_generate_$(date +%Y%m%d_%H%M%S).log"

cd "${OFA_DIR}"

args=(
  -m GraphhopSimhash.generate_real_quant_pools
  --datasets "${DATASET}"
  --llm_name "${LLM_NAME}"
  --configs "${CONFIGS[@]}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --awq_calib_samples "${AWQ_CALIB_SAMPLES}"
  --awq_seqlen "${AWQ_SEQLEN}"
  --awq_q_group_size "${AWQ_Q_GROUP_SIZE}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

{
  echo "[GraphBFP] dataset=${DATASET}"
  echo "[GraphBFP] llm=${LLM_NAME}"
  echo "[GraphBFP] configs=${CONFIGS[*]}"
  echo "[GraphBFP] batch_size=${BATCH_SIZE} max_length=${MAX_LENGTH}"
  echo "[GraphBFP] awq_calib_samples=${AWQ_CALIB_SAMPLES} awq_seqlen=${AWQ_SEQLEN} awq_q_group_size=${AWQ_Q_GROUP_SIZE}"
  echo "[GraphBFP] overwrite=${OVERWRITE}"
  echo "[GraphBFP] log=${LOG}"
  "${PYTHON_BIN}" "${args[@]}"
} 2>&1 | tee "${LOG}"
