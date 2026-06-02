#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

LLM_NAME="${LLM_NAME:-llama2_7b}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
AWQ_CALIB_SAMPLES="${AWQ_CALIB_SAMPLES:-128}"
AWQ_SEQLEN="${AWQ_SEQLEN:-512}"
AWQ_Q_GROUP_SIZE="${AWQ_Q_GROUP_SIZE:-128}"
OVERWRITE="${OVERWRITE:-0}"

# Defaults are chosen for the current cache state:
#   arxiv already has BFPA8/BFPA4 and needs BFPA7/BFPA6/BFPA5.
#   pubmed already has BFPA8/BFPA6/BFPA5/BFPA4 and needs BFPA7.
# The script still checks cache_data and skips existing pools unless OVERWRITE=1.
DATASETS=(${DATASETS:-arxiv pubmed})
ARXIV_CONFIGS=(${ARXIV_CONFIGS:-W4BFPA8_B128 W4BFPA7_B128 W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128})
PUBMED_CONFIGS=(${PUBMED_CONFIGS:-W4BFPA8_B128 W4BFPA7_B128 W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128})

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/graphbfp_pools/missing_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}"

cd "${OFA_DIR}"

configs_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    arxiv)
      printf "%s\n" "${ARXIV_CONFIGS[@]}"
      ;;
    pubmed)
      printf "%s\n" "${PUBMED_CONFIGS[@]}"
      ;;
    *)
      echo "Unsupported dataset for this script: ${dataset}" >&2
      return 2
      ;;
  esac
}

missing_configs_for_dataset() {
  local dataset="$1"
  local cfg
  while IFS= read -r cfg; do
    [[ -z "${cfg}" ]] && continue
    local path="cache_data/${dataset}_${LLM_NAME}_oracle_${cfg}.pt"
    if [[ "${OVERWRITE}" == "1" || ! -s "${path}" ]]; then
      printf "%s\n" "${cfg}"
    else
      echo "[Skip] ${path} exists" >&2
    fi
  done < <(configs_for_dataset "${dataset}")
}

for dataset in "${DATASETS[@]}"; do
  mapfile -t missing < <(missing_configs_for_dataset "${dataset}")
  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "[Done] ${dataset}: no missing BFP pools."
    continue
  fi

  log_path="${OUT_DIR}/${dataset}_${LLM_NAME}_${missing[*]// /_}.log"
  echo
  echo "========================================================================"
  echo "[GraphBFP] dataset=${dataset}"
  echo "[GraphBFP] missing configs=${missing[*]}"
  echo "[GraphBFP] log=${log_path}"
  echo "========================================================================"

  args=(
    -m GraphhopSimhash.generate_real_quant_pools
    --datasets "${dataset}"
    --llm_name "${LLM_NAME}"
    --configs "${missing[@]}"
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
    echo "[GraphBFP] OFA_DIR=${OFA_DIR}"
    echo "[GraphBFP] python=${PYTHON_BIN}"
    echo "[GraphBFP] dataset=${dataset}"
    echo "[GraphBFP] llm=${LLM_NAME}"
    echo "[GraphBFP] configs=${missing[*]}"
    echo "[GraphBFP] batch_size=${BATCH_SIZE} max_length=${MAX_LENGTH}"
    echo "[GraphBFP] awq_calib_samples=${AWQ_CALIB_SAMPLES} awq_seqlen=${AWQ_SEQLEN} awq_q_group_size=${AWQ_Q_GROUP_SIZE}"
    echo "[GraphBFP] overwrite=${OVERWRITE}"
    "${PYTHON_BIN}" "${args[@]}"
  } 2>&1 | tee "${log_path}"
done

echo
echo "[GraphBFP] logs written to ${OUT_DIR}"
