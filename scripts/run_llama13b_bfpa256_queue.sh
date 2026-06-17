#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="$(cd "${REPO_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
MODEL_DIR="${MODEL_DIR:-${OFA_DIR}/models/llama-13b/modelscope/Llama-2-13b-ms}"
DATASETS=(${DATASETS:-cora pubmed arxiv wikics tape_products tape_arxiv23})
CONFIGS=(${CONFIGS:-W4BFPA3_B256 W4BFPA4_B256 W4BFPA5_B256 W4BFPA6_B256})
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/llama13b_awq_bfp_pools/aligned_n128_s512_queue}"
MAX_LENGTH="${MAX_LENGTH:-500}"
FORCE="${FORCE:-0}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/status"
cd "${OFA_DIR}"

timestamp() {
  date +"%F %T"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

pool_path() {
  local dataset="$1"
  local config="$2"
  echo "${OFA_DIR}/cache_data/${dataset}_llama2_13b_oracle_${config}.pt"
}

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] Llama2-13B model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

log_msg "Llama2-13B BFPA-B256 queue starts"
log_msg "datasets=${DATASETS[*]}"
log_msg "configs=${CONFIGS[*]}"
log_msg "awq_calib_samples=128 awq_seqlen=512 q_group=128"
log_msg "model_dir=${MODEL_DIR}"
log_msg "out_root=${OUT_ROOT}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    path="$(pool_path "${dataset}" "${config}")"
    status_prefix="${OUT_ROOT}/status/${dataset}_${config}"
    log="${OUT_ROOT}/logs/${dataset}_${config}.log"

    if [[ -s "${path}" && "${FORCE}" != "1" ]]; then
      log_msg "[Skip] ${dataset}/${config}: ${path}"
      echo "$(timestamp) skip ${path}" > "${status_prefix}.done"
      continue
    fi

    rm -f "${status_prefix}.done" "${status_prefix}.failed"
    log_msg "[Generate] dataset=${dataset} config=${config}"
    log_msg "          log=${log}"

    overwrite_args=()
    if [[ "${FORCE}" == "1" ]]; then
      overwrite_args+=(--overwrite)
    fi

    if GRAPHHOP_LLAMA2_13B_PATH="${MODEL_DIR}" \
      "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
        --datasets "${dataset}" \
        --llm_name llama2_13b \
        --configs "${config}" \
        --batch_size 1 \
        --max_length "${MAX_LENGTH}" \
        --cache_dir cache_data/model \
        --awq_calib_samples 128 \
        --awq_seqlen 512 \
        --awq_q_group_size 128 \
        "${overwrite_args[@]}" \
        > "${log}" 2>&1; then
      echo "$(timestamp) done ${path}" > "${status_prefix}.done"
      log_msg "[Done] ${dataset}/${config}: ${path}"
    else
      echo "$(timestamp) failed ${log}" > "${status_prefix}.failed"
      log_msg "[Failed] ${dataset}/${config}; see ${log}"
      exit 1
    fi
  done
done

log_msg "Llama2-13B BFPA-B256 queue completed"
