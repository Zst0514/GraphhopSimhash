#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="$(cd "${REPO_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
MODEL_DIR="${MODEL_DIR:-${OFA_DIR}/models/llama-13b/modelscope/Llama-2-13b-ms}"
DATASETS=(${DATASETS:-cora pubmed arxiv wikics tape_products tape_arxiv23})
CONFIGS=(${CONFIGS:-W4A8 W4BFPA3_B256 W4BFPA4_B256 W4BFPA5_B256 W4BFPA6_B256})
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/llama13b_awq_bfp_pools/aligned_n128_s512}"
MAX_LENGTH="${MAX_LENGTH:-500}"
FORCE="${FORCE:-0}"

mkdir -p "${OUT_ROOT}/logs"
cd "${OFA_DIR}"

timestamp() {
  date +"%F %T"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

pool_path() {
  local dataset="$1"
  local tag="$2"
  "${PYTHON_BIN}" - <<PY
from GraphhopSimhash.real_quant import default_pool_path
print(default_pool_path("${dataset}", "llama2_13b", "${tag}"))
PY
}

log_msg "Aligned Llama2-13B BFPA-B256 pool generation starts"
log_msg "datasets=${DATASETS[*]}"
log_msg "configs=${CONFIGS[*]}"
log_msg "awq_calib_samples=128 awq_seqlen=512 q_group=128"
log_msg "model_dir=${MODEL_DIR}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    path="$(pool_path "${dataset}" "${config}")"
    if [[ -f "${path}" && "${FORCE}" != "1" ]]; then
      log_msg "[Skip] ${dataset}/llama2_13b/${config}: ${path}"
      continue
    fi

    log="${OUT_ROOT}/logs/${dataset}_${config}.log"
    log_msg "[Generate] dataset=${dataset} config=${config} -> ${path}"
    GRAPHHOP_LLAMA2_13B_PATH="${MODEL_DIR}" \
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
      --overwrite \
      2>&1 | tee "${log}"
  done
done

log_msg "Aligned Llama2-13B BFPA-B256 pool generation completed"
