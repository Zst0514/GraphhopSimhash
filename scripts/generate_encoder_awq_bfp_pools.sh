#!/usr/bin/env bash
set -euo pipefail

# Generate AWQ W4A8 and BFP-B256 embedding pools for non-LLaMA encoder
# backbones used in the paper ablations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

DATASETS=(${DATASETS:-cora pubmed arxiv wikics tape_products tape_arxiv23})
MODELS=(${MODELS:-ST e5_large})
CONFIGS=(${CONFIGS:-W4A8 W4BFPA3_B256 W4BFPA4_B256 W4BFPA5_B256 W4BFPA6_B256})
MAX_LENGTH="${MAX_LENGTH:-500}"
CACHE_DIR="${CACHE_DIR:-cache_data/model}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/encoder_awq_bfp_pools}"
FORCE="${FORCE:-0}"
WAIT_FOR_EXISTING="${WAIT_FOR_EXISTING:-1}"

mkdir -p "${OUT_ROOT}/logs"
cd "${OFA_DIR}"

timestamp() {
  date +"%F %T"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

canon_model() {
  case "$1" in
    st|ST|sentence-transformer|sentence_transformer|multi-qa-distilbert-cos-v1) echo "ST" ;;
    e5|E5) echo "e5_large" ;;
    e5-large|e5-large-v2|e5_large|e5_large_v2) echo "e5_large" ;;
    *) echo "$1" ;;
  esac
}

batch_size() {
  case "$(canon_model "$1")" in
    ST) echo "${BATCH_ST:-64}" ;;
    e5_large) echo "${BATCH_E5_LARGE:-8}" ;;
    *) echo "${BATCH_DEFAULT:-16}" ;;
  esac
}

awq_calib_samples() {
  case "$(canon_model "$1")" in
    ST) echo "${AWQ_CALIB_SAMPLES_ST:-16}" ;;
    e5_large) echo "${AWQ_CALIB_SAMPLES_E5_LARGE:-64}" ;;
    *) echo "${AWQ_CALIB_SAMPLES:-64}" ;;
  esac
}

awq_seqlen() {
  case "$(canon_model "$1")" in
    ST) echo "${AWQ_SEQLEN_ST:-128}" ;;
    *) echo "${AWQ_SEQLEN:-512}" ;;
  esac
}

pool_path() {
  local dataset="$1"
  local model="$2"
  local tag="$3"
  "${PYTHON_BIN}" - <<PY
from GraphhopSimhash.real_quant import default_pool_path
print(default_pool_path("${dataset}", "${model}", "${tag}"))
PY
}

wait_for_generators() {
  [[ "${WAIT_FOR_EXISTING}" == "1" ]] || return 0
  while true; do
    local procs
    procs="$(
      ps -eo pid=,args= \
        | awk '/python/ && /-m GraphhopSimhash.generate_real_quant_pools/ {print}' \
        | grep -v "awk " \
        || true
    )"
    if [[ -z "${procs}" ]]; then
      return 0
    fi
    log_msg "[Wait] existing generate_real_quant_pools process is running; sleep 10 min"
    echo "${procs}" | sed 's/^/[Wait] /'
    sleep 600
  done
}

generate_one() {
  local dataset="$1"
  local model_raw="$2"
  local config="$3"
  local model
  model="$(canon_model "${model_raw}")"

  local path
  path="$(pool_path "${dataset}" "${model}" "${config}")"
  if [[ -f "${path}" && "${FORCE}" != "1" ]]; then
    log_msg "[Skip] ${dataset}/${model}/${config}: ${path}"
    return 0
  fi

  wait_for_generators

  local batch calib seqlen log_path
  batch="$(batch_size "${model}")"
  calib="$(awq_calib_samples "${model}")"
  seqlen="$(awq_seqlen "${model}")"
  log_path="${OUT_ROOT}/logs/${dataset}_${model}_${config}.log"

  local overwrite_args=()
  if [[ "${FORCE}" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  log_msg "[Generate] dataset=${dataset} model=${model} config=${config} batch=${batch} calib=${calib} seqlen=${seqlen}"
  "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
    --datasets "${dataset}" \
    --llm_name "${model}" \
    --configs "${config}" \
    --batch_size "${batch}" \
    --max_length "${MAX_LENGTH}" \
    --cache_dir "${CACHE_DIR}" \
    --awq_calib_samples "${calib}" \
    --awq_seqlen "${seqlen}" \
    --awq_q_group_size 128 \
    "${overwrite_args[@]}" \
    2>&1 | tee "${log_path}"
}

log_msg "Encoder AWQ/BFP pool generation starts"
log_msg "datasets=${DATASETS[*]}"
log_msg "models=${MODELS[*]}"
log_msg "configs=${CONFIGS[*]}"
log_msg "HF_ENDPOINT=${HF_ENDPOINT}"

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      generate_one "${dataset}" "${model}" "${config}"
    done
  done
done

log_msg "Encoder AWQ/BFP pool generation completed"
