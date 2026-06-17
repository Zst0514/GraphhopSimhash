#!/usr/bin/env bash
set -euo pipefail

# Encoder-backbone suite for GFMEngine-style comparisons.
#
# This script does two scoped jobs:
#   1. Generate FP16 target embedding pools for several frontend encoders.
#   2. Run the current SimHash/TSER/residual-reuse frontend on those pools.
#
# Large/gated models:
#   - llama2_13b needs either GRAPHHOP_LLAMA2_13B_PATH or
#     ALLOW_LLAMA2_13B_HF=1 with a working HuggingFace login/token.
#   - e5/e5_large refers to intfloat/e5-large-v2. Set
#     GRAPHHOP_E5_LARGE_PATH for an offline local copy, or ALLOW_REMOTE_MODELS=1.
#
# Typical usage:
#   DATASETS="cora pubmed" MODELS="BERT ST e5_large llama2_7b" RUNS=3 \
#     bash GraphhopSimhash/scripts/run_encoder_model_suite.sh
#
# 13B explicit run:
#   export GRAPHHOP_LLAMA2_13B_PATH=/path/to/Llama-2-13b-hf
#   DATASETS="cora" MODELS="llama2_13b" RUNS=1 \
#     bash GraphhopSimhash/scripts/run_encoder_model_suite.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

MODELS=(${MODELS:-BERT ST e5_large llama2_7b llama2_13b})
DATASETS=(${DATASETS:-cora pubmed arxiv wikics})
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-500}"
CACHE_DIR="${CACHE_DIR:-cache_data/model}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/encoder_model_suite}"
GENERATE_POOLS="${GENERATE_POOLS:-1}"
RUN_RESIDUAL="${RUN_RESIDUAL:-1}"
RUN_ALLFP_SANITY="${RUN_ALLFP_SANITY:-0}"
FORCE_POOL="${FORCE_POOL:-0}"
ALLOW_REMOTE_MODELS="${ALLOW_REMOTE_MODELS:-1}"
ALLOW_LLAMA2_13B_HF="${ALLOW_LLAMA2_13B_HF:-0}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/summaries"
cd "${OFA_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

canon_model() {
  case "${1}" in
    bert|BERT|bert-base|bert-base-uncased) echo "BERT" ;;
    st|ST|sentence-transformer|sentence_transformer|multi-qa-distilbert-cos-v1) echo "ST" ;;
    e5|E5|e5-large|e5-large-v2|e5_large|e5_large_v2) echo "e5_large" ;;
    llama2_7b|llama2-7b|llama-2-7b) echo "llama2_7b" ;;
    llama2_13b|llama2-13b|llama-2-13b) echo "llama2_13b" ;;
    *) echo "${1}" ;;
  esac
}

model_batch_size() {
  case "$(canon_model "$1")" in
    BERT|ST) echo "${BATCH_BERT_ST:-64}" ;;
    e5_large) echo "${BATCH_E5_LARGE:-8}" ;;
    llama2_7b) echo "${BATCH_LLAMA2_7B:-4}" ;;
    llama2_13b) echo "${BATCH_LLAMA2_13B:-1}" ;;
    *) echo "${BATCH_DEFAULT:-16}" ;;
  esac
}

frontend_t() {
  case "$1" in
    cora) echo "${CORA_T:-31}" ;;
    pubmed) echo "${PUBMED_T:-31}" ;;
    arxiv) echo "${ARXIV_T:-18}" ;;
    wikics) echo "${WIKICS_T:-31}" ;;
    tape_products) echo "${PRODUCTS_T:-31}" ;;
    tape_arxiv23) echo "${TAPE_ARXIV23_T:-18}" ;;
    *) echo "${FRONTEND_T:-31}" ;;
  esac
}

pool_path() {
  local dataset="$1"
  local model="$2"
  "${PYTHON_BIN}" - <<PY
from GraphhopSimhash.real_quant import default_pool_path
print(default_pool_path("${dataset}", "${model}", "FP16"))
PY
}

model_is_available() {
  local model
  model="$(canon_model "$1")"
  case "${model}" in
    BERT)
      [[ -n "${GRAPHHOP_BERT_PATH:-}" || -d "${OFA_DIR}/models/bert-base-uncased" ]]
      ;;
    ST)
      [[ -n "${GRAPHHOP_ST_PATH:-}" || -d "${OFA_DIR}/models/multi-qa-distilbert-cos-v1" ]]
      ;;
    e5_large)
      [[ -n "${GRAPHHOP_E5_LARGE_PATH:-}" || -d "${OFA_DIR}/models/e5-large-v2" || "${ALLOW_REMOTE_MODELS}" == "1" ]]
      ;;
    llama2_7b)
      [[ -n "${GRAPHHOP_LLAMA2_7B_PATH:-}" || -d "${OFA_DIR}/models/llama-7b/modelscope/Llama-2-7b-ms" || "${ALLOW_REMOTE_MODELS}" == "1" ]]
      ;;
    llama2_13b)
      [[ -n "${GRAPHHOP_LLAMA2_13B_PATH:-}" || -d "${OFA_DIR}/models/llama-13b" || "${ALLOW_LLAMA2_13B_HF}" == "1" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

prepare_model_env() {
  local model
  model="$(canon_model "$1")"
  case "${model}" in
    e5_large)
      if [[ -z "${GRAPHHOP_E5_LARGE_PATH:-}" && -d "${OFA_DIR}/models/e5-large-v2" ]]; then
        export GRAPHHOP_E5_LARGE_PATH="${OFA_DIR}/models/e5-large-v2"
      fi
      ;;
    llama2_13b)
      if [[ -z "${GRAPHHOP_LLAMA2_13B_PATH:-}" && -d "${OFA_DIR}/models/llama-13b" ]]; then
        export GRAPHHOP_LLAMA2_13B_PATH="${OFA_DIR}/models/llama-13b"
      fi
      ;;
  esac
}

skip_reason() {
  local model
  model="$(canon_model "$1")"
  case "${model}" in
    e5_large)
      echo "missing local e5-large-v2 path; set GRAPHHOP_E5_LARGE_PATH or ALLOW_REMOTE_MODELS=1"
      ;;
    llama2_13b)
      echo "missing 13B path; set GRAPHHOP_LLAMA2_13B_PATH or ALLOW_LLAMA2_13B_HF=1"
      ;;
    *)
      echo "model checkpoint not found"
      ;;
  esac
}

generate_pool_if_needed() {
  local dataset="$1"
  local model="$2"
  local path batch log_path
  path="$(pool_path "${dataset}" "${model}")"
  batch="$(model_batch_size "${model}")"
  log_path="${OUT_ROOT}/logs/generate_${dataset}_${model}_FP16.log"

  if [[ -f "${path}" && "${FORCE_POOL}" != "1" ]]; then
    log "[Pool] ${dataset}/${model}: exists ${path}"
    return 0
  fi
  if [[ "${GENERATE_POOLS}" != "1" ]]; then
    log "[Pool] ${dataset}/${model}: missing and GENERATE_POOLS=0"
    return 1
  fi

  local overwrite_args=()
  if [[ "${FORCE_POOL}" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  log "[Pool] generating ${dataset}/${model} FP16 | batch=${batch} | log=${log_path}"
  "${PYTHON_BIN}" -m GraphhopSimhash.generate_real_quant_pools \
    --datasets "${dataset}" \
    --llm_name "${model}" \
    --configs fp16 \
    --batch_size "${batch}" \
    --max_length "${MAX_LENGTH}" \
    --cache_dir "${CACHE_DIR}" \
    "${overwrite_args[@]}" \
    2>&1 | tee "${log_path}"
}

run_residual_eval() {
  local dataset="$1"
  local model="$2"
  local t log_path
  t="$(frontend_t "${dataset}")"
  log_path="${OUT_ROOT}/logs/residual_${dataset}_${model}_runs${RUNS}_T${t}.log"

  log "[Residual] ${dataset}/${model} | T=${t} | runs=${RUNS} | log=${log_path}"
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets "${dataset}" \
    --runs "${RUNS}" \
    --seed "${SEED}" \
    --experiment_suite residual_reuse \
    --residual_embedding_source real_quant_fp \
    --real_quant_model_name "${model}" \
    --real_quant_fp_tag FP16 \
    --residual_fit_profile auto \
    --learned_hash_epochs 10 \
    --learned_hash_dim 128 \
    --hash_heads_per_route 8 \
    --hamming_only_acceptor \
    --disable_structure_check \
    --enable_score_gate \
    --allow_rare_fuzzy \
    --score_reuse_threshold "${t}" \
    --score_propagation_weight 3 \
    --score_graph_context_weight 1 \
    --score_low_unique_weight 1 \
    --score_pair_confidence_discount 1 \
    --radius 2 \
    --main_hash_head_bits 16 16 16 16 16 16 16 16 \
    --residual_hard_min_support_hits 5 \
    --residual_soft_min_support_hits 3 \
    --residual_rank "${RESIDUAL_RANK:-64}" \
    --residual_epochs "${RESIDUAL_EPOCHS:-160}" \
    --residual_max_train_pairs "${RESIDUAL_MAX_TRAIN_PAIRS:-4096}" \
    --residual_min_dist 1.0 \
    --residual_alpha_grid 0 0.03125 0.0625 0.125 0.25 0.5 \
    --residual_support_aware_alpha \
    --residual_adapter_type mlp \
    --residual_dropout 0.05 \
    --residual_loss_cosine_weight 1.0 \
    --residual_loss_mse_weight 0.5 \
    --residual_loss_delta_weight 0.75 \
    --residual_bucket_mode support_dist \
    --residual_offline_extra_anchors_per_node 8 \
    --residual_offline_extra_query_nodes 4096 \
    --residual_train_split train_val \
    --residual_gate_loss_weight 0.5 \
    --residual_gate_error_scale 0.25 \
    --residual_gate_error_max 0.45 \
    --residual_accept_mode "${RESIDUAL_ACCEPT_MODE:-separate}" \
    --residual_positive_error_max "${RESIDUAL_POSITIVE_ERROR_MAX:-0.40}" \
    --residual_offline_negative_anchors_per_node "${RESIDUAL_NEGATIVE_ANCHORS:-4}" \
    --residual_negative_error_min "${RESIDUAL_NEGATIVE_ERROR_MIN:-0.45}" \
    --residual_negative_gate_weight "${RESIDUAL_NEGATIVE_GATE_WEIGHT:-1.0}" \
    --residual_accept_loss_weight "${RESIDUAL_ACCEPT_LOSS_WEIGHT:-1.0}" \
    --residual_gate_sparsity_weight "${RESIDUAL_GATE_SPARSITY_WEIGHT:-0.02}" \
    --residual_gate_accept_threshold "${RESIDUAL_GATE_ACCEPT_THRESHOLD:-0.5}" \
    2>&1 | tee "${log_path}"
}

run_allfp_sanity() {
  local dataset="$1"
  local model="$2"
  local t log_path
  t="$(frontend_t "${dataset}")"
  log_path="${OUT_ROOT}/logs/allfp_${dataset}_${model}_runs${RUNS}_T${t}.log"

  log "[AllFP sanity] ${dataset}/${model} | T=${t} | runs=${RUNS} | log=${log_path}"
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets "${dataset}" \
    --runs "${RUNS}" \
    --seed "${SEED}" \
    --experiment_suite reuse_real_quant \
    --disable_real_quant_autogen \
    --reuse_real_quant_allfp_only \
    --real_quant_model_name "${model}" \
    --real_quant_fp_tag FP16 \
    --learned_hash_epochs 10 \
    --learned_hash_dim 128 \
    --hash_heads_per_route 8 \
    --hamming_only_acceptor \
    --disable_structure_check \
    --enable_score_gate \
    --allow_rare_fuzzy \
    --score_reuse_threshold "${t}" \
    --score_propagation_weight 3 \
    --score_graph_context_weight 1 \
    --score_low_unique_weight 1 \
    --radius 2 \
    --main_hash_head_bits 16 16 16 16 16 16 16 16 \
    2>&1 | tee "${log_path}"
}

summarize_logs() {
  local summary="${OUT_ROOT}/summaries/model_suite_summary.txt"
  {
    echo "# Encoder Model Suite Summary"
    echo
    echo "Generated at: $(timestamp)"
    echo "Datasets: ${DATASETS[*]}"
    echo "Models: ${MODELS[*]}"
    echo "Runs: ${RUNS}"
    echo
    echo "## Residual reuse"
    rg -n "FINAL RESIDUAL REUSE SUMMARY|Baseline Acc:|DirectReuse|SoftDirectReuse|ResidualReuse" "${OUT_ROOT}/logs" || true
    echo
    echo "## AllFP sanity"
    rg -n "FINAL REUSE \\+ REAL QUANT SUMMARY|Baseline Acc:|AllFP" "${OUT_ROOT}/logs" || true
  } > "${summary}"
  log "[Summary] ${summary}"
}

log "Encoder model suite start | out=${OUT_ROOT}"
for raw_model in "${MODELS[@]}"; do
  model="$(canon_model "${raw_model}")"
  prepare_model_env "${model}"
  if ! model_is_available "${model}"; then
    log "[Skip] ${model}: $(skip_reason "${model}")"
    continue
  fi
  for dataset in "${DATASETS[@]}"; do
    if ! generate_pool_if_needed "${dataset}" "${model}"; then
      log "[Skip] ${dataset}/${model}: pool unavailable"
      continue
    fi
    if [[ "${RUN_RESIDUAL}" == "1" ]]; then
      run_residual_eval "${dataset}" "${model}"
    fi
    if [[ "${RUN_ALLFP_SANITY}" == "1" ]]; then
      run_allfp_sanity "${dataset}" "${model}"
    fi
  done
done

summarize_logs
log "Encoder model suite done"
