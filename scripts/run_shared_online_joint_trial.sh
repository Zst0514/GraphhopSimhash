#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/qiumingzhi/miniconda3/envs/research3/bin/python}"

RUNS="${RUNS:-1}"
SEED="${SEED:-42}"
SHARED_SCORE_T="${SHARED_SCORE_T:-31}"
SHARED_HARD_HITS="${SHARED_HARD_HITS:-4}"
SHARED_SOFT_HITS="${SHARED_SOFT_HITS:-3}"
SHARED_TAU="${SHARED_TAU:-0.65}"
SHARED_ACCEPT_MODE="${SHARED_ACCEPT_MODE:-shared}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-200}"
RESIDUAL_MAX_TRAIN_PAIRS="${RESIDUAL_MAX_TRAIN_PAIRS:-4096}"

ST_CORA_PATH="${ST_CORA_PATH:-cache_data/cora_ST_oracle_W4A16.pt}"
ST_PUBMED_PATH="${ST_PUBMED_PATH:-cache_data/pubmed_ST_oracle_W4A16.pt}"
LLAMA_FP_TAG="${LLAMA_FP_TAG:-W4A16}"
CASES=(${CASES:-st_cora st_pubmed llama_cora llama_pubmed})

TAG="h${SHARED_HARD_HITS}s${SHARED_SOFT_HITS}_T${SHARED_SCORE_T}_tau${SHARED_TAU}_${SHARED_ACCEPT_MODE}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/shared_online_joint_trials/${TAG}}"
CACHE_ROOT="${CACHE_ROOT:-${OFA_DIR}/cache_data/reuse_traces/shared_online_joint/${TAG}}"

mkdir -p "${OUT_DIR}/logs"
mkdir -p "${CACHE_ROOT}"
cd "${OFA_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

base_args=(
  --runs "${RUNS}"
  --seed "${SEED}"
  --experiment_suite residual_reuse
  --learned_hash_epochs 10
  --learned_hash_dim 128
  --hash_heads_per_route 8
  --hamming_only_acceptor
  --disable_structure_check
  --enable_score_gate
  --allow_rare_fuzzy
  --score_reuse_threshold "${SHARED_SCORE_T}"
  --score_propagation_weight 3
  --score_graph_context_weight 1
  --score_low_unique_weight 1
  --score_pair_confidence_discount 1
  --radius 2
  --main_hash_head_bits 16 16 16 16 16 16 16 16
  --residual_hard_min_support_hits "${SHARED_HARD_HITS}"
  --residual_soft_min_support_hits "${SHARED_SOFT_HITS}"
  --residual_rank 64
  --residual_epochs "${RESIDUAL_EPOCHS}"
  --residual_max_train_pairs "${RESIDUAL_MAX_TRAIN_PAIRS}"
  --residual_min_dist 1.0
  --residual_alpha_grid 0 0.03125 0.0625 0.125 0.25 0.5
  --residual_support_aware_alpha
  --residual_adapter_type mlp
  --residual_dropout 0.05
  --residual_loss_cosine_weight 1.0
  --residual_loss_mse_weight 0.5
  --residual_loss_delta_weight 0.75
  --residual_bucket_mode support_dist
  --residual_offline_extra_anchors_per_node 8
  --residual_offline_extra_query_nodes 4096
  --residual_train_split train_val
  --residual_gate_loss_weight 0.5
  --residual_gate_error_scale 0.25
  --residual_gate_error_max 0.45
  --residual_accept_mode "${SHARED_ACCEPT_MODE}"
  --residual_gate_accept_threshold "${SHARED_TAU}"
)

run_case() {
  local case_name="$1"
  local dataset=""
  local case_args=()

  case "${case_name}" in
    st_cora)
      dataset="cora"
      case_args=(
        --datasets "${dataset}"
        --residual_embedding_path "${ST_CORA_PATH}"
        --residual_fit_profile st
        --residual_class_aware_accept
        --residual_classifier_accept_gate
        --residual_classifier_accept_mode both
        --residual_classifier_accept_max_kl 0.2
        --residual_classifier_accept_after_residual
        --residual_classifier_accept_probe_alpha 0.125
        --residual_positive_error_max -1
        --residual_offline_negative_anchors_per_node 0
        --residual_negative_gate_weight 0.0
        --residual_accept_loss_weight 1.0
        --residual_gate_sparsity_weight 0.0
      )
      ;;
    st_pubmed)
      dataset="pubmed"
      case_args=(
        --datasets "${dataset}"
        --residual_embedding_path "${ST_PUBMED_PATH}"
        --residual_fit_profile st
        --residual_class_aware_accept
        --residual_classifier_accept_gate
        --residual_classifier_accept_mode both
        --residual_classifier_accept_max_kl 0.2
        --residual_classifier_accept_after_residual
        --residual_classifier_accept_probe_alpha 0.125
        --residual_positive_error_max 0.40
        --residual_offline_negative_anchors_per_node 4
        --residual_negative_error_min 0.45
        --residual_negative_gate_weight 1.0
        --residual_accept_loss_weight 1.0
        --residual_gate_sparsity_weight 0.02
      )
      ;;
    llama_cora)
      dataset="cora"
      case_args=(
        --datasets "${dataset}"
        --residual_embedding_source real_quant_fp
        --real_quant_model_name llama2_7b
        --real_quant_fp_tag "${LLAMA_FP_TAG}"
        --residual_fit_profile llama
        --residual_positive_error_max 0.40
        --residual_offline_negative_anchors_per_node 4
        --residual_negative_error_min 0.45
        --residual_negative_gate_weight 2.0
        --residual_accept_loss_weight 2.0
        --residual_gate_sparsity_weight 0.02
        --residual_classifier_accept_gate
        --residual_classifier_accept_mode both
        --residual_classifier_accept_max_kl 0.2
        --residual_classifier_accept_after_residual
        --residual_classifier_accept_probe_alpha 0.125
      )
      ;;
    llama_pubmed)
      dataset="pubmed"
      case_args=(
        --datasets "${dataset}"
        --residual_embedding_source real_quant_fp
        --real_quant_model_name llama2_7b
        --real_quant_fp_tag "${LLAMA_FP_TAG}"
        --residual_fit_profile llama
        --residual_positive_error_max 0.40
        --residual_offline_negative_anchors_per_node 4
        --residual_negative_error_min 0.45
        --residual_negative_gate_weight 1.0
        --residual_accept_loss_weight 1.0
        --residual_gate_sparsity_weight 0.02
        --residual_classifier_accept_gate
        --residual_classifier_accept_mode both
        --residual_classifier_accept_max_kl 0.2
        --residual_classifier_accept_after_residual
        --residual_classifier_accept_probe_alpha 0.125
      )
      ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      return 2
      ;;
  esac

  local cache_path="${CACHE_ROOT}/${case_name}_{dataset}_seed{seed}.pt"
  local log_path="${OUT_DIR}/logs/${case_name}.log"

  echo
  echo "================================================================"
  echo "[$(timestamp)] [Run] ${case_name} | dataset=${dataset} | tag=${TAG}"
  echo "[Log] ${log_path}"
  echo "================================================================"

  set +e
  "${PYTHON_BIN}" "${REPO_DIR}/scripts/run_graphhopsimhash_main.py" \
    "${base_args[@]}" \
    "${case_args[@]}" \
    --reuse_trace_cache_path "${cache_path}" 2>&1 | tee "${log_path}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[$(timestamp)] [Failed] ${case_name} status=${status}" >&2
    return "${status}"
  fi
}

for case_name in "${CASES[@]}"; do
  run_case "${case_name}"
done

echo
echo "[$(timestamp)] [Done] tag=${TAG} logs=${OUT_DIR}/logs"
