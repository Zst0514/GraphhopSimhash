#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${SCRIPT_DIR}/run_graphhopsimhash_main.py"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/shared_bucketwise_gate_sweep}"
RUNS="${RUNS:-1}"
SEED="${SEED:-42}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-200}"
RESIDUAL_MAX_TRAIN_PAIRS="${RESIDUAL_MAX_TRAIN_PAIRS:-4096}"
SHARED_HARD_HITS="${SHARED_HARD_HITS:-5}"
SHARED_SOFT_HITS="${SHARED_SOFT_HITS:-3}"
SHARED_ACCEPT_MODE="${SHARED_ACCEPT_MODE:-separate}"

CASES=(${CASES:-st_cora st_pubmed llama_cora llama_pubmed})

CONFIGS_DEFAULT=(
  "T38_A|38|default=1.000 3h_d1=0.22 3h_d2=0.35 4h_d1=0.15 4h_d2=0.25"
  "T40_A|40|default=1.000 3h_d1=0.20 3h_d2=0.30 4h_d1=0.15 4h_d2=0.22"
  "T40_B|40|default=1.000 3h_d1=0.25 3h_d2=0.40 4h_d1=0.18 4h_d2=0.28"
)

mkdir -p "${OUT_DIR}"
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
  --residual_accept_mode "${SHARED_ACCEPT_MODE}"
  --residual_dropout 0.05
  --residual_loss_cosine_weight 1.0
  --residual_loss_mse_weight 0.5
  --residual_loss_delta_weight 0.75
  --residual_bucket_mode support_dist
  --residual_train_split train_val
  --residual_gate_loss_weight 0.5
  --residual_gate_error_scale 0.25
  --residual_gate_error_max 0.45
)

run_case() {
  local tag="$1"
  local score_t="$2"
  local bucket_specs="$3"
  local case_name="$4"
  local dataset=""
  local case_args=()

  case "${case_name}" in
    st_cora)
      dataset="cora"
      case_args=(
        --datasets "${dataset}"
        --residual_embedding_path "${OFA_DIR}/cache_data/cora_ST_oracle_W4A16.pt"
        --residual_fit_profile st
        --residual_positive_error_max 0.40
        --residual_offline_extra_anchors_per_node 8
        --residual_offline_extra_query_nodes 4096
        --residual_offline_negative_anchors_per_node 4
        --residual_negative_error_min 0.45
        --residual_negative_gate_weight 1.0
        --residual_accept_loss_weight 1.0
        --residual_gate_sparsity_weight 0.02
        --residual_class_aware_accept
      )
      ;;
    st_pubmed)
      dataset="pubmed"
      case_args=(
        --datasets "${dataset}"
        --residual_embedding_path "${OFA_DIR}/cache_data/pubmed_ST_oracle_W4A16.pt"
        --residual_fit_profile st
        --residual_positive_error_max 0.40
        --residual_offline_extra_anchors_per_node 8
        --residual_offline_extra_query_nodes 4096
        --residual_offline_negative_anchors_per_node 4
        --residual_negative_error_min 0.45
        --residual_negative_gate_weight 1.0
        --residual_accept_loss_weight 1.0
        --residual_gate_sparsity_weight 0.02
        --residual_class_aware_accept
      )
      ;;
    llama_cora)
      dataset="cora"
      case_args=(
        --datasets "${dataset}"
        --residual_embedding_source real_quant_fp
        --real_quant_model_name llama2_7b
        --real_quant_fp_tag W4A16
        --residual_fit_profile llama
        --residual_positive_error_max 0.40
        --residual_offline_extra_anchors_per_node 8
        --residual_offline_extra_query_nodes 4096
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
        --real_quant_fp_tag W4A16
        --residual_fit_profile llama
        --residual_positive_error_max 0.40
        --residual_offline_extra_anchors_per_node 8
        --residual_offline_extra_query_nodes 4096
        --residual_offline_negative_anchors_per_node 4
        --residual_negative_error_min 0.45
        --residual_negative_gate_weight 1.0
        --residual_accept_loss_weight 2.0
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

  local cfg_dir="${OUT_DIR}/${tag}"
  local log_dir="${cfg_dir}/logs"
  mkdir -p "${log_dir}"
  local log_path="${log_dir}/${case_name}.log"
  local done_path="${log_path}.done"
  if [[ -e "${done_path}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(timestamp)] [Skip] ${tag}/${case_name}; existing ${done_path}"
    return
  fi

  echo
  echo "================================================================"
  echo "[$(timestamp)] [Run] ${tag} | case=${case_name} | dataset=${dataset} | runs=${RUNS}"
  echo "[Log] ${log_path}"
  echo "================================================================"

  read -r -a bucket_args <<< "${bucket_specs}"

  set +e
  "${PYTHON_BIN}" "${RUNNER}" \
    "${base_args[@]}" \
    --score_reuse_threshold "${score_t}" \
    --residual_gate_accept_threshold_by_bucket "${bucket_args[@]}" \
    "${case_args[@]}" 2>&1 | tee "${log_path}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[$(timestamp)] [Failed] ${tag}/${case_name} status=${status}" >&2
    return "${status}"
  fi
  touch "${done_path}"
}

if [[ -n "${CONFIGS:-}" ]]; then
  IFS=';' read -r -a CONFIG_LIST <<< "${CONFIGS}"
else
  CONFIG_LIST=("${CONFIGS_DEFAULT[@]}")
fi

for config in "${CONFIG_LIST[@]}"; do
  IFS='|' read -r tag score_t bucket_specs <<< "${config}"
  for case_name in "${CASES[@]}"; do
    run_case "${tag}" "${score_t}" "${bucket_specs}" "${case_name}"
  done
done

echo
echo "[$(timestamp)] [Done] out=${OUT_DIR}"
