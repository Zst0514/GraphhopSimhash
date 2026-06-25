#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/llama7b_tser_trace_replay}"
DATASETS=(${DATASETS:-tape_products arxiv tape_arxiv23})
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
FORCE_TRACE="${FORCE_TRACE:-0}"
TRACE_THRESHOLD="${TRACE_THRESHOLD:-999}"
REPLAY_THRESHOLDS="${REPLAY_THRESHOLDS:-16 20 24 28 31 35 40 45 50}"
TARGET_REUSE="${TARGET_REUSE:-0.30 0.35 0.40 0.45 0.50}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/traces"
cd "${OFA_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

accept_args_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    cora|arxiv|wikics|tape_products|tape_arxiv23)
      echo "--residual_accept_mode separate --residual_positive_error_max 0.40 --residual_offline_negative_anchors_per_node 1 --residual_negative_error_min 0.45 --residual_negative_gate_weight 2.0 --residual_accept_loss_weight 2.0 --residual_gate_sparsity_weight 0.02 --residual_classifier_accept_gate --residual_classifier_accept_mode both --residual_classifier_accept_max_kl 0.2 --residual_classifier_accept_after_residual --residual_classifier_accept_probe_alpha 0.125 --residual_gate_accept_threshold 0.40"
      ;;
    pubmed)
      echo "--residual_accept_mode shared --residual_positive_error_max 0.40 --residual_offline_negative_anchors_per_node 1 --residual_negative_error_min 0.45 --residual_negative_gate_weight 1.0 --residual_accept_loss_weight 0.0 --residual_gate_sparsity_weight 0.02 --residual_gate_accept_threshold 0.91"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      return 2
      ;;
  esac
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
  --allow_rare_fuzzy
  --score_pair_confidence_discount 1
  --radius 2
  --main_hash_head_bits 16 16 16 16 16 16 16 16
  --residual_hard_min_support_hits 5
  --residual_soft_min_support_hits 3
  --residual_rank 16
  --residual_epochs 1
  --residual_max_train_pairs 128
  --residual_min_dist 1.0
  --residual_alpha_grid 0
  --residual_adapter_type mlp
  --residual_dropout 0.05
  --residual_loss_cosine_weight 1.0
  --residual_loss_mse_weight 0.5
  --residual_loss_delta_weight 0.75
  --residual_bucket_mode support_dist
  --residual_offline_extra_anchors_per_node 0
  --residual_offline_extra_query_nodes 0
  --residual_train_split train_val
  --residual_gate_loss_weight 0.5
  --residual_gate_error_scale 0.25
  --residual_gate_error_max 0.45
  --residual_embedding_source real_quant_fp
  --real_quant_model_name llama2_7b
  --real_quant_fp_tag W4BFPA8_B128
  --residual_fit_profile manual
  --enable_score_gate
  --score_reuse_threshold "${TRACE_THRESHOLD}"
  --score_propagation_weight 3
  --score_graph_context_weight 1
  --score_low_unique_weight 1
)

capture_trace() {
  local dataset="$1"
  local pool="${OFA_DIR}/cache_data/${dataset}_llama2_7b_oracle_W4BFPA8_B128.pt"
  local tag="${dataset}_canonical_T${TRACE_THRESHOLD}_runs${RUNS}"
  local log_path="${OUT_DIR}/logs/${tag}.log"
  local done_path="${log_path}.done"
  local existing
  existing="$(find "${OUT_DIR}/traces" -maxdepth 1 -type f -name "${dataset}_canonical_T${TRACE_THRESHOLD}_run*_reuse_decisions.tsv" | wc -l)"

  if [[ "${existing}" -ge "${RUNS}" && "${FORCE_TRACE}" != "1" ]]; then
    echo "[$(timestamp)] [Trace exists] dataset=${dataset} files=${existing}"
    return
  fi
  if [[ ! -s "${pool}" ]]; then
    echo "[$(timestamp)] [Missing pool] ${pool}" >&2
    return 2
  fi
  if [[ -e "${done_path}" && "${FORCE_TRACE}" != "1" ]]; then
    echo "[$(timestamp)] [Trace log done] ${done_path}"
    return
  fi

  read -r -a accept_args <<< "$(accept_args_for_dataset "${dataset}")"
  echo
  echo "================================================================"
  echo "[$(timestamp)] [Trace capture] dataset=${dataset} T=${TRACE_THRESHOLD} runs=${RUNS}"
  echo "[Log] ${log_path}"
  echo "================================================================"
  "${PYTHON_BIN}" -m GraphhopSimhash \
    "${base_args[@]}" \
    --datasets "${dataset}" \
    --residual_embedding_path "${pool}" \
    --reuse_decision_trace_export_dir "${OUT_DIR}/traces" \
    --reuse_decision_trace_tag "canonical_T${TRACE_THRESHOLD}" \
    "${accept_args[@]}" \
    2>&1 | tee "${log_path}"
  touch "${done_path}"
}

echo "[$(timestamp)] LLaMA2-7B TSER trace replay queue"
echo "OUT_DIR=${OUT_DIR}"
echo "DATASETS=${DATASETS[*]}"
echo "RUNS=${RUNS}"

for dataset in "${DATASETS[@]}"; do
  capture_trace "${dataset}"
done

"${PYTHON_BIN}" "${REPO_DIR}/scripts/replay_llama7b_tser_from_trace.py" \
  --trace_dir "${OUT_DIR}/traces" \
  --trace_tag_contains "canonical_T${TRACE_THRESHOLD}" \
  --datasets "${DATASETS[@]}" \
  --thresholds ${REPLAY_THRESHOLDS} \
  --targets ${TARGET_REUSE} \
  --output_dir "${OUT_DIR}/replay"

echo "[$(timestamp)] [Done] ${OUT_DIR}/replay/trace_replay_summary.md"
