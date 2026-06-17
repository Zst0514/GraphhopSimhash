#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/llama7b_frontend_reuse_six_datasets}"

RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-200}"
RESIDUAL_MAX_TRAIN_PAIRS="${RESIDUAL_MAX_TRAIN_PAIRS:-4096}"

# Space-separated dataset list. All datasets use the LLaMA2-7B W4BFPA8_B128
# target pool so the frontend reuse numbers are comparable across workloads.
DATASETS=(${DATASETS:-cora pubmed arxiv wikics tape_products tape_arxiv23})

# MODE=sweep: run the dataset-specific T sweep below.
# MODE=point: run one threshold from T=<value> for every dataset.
MODE="${MODE:-sweep}"
T="${T:-}"

mkdir -p "${OUT_DIR}/logs"
cd "${OFA_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

thresholds_for_dataset() {
  local dataset="$1"
  if [[ "${MODE}" == "point" ]]; then
    if [[ -z "${T}" ]]; then
      echo "MODE=point requires T=<threshold>" >&2
      return 2
    fi
    echo "${T}"
    return
  fi

  case "${dataset}" in
    cora)
      echo "16 18 20 22 24 28 31"
      ;;
    pubmed)
      echo "8 10 12 15 18 20 24"
      ;;
    arxiv)
      echo "16 18 20 22"
      ;;
    wikics)
      echo "18 20 24 28 31"
      ;;
    tape_products)
      echo "12 15 18 20 24 28"
      ;;
    tape_arxiv23)
      echo "16 18 20 22 24"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      return 2
      ;;
  esac
}

accept_args_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    cora|arxiv|wikics|tape_products|tape_arxiv23)
      echo "--residual_accept_mode separate --residual_positive_error_max 0.40 --residual_offline_negative_anchors_per_node 4 --residual_negative_error_min 0.45 --residual_negative_gate_weight 2.0 --residual_accept_loss_weight 2.0 --residual_gate_sparsity_weight 0.02 --residual_classifier_accept_gate --residual_classifier_accept_mode both --residual_classifier_accept_max_kl 0.2 --residual_classifier_accept_after_residual --residual_classifier_accept_probe_alpha 0.125 --residual_gate_accept_threshold 0.40"
      ;;
    pubmed)
      echo "--residual_accept_mode shared --residual_positive_error_max 0.40 --residual_offline_negative_anchors_per_node 4 --residual_negative_error_min 0.45 --residual_negative_gate_weight 1.0 --residual_accept_loss_weight 0.0 --residual_gate_sparsity_weight 0.02 --residual_gate_accept_threshold 0.91"
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
  --enable_score_gate
  --allow_rare_fuzzy
  --score_propagation_weight 3
  --score_graph_context_weight 1
  --score_low_unique_weight 1
  --score_pair_confidence_discount 1
  --radius 2
  --main_hash_head_bits 16 16 16 16 16 16 16 16
  --residual_hard_min_support_hits 5
  --residual_soft_min_support_hits 3
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
  --residual_embedding_source real_quant_fp
  --real_quant_model_name llama2_7b
  --real_quant_fp_tag W4BFPA8_B128
  --residual_fit_profile llama
)

run_one() {
  local dataset="$1"
  local threshold="$2"
  local pool="${OFA_DIR}/cache_data/${dataset}_llama2_7b_oracle_W4BFPA8_B128.pt"
  local tag="${dataset}_T${threshold}_runs${RUNS}"
  local log_path="${OUT_DIR}/logs/${tag}.log"
  local done_path="${log_path}.done"

  if [[ ! -s "${pool}" ]]; then
    echo "[$(timestamp)] [Missing pool] ${pool}" >&2
    return 2
  fi
  if [[ -e "${done_path}" && "${FORCE}" != "1" ]]; then
    echo "[$(timestamp)] [Skip] ${tag}; existing ${done_path}"
    return
  fi

  read -r -a accept_args <<< "$(accept_args_for_dataset "${dataset}")"

  echo
  echo "================================================================"
  echo "[$(timestamp)] [Run] dataset=${dataset} | T=${threshold} | runs=${RUNS}"
  echo "[Pool] ${pool}"
  echo "[Log]  ${log_path}"
  echo "================================================================"

  set +e
  "${PYTHON_BIN}" -m GraphhopSimhash \
    "${base_args[@]}" \
    --datasets "${dataset}" \
    --score_reuse_threshold "${threshold}" \
    --residual_embedding_path "${pool}" \
    "${accept_args[@]}" \
    2>&1 | tee "${log_path}"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "[$(timestamp)] [Failed] dataset=${dataset} T=${threshold} status=${status}" >&2
    return "${status}"
  fi
  touch "${done_path}"
}

echo "[$(timestamp)] LLaMA2-7B frontend reuse sweep"
echo "OUT_DIR=${OUT_DIR}"
echo "DATASETS=${DATASETS[*]}"
echo "RUNS=${RUNS} MODE=${MODE}"

for dataset in "${DATASETS[@]}"; do
  for threshold in $(thresholds_for_dataset "${dataset}"); do
    run_one "${dataset}" "${threshold}"
  done
done

"${PYTHON_BIN}" "${REPO_DIR}/scripts/summarize_llama7b_frontend_reuse.py" \
  --log_dir "${OUT_DIR}/logs" \
  --output_dir "${OUT_DIR}" || true

echo
echo "[$(timestamp)] [Done] logs=${OUT_DIR}/logs"
