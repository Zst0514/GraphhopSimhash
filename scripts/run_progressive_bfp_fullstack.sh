#!/usr/bin/env bash
set -euo pipefail

# SimHash/residual front-end + progressive BFP encoder back-end.
#
# Online front-end defaults:
#   8 x 16-bit SimHash heads, R=2, T=31, TSER weights=3/1/1
#   support >= 5  -> direct reuse
#   support = 3..4 -> residual-gate candidate
#   rejected/miss  -> progressive BFP encoder
#
# Back-end defaults:
#   BFPA4 base path for all encoder nodes
#   top REFINE_RATIO encoder nodes by risk -> BFPA6 refinement
#
# Examples:
#   DATASET=cora RUNS=1 bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
#   DATASET=pubmed RUNS=3 REFINE_BIT=5 REFINE_RATIO=0.30 bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASET="${DATASET:-cora}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
THRESHOLD="${THRESHOLD:-31}"
HARD_SUPPORT="${HARD_SUPPORT:-5}"
SOFT_SUPPORT="${SOFT_SUPPORT:-3}"

REFERENCE_TAG="${REFERENCE_TAG:-W4BFPA8_B128}"
BASE_BIT="${BASE_BIT:-4}"
BASE_TAG="${BASE_TAG:-W4BFPA4_B128}"
REFINE_BIT="${REFINE_BIT:-6}"
REFINE_TAG="${REFINE_TAG:-W4BFPA${REFINE_BIT}_B128}"
REFINE_RATIO="${REFINE_RATIO:-0.30}"

RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-200}"
RESIDUAL_MAX_TRAIN_PAIRS="${RESIDUAL_MAX_TRAIN_PAIRS:-4096}"
RESIDUAL_ALPHA_GRID=(${RESIDUAL_ALPHA_GRID:-0 0.03125 0.0625 0.125 0.25 0.5})

FRONTEND_ID="${FRONTEND_ID:-h8_${HARD_SUPPORT}${SOFT_SUPPORT}_T${THRESHOLD}}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/progressive_bfp_fullstack/${DATASET}_${FRONTEND_ID}_bfpa${REFINE_BIT}_r${REFINE_RATIO}}"
LOG_DIR="${OUT_DIR}/logs"
LOG_PATH="${LOG_DIR}/${DATASET}_runs${RUNS}.log"
DONE_PATH="${LOG_PATH}.done"

mkdir -p "${LOG_DIR}"
cd "${OFA_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

case "${DATASET}" in
  cora)
    RESIDUAL_ACCEPT_MODE="${RESIDUAL_ACCEPT_MODE:-separate}"
    RESIDUAL_GATE_THRESHOLD="${RESIDUAL_GATE_THRESHOLD:-0.40}"
    RESIDUAL_NEGATIVE_GATE_WEIGHT="${RESIDUAL_NEGATIVE_GATE_WEIGHT:-2.0}"
    RESIDUAL_ACCEPT_LOSS_WEIGHT="${RESIDUAL_ACCEPT_LOSS_WEIGHT:-2.0}"
    RESIDUAL_GATE_SPARSITY_WEIGHT="${RESIDUAL_GATE_SPARSITY_WEIGHT:-0.02}"
    CLASSIFIER_ACCEPT_GATE="${CLASSIFIER_ACCEPT_GATE:-1}"
    ;;
  pubmed)
    RESIDUAL_ACCEPT_MODE="${RESIDUAL_ACCEPT_MODE:-shared}"
    RESIDUAL_GATE_THRESHOLD="${RESIDUAL_GATE_THRESHOLD:-0.91}"
    RESIDUAL_NEGATIVE_GATE_WEIGHT="${RESIDUAL_NEGATIVE_GATE_WEIGHT:-1.0}"
    RESIDUAL_ACCEPT_LOSS_WEIGHT="${RESIDUAL_ACCEPT_LOSS_WEIGHT:-0.0}"
    RESIDUAL_GATE_SPARSITY_WEIGHT="${RESIDUAL_GATE_SPARSITY_WEIGHT:-0.02}"
    CLASSIFIER_ACCEPT_GATE="${CLASSIFIER_ACCEPT_GATE:-0}"
    ;;
  arxiv)
    RESIDUAL_ACCEPT_MODE="${RESIDUAL_ACCEPT_MODE:-shared}"
    RESIDUAL_GATE_THRESHOLD="${RESIDUAL_GATE_THRESHOLD:-0.91}"
    RESIDUAL_NEGATIVE_GATE_WEIGHT="${RESIDUAL_NEGATIVE_GATE_WEIGHT:-1.0}"
    RESIDUAL_ACCEPT_LOSS_WEIGHT="${RESIDUAL_ACCEPT_LOSS_WEIGHT:-0.0}"
    RESIDUAL_GATE_SPARSITY_WEIGHT="${RESIDUAL_GATE_SPARSITY_WEIGHT:-0.02}"
    CLASSIFIER_ACCEPT_GATE="${CLASSIFIER_ACCEPT_GATE:-0}"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

if [[ "${REFINE_BIT}" -eq 8 ]]; then
  PRECISION_DEPTH_TAGS=("${BASE_TAG}")
  PRECISION_DEPTH_BITS=("${BASE_BIT}")
  HIGH_RATIO="${REFINE_RATIO}"
  MID_RATIO="0.0"
  LOW_RATIO="0.0"
else
  PRECISION_DEPTH_TAGS=("${REFINE_TAG}" "${BASE_TAG}")
  PRECISION_DEPTH_BITS=("${REFINE_BIT}" "${BASE_BIT}")
  HIGH_RATIO="0.0"
  MID_RATIO="${REFINE_RATIO}"
  LOW_RATIO="0.0"
fi

classifier_args=()
if [[ "${CLASSIFIER_ACCEPT_GATE}" == "1" ]]; then
  classifier_args=(
    --residual_classifier_accept_gate
    --residual_classifier_accept_mode "${CLASSIFIER_ACCEPT_MODE:-both}"
    --residual_classifier_accept_max_kl "${CLASSIFIER_ACCEPT_MAX_KL:-0.2}"
    --residual_classifier_accept_probe_alpha "${CLASSIFIER_ACCEPT_PROBE_ALPHA:-0.125}"
  )
  if [[ "${CLASSIFIER_ACCEPT_AFTER_RESIDUAL:-1}" == "1" ]]; then
    classifier_args+=(--residual_classifier_accept_after_residual)
  fi
fi

if [[ "${FORCE:-0}" != "1" && -f "${DONE_PATH}" ]]; then
  echo "[$(timestamp)] [Skip] existing ${DONE_PATH}"
  echo "Log: ${LOG_PATH}"
  exit 0
fi

echo "================================================================"
echo "[$(timestamp)] Progressive BFP full-stack"
echo "dataset=${DATASET} runs=${RUNS} frontend=${FRONTEND_ID}"
echo "reference=${REFERENCE_TAG} base=P${BASE_BIT}:${BASE_TAG} refine=P${REFINE_BIT}:${REFINE_TAG} ratio=${REFINE_RATIO}"
echo "log=${LOG_PATH}"
echo "================================================================"

set +e
"${PYTHON_BIN}" -m GraphhopSimhash \
  --datasets "${DATASET}" \
  --runs "${RUNS}" \
  --seed "${SEED}" \
  --experiment_suite residual_precision_depth \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag "${REFERENCE_TAG}" \
  --precision_depth_reference_bits 8 \
  --precision_depth_tags "${PRECISION_DEPTH_TAGS[@]}" \
  --precision_depth_bits "${PRECISION_DEPTH_BITS[@]}" \
  --precision_depth_high_ratio "${HIGH_RATIO}" \
  --precision_depth_mid_ratio "${MID_RATIO}" \
  --precision_depth_low_ratio "${LOW_RATIO}" \
  --precision_depth_cost_scale 0.50 \
  --precision_depth_fixed_cost 0.15 \
  --radius 2 \
  --hash_heads_per_route 8 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --disable_structure_check \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold "${THRESHOLD}" \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --score_pair_confidence_discount 1 \
  --residual_fit_profile llama \
  --residual_rank 64 \
  --residual_epochs "${RESIDUAL_EPOCHS}" \
  --residual_max_train_pairs "${RESIDUAL_MAX_TRAIN_PAIRS}" \
  --residual_hard_min_support_hits "${HARD_SUPPORT}" \
  --residual_soft_min_support_hits "${SOFT_SUPPORT}" \
  --residual_alpha_grid "${RESIDUAL_ALPHA_GRID[@]}" \
  --residual_support_aware_alpha \
  --residual_adapter_type mlp \
  --residual_accept_mode "${RESIDUAL_ACCEPT_MODE}" \
  --residual_dropout 0.05 \
  --residual_loss_cosine_weight 1.0 \
  --residual_loss_mse_weight 0.5 \
  --residual_loss_delta_weight 0.75 \
  --residual_bucket_mode support_dist \
  --residual_offline_extra_anchors_per_node 8 \
  --residual_positive_error_max "${RESIDUAL_POSITIVE_ERROR_MAX:-0.40}" \
  --residual_offline_extra_query_nodes 4096 \
  --residual_offline_negative_anchors_per_node "${RESIDUAL_OFFLINE_NEGATIVE_ANCHORS_PER_NODE:-4}" \
  --residual_negative_error_min "${RESIDUAL_NEGATIVE_ERROR_MIN:-0.45}" \
  --residual_negative_gate_weight "${RESIDUAL_NEGATIVE_GATE_WEIGHT}" \
  --residual_train_split train_val \
  --residual_gate_loss_weight 0.5 \
  --residual_accept_loss_weight "${RESIDUAL_ACCEPT_LOSS_WEIGHT}" \
  --residual_gate_error_scale 0.25 \
  --residual_gate_error_max 0.45 \
  --residual_gate_sparsity_weight "${RESIDUAL_GATE_SPARSITY_WEIGHT}" \
  "${classifier_args[@]}" \
  --residual_gate_accept_threshold "${RESIDUAL_GATE_THRESHOLD}" \
  --residual_min_dist 1.0 2>&1 | tee "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "[$(timestamp)] [Failed] status=${status}" >&2
  exit "${status}"
fi

touch "${DONE_PATH}"
echo "[$(timestamp)] [Done] ${LOG_PATH}"
