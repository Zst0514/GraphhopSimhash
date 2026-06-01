#!/usr/bin/env bash
set -euo pipefail

# Generic Graph-Bit predictor-free validation flow.
#
# Fixed front-end:
#   T31 shared retrieval skeleton, R=2, 8 x 16-bit heads
#   default: hard>=5 direct reuse, soft=3..4 residual candidates
#   accepted misses -> Degree Graph-Bit
#
# Useful overrides:
#   DATASET=pubmed  run PubMed instead of Cora
#   RUNS=3          quick multi-seed check
#   HARD_SUPPORT=6 SOFT_SUPPORT=4  make accepted reuse stricter
#   BUDGET=p8heavy                 safer LLaMA budget: 80% P8 / 20% P6 / 0% P4
#   BUDGET=balanced                older budget: 20% P8 / 50% P6 / 30% P4
#   RESIDUAL_ACCEPT_MODE=separate  test wider accept gate
#   RESIDUAL_GATE_THRESHOLD=0.60   fixed soft-hit accept threshold
#   RUN_ALGO=1      rerun residual_precision_depth even if a log exists
#   BOUND_ENABLE=1  use graph-conditioned runtime-bound depth policies
#   BOUND_RULE=tile_score  use node risk * W strength * low-bit budget stop score
#   BOUND_SCORE_TAU=0.001  threshold for tile_score
#   TRACE_EXPORT=1  export per-node Graph-Bit replay traces
#   RUN_ONNXIM=1    rerun ONNXim LLaMA GEMM microbenchmarks
#   BUILD_ONNXIM=1  try to build ONNXim before running microbenchmarks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASET="${DATASET:-cora}"
HEADS="${HEADS:-h8}"
THRESHOLD="${THRESHOLD:-31}"
BUDGET="${BUDGET:-p8heavy}"
HARD_SUPPORT="${HARD_SUPPORT:-5}"
SOFT_SUPPORT="${SOFT_SUPPORT:-3}"
FRONTEND_ID="${FRONTEND_ID:-h8_${HARD_SUPPORT}${SOFT_SUPPORT}_T${THRESHOLD}}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/graphbit_predictor_free/${DATASET}_${FRONTEND_ID}}"
LOG_DIR="${OUT_DIR}/logs/${DATASET}/${HEADS}"
RUNS="${RUNS:-10}"
RUN_ALGO="${RUN_ALGO:-0}"
RUN_ONNXIM="${RUN_ONNXIM:-0}"
BUILD_ONNXIM="${BUILD_ONNXIM:-0}"
SEQ_LEN="${SEQ_LEN:-64}"
if [[ -z "${RESIDUAL_ACCEPT_MODE+x}" ]]; then
  if [[ "${DATASET}" == "cora" ]]; then
    RESIDUAL_ACCEPT_MODE="separate"
  else
    RESIDUAL_ACCEPT_MODE="shared"
  fi
fi
if [[ -z "${RESIDUAL_GATE_THRESHOLD+x}" ]]; then
  if [[ "${DATASET}" == "cora" ]]; then
    RESIDUAL_GATE_THRESHOLD="0.40"
  else
    RESIDUAL_GATE_THRESHOLD="0.91"
  fi
fi
CLASSIFIER_ACCEPT_GATE="${CLASSIFIER_ACCEPT_GATE:-$([[ "${DATASET}" == "cora" ]] && echo 1 || echo 0)}"
CLASSIFIER_ACCEPT_MODE="${CLASSIFIER_ACCEPT_MODE:-both}"
CLASSIFIER_ACCEPT_MAX_KL="${CLASSIFIER_ACCEPT_MAX_KL:-0.2}"
CLASSIFIER_ACCEPT_AFTER_RESIDUAL="${CLASSIFIER_ACCEPT_AFTER_RESIDUAL:-1}"
CLASSIFIER_ACCEPT_PROBE_ALPHA="${CLASSIFIER_ACCEPT_PROBE_ALPHA:-0.125}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-200}"
RESIDUAL_ALPHA_GRID=(${RESIDUAL_ALPHA_GRID:-0 0.03125 0.0625 0.125 0.25 0.5})
RESIDUAL_POSITIVE_ERROR_MAX="${RESIDUAL_POSITIVE_ERROR_MAX:-0.40}"
RESIDUAL_OFFLINE_NEGATIVE_ANCHORS_PER_NODE="${RESIDUAL_OFFLINE_NEGATIVE_ANCHORS_PER_NODE:-4}"
RESIDUAL_NEGATIVE_ERROR_MIN="${RESIDUAL_NEGATIVE_ERROR_MIN:-0.45}"
if [[ -z "${RESIDUAL_NEGATIVE_GATE_WEIGHT+x}" ]]; then
  if [[ "${DATASET}" == "cora" ]]; then
    RESIDUAL_NEGATIVE_GATE_WEIGHT="2.0"
  else
    RESIDUAL_NEGATIVE_GATE_WEIGHT="1.0"
  fi
fi
if [[ -z "${RESIDUAL_ACCEPT_LOSS_WEIGHT+x}" ]]; then
  if [[ "${DATASET}" == "cora" ]]; then
    RESIDUAL_ACCEPT_LOSS_WEIGHT="2.0"
  else
    RESIDUAL_ACCEPT_LOSS_WEIGHT="0.0"
  fi
fi
RESIDUAL_GATE_SPARSITY_WEIGHT="${RESIDUAL_GATE_SPARSITY_WEIGHT:-0.02}"
PRECISION_DEPTH_TAGS=(${PRECISION_DEPTH_TAGS:-W4A8_TRUNC7 W4A8_TRUNC6 W4A8_TRUNC5 W4A8_TRUNC4})
PRECISION_DEPTH_BITS=(${PRECISION_DEPTH_BITS:-7 6 5 4})
BOUND_ENABLE="${BOUND_ENABLE:-1}"
BOUND_PRIORITIES=(${BOUND_PRIORITIES:-degree tser})
BOUND_ASSIGNMENT="${BOUND_ASSIGNMENT:-bucket_ratio}"
BOUND_RULE="${BOUND_RULE:-remaining_bound}"
BOUND_HIGH_MIN="${BOUND_HIGH_MIN:-8}"
BOUND_MID_MIN="${BOUND_MID_MIN:-6}"
BOUND_LOW_MIN="${BOUND_LOW_MIN:-4}"
BOUND_HIGH_TOL="${BOUND_HIGH_TOL:-0.0}"
BOUND_MID_TOL="${BOUND_MID_TOL:-0.02}"
BOUND_LOW_TOL="${BOUND_LOW_TOL:-0.04}"
BOUND_SCALE="${BOUND_SCALE:-1.0}"
BOUND_TILE_K="${BOUND_TILE_K:-128}"
BOUND_W_STRENGTH="${BOUND_W_STRENGTH:-1.0}"
BOUND_NODEWISE_MIN_DEPTH="${BOUND_NODEWISE_MIN_DEPTH:-4}"
BOUND_NODEWISE_MIN_TOL="${BOUND_NODEWISE_MIN_TOL:-0.0}"
BOUND_NODEWISE_MAX_TOL="${BOUND_NODEWISE_MAX_TOL:-0.04}"
BOUND_NODEWISE_GAMMA="${BOUND_NODEWISE_GAMMA:-1.0}"
BOUND_NODEWISE_RISK_MAX="${BOUND_NODEWISE_RISK_MAX:-15.0}"
BOUND_SCORE_TAU="${BOUND_SCORE_TAU:-0.001}"
BOUND_SCORE_ALPHA="${BOUND_SCORE_ALPHA:-1.0}"
BOUND_SCORE_BETA="${BOUND_SCORE_BETA:-1.0}"
BOUND_SCORE_W_CAP="${BOUND_SCORE_W_CAP:-2.0}"
BOUND_SCORE_W_REFERENCE="${BOUND_SCORE_W_REFERENCE:-1.0}"
BOUND_SCORE_NODE_FLOOR="${BOUND_SCORE_NODE_FLOOR:-0.0}"
TRACE_EXPORT="${TRACE_EXPORT:-0}"
TRACE_EXPORT_DIR="${TRACE_EXPORT_DIR:-${OUT_DIR}/node_traces}"
TRACE_EXPORT_CONFIGS=(${TRACE_EXPORT_CONFIGS:-DegBound})

case "${BUDGET}" in
  p8heavy)
    HIGH_RATIO="${HIGH_RATIO:-0.80}"
    MID_RATIO="${MID_RATIO:-0.20}"
    LOW_RATIO="${LOW_RATIO:-0.00}"
    ;;
  conservative)
    HIGH_RATIO="${HIGH_RATIO:-0.60}"
    MID_RATIO="${MID_RATIO:-0.30}"
    LOW_RATIO="${LOW_RATIO:-0.10}"
    ;;
  balanced)
    HIGH_RATIO="${HIGH_RATIO:-0.20}"
    MID_RATIO="${MID_RATIO:-0.50}"
    LOW_RATIO="${LOW_RATIO:-0.30}"
    ;;
  *)
    HIGH_RATIO="${HIGH_RATIO:-0.20}"
    MID_RATIO="${MID_RATIO:-0.50}"
    LOW_RATIO="${LOW_RATIO:-0.30}"
    ;;
esac

SUMMARY_TSV="${OUT_DIR}/summary.tsv"
SUMMARY_TXT="${OUT_DIR}/summary.txt"
MICROBENCH_JSON="${OFA_DIR}/output/onnxim_graphbit/microbench_s${SEQ_LEN}/aggregate.json"
BOUND_ID="${BOUND_ID:-${BOUND_RULE}}"
if [[ "${BOUND_RULE}" == "tile_score" ]]; then
  BOUND_ID="score_tau${BOUND_SCORE_TAU}"
fi
BUDGET_ID="${BUDGET}_${BOUND_ID}"
MAIN_LOG="${LOG_DIR}/T${THRESHOLD}_${BUDGET_ID}_runs${RUNS}.log"
DONE_PATH="${MAIN_LOG}.done"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${OFA_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

maybe_seed_existing_log() {
  if [[ -f "${MAIN_LOG}" ]]; then
    return
  fi
  local existing="${OFA_DIR}/output/residual_graphbit_main/${DATASET}_${FRONTEND_ID}/${DATASET}_${FRONTEND_ID}_runs${RUNS}.log"
  if [[ "${RUNS}" == "10" && -f "${existing}" ]]; then
    echo "[$(timestamp)] [Seed] using existing ${DATASET} ${FRONTEND_ID} log: ${existing}"
    cp "${existing}" "${MAIN_LOG}"
    touch "${DONE_PATH}"
  fi
}

run_algo() {
  local head_bits=(16 16 16 16 16 16 16 16)
  local bound_args=()
  if [[ "${BOUND_ENABLE}" == "1" ]]; then
    bound_args=(
      --precision_depth_bound_enable
      --precision_depth_bound_priorities "${BOUND_PRIORITIES[@]}"
      --precision_depth_bound_assignment "${BOUND_ASSIGNMENT}"
      --precision_depth_bound_rule "${BOUND_RULE}"
      --precision_depth_bound_high_min_depth "${BOUND_HIGH_MIN}"
      --precision_depth_bound_mid_min_depth "${BOUND_MID_MIN}"
      --precision_depth_bound_low_min_depth "${BOUND_LOW_MIN}"
      --precision_depth_bound_high_tolerance "${BOUND_HIGH_TOL}"
      --precision_depth_bound_mid_tolerance "${BOUND_MID_TOL}"
      --precision_depth_bound_low_tolerance "${BOUND_LOW_TOL}"
      --precision_depth_bound_scale "${BOUND_SCALE}"
      --precision_depth_bound_tile_k "${BOUND_TILE_K}"
      --precision_depth_bound_w_strength "${BOUND_W_STRENGTH}"
      --precision_depth_bound_node_risk_weight "${BOUND_NODE_RISK_WEIGHT:-1.0}"
      --precision_depth_bound_w_risk_weight "${BOUND_W_RISK_WEIGHT:-0.0}"
      --precision_depth_bound_w_risk_reference "${BOUND_W_RISK_REFERENCE:-1.5}"
      --precision_depth_bound_nodewise_min_depth "${BOUND_NODEWISE_MIN_DEPTH}"
      --precision_depth_bound_nodewise_min_tolerance "${BOUND_NODEWISE_MIN_TOL}"
      --precision_depth_bound_nodewise_max_tolerance "${BOUND_NODEWISE_MAX_TOL}"
      --precision_depth_bound_nodewise_gamma "${BOUND_NODEWISE_GAMMA}"
      --precision_depth_bound_nodewise_risk_max "${BOUND_NODEWISE_RISK_MAX}"
      --precision_depth_score_tau "${BOUND_SCORE_TAU}"
      --precision_depth_score_alpha "${BOUND_SCORE_ALPHA}"
      --precision_depth_score_beta "${BOUND_SCORE_BETA}"
      --precision_depth_score_w_cap "${BOUND_SCORE_W_CAP}"
      --precision_depth_score_w_reference "${BOUND_SCORE_W_REFERENCE}"
      --precision_depth_score_node_floor "${BOUND_SCORE_NODE_FLOOR}"
    )
  fi
  local trace_args=()
  if [[ "${TRACE_EXPORT}" == "1" ]]; then
    trace_args=(
      --precision_depth_trace_export_dir "${TRACE_EXPORT_DIR}"
      --precision_depth_trace_export_configs "${TRACE_EXPORT_CONFIGS[@]}"
    )
  fi
  local classifier_args=()
  if [[ "${CLASSIFIER_ACCEPT_GATE}" == "1" ]]; then
    classifier_args=(
      --residual_classifier_accept_gate
      --residual_classifier_accept_mode "${CLASSIFIER_ACCEPT_MODE}"
      --residual_classifier_accept_max_kl "${CLASSIFIER_ACCEPT_MAX_KL}"
      --residual_classifier_accept_probe_alpha "${CLASSIFIER_ACCEPT_PROBE_ALPHA}"
    )
    if [[ "${CLASSIFIER_ACCEPT_AFTER_RESIDUAL}" == "1" ]]; then
      classifier_args+=(--residual_classifier_accept_after_residual)
    fi
  fi
  echo "[$(timestamp)] [Algo] running ${DATASET} ${FRONTEND_ID} residual + Graph-Bit, runs=${RUNS}"
  set +e
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets "${DATASET}" \
    --runs "${RUNS}" \
    --experiment_suite residual_precision_depth \
    --real_quant_model_name llama2_7b \
    --precision_depth_reference_tag W4A8 \
    --precision_depth_tags "${PRECISION_DEPTH_TAGS[@]}" \
    --precision_depth_bits "${PRECISION_DEPTH_BITS[@]}" \
    --precision_depth_reference_bits 8 \
    --precision_depth_high_ratio "${HIGH_RATIO}" \
    --precision_depth_mid_ratio "${MID_RATIO}" \
    --precision_depth_low_ratio "${LOW_RATIO}" \
    --precision_depth_cost_scale 0.50 \
    --precision_depth_fixed_cost 0.15 \
    "${bound_args[@]}" \
    "${trace_args[@]}" \
    --radius 2 \
    --hash_heads_per_route 8 \
    --main_hash_head_bits "${head_bits[@]}" \
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
    --residual_max_train_pairs 4096 \
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
    --residual_positive_error_max "${RESIDUAL_POSITIVE_ERROR_MAX}" \
    --residual_offline_extra_query_nodes 4096 \
    --residual_offline_negative_anchors_per_node "${RESIDUAL_OFFLINE_NEGATIVE_ANCHORS_PER_NODE}" \
    --residual_negative_error_min "${RESIDUAL_NEGATIVE_ERROR_MIN}" \
    --residual_negative_gate_weight "${RESIDUAL_NEGATIVE_GATE_WEIGHT}" \
    --residual_train_split train_val \
    --residual_gate_loss_weight 0.5 \
    --residual_accept_loss_weight "${RESIDUAL_ACCEPT_LOSS_WEIGHT}" \
    --residual_gate_error_scale 0.25 \
    --residual_gate_error_max 0.45 \
    --residual_gate_sparsity_weight "${RESIDUAL_GATE_SPARSITY_WEIGHT}" \
    "${classifier_args[@]}" \
    --residual_gate_accept_threshold "${RESIDUAL_GATE_THRESHOLD}" \
    --residual_min_dist 1.0 2>&1 | tee "${MAIN_LOG}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[$(timestamp)] [Algo] failed with status=${status}" >&2
    exit "${status}"
  fi
  touch "${DONE_PATH}"
}

maybe_run_onnxim() {
  if [[ "${BUILD_ONNXIM}" == "1" ]]; then
    "${SCRIPT_DIR}/build_onnxim.sh"
  fi

  if [[ "${RUN_ONNXIM}" != "1" && -f "${MICROBENCH_JSON}" ]]; then
    echo "[$(timestamp)] [ONNXim] using existing microbench: ${MICROBENCH_JSON}"
    return
  fi

  echo "[$(timestamp)] [ONNXim] running LLaMA GEMM microbench seq_len=${SEQ_LEN}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --action all \
    --log-level info
}

if [[ "${RUN_ALGO}" == "1" || ! -f "${DONE_PATH}" ]]; then
  maybe_seed_existing_log
fi

if [[ "${RUN_ALGO}" == "1" || ! -f "${DONE_PATH}" ]]; then
  run_algo
else
  echo "[$(timestamp)] [Algo] using existing log: ${MAIN_LOG}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_residual_graphbit.py" \
  --log-dir "${OUT_DIR}/logs" \
  --output-tsv "${SUMMARY_TSV}" \
  --output-txt "${SUMMARY_TXT}"

maybe_run_onnxim

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_graphbit_predictor_free_flow.py" \
  --residual-summary "${SUMMARY_TSV}" \
  --microbench "${MICROBENCH_JSON}" \
  --output-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --heads "${HEADS}" \
  --threshold "${THRESHOLD}" \
  --budget "${BUDGET_ID}" \
  --runs "${RUNS}" \
  --frontend-id "${FRONTEND_ID}" \
  --hard-support "${HARD_SUPPORT}" \
  --soft-support "${SOFT_SUPPORT}" \
  --bounded-save-p6 0.50 \
  --bounded-save-p4 0.25

echo "[$(timestamp)] [Done] ${OUT_DIR}/predictor_free_main.txt"
