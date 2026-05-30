#!/usr/bin/env bash
set -euo pipefail

# Generic Graph-Bit predictor-free validation flow.
#
# Fixed front-end:
#   h8_{hard}{soft}_T{threshold}, R=2, 8 x 16-bit heads
#   default: LLaMA-aware gate, hard>=5 direct reuse, soft=3..4 residual candidates
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
#   RUN_ONNXIM=1    rerun ONNXim LLaMA GEMM microbenchmarks
#   BUILD_ONNXIM=1  try to build ONNXim before running microbenchmarks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASET="${DATASET:-cora}"
HEADS="${HEADS:-h8}"
THRESHOLD="${THRESHOLD:-30}"
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
RESIDUAL_ACCEPT_MODE="${RESIDUAL_ACCEPT_MODE:-shared}"
RESIDUAL_GATE_THRESHOLD="${RESIDUAL_GATE_THRESHOLD:-0.60}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-200}"
RESIDUAL_ALPHA_GRID=(${RESIDUAL_ALPHA_GRID:-0 0.03125 0.0625 0.125 0.25 0.5})

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
MAIN_LOG="${LOG_DIR}/T${THRESHOLD}_${BUDGET}_runs${RUNS}.log"
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
  echo "[$(timestamp)] [Algo] running ${DATASET} ${FRONTEND_ID} residual + Graph-Bit, runs=${RUNS}"
  set +e
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets "${DATASET}" \
    --runs "${RUNS}" \
    --experiment_suite residual_precision_depth \
    --real_quant_model_name llama2_7b \
    --precision_depth_reference_tag W4A8 \
    --precision_depth_tags W4A6 W4A4 \
    --precision_depth_bits 6 4 \
    --precision_depth_reference_bits 8 \
    --precision_depth_high_ratio "${HIGH_RATIO}" \
    --precision_depth_mid_ratio "${MID_RATIO}" \
    --precision_depth_low_ratio "${LOW_RATIO}" \
    --precision_depth_cost_scale 0.50 \
    --precision_depth_fixed_cost 0.15 \
    --radius 2 \
    --hash_heads_per_route 8 \
    --main_hash_head_bits "${head_bits[@]}" \
    --learned_hash_epochs 10 \
    --learned_hash_dim 128 \
    --hamming_only_acceptor \
    --enable_score_gate \
    --allow_rare_fuzzy \
    --score_reuse_threshold "${THRESHOLD}" \
    --score_propagation_weight 3 \
    --score_graph_context_weight 1 \
    --score_low_unique_weight 1 \
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
    --residual_positive_error_max 0.40 \
    --residual_offline_extra_query_nodes 4096 \
    --residual_offline_negative_anchors_per_node 4 \
    --residual_negative_error_min 0.45 \
    --residual_negative_gate_weight 1.0 \
    --residual_train_split train_val \
    --residual_gate_loss_weight 0.5 \
    --residual_accept_loss_weight 0.0 \
    --residual_gate_error_scale 0.25 \
    --residual_gate_error_max 0.45 \
    --residual_gate_sparsity_weight 0.02 \
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
  --budget "${BUDGET}" \
  --runs "${RUNS}" \
  --frontend-id "${FRONTEND_ID}" \
  --hard-support "${HARD_SUPPORT}" \
  --soft-support "${SOFT_SUPPORT}" \
  --bounded-save-p6 0.50 \
  --bounded-save-p4 0.25

echo "[$(timestamp)] [Done] ${OUT_DIR}/predictor_free_main.txt"
