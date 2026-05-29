#!/usr/bin/env bash
set -euo pipefail

# Cora fast hardware-validation flow for the current Graph-Bit NPU line.
#
# Fixed front-end:
#   h8_54_T40, R=2, 8 x 16-bit heads
#   hard>=5 direct reuse, soft==4 residual correction
#   miss nodes -> Degree Graph-Bit
#
# Outputs:
#   output/graphbit_predictor_free/cora_h8_54_T40/summary.tsv
#   output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_main.txt
#
# Useful overrides:
#   RUN_ALGO=1      rerun residual_precision_depth instead of using an existing log
#   RUN_ONNXIM=1    rerun ONNXim LLaMA GEMM microbenchmarks
#   BUILD_ONNXIM=1  try to build ONNXim before running microbenchmarks
#   RUNS=1          quick smoke test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/graphbit_predictor_free/cora_h8_54_T40}"
LOG_DIR="${OUT_DIR}/logs/cora/h8"
RUNS="${RUNS:-10}"
RUN_ALGO="${RUN_ALGO:-0}"
RUN_ONNXIM="${RUN_ONNXIM:-0}"
BUILD_ONNXIM="${BUILD_ONNXIM:-0}"
SEQ_LEN="${SEQ_LEN:-64}"

SUMMARY_TSV="${OUT_DIR}/summary.tsv"
SUMMARY_TXT="${OUT_DIR}/summary.txt"
MICROBENCH_JSON="${OFA_DIR}/output/onnxim_graphbit/microbench_s${SEQ_LEN}/aggregate.json"
MAIN_LOG="${LOG_DIR}/T40_balanced_runs${RUNS}.log"
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
  local existing="${OFA_DIR}/output/residual_graphbit_main/cora_h8_54_T40/cora_h8_54_T40_runs10.log"
  if [[ "${RUNS}" == "10" && -f "${existing}" ]]; then
    echo "[$(timestamp)] [Seed] using existing Cora h8_54_T40 log: ${existing}"
    cp "${existing}" "${MAIN_LOG}"
    touch "${DONE_PATH}"
  fi
}

run_algo() {
  local head_bits=(16 16 16 16 16 16 16 16)
  echo "[$(timestamp)] [Algo] running Cora h8_54_T40 residual + Graph-Bit, runs=${RUNS}"
  set +e
  "${PYTHON_BIN}" -m GraphhopSimhash \
    --datasets cora \
    --runs "${RUNS}" \
    --experiment_suite residual_precision_depth \
    --real_quant_model_name llama2_7b \
    --precision_depth_reference_tag W4A8 \
    --precision_depth_tags W4A6 W4A4 \
    --precision_depth_bits 6 4 \
    --precision_depth_reference_bits 8 \
    --precision_depth_high_ratio 0.20 \
    --precision_depth_mid_ratio 0.50 \
    --precision_depth_low_ratio 0.30 \
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
    --score_reuse_threshold 40 \
    --score_propagation_weight 3 \
    --score_graph_context_weight 1 \
    --score_low_unique_weight 1 \
    --residual_fit_profile llama \
    --residual_rank 64 \
    --residual_epochs 120 \
    --residual_max_train_pairs 4096 \
    --residual_hard_min_support_hits 5 \
    --residual_soft_min_support_hits 4 \
    --residual_alpha_grid 0 0.125 0.25 0.5 \
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

  for depth in 8 6 4; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${SEQ_LEN}" \
      --workspace "${OFA_DIR}/output/onnxim_graphbit/microbench_s${SEQ_LEN}_internal_p${depth}" \
      --graphbit-depth "${depth}" \
      --action all \
      --log-level info
  done

  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --workspace "${OFA_DIR}/output/onnxim_graphbit/microbench_s${SEQ_LEN}_internal_bound_t006" \
    --graphbit-depth 8 \
    --graphbit-bound-enable \
    --graphbit-bound-tolerance 0.06 \
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
  --dataset cora \
  --heads h8 \
  --threshold 40 \
  --budget balanced \
  --runs "${RUNS}" \
  --bounded-save-p6 0.50 \
  --bounded-save-p4 0.25

echo "[$(timestamp)] [Done] ${OUT_DIR}/predictor_free_main.txt"
