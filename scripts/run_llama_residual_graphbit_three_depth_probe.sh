#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/output/residual_graphbit_three_depth_probe}"

RUNS="${RUNS:-3}"
RESIDUAL_FIT_PROFILE="${RESIDUAL_FIT_PROFILE:-llama}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-120}"
RESIDUAL_RANK="${RESIDUAL_RANK:-64}"
RESIDUAL_MAX_TRAIN_PAIRS="${RESIDUAL_MAX_TRAIN_PAIRS:-4096}"
RESIDUAL_HARD_MIN_SUPPORT_HITS="${RESIDUAL_HARD_MIN_SUPPORT_HITS:-5}"
RESIDUAL_SOFT_MIN_SUPPORT_HITS="${RESIDUAL_SOFT_MIN_SUPPORT_HITS:-4}"

# dataset heads threshold
CASES=(
  "cora 8 40"
  "pubmed 8 40"
)

# name high mid low. Mapping with tags W4A6/W4A4:
# high -> P8/W4A8, mid -> P6/W4A6, low + remainder -> P4/W4A4.
BUDGETS=(
  "conservative 0.30 0.50 0.20"
  "balanced 0.20 0.50 0.30"
  "aggressive 0.20 0.40 0.40"
)

mkdir -p "${OUT_DIR}/logs"
cd "$ROOT_DIR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_case() {
  local dataset="$1"
  local heads="$2"
  local threshold="$3"
  local budget_name="$4"
  local high="$5"
  local mid="$6"
  local low="$7"

  local log_dir="${OUT_DIR}/logs/${dataset}/h${heads}"
  local log_path="${log_dir}/T${threshold}_${budget_name}_runs${RUNS}.log"
  local done_path="${log_path}.done"
  mkdir -p "$log_dir"

  if [[ -e "$done_path" ]]; then
    echo "[$(timestamp)] [Skip] dataset=${dataset} heads=${heads} T=${threshold} budget=${budget_name}"
    return
  fi

  local head_bits=()
  for ((i = 0; i < heads; i++)); do
    head_bits+=(16)
  done

  echo
  echo "================================================================"
  echo "[$(timestamp)] [Run] dataset=${dataset} heads=${heads}x16 T=${threshold} hard>=${RESIDUAL_HARD_MIN_SUPPORT_HITS} soft=${RESIDUAL_SOFT_MIN_SUPPORT_HITS} budget=${budget_name} P8/P6/P4=${high}/${mid}/${low}"
  echo "[Log] ${log_path}"
  echo "================================================================"

  set +e
  "$PYTHON_BIN" -m GraphhopSimhash \
    --datasets "$dataset" \
    --runs "$RUNS" \
    --experiment_suite residual_precision_depth \
    --real_quant_model_name llama2_7b \
    --precision_depth_reference_tag W4A8 \
    --precision_depth_tags W4A6 W4A4 \
    --precision_depth_bits 6 4 \
    --precision_depth_reference_bits 8 \
    --precision_depth_high_ratio "$high" \
    --precision_depth_mid_ratio "$mid" \
    --precision_depth_low_ratio "$low" \
    --precision_depth_cost_scale 0.50 \
    --precision_depth_fixed_cost 0.15 \
    --radius 2 \
    --hash_heads_per_route "$heads" \
    --main_hash_head_bits "${head_bits[@]}" \
    --learned_hash_epochs 10 \
    --learned_hash_dim 128 \
    --hamming_only_acceptor \
    --enable_score_gate \
    --allow_rare_fuzzy \
    --score_reuse_threshold "$threshold" \
    --score_propagation_weight 3 \
    --score_graph_context_weight 1 \
    --score_low_unique_weight 1 \
    --residual_fit_profile "$RESIDUAL_FIT_PROFILE" \
    --residual_rank "$RESIDUAL_RANK" \
    --residual_epochs "$RESIDUAL_EPOCHS" \
    --residual_max_train_pairs "$RESIDUAL_MAX_TRAIN_PAIRS" \
    --residual_hard_min_support_hits "$RESIDUAL_HARD_MIN_SUPPORT_HITS" \
    --residual_soft_min_support_hits "$RESIDUAL_SOFT_MIN_SUPPORT_HITS" \
    --residual_alpha_grid 0 0.125 0.25 0.5 \
    --residual_min_dist 1.0 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[$(timestamp)] [Failed] dataset=${dataset} heads=${heads} T=${threshold} budget=${budget_name} status=${status}"
    return "$status"
  fi
  touch "$done_path"
}

for case_spec in "${CASES[@]}"; do
  read -r dataset heads threshold <<< "$case_spec"
  for budget_spec in "${BUDGETS[@]}"; do
    read -r budget_name high mid low <<< "$budget_spec"
    run_case "$dataset" "$heads" "$threshold" "$budget_name" "$high" "$mid" "$low"
  done
done

echo
echo "================================================================"
echo "[$(timestamp)] [Done] Logs under ${OUT_DIR}/logs"
echo "================================================================"
