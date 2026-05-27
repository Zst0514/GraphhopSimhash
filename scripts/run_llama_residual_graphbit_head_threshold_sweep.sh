#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ROOT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${DEFAULT_ROOT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/output/residual_graphbit_head_threshold_sweep}"

DATASETS_STR="${DATASETS:-cora pubmed arxiv}"
HEADS_STR="${HEADS:-4 8}"
CORA_THRESHOLDS_STR="${CORA_THRESHOLDS:-12 15 18 20 22 25 30 35}"
PUBMED_THRESHOLDS_STR="${PUBMED_THRESHOLDS:-12 16 20 24}"
ARXIV_THRESHOLDS_STR="${ARXIV_THRESHOLDS:-12 16 20}"

CORA_RUNS="${CORA_RUNS:-3}"
PUBMED_RUNS="${PUBMED_RUNS:-3}"
ARXIV_RUNS="${ARXIV_RUNS:-1}"

RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-60}"
RESIDUAL_RANK="${RESIDUAL_RANK:-32}"
RESIDUAL_MAX_TRAIN_PAIRS="${RESIDUAL_MAX_TRAIN_PAIRS:-1024}"

mkdir -p "${OUT_DIR}/logs"
cd "$ROOT_DIR"

read -r -a DATASETS_ARR <<< "$DATASETS_STR"
read -r -a HEADS_ARR <<< "$HEADS_STR"
read -r -a CORA_THRESHOLDS_ARR <<< "$CORA_THRESHOLDS_STR"
read -r -a PUBMED_THRESHOLDS_ARR <<< "$PUBMED_THRESHOLDS_STR"
read -r -a ARXIV_THRESHOLDS_ARR <<< "$ARXIV_THRESHOLDS_STR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

dataset_runs() {
  case "$1" in
    cora) echo "$CORA_RUNS" ;;
    pubmed) echo "$PUBMED_RUNS" ;;
    arxiv) echo "$ARXIV_RUNS" ;;
    *) echo "1" ;;
  esac
}

check_required_pools() {
  local dataset="$1"
  local missing=0
  for tag in W4A8 W4A6 W4A5 W4A4; do
    local path="${ROOT_DIR}/cache_data/${dataset}_llama2_7b_oracle_${tag}.pt"
    if [[ ! -s "$path" ]]; then
      echo "[$(timestamp)] [MissingPool] $path"
      missing=1
    fi
  done
  return "$missing"
}

run_case() {
  local dataset="$1"
  local heads="$2"
  local threshold="$3"
  local runs="$4"
  local log_dir="${OUT_DIR}/logs/${dataset}/h${heads}"
  local log_path="${log_dir}/T${threshold}_runs${runs}.log"
  local done_path="${log_path}.done"
  mkdir -p "$log_dir"

  if [[ -e "$done_path" ]]; then
    echo "[$(timestamp)] [Skip] dataset=${dataset} heads=${heads} T=${threshold} runs=${runs}"
    return
  fi

  local head_bits=()
  for ((i = 0; i < heads; i++)); do
    head_bits+=(16)
  done

  echo
  echo "================================================================"
  echo "[$(timestamp)] [Run] dataset=${dataset} heads=${heads}x16 radius=2 T=${threshold} runs=${runs}"
  echo "[Log] ${log_path}"
  echo "================================================================"

  set +e
  "$PYTHON_BIN" -m GraphhopSimhash \
    --datasets "$dataset" \
    --runs "$runs" \
    --experiment_suite residual_precision_depth \
    --real_quant_model_name llama2_7b \
    --precision_depth_reference_tag W4A8 \
    --precision_depth_tags W4A6 W4A5 W4A4 \
    --precision_depth_bits 6 5 4 \
    --precision_depth_reference_bits 8 \
    --precision_depth_high_ratio 0.20 \
    --precision_depth_mid_ratio 0.30 \
    --precision_depth_low_ratio 0.30 \
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
    --residual_rank "$RESIDUAL_RANK" \
    --residual_epochs "$RESIDUAL_EPOCHS" \
    --residual_max_train_pairs "$RESIDUAL_MAX_TRAIN_PAIRS" \
    --residual_min_dist 1.0 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[$(timestamp)] [Failed] dataset=${dataset} heads=${heads} T=${threshold} status=${status}"
    return "$status"
  fi
  touch "$done_path"
}

for dataset in "${DATASETS_ARR[@]}"; do
  check_required_pools "$dataset"
  runs="$(dataset_runs "$dataset")"
  case "$dataset" in
    cora) thresholds=("${CORA_THRESHOLDS_ARR[@]}") ;;
    pubmed) thresholds=("${PUBMED_THRESHOLDS_ARR[@]}") ;;
    arxiv) thresholds=("${ARXIV_THRESHOLDS_ARR[@]}") ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac

  for heads in "${HEADS_ARR[@]}"; do
    for threshold in "${thresholds[@]}"; do
      run_case "$dataset" "$heads" "$threshold" "$runs"
    done
  done
done

"${PYTHON_BIN}" "${REPO_DIR}/scripts/summarize_residual_graphbit_head_threshold_sweep.py" "$OUT_DIR"

echo
echo "================================================================"
echo "[$(timestamp)] [Done] Logs and summaries under ${OUT_DIR}"
echo "  ${OUT_DIR}/summary.tsv"
echo "  ${OUT_DIR}/head_threshold_pivot.txt"
echo "================================================================"
