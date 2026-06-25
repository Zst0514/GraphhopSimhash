#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_ROOT="${OUT_ROOT:-output/motivation_block_lift_profile_remaining}"
RUNS="${RUNS:-5}"
GEN_RUNS="${GEN_RUNS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "$ROOT_DIR"
mkdir -p "$OUT_ROOT/logs"

log_step() {
  echo
  echo "================================================================"
  date
  echo "$*"
  echo "================================================================"
}

run_cmd() {
  local name="$1"
  shift
  local log="$OUT_ROOT/logs/${name}.log"
  log_step "$name" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[FAILED] $name status=$status" | tee -a "$log"
    return "$status"
  fi
  echo "[DONE] $name" | tee -a "$log"
}

generate_pool() {
  local dataset="$1"
  local policy="$2"
  local tag_name="$3"
  run_cmd "generate_${dataset}_${tag_name}" \
    "$PYTHON_BIN" GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py \
      --dataset "$dataset" \
      --llm_name llama2_7b \
      --block_size 256 \
      --base_mantissa 4 \
      --refine_mantissa 6 \
      --selection_policy "$policy" \
      --lift_ratio 0.20 \
      --save_to_cache \
      --runs "$GEN_RUNS" \
      --seed 42 \
      --batch_size "$BATCH_SIZE"
}

profile_tasks() {
  local dataset_key="$1"
  local tasks="$2"
  local dynamic_tag="$3"
  local tag_name="$4"
  run_cmd "profile_${dataset_key}_${tag_name}" \
    "$PYTHON_BIN" GraphhopSimhash/scripts/profile_dual_granularity_bfp_oracle.py \
      --tasks $tasks \
      --runs "$RUNS" \
      --dynamic_tag "$dynamic_tag" \
      --output_dir "$OUT_ROOT/${dataset_key}_${tag_name}"
}

RANDOM_TAG="W4BlockBFPA4to6_B256_random20"
STRESS_TAG="W4BlockBFPA4to6_B256_oracle20"

# Cora pools already exist from the CN profiling; only profile the CL task.
profile_tasks "cora_CL" "CL" "$RANDOM_TAG" "random20"
profile_tasks "cora_CL" "CL" "$STRESS_TAG" "stress20"

# PubMed covers PN and PL.
generate_pool "pubmed" "random" "random20"
profile_tasks "pubmed_PN_PL" "PN PL" "$RANDOM_TAG" "random20"
generate_pool "pubmed" "oracle_error" "stress20"
profile_tasks "pubmed_PN_PL" "PN PL" "$STRESS_TAG" "stress20"

# Wiki-CS covers WK.
generate_pool "wikics" "random" "random20"
profile_tasks "wikics_WK" "WK" "$RANDOM_TAG" "random20"
generate_pool "wikics" "oracle_error" "stress20"
profile_tasks "wikics_WK" "WK" "$STRESS_TAG" "stress20"

# OGBN-Arxiv covers AR and is expected to dominate runtime.
generate_pool "arxiv" "random" "random20"
profile_tasks "arxiv_AR" "AR" "$RANDOM_TAG" "random20"
generate_pool "arxiv" "oracle_error" "stress20"
profile_tasks "arxiv_AR" "AR" "$STRESS_TAG" "stress20"

log_step "All remaining block-lift profiling jobs finished" | tee "$OUT_ROOT/logs/done.log"
