#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS="${DATASETS:-pubmed wikics}"
POINTS="${POINTS:-0.25:0.20}"
RUNS="${RUNS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
RISK_MODE="${RISK_MODE:-tser}"
STRESS_SCALE="${STRESS_SCALE:-8.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_ROOT="${OUT_ROOT:-output/dual_granularity_bfp_oracle}"

cd "$ROOT_DIR"
mkdir -p "$OUT_ROOT/logs"

tasks_for_dataset() {
  case "$1" in
    cora) echo "CN CL" ;;
    pubmed) echo "PN PL" ;;
    wikics) echo "WK" ;;
    arxiv) echo "AR" ;;
    *) echo "Unknown dataset: $1" >&2; return 1 ;;
  esac
}

for dataset in $DATASETS; do
  tasks="$(tasks_for_dataset "$dataset")"
  for point in $POINTS; do
    top_frac="${point%%:*}"
    threshold="${point##*:}"
    top_tag="$("$PYTHON_BIN" -c "print(str(int(round(float('$top_frac') * 100))))")"
    threshold_tag="$("$PYTHON_BIN" -c "print(str(float('$threshold')))")"
    # The generator's canonical tag keeps threshold formatting from Python.
    dyn_tag="W4GraphBFPA4to6_B${BLOCK_SIZE}_${RISK_MODE}_top${top_tag}_t${threshold_tag}"

    log="$OUT_ROOT/logs/${dataset}_${dyn_tag}.log"
    out_dir="$OUT_ROOT/${dataset}_${dyn_tag}"

    {
      echo "================================================================"
      date
      echo "dataset=$dataset tasks=$tasks top_frac=$top_frac threshold=$threshold"
      echo "dynamic_tag=$dyn_tag"
      echo "================================================================"

      "$PYTHON_BIN" GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py \
        --dataset "$dataset" \
        --llm_name llama2_7b \
        --risk_mode "$RISK_MODE" \
        --top_risk_frac "$top_frac" \
        --threshold "$threshold" \
        --stress_scale "$STRESS_SCALE" \
        --block_size "$BLOCK_SIZE" \
        --base_mantissa 4 \
        --refine_mantissa 6 \
        --save_to_cache \
        --runs 3 \
        --seed 42 \
        --batch_size "$BATCH_SIZE"

      "$PYTHON_BIN" GraphhopSimhash/scripts/profile_dual_granularity_bfp_oracle.py \
        --tasks $tasks \
        --runs "$RUNS" \
        --dynamic_tag "$dyn_tag" \
        --output_dir "$out_dir"
    } 2>&1 | tee "$log"
  done
done
