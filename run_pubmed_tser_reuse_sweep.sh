#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNS="${RUNS:-3}"
OUT_DIR="${OUT_DIR:-output/tser_reuse_sweep/pubmed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --datasets pubmed
  --runs "$RUNS"
  --radius 2
  --hash_heads_per_route 4
  --main_hash_head_bits 16 16 16 16
  --learned_hash_epochs 10
  --learned_hash_dim 128
  --hamming_only_acceptor
)

run_case() {
  local name="$1"
  shift
  local log_path="$OUT_DIR/${name}.log"
  echo
  echo "================================================================"
  echo "[Run] $name | runs=$RUNS"
  echo "[Log] $log_path"
  echo "================================================================"
  "$PYTHON_BIN" -m GraphhopSimhash "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$log_path"
}

# 1) Baseline: hash reuse only, no TSER gate.
run_case "00_no_score" \
  --experiment_suite single \
  --disable_score_gate

# 2) Degree-only and TSER component ablations at T=45.
run_case "01_degree_only_T45_w300" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 0 \
  --score_low_unique_weight 0 \
  --allow_rare_fuzzy

run_case "02_prop_context_T45_w310" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 0

run_case "03_prop_unique_T45_w301" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 0 \
  --score_low_unique_weight 1

run_case "04_tser_light_T45_w311" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "05_tser_light_T45_w211" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 2 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "06_tser_light_T45_w111" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 1 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

# 3) No-propagation variants.
run_case "07_context_unique_T45_w011" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 0 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "08_context_only_T45_w010" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 0 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 0

run_case "09_unique_only_T45_w001" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 0 \
  --score_graph_context_weight 0 \
  --score_low_unique_weight 1

# 4) Stronger scores at T=45.
run_case "10_tser_mid_T45_w322" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 2 \
  --score_low_unique_weight 2

run_case "11_tser_prop4_T45_w411" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --score_propagation_weight 4 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

# 5) PubMed-specific threshold sweep for 3/1/1.
# PubMed's hash-only reuse is very high, so stricter thresholds are important.
run_case "12_tser_light_T15_w311" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 15 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "13_tser_light_T20_w311" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 20 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "14_tser_light_T25_w311" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 25 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "15_tser_light_T30_w311" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 30 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "16_tser_light_T35_w311" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 35 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

run_case "17_tser_light_T60_w311" \
  --experiment_suite single \
  --enable_score_gate \
  --score_reuse_threshold 60 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1

echo
echo "================================================================"
echo "[Done] Logs saved under $OUT_DIR"
echo "Quick summary command:"
echo "  rg 'FINAL SUMMARY|R2[[:space:]]+\\||Reuse=' $OUT_DIR/*.log"
echo "================================================================"
