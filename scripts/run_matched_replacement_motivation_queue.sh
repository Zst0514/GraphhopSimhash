#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangshangtong/Transformer/OFA

PY=/home/zhangshangtong/.conda/envs/OFA/bin/python
SCRIPT=GraphhopSimhash/scripts/profile_topology_risk_sensitivity.py
LOG_DIR=output/matched_replacement_motivation_queue/logs
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --runs 5
  --perturbation anchor
  --matched_quality
  --replace_frac 0.10
  --min_support 3
  --risk_pool_frac 0.35
)

echo "[Start] $(date) WK node matched replacement"
"$PY" "$SCRIPT" \
  --datasets wikics \
  "${COMMON_ARGS[@]}" \
  --output_dir output/matched_replacement_wikics_runs5 \
  > "$LOG_DIR/wikics_node.log" 2>&1

echo "[Start] $(date) Cora link matched replacement"
"$PY" "$SCRIPT" \
  --datasets cora \
  --downstream_task link \
  --link_epochs 300 \
  "${COMMON_ARGS[@]}" \
  --output_dir output/matched_replacement_cora_link_runs5 \
  > "$LOG_DIR/cora_link.log" 2>&1

echo "[Start] $(date) PubMed link matched replacement"
"$PY" "$SCRIPT" \
  --datasets pubmed \
  --downstream_task link \
  --link_epochs 300 \
  "${COMMON_ARGS[@]}" \
  --output_dir output/matched_replacement_pubmed_link_runs5 \
  > "$LOG_DIR/pubmed_link.log" 2>&1

echo "[Summarize] $(date)"
"$PY" GraphhopSimhash/scripts/summarize_matched_replacement_motivation.py \
  --output GraphhopSimhash/docs/results/MOTIVATION_MATCHED_REPLACEMENT_FIVE_TASKS.md \
  > "$LOG_DIR/summarize.log" 2>&1

echo "[Done] $(date)"
