#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangshangtong/Transformer/OFA

PY=/home/zhangshangtong/.conda/envs/OFA/bin/python
SCRIPT=GraphhopSimhash/scripts/profile_topology_risk_sensitivity.py
LOG_DIR=output/matched_replacement_ar_after_bfp/logs
mkdir -p "$LOG_DIR"

WAIT_PID="${WAIT_PID:-416397}"
if [[ -n "$WAIT_PID" ]] && ps -p "$WAIT_PID" >/dev/null 2>&1; then
  echo "[Wait] $(date) waiting for PID=$WAIT_PID"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do
    sleep 300
  done
fi

COMMON_ARGS=(
  --runs 5
  --perturbation anchor
  --matched_quality
  --replace_frac 0.10
  --min_support 3
  --risk_pool_frac 0.35
)

echo "[Start] $(date) AR node matched replacement"
"$PY" "$SCRIPT" \
  --datasets arxiv \
  "${COMMON_ARGS[@]}" \
  --output_dir output/matched_replacement_arxiv_runs5 \
  > "$LOG_DIR/arxiv_node.log" 2>&1

echo "[Summarize] $(date)"
"$PY" GraphhopSimhash/scripts/summarize_matched_replacement_motivation.py \
  --output GraphhopSimhash/docs/results/MOTIVATION_MATCHED_REPLACEMENT_FIVE_TASKS.md \
  > "$LOG_DIR/summarize.log" 2>&1

echo "[Done] $(date)"
