#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhangshangtong/Transformer/OFA}"
REPO="$ROOT/GraphhopSimhash"
WAIT_PID="${WAIT_PID:-3645474}"
POLL_SEC="${POLL_SEC:-300}"

LOG_DIR="${LOG_DIR:-$ROOT/output/progressive_bfp_policy_eval/logs}"
mkdir -p "$LOG_DIR"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

if [[ -n "$WAIT_PID" && "$WAIT_PID" != "0" ]]; then
  log "[Wait] waiting for predecessor pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SEC"
  done
  log "[Wait done] predecessor pid=$WAIT_PID exited"
fi

cd "$REPO"

log "[Arxiv queue start] DATASETS=arxiv TASKS=AR"
env \
  DATASETS="${DATASETS:-arxiv}" \
  TASKS="${TASKS:-AR}" \
  RATIOS="${RATIOS:-0.10 0.15 0.20 0.25 0.30}" \
  RUNS="${RUNS:-10}" \
  MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}" \
  GPU_WAIT_SEC="${GPU_WAIT_SEC:-120}" \
  EVAL_EXISTING_THRESHOLD="${EVAL_EXISTING_THRESHOLD:-0}" \
  bash "$REPO/scripts/run_progressive_bfp_policy_eval_queue.sh"

log "[Arxiv queue done]"
