#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
RUNS="${RUNS:-3}"
FORCE="${FORCE:-0}"
EXPORT_TRACE="${EXPORT_TRACE:-0}"
TARGET_REUSE="${TARGET_REUSE:-0.30 0.35 0.40 0.45}"
WAIT_SESSIONS="${WAIT_SESSIONS:-llama7b_reuse_six reuse_safety_no_tser_rest llama7b_tser_score_ablation}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

wait_for_sessions() {
  for sess in ${WAIT_SESSIONS}; do
    while tmux has-session -t "${sess}" 2>/dev/null; do
      echo "[$(timestamp)] waiting for tmux session: ${sess}"
      sleep 300
    done
  done
}

run_one() {
  local dataset="$1"
  local ts_list="$2"
  local out_dir="${OFA_DIR}/output/llama7b_tser_equal_reuse_sweep_${dataset}"

  echo
  echo "================================================================"
  echo "[$(timestamp)] equal-reuse sweep | dataset=${dataset}"
  echo "TS_LIST=${ts_list}"
  echo "OUT_DIR=${out_dir}"
  echo "================================================================"

  DATASETS="${dataset}" \
  RUNS="${RUNS}" \
  FORCE="${FORCE}" \
  EXPORT_TRACE="${EXPORT_TRACE}" \
  TS_LIST="${ts_list}" \
  TARGET_REUSE="${TARGET_REUSE}" \
  OUT_DIR="${out_dir}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO_DIR}/scripts/run_llama7b_tser_equal_reuse_sweep.sh"
}

cd "${REPO_DIR}"

echo "[$(timestamp)] queued equal-reuse TSER sweep for remaining datasets"
echo "WAIT_SESSIONS=${WAIT_SESSIONS}"
echo "TARGET_REUSE=${TARGET_REUSE}"
echo "RUNS=${RUNS}"

wait_for_sessions

# Coarse threshold grids centered around the operating regions observed in the
# frontend sweeps. These are intended to find comparable points near 30%-40%
# reuse first; local refinement can be added after reviewing the closest table.
run_one "pubmed" "16 20 24 28 31 35"
run_one "wikics" "20 24 28 31 35 40"
run_one "tape_products" "16 20 24 28 31 35"
run_one "tape_arxiv23" "16 18 20 22 24 28"
run_one "arxiv" "16 18 20 22 24 28"

echo "[$(timestamp)] [Done] remaining equal-reuse sweeps"
