#!/usr/bin/env bash
set -euo pipefail

# Sequential Arxiv low-T policy sweep.
# This script waits for optional running jobs, then launches T values below 24.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"

T_VALUES=(${T_VALUES:-20 22 23})
RUNS="${RUNS:-1}"
REFINE_RATIO="${REFINE_RATIO:-0.25}"
WAIT_FOR_PIDS="${WAIT_FOR_PIDS:-}"
WAIT_PATTERN="${WAIT_PATTERN:-}"
ROOT_OUT="${ROOT_OUT:-${OFA_DIR}/output/progressive_bfp_fullstack_unified_t_sweep}"
QUEUE_LOG="${QUEUE_LOG:-${ROOT_OUT}/arxiv_T20_22_23_sweep.launch.log}"

mkdir -p "${ROOT_OUT}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

wait_for_pid_list() {
  if [[ -z "${WAIT_FOR_PIDS}" ]]; then
    return 0
  fi
  for pid in ${WAIT_FOR_PIDS}; do
    while kill -0 "${pid}" 2>/dev/null; do
      echo "[$(timestamp)] waiting for PID ${pid}"
      sleep 60
    done
  done
}

wait_for_pattern() {
  if [[ -z "${WAIT_PATTERN}" ]]; then
    return 0
  fi
  while pgrep -af "${WAIT_PATTERN}" >/dev/null 2>&1; do
    echo "[$(timestamp)] waiting for pattern: ${WAIT_PATTERN}"
    sleep 60
  done
}

{
  echo "================================================================"
  echo "[$(timestamp)] Arxiv lower-T policy sweep"
  echo "T_VALUES=${T_VALUES[*]} RUNS=${RUNS} REFINE_RATIO=${REFINE_RATIO}"
  echo "ROOT_OUT=${ROOT_OUT}"
  echo "================================================================"

  wait_for_pid_list
  wait_for_pattern

  for t in "${T_VALUES[@]}"; do
    out_dir="${ROOT_OUT}/arxiv_T${t}_r${REFINE_RATIO}_runs${RUNS}"
    echo "[$(timestamp)] [Start] Arxiv T=${t} -> ${out_dir}"
    DATASET=arxiv \
      RUNS="${RUNS}" \
      THRESHOLD="${t}" \
      REFINE_RATIO="${REFINE_RATIO}" \
      FORCE="${FORCE:-0}" \
      OUT_DIR="${out_dir}" \
      bash "${SCRIPT_DIR}/run_progressive_bfp_fullstack.sh"
    echo "[$(timestamp)] [Done] Arxiv T=${t}"
  done

  echo "[$(timestamp)] [All done]"
} 2>&1 | tee -a "${QUEUE_LOG}"

