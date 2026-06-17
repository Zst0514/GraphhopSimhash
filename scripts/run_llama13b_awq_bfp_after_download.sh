#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="$(cd "${REPO_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
DOWNLOAD_PID_FILE="${DOWNLOAD_PID_FILE:-${OFA_DIR}/output/download_llama2_13b_modelscope.pid}"
CURRENT_POOL_PID_FILE="${CURRENT_POOL_PID_FILE:-${OFA_DIR}/output/encoder_awq_bfp_pools/run_all.pid}"
MODEL_DIR="${MODEL_DIR:-${OFA_DIR}/models/llama-13b/modelscope/Llama-2-13b-ms}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/llama13b_awq_bfp_pools}"
DATASETS="${DATASETS:-cora pubmed arxiv wikics tape_products tape_arxiv23}"
CONFIGS="${CONFIGS:-W4A8 W4BFPA3_B256 W4BFPA4_B256 W4BFPA5_B256 W4BFPA6_B256}"

timestamp() {
  date +"%F %T"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

wait_pid_file() {
  local pid_file="$1"
  local name="$2"
  if [[ ! -f "${pid_file}" ]]; then
    log_msg "[Wait] ${name}: pid file not found, skip (${pid_file})"
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    log_msg "[Wait] ${name}: empty pid file, skip"
    return 0
  fi
  while kill -0 "${pid}" 2>/dev/null; do
    log_msg "[Wait] ${name} pid=${pid} still running"
    sleep 300
  done
  log_msg "[Wait] ${name} pid=${pid} finished"
}

check_model_dir() {
  log_msg "[Check] model_dir=${MODEL_DIR}"
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json
root = Path("${MODEL_DIR}")
index = root / "model.safetensors.index.json"
if not index.exists():
    raise SystemExit(f"Missing {index}")
data = json.loads(index.read_text())
files = sorted(set(data.get("weight_map", {}).values()))
missing = [f for f in files if not (root / f).exists()]
if missing:
    raise SystemExit("Missing safetensor shards: " + ", ".join(missing))
print("OK", root, len(files), "safetensor shards")
PY
}

log_msg "Llama2-13B AWQ/BFP generation watcher starts"
wait_pid_file "${DOWNLOAD_PID_FILE}" "llama2_13b_download"
check_model_dir

# Avoid competing with the currently running ST/e5/e5-large pool generation.
wait_pid_file "${CURRENT_POOL_PID_FILE}" "existing_encoder_pool_generation"

log_msg "[Generate] llama2_13b datasets=${DATASETS}"
log_msg "[Generate] configs=${CONFIGS}"

cd "${REPO_DIR}"
GRAPHHOP_LLAMA2_13B_PATH="${MODEL_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUT_ROOT="${OUT_ROOT}" \
DATASETS="${DATASETS}" \
MODELS="llama2_13b" \
CONFIGS="${CONFIGS}" \
BATCH_DEFAULT=1 \
AWQ_CALIB_SAMPLES="${AWQ_CALIB_SAMPLES:-128}" \
AWQ_SEQLEN="${AWQ_SEQLEN:-512}" \
WAIT_FOR_EXISTING=1 \
bash scripts/generate_encoder_awq_bfp_pools.sh

log_msg "Llama2-13B AWQ/BFP generation watcher completed"
