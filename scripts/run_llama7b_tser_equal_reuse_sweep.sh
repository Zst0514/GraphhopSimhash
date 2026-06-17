#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/llama7b_tser_equal_reuse_sweep}"

DATASETS="${DATASETS:-cora}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
EXPORT_TRACE="${EXPORT_TRACE:-0}"
TS_LIST="${TS_LIST:-20 24 28 31 35 40 45 50}"
TARGET_REUSE="${TARGET_REUSE:-0.35 0.40 0.45 0.50}"

mkdir -p "${OUT_DIR}"
cd "${REPO_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

echo "[$(timestamp)] Equal-reuse TSER sweep"
echo "OUT_DIR=${OUT_DIR}"
echo "DATASETS=${DATASETS}"
echo "TS_LIST=${TS_LIST}"
echo "TARGET_REUSE=${TARGET_REUSE}"

for t in ${TS_LIST}; do
  echo
  echo "[$(timestamp)] sweep T=${t}"
  T_CORA="${t}" \
  T_PUBMED="${t}" \
  T_ARXIV="${t}" \
  T_WIKICS="${t}" \
  T_PRODUCTS="${t}" \
  T_ARXIV23="${t}" \
  DATASETS="${DATASETS}" \
  RUNS="${RUNS}" \
  SEED="${SEED}" \
  FORCE="${FORCE}" \
  EXPORT_TRACE="${EXPORT_TRACE}" \
  OUT_DIR="${OUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO_DIR}/scripts/run_llama7b_tser_score_ablation.sh"
done

"${PYTHON_BIN}" "${REPO_DIR}/scripts/summarize_llama7b_tser_equal_reuse_sweep.py" \
  --log_dir "${OUT_DIR}/logs" \
  --output_dir "${OUT_DIR}" \
  --targets ${TARGET_REUSE}

echo "[$(timestamp)] [Done] ${OUT_DIR}"
