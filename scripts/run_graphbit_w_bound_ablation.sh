#!/usr/bin/env bash
set -euo pipefail

# W-bound ablation for nodewise predictor-free Graph-Bit:
#   1. No-W-bound: W strength = 1.0
#   2. Global-W-bound: global W-tile p75/p90/p95
#   3. Module-weighted-W-bound: MAC-weighted module-kind p75/p90/p95

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS="${DATASETS:-cora}"
RUNS="${RUNS:-1}"
TILE_K="${TILE_K:-128}"
TILE_N="${TILE_N:-128}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/graphbit_w_bound_ablation}"
PROFILE_DIR="${PROFILE_DIR:-${OFA_DIR}/output/graphbit_w_tile_strength/llama2_7b_k${TILE_K}_n${TILE_N}}"
SWEEP_DIR="${SWEEP_DIR:-${OUT_ROOT}/sweep}"

MIN_DEPTH="${MIN_DEPTH:-4}"
MIN_TOL="${MIN_TOL:-0.0}"
MAX_TOL="${MAX_TOL:-0.04}"
GAMMA="${GAMMA:-1.0}"
RISK_MAX="${RISK_MAX:-15.0}"
BOUND_SCALE="${BOUND_SCALE:-1.0}"

mkdir -p "${OUT_ROOT}" "${PROFILE_DIR}"

if [[ ! -f "${PROFILE_DIR}/global_summary.tsv" ]]; then
  echo "[WBoundAblation] profiling LLaMA W tiles -> ${PROFILE_DIR}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_llama_w_tile_strength.py" \
    --tile_k "${TILE_K}" \
    --tile_n "${TILE_N}" \
    --output_dir "${PROFILE_DIR}"
else
  echo "[WBoundAblation] reuse existing W profile: ${PROFILE_DIR}"
fi

POLICY_FILE="${OUT_ROOT}/policies.txt"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_graphbit_w_bound_policies.py" \
  --profile_dir "${PROFILE_DIR}" \
  --min_depth "${MIN_DEPTH}" \
  --min_tol "${MIN_TOL}" \
  --max_tol "${MAX_TOL}" \
  --gamma "${GAMMA}" \
  --risk_max "${RISK_MAX}" \
  --scale "${BOUND_SCALE}" \
  --output "${POLICY_FILE}" \
  > "${OUT_ROOT}/policies.stdout"

echo "[WBoundAblation] policies:"
cat "${POLICY_FILE}"

POLICIES="$(cat "${POLICY_FILE}")" \
DATASETS="${DATASETS}" \
RUNS="${RUNS}" \
OUT_ROOT="${SWEEP_DIR}" \
BOUND_TILE_K="${TILE_K}" \
bash "${SCRIPT_DIR}/run_t31_graphbit_nodewise_bound_sweep.sh"

cp "${PROFILE_DIR}/global_summary.tsv" "${OUT_ROOT}/w_global_summary.tsv"
cp "${PROFILE_DIR}/manifest.json" "${OUT_ROOT}/w_profile_manifest.json"
cp "${SWEEP_DIR}/summary.tsv" "${OUT_ROOT}/summary.tsv"
cp "${SWEEP_DIR}/summary.txt" "${OUT_ROOT}/summary.txt"
cp "${SWEEP_DIR}/pareto.tsv" "${OUT_ROOT}/pareto.tsv"
cp "${SWEEP_DIR}/pareto.txt" "${OUT_ROOT}/pareto.txt"

echo "[WBoundAblation] done: ${OUT_ROOT}"
