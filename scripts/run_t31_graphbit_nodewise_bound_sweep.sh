#!/usr/bin/env bash
set -euo pipefail

# Fixed T31 reuse/residual front-end with nodewise predictor-free Graph-Bit.
#
# This is the ratio-free path: no high/mid/low percentage budget is used to
# assign P8/P6/P5/P4.  Each miss node maps its graph risk to a tolerance and
# the runtime bound selects the first acceptable activation depth.
#
# Policy format:
#   id:min_depth:min_tol:max_tol:gamma:risk_max:scale
#
# Example:
#   mild:4:0.0:0.02:1.0:15:1.0
#   normal:4:0.0:0.04:1.0:15:1.0
#   strong:4:0.0:0.08:1.0:15:1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASETS=(${DATASETS:-cora})
RUNS="${RUNS:-1}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/t31_graphbit_nodewise_bound_sweep}"
BOUND_PRIORITIES="${BOUND_PRIORITIES:-degree}"

DEFAULT_POLICIES=$'mild:4:0.0:0.02:1.0:15:1.0\nnormal:4:0.0:0.04:1.0:15:1.0\nstrong:4:0.0:0.08:1.0:15:1.0\nsteep:4:0.0:0.04:2.0:15:1.0'
POLICIES="${POLICIES:-${DEFAULT_POLICIES}}"

mkdir -p "${OUT_ROOT}"

MANIFEST="${OUT_ROOT}/manifest.tsv"
printf "policy\tnodewise_min_depth\tnodewise_min_tol\tnodewise_max_tol\tnodewise_gamma\tnodewise_risk_max\tscale\n" > "${MANIFEST}"

run_one() {
  local dataset="$1"
  local policy="$2"
  local min_depth="$3"
  local min_tol="$4"
  local max_tol="$5"
  local gamma="$6"
  local risk_max="$7"
  local scale="$8"

  local out_dir="${OUT_ROOT}/${dataset}_h8_53_T31/${policy}"
  echo "[T31NodewiseBoundSweep] dataset=${dataset} policy=${policy} runs=${RUNS}"

  RUNS="${RUNS}" \
  RUN_ALGO="${RUN_ALGO:-1}" \
  RUN_ONNXIM=0 \
  DATASET="${dataset}" \
  THRESHOLD=31 \
  HARD_SUPPORT=5 \
  SOFT_SUPPORT=3 \
  FRONTEND_ID=h8_53_T31 \
  BUDGET="node_${policy}" \
  HIGH_RATIO=0 \
  MID_RATIO=0 \
  LOW_RATIO=0 \
  OUT_DIR="${out_dir}" \
  PRECISION_DEPTH_TAGS="${PRECISION_DEPTH_TAGS:-W4A7 W4A6 W4A5 W4A4}" \
  PRECISION_DEPTH_BITS="${PRECISION_DEPTH_BITS:-7 6 5 4}" \
  BOUND_ENABLE=1 \
  BOUND_ASSIGNMENT=nodewise \
  BOUND_PRIORITIES="${BOUND_PRIORITIES}" \
  BOUND_SCALE="${scale}" \
  BOUND_TILE_K="${BOUND_TILE_K:-128}" \
  BOUND_NODEWISE_MIN_DEPTH="${min_depth}" \
  BOUND_NODEWISE_MIN_TOL="${min_tol}" \
  BOUND_NODEWISE_MAX_TOL="${max_tol}" \
  BOUND_NODEWISE_GAMMA="${gamma}" \
  BOUND_NODEWISE_RISK_MAX="${risk_max}" \
  bash "${SCRIPT_DIR}/run_graphbit_predictor_free_flow.sh"
}

while IFS= read -r spec; do
  [[ -z "${spec}" ]] && continue
  [[ "${spec}" =~ ^# ]] && continue
  IFS=":" read -r policy min_depth min_tol max_tol gamma risk_max scale <<< "${spec}"
  if [[ -z "${policy:-}" || -z "${scale:-}" ]]; then
    echo "[T31NodewiseBoundSweep] invalid policy spec: ${spec}" >&2
    exit 2
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${policy}" "${min_depth}" "${min_tol}" "${max_tol}" "${gamma}" "${risk_max}" "${scale}" >> "${MANIFEST}"
  for dataset in "${DATASETS[@]}"; do
    run_one "${dataset}" "${policy}" "${min_depth}" "${min_tol}" "${max_tol}" "${gamma}" "${risk_max}" "${scale}"
  done
done <<< "${POLICIES}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_t31_graphbit_bound_policy_sweep.py" \
  --root "${OUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-tsv "${OUT_ROOT}/summary.tsv" \
  --output-txt "${OUT_ROOT}/summary.txt" \
  --pareto-tsv "${OUT_ROOT}/pareto.tsv" \
  --pareto-txt "${OUT_ROOT}/pareto.txt"

echo "[T31NodewiseBoundSweep] done: ${OUT_ROOT}"
