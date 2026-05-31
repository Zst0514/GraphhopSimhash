#!/usr/bin/env bash
set -euo pipefail

# Export a real residual+Graph-Bit per-node trace and replay it through the
# trace-driven Graph-Bit scheduler.
#
# Default target:
#   Cora / LLaMA-7B / T31 shared retrieval front-end / Degree runtime-bound
#
# Useful overrides:
#   DATASET=pubmed
#   RUNS=3
#   OUT_DIR=output/graphbit_trace_replay/my_run
#   CANDIDATE_BATCHES="32 64 128"
#   SKIP_EXPORT=1   reuse an existing TRACE_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${REPO_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"

DATASET="${DATASET:-cora}"
RUNS="${RUNS:-3}"
THRESHOLD="${THRESHOLD:-31}"
HARD_SUPPORT="${HARD_SUPPORT:-5}"
SOFT_SUPPORT="${SOFT_SUPPORT:-3}"
FRONTEND_ID="${FRONTEND_ID:-h8_${HARD_SUPPORT}${SOFT_SUPPORT}_T${THRESHOLD}}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/graphbit_trace_replay/${DATASET}_${FRONTEND_ID}_t31}"
TRACE_DIR="${TRACE_DIR:-${OUT_DIR}/node_traces}"
TRACE_PATH="${TRACE_PATH:-${TRACE_DIR}/${DATASET}_seed42_DegBound.jsonl}"
REPLAY_DIR="${REPLAY_DIR:-${OUT_DIR}/replay}"
COMPONENTS_ROOT="${COMPONENTS_ROOT:-${OFA_DIR}/output/onnxim_graphbit/risk_bucket_components_s8}"
CANDIDATE_BATCHES=(${CANDIDATE_BATCHES:-32 64})
BASELINE_TILE_BATCH="${BASELINE_TILE_BATCH:-16}"
FULLP8_DROP="${FULLP8_DROP:-}"
GRAPHBIT_DROP="${GRAPHBIT_DROP:-}"
SUMMARY_TSV="${SUMMARY_TSV:-${OUT_DIR}/summary.tsv}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

cd "${OFA_DIR}"

if [[ "${SKIP_EXPORT}" != "1" ]]; then
  RUNS="${RUNS}" \
  RUN_ALGO=1 \
  RUN_ONNXIM=0 \
  DATASET="${DATASET}" \
  THRESHOLD="${THRESHOLD}" \
  HARD_SUPPORT="${HARD_SUPPORT}" \
  SOFT_SUPPORT="${SOFT_SUPPORT}" \
  FRONTEND_ID="${FRONTEND_ID}" \
  BUDGET=boundclean \
  HIGH_RATIO=0.20 \
  MID_RATIO=0.50 \
  LOW_RATIO=0.0 \
  OUT_DIR="${OUT_DIR}" \
  TRACE_EXPORT=1 \
  TRACE_EXPORT_DIR="${TRACE_DIR}" \
  TRACE_EXPORT_CONFIGS='DegBound' \
  BOUND_ENABLE=1 \
  BOUND_PRIORITIES='degree' \
  BOUND_MID_TOL=0.02 \
  BOUND_LOW_TOL=0.04 \
  bash "${SCRIPT_DIR}/run_graphbit_predictor_free_flow.sh"
fi

derive_drop_from_summary() {
  local config="$1"
  "${PYTHON_BIN}" - "${SUMMARY_TSV}" "${config}" <<'PY'
import csv
import sys

path, config = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

matches = [row for row in rows if row.get("config") == config]
if not matches:
    raise SystemExit(f"missing config={config} in {path}")

value = matches[-1].get("drop", "")
print(value.rstrip("%"))
PY
}

if [[ -z "${FULLP8_DROP}" ]]; then
  FULLP8_DROP="$(derive_drop_from_summary FullP8)"
fi
if [[ -z "${GRAPHBIT_DROP}" ]]; then
  if ! GRAPHBIT_DROP="$(derive_drop_from_summary DegBound 2>/dev/null)"; then
    GRAPHBIT_DROP="$(derive_drop_from_summary Deg)"
  fi
fi

echo "[GraphBitTraceReplay] drop profile: FullP8=${FULLP8_DROP}% GraphBit=${GRAPHBIT_DROP}%"

"${PYTHON_BIN}" "${SCRIPT_DIR}/replay_graphbit_trace_scheduler.py" \
  --trace "${TRACE_PATH}" \
  --components-root "${COMPONENTS_ROOT}" \
  --output-dir "${REPLAY_DIR}" \
  --fullp8-drop-percent "${FULLP8_DROP}" \
  --drop-percent "${GRAPHBIT_DROP}" \
  --baseline-tile-batch "${BASELINE_TILE_BATCH}" \
  --candidate-batches "${CANDIDATE_BATCHES[@]}"

echo "[GraphBitTraceReplay] done: ${REPLAY_DIR}"
