#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

TRACE_PATH="${ROOT_DIR}/traces/pubmed_8h16b_r2.trace"
BUILD_DIR="${ROOT_DIR}/cmake-build-release"
REPORT_DIR="${ROOT_DIR}/reports"

python -m CAM_sim.tools.export_graphhop_trace \
  --datasets pubmed \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --route_min_support_hits 3 \
  --radius 2 \
  --output "${TRACE_PATH}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

"${BUILD_DIR}/digital_logic_cpp/digital_hash_reuse" \
  --trace "${TRACE_PATH}" \
  --config "${ROOT_DIR}/digital_logic_cpp/configs/digital_default.json" \
  --out "${REPORT_DIR}/pubmed_digital.json"

"${BUILD_DIR}/analog_cam_cpp/analog_cam_reuse" \
  --trace "${TRACE_PATH}" \
  --config "${ROOT_DIR}/analog_cam_cpp/configs/analog_cam_default.json" \
  --out "${REPORT_DIR}/pubmed_analog_cam.json"

python "${ROOT_DIR}/tools/compare_reports.py" \
  "${REPORT_DIR}/pubmed_digital.json" \
  "${REPORT_DIR}/pubmed_analog_cam.json" \
  --out "${REPORT_DIR}/pubmed_compare.md"

echo "[run] report: ${REPORT_DIR}/pubmed_compare.md"
