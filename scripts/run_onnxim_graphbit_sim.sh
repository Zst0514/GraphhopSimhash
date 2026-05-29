#!/usr/bin/env bash
set -euo pipefail

# End-to-end ONNXim + Graph-Bit hardware flow.
#
# Steps:
#   1. Generate/run LLaMA GEMM microbenchmarks in ONNXim.
#   2. Optionally run ONNXim internal GemmWS Graph-Bit P8/P6/P4.
#   3. Export residual-reuse + Graph-Bit workload profiles.
#   4. Combine both into normalized cycles/traffic/energy proxy tables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LEN="${SEQ_LEN:-64}"
SOURCE_TSV="${SOURCE_TSV:-${OFA_DIR}/output/residual_graphbit_three_depth_probe/three_depth_summary.tsv}"
OUT_DIR="${OUT_DIR:-${OFA_DIR}/output/onnxim_graphbit}"
WORKLOAD_JSON="${OUT_DIR}/workloads/three_depth_deg_profiles.json"
MICROBENCH_JSON="${OUT_DIR}/microbench_s${SEQ_LEN}/aggregate.json"
RUN_INTERNAL="${RUN_INTERNAL:-1}"

mkdir -p "${OUT_DIR}/workloads" "${OUT_DIR}/summary"

echo "[ONNXimGraphBitFlow] seq_len=${SEQ_LEN}"
echo "[ONNXimGraphBitFlow] source=${SOURCE_TSV}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
  --seq-len "${SEQ_LEN}" \
  --action all \
  --log-level info

if [[ "${RUN_INTERNAL}" == "1" ]]; then
  for depth in 8 6 4; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
      --seq-len "${SEQ_LEN}" \
      --workspace "${OUT_DIR}/microbench_s${SEQ_LEN}_internal_p${depth}" \
      --graphbit-depth "${depth}" \
      --action all \
      --log-level info
  done

  "${PYTHON_BIN}" "${SCRIPT_DIR}/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --workspace "${OUT_DIR}/microbench_s${SEQ_LEN}_internal_bound_t006" \
    --graphbit-depth 8 \
    --graphbit-bound-enable \
    --graphbit-bound-tolerance 0.06 \
    --action all \
    --log-level info
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_graphbit_workload.py" \
  --source "${SOURCE_TSV}" \
  --config Deg \
  --output "${WORKLOAD_JSON}" \
  --pretty

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_onnxim_graphbit.py" \
  --workload "${WORKLOAD_JSON}" \
  --microbench "${MICROBENCH_JSON}" \
  --bounded-save-p6 0.5 \
  --bounded-save-p4 0.25

echo "[ONNXimGraphBitFlow] compact summary:"
cat "${OUT_DIR}/summary/three_depth_deg_profiles_compact.txt"

if [[ "${RUN_INTERNAL}" == "1" ]]; then
  echo "[ONNXimGraphBitFlow] internal GemmWS summary:"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

out = Path("${OUT_DIR}")
seq = "${SEQ_LEN}"
modes = ["p8", "p6", "p4", "bound_t006"]
rows = []
for mode in modes:
    path = out / f"microbench_s{seq}_internal_{mode}" / "aggregate.json"
    if not path.exists():
        continue
    enc = json.loads(path.read_text())["encoder"]
    rows.append((mode, int(enc["cycles"]), int(enc["dram_read_requests"]), int(enc["dram_write_requests"])))
if rows:
    base_cycles = rows[0][1]
    base_reads = rows[0][2]
    print("mode        cycles      cyc_ratio   read_req     read_ratio  write_req")
    for mode, cycles, reads, writes in rows:
        print(f"{mode:<10} {cycles:<11} {cycles/base_cycles:>8.3f}   {reads:<11} {reads/base_reads:>8.3f}   {writes}")
PY
fi
