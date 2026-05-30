#!/usr/bin/env bash
set -euo pipefail

# Run a small ONNXim suite that separates Graph-Bit datapath mechanisms:
#
#   1. byte-major + mask only
#   2. byte-major + issue/RF/psum gating
#   3. plane-group activation demand fetch
#   4. predictor-free bound stop
#   5. risk-bucket disabled control
#   6. weight-stationary HBM amortization
#
# This is intentionally a microbenchmark suite.  The full graph workload mix is
# still supplied by residual/Graph-Bit summaries; this script checks whether
# GemmWS/SystolicWS expose the right hardware counters.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFA_DIR="$(cd "${REPO_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/home/zhangshangtong/.conda/envs/OFA/bin/python}"
SEQ_LEN="${SEQ_LEN:-16}"
LOG_LEVEL="${LOG_LEVEL:-info}"
OUT_ROOT="${OUT_ROOT:-${OFA_DIR}/output/onnxim_graphbit/datapath_suite_s${SEQ_LEN}}"

run_case() {
  local name="$1"
  shift
  local workspace="${OUT_ROOT}/${name}"
  echo "[GraphBitDatapath] ${name} -> ${workspace}"
  "${PYTHON}" "${REPO_DIR}/scripts/onnxim_graphbit_microbench.py" \
    --seq-len "${SEQ_LEN}" \
    --workspace "${workspace}" \
    --action all \
    --log-level "${LOG_LEVEL}" \
    "$@"
}

mkdir -p "${OUT_ROOT}"

run_case "full_p8" \
  --graphbit-depth 8

run_case "byte_major_mask_only_p6" \
  --graphbit-depth 6 \
  --graphbit-activation-layout byte_major \
  --disable-graphbit-issue-gate

run_case "byte_major_issue_rf_psum_p6" \
  --graphbit-depth 6 \
  --graphbit-activation-layout byte_major \
  --graphbit-weight-rf-gate \
  --graphbit-psum-gate

run_case "plane_group2_issue_rf_psum_p6" \
  --graphbit-depth 6 \
  --graphbit-activation-layout plane_group \
  --graphbit-plane-group-bits 2 \
  --graphbit-weight-rf-gate \
  --graphbit-psum-gate

run_case "plane_group2_bound_low" \
  --graphbit-depth 8 \
  --graphbit-min-depth 4 \
  --graphbit-bound-enable \
  --graphbit-bound-tolerance 0.04 \
  --graphbit-activation-layout plane_group \
  --graphbit-plane-group-bits 2 \
  --graphbit-weight-rf-gate \
  --graphbit-psum-gate

run_case "no_risk_bucket_p6" \
  --graphbit-depth 6 \
  --graphbit-activation-layout plane_group \
  --graphbit-plane-group-bits 2 \
  --graphbit-weight-rf-gate \
  --graphbit-psum-gate \
  --disable-graphbit-risk-bucket

# Sensitivity only: this assumes a 4x larger effective weight-stationary
# bucket.  It is useful to show the upper-bound impact of W HBM amortization,
# but it is not the default Graph-Bit datapath claim.
run_case "ws_sensitivity_4x_p6" \
  --graphbit-depth 6 \
  --graphbit-activation-layout plane_group \
  --graphbit-plane-group-bits 2 \
  --graphbit-weight-rf-gate \
  --graphbit-psum-gate \
  --graphbit-weight-stationary \
  --graphbit-baseline-weight-tile-batch 16 \
  --graphbit-weight-stationary-tile-batch 64

"${PYTHON}" - <<'PY' "${OUT_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for agg in sorted(root.glob("*/aggregate.json")):
    payload = json.loads(agg.read_text())
    enc = payload["encoder"]
    rows.append(
        {
            "case": agg.parent.name,
            "cycles": enc.get("cycles", 0),
            "read": enc.get("dram_read_requests", 0),
            "act_read": enc.get("mem_read_input_actual", 0),
            "act_orig": enc.get("mem_read_input_original", 0),
            "w_read": enc.get("mem_read_weight", 0),
            "w_orig": enc.get("mem_read_weight_original", 0),
            "avg_depth": enc.get("graphbit_avg_depth"),
            "fetch": enc.get("graphbit_avg_fetch_depth"),
            "issue": enc.get("graphbit_avg_issue_depth"),
            "wrf": enc.get("graphbit_avg_weight_rf_depth"),
            "psum": enc.get("graphbit_avg_psum_depth"),
        }
    )

out = root / "datapath_summary.tsv"
with out.open("w") as f:
    headers = list(rows[0])
    f.write("\t".join(headers) + "\n")
    for row in rows:
        f.write("\t".join(str(row[h]) for h in headers) + "\n")

txt = root / "datapath_summary.txt"
with txt.open("w") as f:
    f.write("Graph-Bit ONNXim datapath suite\n")
    f.write(f"Source: {root}\n\n")
    f.write(
        f"{'case':34s} {'cycles':>12s} {'act':>12s} {'act/orig':>9s} "
        f"{'w':>12s} {'w/orig':>9s} {'fetch':>7s} {'issue':>7s} {'wrf':>7s} {'psum':>7s}\n"
    )
    f.write("-" * 124 + "\n")
    for r in rows:
        act_ratio = (r["act_read"] / r["act_orig"]) if r["act_orig"] else 0.0
        w_ratio = (r["w_read"] / r["w_orig"]) if r["w_orig"] else 0.0
        f.write(
            f"{r['case']:34s} {r['cycles']:12.0f} {r['act_read']:12.0f} {act_ratio:9.3f} "
            f"{r['w_read']:12.0f} {w_ratio:9.3f} "
            f"{(r['fetch'] or 0):7.2f} {(r['issue'] or 0):7.2f} "
            f"{(r['wrf'] or 0):7.2f} {(r['psum'] or 0):7.2f}\n"
        )
print(f"[GraphBitDatapath] wrote {out}")
print(f"[GraphBitDatapath] wrote {txt}")
PY

echo "[GraphBitDatapath] done: ${OUT_ROOT}"
