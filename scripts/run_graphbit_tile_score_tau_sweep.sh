#!/usr/bin/env bash
set -euo pipefail

# Sweep the Graph-Bit tile-score threshold.
#
# This uses the shared residual/reuse front-end and changes only the miss-node
# predictor-free stop threshold:
#
#   score = node_risk^alpha * W_strength^beta * low_bit_budget(depth)
#   stop at the lowest depth whose score <= tau
#
# Typical quick run:
#   RUNS=1 TAUS="0.0005 0.001 0.002 0.005" \
#     scripts/run_graphbit_tile_score_tau_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFA_DIR="${OFA_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

DATASET="${DATASET:-cora}"
RUNS="${RUNS:-1}"
TAUS=(${TAUS:-0.0005 0.001 0.002 0.005 0.01})
BASE_OUT_DIR="${BASE_OUT_DIR:-${OFA_DIR}/output/graphbit_tile_score_tau_sweep/${DATASET}}"
export BASE_OUT_DIR

mkdir -p "${BASE_OUT_DIR}"

for tau in "${TAUS[@]}"; do
  safe_tau="${tau//./p}"
  echo "[GraphBitTauSweep] dataset=${DATASET} tau=${tau} runs=${RUNS}"
  DATASET="${DATASET}" \
  RUNS="${RUNS}" \
  RUN_ALGO=1 \
  RUN_ONNXIM=0 \
  BOUND_ENABLE=1 \
  BOUND_RULE=tile_score \
  BOUND_ASSIGNMENT=nodewise \
  BOUND_PRIORITIES="${BOUND_PRIORITIES:-degree}" \
  BOUND_SCORE_TAU="${tau}" \
  BOUND_SCORE_ALPHA="${BOUND_SCORE_ALPHA:-1.0}" \
  BOUND_SCORE_BETA="${BOUND_SCORE_BETA:-1.0}" \
  BOUND_SCORE_W_CAP="${BOUND_SCORE_W_CAP:-2.0}" \
  BOUND_SCORE_W_REFERENCE="${BOUND_SCORE_W_REFERENCE:-1.0}" \
  BOUND_SCORE_NODE_FLOOR="${BOUND_SCORE_NODE_FLOOR:-0.0}" \
  BOUND_W_STRENGTH="${BOUND_W_STRENGTH:-1.0}" \
  FRONTEND_ID="${FRONTEND_ID:-h8_53_T31}" \
  THRESHOLD="${THRESHOLD:-31}" \
  HARD_SUPPORT="${HARD_SUPPORT:-5}" \
  SOFT_SUPPORT="${SOFT_SUPPORT:-3}" \
  OUT_DIR="${BASE_OUT_DIR}/tau_${safe_tau}" \
  "${SCRIPT_DIR}/run_graphbit_predictor_free_flow.sh"
done

python - <<'PY'
import csv
import os
from pathlib import Path

base = Path(os.environ.get("BASE_OUT_DIR", ""))
if not base:
    raise SystemExit(0)

rows = []
for summary in sorted(base.glob("tau_*/summary.tsv")):
    tau = summary.parent.name.replace("tau_", "").replace("p", ".")
    with summary.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("config") in {"FullP8", "DegBoundNode"}:
                rows.append({
                    "tau": tau,
                    "config": row.get("config", ""),
                    "reuse": row.get("reuse", ""),
                    "cost": row.get("cost", ""),
                    "acc": row.get("acc", ""),
                    "drop": row.get("drop", ""),
                    "p8": row.get("P8", ""),
                    "p6": row.get("P6", ""),
                    "p5": row.get("P5", ""),
                    "p4": row.get("P4", ""),
                })

out = base / "tau_sweep_summary.tsv"
with out.open("w", newline="") as f:
    fieldnames = ["tau", "config", "reuse", "cost", "acc", "drop", "p8", "p6", "p5", "p4"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"[GraphBitTauSweep] wrote {out}")
PY
