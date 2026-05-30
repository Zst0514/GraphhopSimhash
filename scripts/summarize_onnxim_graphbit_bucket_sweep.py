#!/usr/bin/env python3
"""Summarize Graph-Bit bucket-size and weight-stationary reuse sweeps.

The sweep separates two effects:

1. `seq_len` changes the GEMM M dimension and represents a real risk-bucket
   micro-batch size.
2. `stationary_tile_batch` is an explicit weight-stationary scheduling
   assumption: how many same-risk node blocks reuse a loaded W tile.

This script does not hide that distinction.  Tables show both dimensions so the
mainline result can use stationary_tile_batch == baseline, while larger values
remain sensitivity points.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing aggregate: {path}")
    return json.loads(path.read_text())


def enc(workspace: Path) -> dict[str, Any]:
    payload = load_json(workspace / "aggregate.json")
    return payload["encoder"]


def val(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0.0 else num / den


def pct_drop(norm: float) -> float:
    return 100.0 * (1.0 - norm)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def make_row(root: Path, seq_len: int, case: str, baseline: dict[str, Any]) -> dict[str, Any]:
    workspace = root / f"s{seq_len}" / case
    row = enc(workspace)
    cycles = val(row, "cycles")
    traffic = val(row, "dram_read_requests") + val(row, "dram_write_requests")
    base_cycles = val(baseline, "cycles")
    base_traffic = val(baseline, "dram_read_requests") + val(baseline, "dram_write_requests")
    act_actual = val(row, "mem_read_input_actual")
    act_orig = val(row, "mem_read_input_original")
    w_actual = val(row, "mem_read_weight")
    w_orig = val(row, "mem_read_weight_original")
    cycle_norm = safe_div(cycles, base_cycles)
    traffic_norm = safe_div(traffic, base_traffic)
    # Simple normalized proxy.  Keep it explicit and conservative: average of
    # cycle and DRAM-request normalization relative to FullP8 at the same seq.
    energy_norm = 0.5 * cycle_norm + 0.5 * traffic_norm
    parts = case.split("_")
    stationary = "-"
    if case.startswith("gb_ws"):
        stationary = parts[-1].removeprefix("b")
    return {
        "seq_len": seq_len,
        "case": case,
        "stationary_tile_batch": stationary,
        "cycles": cycles,
        "traffic": traffic,
        "cycle_norm": cycle_norm,
        "traffic_norm": traffic_norm,
        "energy_norm": energy_norm,
        "cycle_reduction_pct": pct_drop(cycle_norm),
        "traffic_reduction_pct": pct_drop(traffic_norm),
        "energy_reduction_pct": pct_drop(energy_norm),
        "act_read": act_actual,
        "act_orig": act_orig,
        "act_ratio": safe_div(act_actual, act_orig),
        "weight_read": w_actual,
        "weight_orig": w_orig,
        "weight_ratio": safe_div(w_actual, w_orig),
        "avg_depth": row.get("graphbit_avg_depth"),
        "fetch_depth": row.get("graphbit_avg_fetch_depth"),
        "issue_depth": row.get("graphbit_avg_issue_depth"),
        "wrf_depth": row.get("graphbit_avg_weight_rf_depth"),
        "psum_depth": row.get("graphbit_avg_psum_depth"),
        "workspace": str(root / f"s{seq_len}" / case),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for seq_len in args.seq_lens:
        baseline = enc(args.root / f"s{seq_len}" / "full_p8")
        for case in args.cases:
            rows.append(make_row(args.root, seq_len, case, baseline))

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "bucket_sweep_summary.tsv"
    txt_path = output_dir / "bucket_sweep_summary.txt"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Graph-Bit bucket-size / weight-stationary sweep",
        f"Source: {args.root}",
        "",
        "EnergyNorm is a simple proxy: 0.5 * cycle_norm + 0.5 * traffic_norm.",
        "stationary_tile_batch='-' means no extra W HBM amortization assumption.",
        "",
        (
            f"{'seq':>4s} {'case':>14s} {'tileB':>6s} {'cycles':>11s} "
            f"{'CycRed':>7s} {'TrafRed':>7s} {'EnerRed':>7s} "
            f"{'act/orig':>8s} {'w/orig':>7s} {'fetch':>5s} {'issue':>5s}"
        ),
        "-" * 100,
    ]
    for row in rows:
        lines.append(
            f"{row['seq_len']:4d} {row['case']:>14s} {str(row['stationary_tile_batch']):>6s} "
            f"{row['cycles']:11.0f} {fmt_pct(row['cycle_reduction_pct']):>7s} "
            f"{fmt_pct(row['traffic_reduction_pct']):>7s} {fmt_pct(row['energy_reduction_pct']):>7s} "
            f"{row['act_ratio']:8.3f} {row['weight_ratio']:7.3f} "
            f"{float(row['fetch_depth'] or 0.0):5.2f} {float(row['issue_depth'] or 0.0):5.2f}"
        )
    lines.extend(
        [
            "",
            "How to read:",
            "- gb_now is the conservative Graph-Bit datapath: plane-group activation + runtime tile bound, no extra W HBM reuse.",
            "- gb_ws_bX keeps the same bound but assumes the scheduler can reuse a W tile across X same-risk node blocks.",
            "- If gb_now helps little but gb_ws_bX improves with X, the bottleneck is weight-side reuse, not the early-stop bound.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[GraphBitBucketSweep] wrote {tsv_path}")
    print(f"[GraphBitBucketSweep] wrote {txt_path}")
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
