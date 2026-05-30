#!/usr/bin/env python3
"""Summarize Cora h8_54_T40 Graph-Bit predictor-free early-stop sweep.

The key difference from the earlier static P8/P6/P4 table is that the miss-node
classes are not assigned a fixed bit-depth.  Instead, each class starts from
P8 and lets the ONNXim Graph-Bit bound logic stop at runtime:

  high-risk miss: full P8
  mid-risk miss:  max_depth=8, min_depth=6, tolerance=...
  low-risk miss:  max_depth=8, min_depth=4, tolerance=...

The algorithmic drop is reported from the existing static Degree proxy as a
conservative accuracy anchor; the hardware metrics come from ONNXim internal
bit-plane execution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_percent(value: str) -> float:
    text = str(value or "").strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    value_f = float(text or 0)
    return value_f / 100.0 if abs(value_f) > 1.0 else value_f


def load_summary_row(path: Path, config: str) -> dict[str, str]:
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("dataset") == "cora" and row.get("heads") == "h8" and row.get("T") == "40":
                if row.get("budget") == "balanced" and row.get("config") == config:
                    return row
    raise SystemExit(f"Missing Cora h8 T40 balanced config={config} in {path}")


def load_agg(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing ONNXim aggregate: {path}")
    return json.loads(path.read_text())


def enc(agg: dict[str, Any]) -> dict[str, Any]:
    return agg["encoder"]


def reqs(agg: dict[str, Any]) -> float:
    e = enc(agg)
    return float(e.get("dram_read_requests", 0.0)) + float(e.get("dram_write_requests", 0.0))


def metric(agg: dict[str, Any], key: str) -> float:
    return float(enc(agg).get(key, 0.0) or 0.0)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def bitcomp_ratio(agg: dict[str, Any]) -> float:
    raw = metric(agg, "graphbit_raw_compute_cycles")
    if raw <= 0:
        return 1.0
    return metric(agg, "graphbit_effective_compute_cycles") / raw


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "output" / "graphbit_predictor_free" / "cora_h8_54_T40" / "summary.tsv",
    )
    parser.add_argument(
        "--microbench-dir",
        type=Path,
        default=root / "output" / "onnxim_graphbit",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "graphbit_predictor_free" / "cora_h8_54_T40" / "earlystop_sweep",
    )
    parser.add_argument("--cache-compute-cost", type=float, default=0.001)
    parser.add_argument("--residual-compute-cost", type=float, default=0.005)
    parser.add_argument("--cache-traffic-cost", type=float, default=0.003)
    parser.add_argument("--residual-traffic-cost", type=float, default=0.005)
    args = parser.parse_args()

    full_row = load_summary_row(args.summary, "FullP8")
    degree_row = load_summary_row(args.summary, "Deg")
    ratios = {
        "reuse": parse_percent(degree_row["reuse"]),
        "direct": parse_percent(degree_row["direct"]),
        "residual": parse_percent(degree_row["residual"]),
        "high": parse_percent(degree_row["P8"]),
        "mid": parse_percent(degree_row["P6"]),
        "low": parse_percent(degree_row["P4"]),
    }

    base = load_agg(args.microbench_dir / f"microbench_s{args.seq_len}" / "aggregate.json")
    full = load_agg(args.microbench_dir / f"microbench_s{args.seq_len}_internal_p8" / "aggregate.json")
    static_p6 = load_agg(args.microbench_dir / f"microbench_s{args.seq_len}_internal_p6" / "aggregate.json")
    static_p4 = load_agg(args.microbench_dir / f"microbench_s{args.seq_len}_internal_p4" / "aggregate.json")

    base_cycles = float(enc(base)["cycles"])
    base_reqs = reqs(base)
    base_matmul = metric(base, "matmul_active_cycles")
    base_input = metric(base, "mem_read_input_actual")
    base_weight = metric(base, "mem_read_weight")
    base_output = metric(base, "mem_write_output")

    def class_agg(name: str) -> dict[str, Any]:
        return load_agg(args.microbench_dir / f"microbench_s{args.seq_len}_internal_{name}" / "aggregate.json")

    variants = [
        {
            "method": "FullP8-miss",
            "mid": full,
            "low": full,
            "drop": float(full_row["drop"].rstrip("%")),
            "acc": float(full_row["acc"]),
            "note": "all miss nodes execute full P8",
        },
        {
            "method": "Static Degree P8/P6/P4",
            "mid": static_p6,
            "low": static_p4,
            "drop": float(degree_row["drop"].rstrip("%")),
            "acc": float(degree_row["acc"]),
            "note": "old fixed-depth proxy",
        },
        {
            "method": "EarlyStop conservative",
            "mid": class_agg("bound_mid_min6_t0p006"),
            "low": class_agg("bound_low_min4_t0p02"),
            "drop": float(degree_row["drop"].rstrip("%")),
            "acc": float(degree_row["acc"]),
            "note": "mid min6 tol0.006; low min4 tol0.02",
        },
        {
            "method": "EarlyStop balanced",
            "mid": class_agg("bound_mid_min6_t0p02"),
            "low": class_agg("bound_low_min4_t0p04"),
            "drop": float(degree_row["drop"].rstrip("%")),
            "acc": float(degree_row["acc"]),
            "note": "mid min6 tol0.02; low min4 tol0.04",
        },
        {
            "method": "EarlyStop aggressive",
            "mid": class_agg("bound_mid_min6_t0p06"),
            "low": class_agg("bound_low_min4_t0p06"),
            "drop": float(degree_row["drop"].rstrip("%")),
            "acc": float(degree_row["acc"]),
            "note": "mid min6 tol0.06; low min4 tol0.06",
        },
    ]

    rows = []
    for variant in variants:
        high = full
        mid = variant["mid"]
        low = variant["low"]
        miss_total = max(1e-12, ratios["high"] + ratios["mid"] + ratios["low"])
        miss_ratios = {
            "high": ratios["high"] / miss_total,
            "mid": ratios["mid"] / miss_total,
            "low": ratios["low"] / miss_total,
        }

        def miss_weighted(key: str) -> float:
            return (
                miss_ratios["high"] * metric(high, key)
                + miss_ratios["mid"] * metric(mid, key)
                + miss_ratios["low"] * metric(low, key)
            )

        miss_cycles = miss_weighted("cycles")
        miss_matmul = miss_weighted("matmul_active_cycles")
        miss_bitcomp = (
            miss_ratios["high"] * bitcomp_ratio(high)
            + miss_ratios["mid"] * bitcomp_ratio(mid)
            + miss_ratios["low"] * bitcomp_ratio(low)
        )
        miss_input_actual = miss_weighted("mem_read_input_actual")
        miss_input_original = miss_weighted("mem_read_input_original")
        miss_weight = miss_weighted("mem_read_weight")
        miss_output = miss_weighted("mem_write_output")
        miss_total_traffic = (
            miss_weighted("dram_read_requests") + miss_weighted("dram_write_requests")
        )
        cycle_norm = (
            ratios["direct"] * args.cache_compute_cost
            + ratios["residual"] * args.residual_compute_cost
            + ratios["high"] * float(enc(high)["cycles"]) / base_cycles
            + ratios["mid"] * float(enc(mid)["cycles"]) / base_cycles
            + ratios["low"] * float(enc(low)["cycles"]) / base_cycles
        )
        traffic_norm = (
            ratios["direct"] * args.cache_traffic_cost
            + ratios["residual"] * (args.cache_traffic_cost + args.residual_traffic_cost)
            + ratios["high"] * reqs(high) / base_reqs
            + ratios["mid"] * reqs(mid) / base_reqs
            + ratios["low"] * reqs(low) / base_reqs
        )
        avg_depth = (
            ratios["high"] * float(enc(high).get("graphbit_avg_depth") or 8.0)
            + ratios["mid"] * float(enc(mid).get("graphbit_avg_depth") or 8.0)
            + ratios["low"] * float(enc(low).get("graphbit_avg_depth") or 8.0)
        ) / max(1e-12, ratios["high"] + ratios["mid"] + ratios["low"])
        bound_stops = (
            float(enc(mid).get("graphbit_bound_stops") or 0.0)
            + float(enc(low).get("graphbit_bound_stops") or 0.0)
        )
        graphbit_inst = (
            float(enc(mid).get("graphbit_inst") or 0.0)
            + float(enc(low).get("graphbit_inst") or 0.0)
        )
        rows.append(
            {
                "method": variant["method"],
                "reuse": ratios["reuse"],
                "direct": ratios["direct"],
                "residual": ratios["residual"],
                "high_p8": ratios["high"],
                "mid": ratios["mid"],
                "low": ratios["low"],
                "avg_miss_depth": avg_depth,
                "saved_bitplanes": 8.0 - avg_depth,
                "bound_stop_rate": 0.0 if graphbit_inst == 0 else bound_stops / graphbit_inst,
                "cycles_norm": cycle_norm,
                "traffic_norm": traffic_norm,
                "energy_proxy": 0.55 * cycle_norm + 0.45 * traffic_norm,
                "miss_cycles_norm": safe_div(miss_cycles, base_cycles),
                "miss_matmul_norm": safe_div(miss_matmul, base_matmul),
                "miss_bitcomp_norm": miss_bitcomp,
                "miss_input_read_norm": safe_div(miss_input_actual, base_input),
                "miss_input_read_saved": 1.0 - safe_div(miss_input_actual, miss_input_original),
                "miss_weight_read_norm": safe_div(miss_weight, base_weight),
                "miss_output_write_norm": safe_div(miss_output, base_output),
                "miss_traffic_norm": safe_div(miss_total_traffic, base_reqs),
                "drop": variant["drop"],
                "acc": variant["acc"],
                "note": variant["note"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "earlystop_sweep.tsv"
    txt_path = args.output_dir / "earlystop_sweep.txt"
    miss_tsv_path = args.output_dir / "miss_only_breakdown.tsv"
    miss_txt_path = args.output_dir / "miss_only_breakdown.txt"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    miss_fields = [
        "method",
        "avg_miss_depth",
        "saved_bitplanes",
        "miss_cycles_norm",
        "miss_matmul_norm",
        "miss_bitcomp_norm",
        "miss_input_read_norm",
        "miss_input_read_saved",
        "miss_weight_read_norm",
        "miss_output_write_norm",
        "miss_traffic_norm",
        "bound_stop_rate",
        "drop",
        "note",
    ]
    with miss_tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=miss_fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in miss_fields})

    lines = [
        "Cora h8_54_T40 Graph-Bit predictor-free early-stop sweep",
        "All miss classes start from max_depth=8; only min_depth/tolerance changes.",
        "",
        "Method                     Reuse  AvgD  Saved  Stop   Cycles Traffic Energy Drop",
        "--------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<26} {fmt_pct(row['reuse']):>6} "
            f"{row['avg_miss_depth']:>5.2f} {row['saved_bitplanes']:>6.2f} "
            f"{fmt_pct(row['bound_stop_rate']):>6} {row['cycles_norm']:>7.3f} "
            f"{row['traffic_norm']:>7.3f} {row['energy_proxy']:>6.3f} "
            f"{row['drop']:>5.2f}%"
        )
    lines.extend(["", "Notes:"])
    for row in rows:
        lines.append(f"- {row['method']}: {row['note']}")
    lines.append("")
    lines.append(
        "Drop is the existing Cora static Degree proxy; early-stop rows report ONNXim internal "
        "bit-plane savings and should be treated as hardware validation before generating "
        "extra dynamic-depth embedding pools."
    )
    txt_path.write_text("\n".join(lines) + "\n")

    miss_lines = [
        "Cora h8_54_T40 ONNXim miss-only Graph-Bit breakdown",
        "Rows ignore reuse/residual ratios and evaluate only nodes that must execute the encoder.",
        "",
        "Method                     AvgD  Saved  Cycles MatMul BitComp ActRd ActSave WeightRd OutWr Traffic",
        "---------------------------------------------------------------------------------------------------",
    ]
    for row in rows:
        miss_lines.append(
            f"{row['method']:<26} {row['avg_miss_depth']:>5.2f} "
            f"{row['saved_bitplanes']:>6.2f} {row['miss_cycles_norm']:>7.3f} "
            f"{row['miss_matmul_norm']:>6.3f} {row['miss_bitcomp_norm']:>7.3f} "
            f"{row['miss_input_read_norm']:>5.3f} "
            f"{fmt_pct(row['miss_input_read_saved']):>7} {row['miss_weight_read_norm']:>8.3f} "
            f"{row['miss_output_write_norm']:>5.3f} {row['miss_traffic_norm']:>7.3f}"
        )
    miss_lines.extend(
        [
            "",
            "Column meaning:",
            "- AvgD: average executed activation bit-depth for miss-node GEMM tiles.",
            "- Cycles: encoder-miss cycles normalized to FullP8 ONNXim microbench.",
            "- MatMul: systolic matmul active cycles normalized to FullP8.",
            "- BitComp: Graph-Bit effective bit-serial compute cycles divided by raw full-depth compute cycles.",
            "- ActRd: activation/input read requests normalized to FullP8.",
            "- ActSave: activation/input request reduction relative to original full-depth input requests.",
            "- WeightRd and OutWr: weight read and output write requests normalized to FullP8.",
            "- Traffic: total DRAM read+write requests normalized to FullP8.",
        ]
    )
    miss_txt_path.write_text("\n".join(miss_lines) + "\n")
    print(f"[GraphBitEarlyStop] wrote {tsv_path}")
    print(f"[GraphBitEarlyStop] wrote {txt_path}")
    print(f"[GraphBitEarlyStop] wrote {miss_tsv_path}")
    print(f"[GraphBitEarlyStop] wrote {miss_txt_path}")
    print(txt_path.read_text())
    print(miss_txt_path.read_text())


if __name__ == "__main__":
    main()
