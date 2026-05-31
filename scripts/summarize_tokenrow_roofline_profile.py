#!/usr/bin/env python3
"""Summarize token-row LLaMA ONNXim Graph-Bit component sweeps.

The input directory is expected to contain folders named like:

    m128_p8/summary.tsv
    m128_p6/summary.tsv
    m256_p8/summary.tsv

Here `m` is the real Transformer GEMM row count:

    M = batch_nodes * padded_sequence_length

The script aggregates the three LLaMA-7B GEMM components into a weighted
per-layer total and emits both TSV and a compact human-readable pivot table.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DEPTH_ORDER = ("p8", "p6", "p5", "p4")
SEQ_BUCKETS = (128, 256, 512)
NODE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(f(row, key, default))


def possible_bs(m: int) -> str:
    pairs = []
    for seq in SEQ_BUCKETS:
        if m % seq == 0:
            b = m // seq
            if b in NODE_BATCHES:
                pairs.append(f"B{b}xS{seq}")
    return ",".join(pairs) if pairs else "-"


def aggregate_summary(path: Path) -> dict[str, float | int | str]:
    rows = read_tsv(path)
    if not rows:
        raise ValueError(f"empty summary: {path}")
    weighted_cycles = 0.0
    weighted_gflops = 0.0
    weighted_gb = 0.0
    weighted_read = 0.0
    weighted_write = 0.0
    weighted_w = 0.0
    weighted_a = 0.0
    weighted_y = 0.0
    weighted_eff_compute = 0.0
    weighted_raw_compute = 0.0
    weighted_depth_numer = 0.0
    weighted_depth_denom = 0.0
    m = i(rows[0], "m")
    for row in rows:
        count = max(1, i(row, "count_per_layer", 1))
        cycles = f(row, "cycles")
        weighted_cycles += cycles * count
        weighted_gflops += f(row, "gflops") * count
        weighted_gb += f(row, "gb") * count
        weighted_read += f(row, "dram_read_requests") * count
        weighted_write += f(row, "dram_write_requests") * count
        weighted_w += f(row, "mem_read_weight") * count
        weighted_a += f(row, "mem_read_input_actual") * count
        weighted_y += f(row, "mem_write_output") * count
        weighted_eff_compute += f(row, "graphbit_effective_compute_cycles") * count
        weighted_raw_compute += f(row, "graphbit_raw_compute_cycles") * count
        inst = f(row, "graphbit_inst")
        avg_depth = f(row, "graphbit_avg_depth", 8.0)
        weighted_depth_numer += avg_depth * inst * count
        weighted_depth_denom += inst * count
    total_req = max(1.0, weighted_read + weighted_write)
    return {
        "m": m,
        "cycles": weighted_cycles,
        "gflops": weighted_gflops,
        "gb": weighted_gb,
        "oi": weighted_gflops / weighted_gb if weighted_gb > 0 else 0.0,
        "read_req": weighted_read,
        "write_req": weighted_write,
        "weight_req": weighted_w,
        "act_req": weighted_a,
        "out_req": weighted_y,
        "weight_share": weighted_w / total_req,
        "act_share": weighted_a / total_req,
        "out_share": weighted_y / total_req,
        "eff_compute": weighted_eff_compute,
        "raw_compute": weighted_raw_compute,
        "avg_depth": weighted_depth_numer / weighted_depth_denom if weighted_depth_denom else 8.0,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/onnxim_graphbit/tokenrow_components_m128_m32768"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    output_dir = args.output_dir or root
    output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[int, dict[str, dict[str, float | int | str]]] = defaultdict(dict)
    pattern = re.compile(r"m(\d+)_(p[0-9]+)$")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        m = int(match.group(1))
        depth = match.group(2)
        summary = child / "summary.tsv"
        if summary.exists():
            records[m][depth] = aggregate_summary(summary)

    fields = [
        "m",
        "depth",
        "batch_seq",
        "cycles",
        "cycles_vs_p8",
        "cycle_save_vs_p8",
        "traffic_req",
        "traffic_vs_p8",
        "traffic_save_vs_p8",
        "oi",
        "weight_share",
        "act_share",
        "out_share",
        "avg_depth",
        "eff_compute_vs_p8",
    ]
    rows_out: list[dict[str, str | int | float]] = []
    for m in sorted(records):
        base = records[m].get("p8")
        if not base:
            continue
        base_cycles = float(base["cycles"])
        base_traffic = float(base["read_req"]) + float(base["write_req"])
        base_eff = float(base["eff_compute"]) or 1.0
        for depth in DEPTH_ORDER:
            rec = records[m].get(depth)
            if not rec:
                continue
            traffic = float(rec["read_req"]) + float(rec["write_req"])
            cycles_vs = float(rec["cycles"]) / base_cycles if base_cycles else 0.0
            traffic_vs = traffic / base_traffic if base_traffic else 0.0
            rows_out.append(
                {
                    "m": m,
                    "depth": depth,
                    "batch_seq": possible_bs(m),
                    "cycles": f"{float(rec['cycles']):.0f}",
                    "cycles_vs_p8": f"{cycles_vs:.4f}",
                    "cycle_save_vs_p8": f"{1.0 - cycles_vs:.4f}",
                    "traffic_req": f"{traffic:.0f}",
                    "traffic_vs_p8": f"{traffic_vs:.4f}",
                    "traffic_save_vs_p8": f"{1.0 - traffic_vs:.4f}",
                    "oi": f"{float(rec['oi']):.2f}",
                    "weight_share": f"{float(rec['weight_share']):.4f}",
                    "act_share": f"{float(rec['act_share']):.4f}",
                    "out_share": f"{float(rec['out_share']):.4f}",
                    "avg_depth": f"{float(rec['avg_depth']):.2f}",
                    "eff_compute_vs_p8": f"{float(rec['eff_compute']) / base_eff:.4f}",
                }
            )

    tsv_path = output_dir / "tokenrow_depth_summary.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)

    lines = [
        "Token-row Graph-Bit ONNXim profile",
        f"Source: {root}",
        "",
        (
            f"{'M':>7s} {'B*S examples':<27s} {'P8 cyc':>12s} "
            f"{'P6 save':>9s} {'P5 save':>9s} {'P4 save':>9s} "
            f"{'Wshare(P8)':>11s} {'Ashare(P8)':>11s} {'OI(P8)':>8s}"
        ),
        "-" * 110,
    ]
    for m in sorted(records):
        base = records[m].get("p8")
        if not base:
            continue
        base_cycles = float(base["cycles"])
        saves = {}
        for depth in ("p6", "p5", "p4"):
            rec = records[m].get(depth)
            saves[depth] = 1.0 - float(rec["cycles"]) / base_cycles if rec else None
        lines.append(
            f"{m:7d} {possible_bs(m):<27s} {base_cycles:12.0f} "
            f"{pct(saves['p6']) if saves['p6'] is not None else '-':>9s} "
            f"{pct(saves['p5']) if saves['p5'] is not None else '-':>9s} "
            f"{pct(saves['p4']) if saves['p4'] is not None else '-':>9s} "
            f"{pct(float(base['weight_share'])):>11s} "
            f"{pct(float(base['act_share'])):>11s} "
            f"{float(base['oi']):8.1f}"
        )

    pivot_path = output_dir / "tokenrow_depth_pivot.txt"
    pivot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TokenRowProfile] wrote {tsv_path}")
    print(f"[TokenRowProfile] wrote {pivot_path}")


if __name__ == "__main__":
    main()
