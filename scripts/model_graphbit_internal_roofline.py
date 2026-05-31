#!/usr/bin/env python3
"""Analytical NPU-internal roofline/activity model for Graph-Bit.

This model is intentionally lower-level than the end-to-end GNN accuracy
experiments.  It answers a hardware question:

    Given M = batch_nodes * padded_sequence_length, which part of a LLaMA
    encoder GEMM is dominated by W HBM, activation movement, PE bit-serial
    compute, RF/broadcast, psum update, or output write?

It reports Q/K/V/O projection separately from FFN gate/up and FFN down, and
compares A8/A6/A5/A4 execution under the same W4 weight format.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage:
    name: str
    k: int
    n: int
    count: int
    input_read_count: float


def default_stages(hidden: int, intermediate: int, *, qkv_fused: bool) -> list[Stage]:
    # Q/K/V/O uses four GEMMs.  With qkv_fused=True, Q/K/V share one activation
    # read and O reads the attention output, so input_read_count=2 rather than 4.
    proj_input_reads = 2.0 if qkv_fused else 4.0
    return [
        Stage("qkvo_proj", hidden, hidden, 4, proj_input_reads),
        Stage("ffn_gate_up", hidden, intermediate, 2, 2.0),
        Stage("ffn_down", intermediate, hidden, 1, 1.0),
    ]


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0.0 else num / den


def ceil_to_group(depth: int, group_bits: int, full_depth: int) -> int:
    if group_bits <= 1:
        return depth
    groups = (depth + group_bits - 1) // group_bits
    return min(full_depth, groups * group_bits)


def ceil_div(num: int, den: int) -> int:
    return (num + den - 1) // den


def default_bound_checks(depth: int, full_depth: int, low_min_depth: int, group_bits: int) -> int:
    """Runtime predictor-free bound checks for a selected stop depth."""
    if depth >= full_depth:
        return 0
    first_check = 6 if depth >= 6 else min(low_min_depth, depth)
    if depth < first_check:
        return 0
    return 1 + max(0, math.ceil((depth - first_check) / max(1, group_bits)))


def stage_metrics(
    stage: Stage,
    *,
    m: int,
    depth: int,
    full_depth: int,
    weight_bits: int,
    output_bits: int,
    activation_hbm_mode: str,
    plane_group_bits: int,
    peak_tops: float,
    mem_gbps: float,
    freq_ghz: float,
    bound_enable: bool,
    bound_low_min_depth: int,
    bound_check_group_bits: int,
    bound_ops_per_output: float,
    bound_tops: float,
    bound_overlap: float,
    bound_control_cycles_per_check: float,
    m_tile: int,
    n_tile: int,
) -> dict[str, float | str | int]:
    macs = float(m) * stage.k * stage.n * stage.count
    p8_macs = macs
    bit_compute_ops_equiv = macs * depth / full_depth

    weight_bytes = stage.k * stage.n * stage.count * weight_bits / 8.0
    if activation_hbm_mode == "plane_group":
        fetch_depth = ceil_to_group(depth, plane_group_bits, full_depth)
    else:
        fetch_depth = full_depth
    activation_bytes = float(m) * stage.k * stage.input_read_count * fetch_depth / 8.0
    output_bytes = float(m) * stage.n * stage.count * output_bits / 8.0
    hbm_bytes = weight_bytes + activation_bytes + output_bytes

    # Roofline latency.  peak_tops is effective P8 MAC throughput; lower
    # activation depth reduces bit-serial compute proportionally.
    bit_compute_time_s = bit_compute_ops_equiv / (peak_tops * 1.0e12)
    bound_checks = default_bound_checks(
        depth,
        full_depth,
        bound_low_min_depth,
        bound_check_group_bits,
    ) if bound_enable else 0
    bound_ops = float(m) * stage.n * stage.count * bound_checks * bound_ops_per_output
    bound_unit_tops = max(1.0e-9, bound_tops)
    bound_time_s = bound_ops / (bound_unit_tops * 1.0e12)
    bound_cycles_raw = bound_time_s * freq_ghz * 1.0e9
    visible_bound_cycles = bound_cycles_raw * max(0.0, min(1.0, 1.0 - bound_overlap))
    tile_checks = (
        ceil_div(max(1, m), max(1, m_tile))
        * ceil_div(max(1, stage.n), max(1, n_tile))
        * stage.count
        * bound_checks
    )
    bound_control_cycles = float(tile_checks) * bound_control_cycles_per_check
    bit_compute_cycles = bit_compute_time_s * freq_ghz * 1.0e9
    cycles_bound = visible_bound_cycles + bound_control_cycles
    memory_time_s = hbm_bytes / (mem_gbps * 1.0e9)
    cycles_compute = bit_compute_cycles + cycles_bound
    cycles_memory = memory_time_s * freq_ghz * 1.0e9
    cycles = max(cycles_compute, cycles_memory)

    # Activity proxy: on-chip terms that should scale with bit-plane depth.
    # HBM output and byte-major activation do not scale with depth.
    depth_scale = depth / full_depth
    fetch_scale = fetch_depth / full_depth
    weight_hbm_activity = weight_bytes
    activation_hbm_activity = activation_bytes
    output_hbm_activity = output_bytes
    a_rf_activity = float(m) * stage.k * stage.input_read_count * depth_scale
    pe_activity = macs * depth_scale
    w_rf_activity = macs * depth_scale
    psum_activity = float(m) * stage.n * stage.count * depth_scale
    bound_activity = bound_ops + float(tile_checks)

    total_activity = (
        weight_hbm_activity
        + activation_hbm_activity
        + output_hbm_activity
        + a_rf_activity
        + pe_activity
        + w_rf_activity
        + psum_activity
        + bound_activity
    )

    oi = (2.0 * p8_macs) / hbm_bytes if hbm_bytes > 0.0 else 0.0
    ridge = (peak_tops * 1000.0) / mem_gbps if mem_gbps > 0.0 else 0.0
    bound = "compute" if cycles_compute >= cycles_memory else "memory"
    return {
        "stage": stage.name,
        "m": m,
        "depth": depth,
        "macs": p8_macs,
        "compute_ops_equiv": bit_compute_ops_equiv + bound_ops,
        "bit_compute_ops_equiv": bit_compute_ops_equiv,
        "bound_ops": bound_ops,
        "bound_checks": bound_checks,
        "tile_checks": tile_checks,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "output_bytes": output_bytes,
        "hbm_bytes": hbm_bytes,
        "oi": oi,
        "ridge": ridge,
        "bit_compute_cycles": bit_compute_cycles,
        "bound_cycles_raw": bound_cycles_raw,
        "bound_cycles": cycles_bound,
        "bound_control_cycles": bound_control_cycles,
        "cycles_compute": cycles_compute,
        "cycles_memory": cycles_memory,
        "cycles": cycles,
        "bound": bound,
        "fetch_depth": fetch_depth,
        "depth_scale": depth_scale,
        "fetch_scale": fetch_scale,
        "weight_hbm_activity": weight_hbm_activity,
        "activation_hbm_activity": activation_hbm_activity,
        "output_hbm_activity": output_hbm_activity,
        "a_rf_activity": a_rf_activity,
        "pe_activity": pe_activity,
        "w_rf_activity": w_rf_activity,
        "psum_activity": psum_activity,
        "bound_activity": bound_activity,
        "total_activity": total_activity,
    }


def aggregate_layer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "stage": "layer_total",
        "m": rows[0]["m"],
        "depth": rows[0]["depth"],
        "macs": sum(float(r["macs"]) for r in rows),
        "compute_ops_equiv": sum(float(r["compute_ops_equiv"]) for r in rows),
        "bit_compute_ops_equiv": sum(float(r["bit_compute_ops_equiv"]) for r in rows),
        "bound_ops": sum(float(r["bound_ops"]) for r in rows),
        "bound_checks": sum(float(r["bound_checks"]) for r in rows),
        "tile_checks": sum(float(r["tile_checks"]) for r in rows),
        "weight_bytes": sum(float(r["weight_bytes"]) for r in rows),
        "activation_bytes": sum(float(r["activation_bytes"]) for r in rows),
        "output_bytes": sum(float(r["output_bytes"]) for r in rows),
        "hbm_bytes": sum(float(r["hbm_bytes"]) for r in rows),
        "bit_compute_cycles": sum(float(r["bit_compute_cycles"]) for r in rows),
        "bound_cycles_raw": sum(float(r["bound_cycles_raw"]) for r in rows),
        "bound_cycles": sum(float(r["bound_cycles"]) for r in rows),
        "bound_control_cycles": sum(float(r["bound_control_cycles"]) for r in rows),
        "cycles_compute": sum(float(r["cycles_compute"]) for r in rows),
        "cycles_memory": sum(float(r["cycles_memory"]) for r in rows),
        # A layer executes these GEMM stages sequentially in this simple model.
        "cycles": sum(float(r["cycles"]) for r in rows),
        "weight_hbm_activity": sum(float(r["weight_hbm_activity"]) for r in rows),
        "activation_hbm_activity": sum(float(r["activation_hbm_activity"]) for r in rows),
        "output_hbm_activity": sum(float(r["output_hbm_activity"]) for r in rows),
        "a_rf_activity": sum(float(r["a_rf_activity"]) for r in rows),
        "pe_activity": sum(float(r["pe_activity"]) for r in rows),
        "w_rf_activity": sum(float(r["w_rf_activity"]) for r in rows),
        "psum_activity": sum(float(r["psum_activity"]) for r in rows),
        "bound_activity": sum(float(r["bound_activity"]) for r in rows),
        "total_activity": sum(float(r["total_activity"]) for r in rows),
    }
    total["oi"] = safe_div(2.0 * float(total["macs"]), float(total["hbm_bytes"]))
    total["bound"] = "compute" if float(total["cycles_compute"]) >= float(total["cycles_memory"]) else "memory"
    total["fetch_depth"] = "-"
    total["depth_scale"] = rows[0]["depth_scale"]
    total["fetch_scale"] = "-"
    return total


def normalize_rows(rows: list[dict[str, Any]]) -> None:
    base_by_key = {
        (row["m"], row["stage"]): row
        for row in rows
        if row["depth"] == 8
    }
    for row in rows:
        base = base_by_key.get((row["m"], row["stage"]))
        if base is None:
            continue
        row["cycles_vs_p8"] = safe_div(float(row["cycles"]), float(base["cycles"]))
        row["cycle_save_vs_p8"] = 1.0 - float(row["cycles_vs_p8"])
        row["hbm_vs_p8"] = safe_div(float(row["hbm_bytes"]), float(base["hbm_bytes"]))
        row["hbm_save_vs_p8"] = 1.0 - float(row["hbm_vs_p8"])
        row["activity_vs_p8"] = safe_div(float(row["total_activity"]), float(base["total_activity"]))
        row["activity_save_vs_p8"] = 1.0 - float(row["activity_vs_p8"])
        row["pe_vs_p8"] = safe_div(float(row["pe_activity"]), float(base["pe_activity"]))
        row["pe_save_vs_p8"] = 1.0 - float(row["pe_vs_p8"])
        row["bound_cycles_vs_p8"] = safe_div(float(row["bound_cycles"]), float(base["cycles"]))
        row["net_cycle_save_after_bound"] = row["cycle_save_vs_p8"]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "m",
        "stage",
        "depth",
        "bound",
        "cycles",
        "bit_compute_cycles",
        "bound_cycles",
        "bound_cycles_raw",
        "bound_control_cycles",
        "cycles_compute",
        "cycles_memory",
        "cycles_vs_p8",
        "cycle_save_vs_p8",
        "bound_cycles_vs_p8",
        "hbm_bytes",
        "weight_bytes",
        "activation_bytes",
        "output_bytes",
        "hbm_vs_p8",
        "hbm_save_vs_p8",
        "oi",
        "fetch_depth",
        "activity_vs_p8",
        "activity_save_vs_p8",
        "pe_vs_p8",
        "pe_save_vs_p8",
        "weight_hbm_activity",
        "activation_hbm_activity",
        "a_rf_activity",
        "pe_activity",
        "w_rf_activity",
        "psum_activity",
        "output_hbm_activity",
        "bound_activity",
        "bound_ops",
        "bound_checks",
        "tile_checks",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_report(path: Path, rows: list[dict[str, Any]], *, batch_nodes: list[int], seq_lens: list[int], args: argparse.Namespace) -> None:
    layer_rows = [r for r in rows if r["stage"] == "layer_total"]
    stage_rows = [r for r in rows if r["stage"] != "layer_total"]
    lines: list[str] = [
        "Graph-Bit NPU internal roofline/activity model",
        f"batch_nodes={batch_nodes} | seq_lens={seq_lens}",
        f"activation_hbm_mode={args.activation_hbm_mode} | qkv_fused={args.qkv_fused}",
        f"peak={args.peak_tops:.1f} TOPS | mem={args.mem_gbps:.1f} GB/s | freq={args.freq_ghz:.2f} GHz",
        (
            "bound_overhead="
            f"{args.bound_enable} | ops/output={args.bound_ops_per_output:.1f} | "
            f"bound_tops={args.bound_tops:.1f} | overlap={args.bound_overlap:.2f} | "
            f"ctrl/check={args.bound_control_cycles_per_check:.1f}"
        ),
        "",
        "Layer total by M",
        (
            f"{'M':>6s} {'B*S':<12s} {'P8 cyc':>10s} {'P6 net':>8s} {'P5 net':>8s} {'P4 net':>8s} "
            f"{'P6 bnd':>8s} {'P5 bnd':>8s} {'P4 bnd':>8s} {'P8 OI':>8s} {'P8 bound':>8s}"
        ),
        "-" * 124,
    ]
    p8_layer = {(r["m"], r["stage"]): r for r in layer_rows if r["depth"] == 8}
    for m in sorted({int(r["m"]) for r in layer_rows}):
        base = p8_layer[(m, "layer_total")]
        by_depth = {r["depth"]: r for r in layer_rows if int(r["m"]) == m}
        def save(depth: int) -> str:
            row = by_depth.get(depth)
            return "-" if row is None else pct(float(row["cycle_save_vs_p8"]))
        def bnd(depth: int) -> str:
            row = by_depth.get(depth)
            return "-" if row is None else pct(float(row["bound_cycles_vs_p8"]))

        w_share = safe_div(float(base["weight_bytes"]), float(base["hbm_bytes"]))
        a_share = safe_div(float(base["activation_bytes"]), float(base["hbm_bytes"]))
        lines.append(
            f"{m:6d} {str(base.get('batch_seq', '-')):<12s} {float(base['cycles']):10.1f} "
            f"{save(6):>8s} {save(5):>8s} {save(4):>8s} "
            f"{bnd(6):>8s} {bnd(5):>8s} {bnd(4):>8s} "
            f"{float(base['oi']):8.1f} {str(base['bound']):>8s}"
        )

    lines.extend(
        [
            "",
            "Per-stage P8 bottleneck",
            (
                f"{'M':>6s} {'stage':<12s} {'cycles%':>8s} {'OI':>8s} {'bound':>8s} "
                f"{'W%':>7s} {'A%':>7s} {'P6 net':>8s} {'P6 bnd':>8s} {'P6 act':>8s}"
            ),
            "-" * 116,
        ]
    )
    totals = {int(r["m"]): float(r["cycles"]) for r in layer_rows if r["depth"] == 8}
    for row in stage_rows:
        if row["depth"] != 8:
            continue
        m = int(row["m"])
        p6 = next((r for r in stage_rows if r["m"] == row["m"] and r["stage"] == row["stage"] and r["depth"] == 6), None)
        w_share = safe_div(float(row["weight_bytes"]), float(row["hbm_bytes"]))
        a_share = safe_div(float(row["activation_bytes"]), float(row["hbm_bytes"]))
        lines.append(
            f"{m:6d} {str(row['stage']):<12s} {pct(float(row['cycles']) / totals[m]):>8s} "
            f"{float(row['oi']):8.1f} {str(row['bound']):>8s} {pct(w_share):>7s} {pct(a_share):>7s} "
            f"{pct(float(p6['cycle_save_vs_p8'])) if p6 else '-':>8s} "
            f"{pct(float(p6['bound_cycles_vs_p8'])) if p6 else '-':>8s} "
            f"{pct(float(p6['activity_save_vs_p8'])) if p6 else '-':>8s}"
        )

    lines.extend(
        [
            "",
            "Reading guide:",
            "- M = batch_nodes * padded_sequence_length. This is the real row count of each Transformer linear GEMM.",
            "- P6/P5/P4 cycle save only appears when compute cycles are on the critical path.",
            "- P6/P5/P4 net save includes predictor-free bound estimator and control overhead.",
            "- bnd columns show visible bound-estimator overhead normalized to the P8 cycle count.",
            "- activity_save captures PE/RF/psum bit-plane reduction even when HBM dominates latency.",
            "- byte_major activation keeps A HBM reads at A8; plane_group allows activation HBM demand fetch.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-nodes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 6, 5, 4])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--output-bits", type=int, default=16)
    parser.add_argument("--full-depth", type=int, default=8)
    parser.add_argument("--activation-hbm-mode", choices=["byte_major", "plane_group"], default="byte_major")
    parser.add_argument("--plane-group-bits", type=int, default=2)
    parser.add_argument("--qkv-fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--peak-tops", type=float, default=131.1)
    parser.add_argument("--mem-gbps", type=float, default=614.4)
    parser.add_argument("--freq-ghz", type=float, default=1.0)
    parser.add_argument("--bound-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bound-low-min-depth", type=int, default=4)
    parser.add_argument("--bound-check-group-bits", type=int, default=1)
    parser.add_argument(
        "--bound-ops-per-output",
        type=float,
        default=8.0,
        help="Conservative bound-estimator ops per output element per runtime check.",
    )
    parser.add_argument(
        "--bound-tops",
        type=float,
        default=16.0,
        help="Effective throughput of the lightweight bound/check unit.",
    )
    parser.add_argument(
        "--bound-overlap",
        type=float,
        default=0.5,
        help="Fraction of bound-estimator cycles overlapped with bit-plane GEMM work.",
    )
    parser.add_argument("--bound-control-cycles-per-check", type=float, default=4.0)
    parser.add_argument("--m-tile", type=int, default=128)
    parser.add_argument("--n-tile", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/graphbit_internal_roofline"),
    )
    args = parser.parse_args()

    stages = default_stages(args.hidden, args.intermediate, qkv_fused=args.qkv_fused)
    rows: list[dict[str, Any]] = []
    seen_m: set[int] = set()
    for batch in args.batch_nodes:
        for seq_len in args.seq_lens:
            m = batch * seq_len
            if m in seen_m:
                continue
            seen_m.add(m)
            for depth in args.depths:
                stage_rows = [
                    stage_metrics(
                        stage,
                        m=m,
                        depth=depth,
                        full_depth=args.full_depth,
                        weight_bits=args.weight_bits,
                        output_bits=args.output_bits,
                        activation_hbm_mode=args.activation_hbm_mode,
                        plane_group_bits=args.plane_group_bits,
                        peak_tops=args.peak_tops,
                        mem_gbps=args.mem_gbps,
                        freq_ghz=args.freq_ghz,
                        bound_enable=args.bound_enable,
                        bound_low_min_depth=args.bound_low_min_depth,
                        bound_check_group_bits=args.bound_check_group_bits,
                        bound_ops_per_output=args.bound_ops_per_output,
                        bound_tops=args.bound_tops,
                        bound_overlap=args.bound_overlap,
                        bound_control_cycles_per_check=args.bound_control_cycles_per_check,
                        m_tile=args.m_tile,
                        n_tile=args.n_tile,
                    )
                    for stage in stages
                ]
                for row in stage_rows:
                    row["batch_seq"] = ",".join(
                        f"B{b}xS{s}" for b in args.batch_nodes for s in args.seq_lens if b * s == m
                    )
                layer = aggregate_layer(stage_rows)
                layer["batch_seq"] = stage_rows[0]["batch_seq"]
                rows.extend(stage_rows)
                rows.append(layer)

    normalize_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "graphbit_internal_roofline.tsv", rows)
    write_report(
        args.output_dir / "graphbit_internal_roofline.txt",
        rows,
        batch_nodes=args.batch_nodes,
        seq_lens=args.seq_lens,
        args=args,
    )
    payload_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {"args": payload_args, "rows": rows}
    (args.output_dir / "graphbit_internal_roofline.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[GraphBitInternalRoofline] wrote {args.output_dir / 'graphbit_internal_roofline.txt'}")
    print((args.output_dir / "graphbit_internal_roofline.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
