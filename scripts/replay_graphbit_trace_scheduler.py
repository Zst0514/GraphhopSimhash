#!/usr/bin/env python3
"""Replay a per-node Graph-Bit trace through a bucket scheduler model.

This script is the trace-driven layer between software workload validation and
ONNXim component simulation.  It reads real per-node routing/actions exported by
`residual_precision_depth`, forms micro-batches, derives W-tile load counts from
the replayed schedule, and composes ONNXim component costs.

It intentionally does not claim full-system cycle accuracy.  The cycle numbers
come from ONNXim GEMM component traces; the full workload is replayed by this
node trace scheduler.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEPTH_KEYS = ("p8", "p6", "p5", "p4")
DEPTH_BY_KEY = {"p8": 8, "p6": 6, "p5": 5, "p4": 4}
KEY_BY_DEPTH = {8: "p8", 6: "p6", 5: "p5", 4: "p4"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing json: {path}")
    return json.loads(path.read_text())


def load_encoder(root: Path, case: str) -> dict[str, Any]:
    return load_json(root / case / "aggregate.json")["encoder"]


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if line_no == 1 and "meta" in obj:
                meta = obj["meta"]
                continue
            rows.append(obj)
    if not rows:
        raise SystemExit(f"No node rows in trace: {path}")
    return meta, rows


def traffic(row: dict[str, Any]) -> float:
    return float(row.get("dram_read_requests", 0.0) or 0.0) + float(
        row.get("dram_write_requests", 0.0) or 0.0
    )


def norm(row: dict[str, Any], base: dict[str, Any], key: str) -> float:
    den = float(base.get(key, 0.0) or 0.0)
    return 0.0 if den <= 0.0 else float(row.get(key, 0.0) or 0.0) / den


def traffic_norm(row: dict[str, Any], base: dict[str, Any]) -> float:
    den = traffic(base)
    return 0.0 if den <= 0.0 else traffic(row) / den


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_hist(hist: dict[int, float]) -> str:
    if not hist:
        return "-"
    return ",".join(f"D{depth}:{100.0 * share:.1f}%" for depth, share in sorted(hist.items()))


class BufferModel:
    def __init__(
        self,
        *,
        sram_kb: float,
        tile_k: int,
        tile_n: int,
        weight_bits: int,
        fetch_depth: int,
        psum_bits: int,
        output_bits: int,
        buffer_factor: float,
    ) -> None:
        self.sram_bytes = float(sram_kb) * 1024.0
        self.tile_k = int(tile_k)
        self.tile_n = int(tile_n)
        self.weight_bits = int(weight_bits)
        self.fetch_depth = int(fetch_depth)
        self.psum_bits = int(psum_bits)
        self.output_bits = int(output_bits)
        self.buffer_factor = float(buffer_factor)

    @property
    def weight_bytes(self) -> float:
        return self.tile_k * self.tile_n * self.weight_bits / 8.0

    def total_bytes(self, batch: int) -> float:
        activation = batch * self.tile_k * self.fetch_depth / 8.0
        psum = batch * self.tile_n * self.psum_bits / 8.0
        output = batch * self.tile_n * self.output_bits / 8.0
        return (self.weight_bytes + activation + psum + output) * self.buffer_factor

    def max_batch(self) -> int:
        per_node = (
            self.tile_k * self.fetch_depth / 8.0
            + self.tile_n * self.psum_bits / 8.0
            + self.tile_n * self.output_bits / 8.0
        )
        available = self.sram_bytes / self.buffer_factor - self.weight_bytes
        if available <= 0.0:
            return 0
        return max(0, int(math.floor(available / per_node)))


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def depth_key(row: dict[str, Any]) -> str:
    bucket = str(row.get("depth_bucket", "")).lower()
    if bucket in DEPTH_KEYS:
        return bucket
    depth = int(row.get("stop_depth", row.get("action_bit", 8)) or 8)
    return KEY_BY_DEPTH.get(depth, "p8")


def depth_value(row: dict[str, Any]) -> int:
    return DEPTH_BY_KEY[depth_key(row)]


def hist_from_counts(counts: dict[int, int], denom: int) -> dict[int, float]:
    if denom <= 0:
        return {}
    return {depth: count / denom for depth, count in sorted(counts.items()) if count > 0}


def component_case(depth_key_name: str, suffix: str) -> str:
    if suffix == "now":
        return f"{depth_key_name}_now"
    return f"{depth_key_name}_{suffix}"


def load_components(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base = load_encoder(root, "full_p8")
    components: dict[str, dict[str, Any]] = {}
    for key in DEPTH_KEYS:
        components[f"{key}_now"] = load_encoder(root, f"{key}_now")
        for batch in (32, 64):
            components[f"{key}_ws_b{batch}"] = load_encoder(root, f"{key}_ws_b{batch}")
    return base, components


def component_lookup_rows(base: dict[str, Any], components: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(case: str, row: dict[str, Any]) -> None:
        parts = case.split("_")
        depth = parts[0].upper() if case != "full_p8" else "P8"
        mode = "full" if case == "full_p8" else ("ws" if "ws" in parts else "now")
        batch = "-"
        for part in parts:
            if part.startswith("b") and part[1:].isdigit():
                batch = part[1:]
        cycles_norm = norm(row, base, "cycles")
        traff_norm = traffic_norm(row, base)
        rows.append(
            {
                "case": case,
                "depth": depth,
                "mode": mode,
                "batch": batch,
                "cycles_norm": cycles_norm,
                "traffic_norm": traff_norm,
                "energy_norm": 0.5 * cycles_norm + 0.5 * traff_norm,
                "cycles": float(row.get("cycles", 0.0) or 0.0),
                "dram_read_requests": float(row.get("dram_read_requests", 0.0) or 0.0),
                "dram_write_requests": float(row.get("dram_write_requests", 0.0) or 0.0),
            }
        )

    add_row("full_p8", base)
    for case in sorted(components):
        add_row(case, components[case])
    return rows


def compose_from_exec_counts(
    *,
    exec_counts: dict[str, int],
    total_nodes: int,
    suffix: str,
    base: dict[str, Any],
    components: dict[str, dict[str, Any]],
) -> dict[str, float]:
    cycles = 0.0
    traff = 0.0
    depth_sum = 0.0
    for key, count in exec_counts.items():
        if count <= 0:
            continue
        row = components[component_case(key, suffix)]
        ratio = count / max(1, total_nodes)
        cycles += ratio * norm(row, base, "cycles")
        traff += ratio * traffic_norm(row, base)
        depth_sum += count * DEPTH_BY_KEY[key]
    energy = 0.5 * cycles + 0.5 * traff
    total_exec = sum(exec_counts.values())
    return {
        "cycles": cycles,
        "traffic": traff,
        "energy": energy,
        "avg_depth": depth_sum / total_exec if total_exec else 0.0,
    }


def replay_original_order(miss_rows: list[dict[str, Any]], batch: int) -> tuple[dict[str, int], int, float]:
    exec_counts = {key: 0 for key in DEPTH_KEYS}
    blocks = chunked(miss_rows, batch)
    useful = 0
    padded = 0
    for block in blocks:
        if not block:
            continue
        max_depth = max(depth_value(row) for row in block)
        key = KEY_BY_DEPTH.get(max_depth, "p8")
        exec_counts[key] += len(block)
        useful += len(block)
        padded += batch
    tail_util = useful / padded if padded else 1.0
    return exec_counts, len(blocks), tail_util


def replay_risk_bucket(miss_rows: list[dict[str, Any]], batch: int) -> tuple[dict[str, int], int, float]:
    exec_counts = {key: 0 for key in DEPTH_KEYS}
    loads = 0
    useful = 0
    padded = 0
    buckets = {key: [] for key in DEPTH_KEYS}
    for row in miss_rows:
        buckets[depth_key(row)].append(row)
    for key, bucket_rows in buckets.items():
        exec_counts[key] += len(bucket_rows)
        if bucket_rows:
            bucket_loads = math.ceil(len(bucket_rows) / batch)
            loads += bucket_loads
            useful += len(bucket_rows)
            padded += bucket_loads * batch
    tail_util = useful / padded if padded else 1.0
    return exec_counts, loads, tail_util


def make_row(
    *,
    method: str,
    reuse_count: int,
    direct_count: int,
    residual_count: int,
    miss_count: int,
    total_nodes: int,
    exec_counts: dict[str, int],
    loads: int,
    baseline_loads: int,
    tail_util: float,
    suffix: str,
    drop: float,
    base: dict[str, Any],
    components: dict[str, dict[str, Any]],
    sram_fit: bool,
) -> dict[str, Any]:
    composed = compose_from_exec_counts(
        exec_counts=exec_counts,
        total_nodes=total_nodes,
        suffix=suffix,
        base=base,
        components=components,
    )
    hist_counts = {DEPTH_BY_KEY[key]: count for key, count in exec_counts.items()}
    return {
        "method": method,
        "reuse": reuse_count / total_nodes,
        "direct": direct_count / total_nodes,
        "residual": residual_count / total_nodes,
        "miss": miss_count / total_nodes,
        "cycles": composed["cycles"],
        "traffic": composed["traffic"],
        "energy": composed["energy"],
        "drop": drop,
        "avg_depth": composed["avg_depth"],
        "depth_hist": hist_from_counts(hist_counts, max(1, miss_count)),
        "wloads": loads,
        "baseline_wloads": baseline_loads,
        "wscale": loads / baseline_loads if baseline_loads else 1.0,
        "tail_util": tail_util,
        "sram_fit": "yes" if sram_fit else "no",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-percent", type=float, default=0.0)
    parser.add_argument("--fullp8-drop-percent", type=float, default=0.0)
    parser.add_argument("--baseline-tile-batch", type=int, default=16)
    parser.add_argument("--candidate-batches", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--sram-kb", type=float, default=512.0)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--fetch-depth", type=int, default=6)
    parser.add_argument("--psum-bits", type=int, default=32)
    parser.add_argument("--output-bits", type=int, default=16)
    parser.add_argument("--buffer-factor", type=float, default=2.0)
    args = parser.parse_args()

    meta, rows = load_trace(args.trace)
    base, components = load_components(args.components_root)
    total_nodes = len(rows)
    direct_rows = [row for row in rows if row.get("role") == "direct"]
    residual_rows = [row for row in rows if row.get("role") == "residual"]
    miss_rows = [row for row in rows if bool(row.get("is_miss", False))]
    reuse_count = total_nodes - len(miss_rows)
    direct_count = len(direct_rows)
    residual_count = len(residual_rows)
    miss_count = len(miss_rows)
    baseline_loads = math.ceil(miss_count / max(1, args.baseline_tile_batch))
    buffers = BufferModel(
        sram_kb=args.sram_kb,
        tile_k=args.tile_k,
        tile_n=args.tile_n,
        weight_bits=args.weight_bits,
        fetch_depth=args.fetch_depth,
        psum_bits=args.psum_bits,
        output_bits=args.output_bits,
        buffer_factor=args.buffer_factor,
    )

    full_exec = {"p8": miss_count, "p6": 0, "p5": 0, "p4": 0}
    out_rows: list[dict[str, Any]] = [
        make_row(
            method="FullP8-miss",
            reuse_count=reuse_count,
            direct_count=direct_count,
            residual_count=residual_count,
            miss_count=miss_count,
            total_nodes=total_nodes,
            exec_counts=full_exec,
            loads=baseline_loads,
            baseline_loads=baseline_loads,
            tail_util=miss_count / (baseline_loads * args.baseline_tile_batch)
            if baseline_loads
            else 1.0,
            suffix="now",
            drop=args.fullp8_drop_percent,
            base=base,
            components=components,
            sram_fit=args.baseline_tile_batch <= buffers.max_batch(),
        )
    ]

    depth_counts = {key: 0 for key in DEPTH_KEYS}
    for row in miss_rows:
        depth_counts[depth_key(row)] += 1
    out_rows.append(
        make_row(
            method="GraphBit-now",
            reuse_count=reuse_count,
            direct_count=direct_count,
            residual_count=residual_count,
            miss_count=miss_count,
            total_nodes=total_nodes,
            exec_counts=depth_counts,
            loads=baseline_loads,
            baseline_loads=baseline_loads,
            tail_util=miss_count / (baseline_loads * args.baseline_tile_batch)
            if baseline_loads
            else 1.0,
            suffix="now",
            drop=args.drop_percent,
            base=base,
            components=components,
            sram_fit=args.baseline_tile_batch <= buffers.max_batch(),
        )
    )

    for batch in args.candidate_batches:
        exec_original, loads_original, tail_original = replay_original_order(miss_rows, batch)
        out_rows.append(
            make_row(
                method=f"OriginalOrder-b{batch}",
                reuse_count=reuse_count,
                direct_count=direct_count,
                residual_count=residual_count,
                miss_count=miss_count,
                total_nodes=total_nodes,
                exec_counts=exec_original,
                loads=loads_original,
                baseline_loads=baseline_loads,
                tail_util=tail_original,
                suffix=f"ws_b{batch}",
                drop=args.drop_percent,
                base=base,
                components=components,
                sram_fit=batch <= buffers.max_batch(),
            )
        )
        exec_bucket, loads_bucket, tail_bucket = replay_risk_bucket(miss_rows, batch)
        out_rows.append(
            make_row(
                method=f"RiskBucket-b{batch}",
                reuse_count=reuse_count,
                direct_count=direct_count,
                residual_count=residual_count,
                miss_count=miss_count,
                total_nodes=total_nodes,
                exec_counts=exec_bucket,
                loads=loads_bucket,
                baseline_loads=baseline_loads,
                tail_util=tail_bucket,
                suffix=f"ws_b{batch}",
                drop=args.drop_percent,
                base=base,
                components=components,
                sram_fit=batch <= buffers.max_batch(),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{meta.get('dataset', 'trace')}_seed{meta.get('seed', 'x')}_{meta.get('config', 'config')}"
    tsv_path = args.output_dir / f"{prefix}_trace_replay.tsv"
    txt_path = args.output_dir / f"{prefix}_trace_replay.txt"
    json_path = args.output_dir / f"{prefix}_trace_replay.json"
    component_rows = component_lookup_rows(base, components)
    component_path = args.output_dir / f"{prefix}_component_lookup.tsv"
    payload = {"meta": meta, "rows": out_rows, "component_lookup": component_rows}
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    fieldnames = [
        "method",
        "reuse",
        "direct",
        "residual",
        "miss",
        "cycles",
        "traffic",
        "energy",
        "drop",
        "avg_depth",
        "depth_hist",
        "wloads",
        "baseline_wloads",
        "wscale",
        "tail_util",
        "sram_fit",
    ]
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in out_rows:
            out = dict(row)
            out["depth_hist"] = fmt_hist(row["depth_hist"])
            for key in ("reuse", "direct", "residual", "miss", "cycles", "traffic", "energy", "avg_depth", "wscale", "tail_util"):
                out[key] = f"{float(out[key]):.6f}"
            out["drop"] = f"{float(out['drop']):.3f}"
            writer.writerow(out)

    component_fields = [
        "case",
        "depth",
        "mode",
        "batch",
        "cycles_norm",
        "traffic_norm",
        "energy_norm",
        "cycles",
        "dram_read_requests",
        "dram_write_requests",
    ]
    with component_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=component_fields, delimiter="\t")
        writer.writeheader()
        for row in component_rows:
            out = dict(row)
            for key in (
                "cycles_norm",
                "traffic_norm",
                "energy_norm",
                "cycles",
                "dram_read_requests",
                "dram_write_requests",
            ):
                out[key] = f"{float(out[key]):.6f}"
            writer.writerow(out)

    lines = [
        "Graph-Bit trace-driven scheduler replay",
        f"trace: {args.trace}",
        f"components: {args.components_root}",
        f"dataset={meta.get('dataset')} | seed={meta.get('seed')} | config={meta.get('config')} | priority={meta.get('priority')}",
        f"nodes={total_nodes} | reuse={fmt_pct(reuse_count/total_nodes)} | miss={fmt_pct(miss_count/total_nodes)} | max_sram_batch={buffers.max_batch()}",
        "",
        (
            f"{'Method':<18s} {'Reuse':>7s} {'Miss':>7s} {'Cycles':>8s} "
            f"{'Traffic':>8s} {'Energy':>8s} {'Drop':>7s} {'AvgD':>6s} "
            f"{'Hist(miss)':<24s} {'Wloads':>7s} {'Wscale':>7s} {'Tail':>7s} {'SRAM':>5s}"
        ),
        "-" * 132,
    ]
    for row in out_rows:
        lines.append(
            f"{row['method']:<18s} {fmt_pct(row['reuse']):>7s} {fmt_pct(row['miss']):>7s} "
            f"{row['cycles']:8.3f} {row['traffic']:8.3f} {row['energy']:8.3f} "
            f"{row['drop']:6.2f}% {row['avg_depth']:6.2f} {fmt_hist(row['depth_hist']):<24s} "
            f"{int(row['wloads']):7d} {row['wscale']:7.3f} {fmt_pct(row['tail_util']):>7s} {row['sram_fit']:>5s}"
        )
    lines.extend(
        [
            "",
            "ONNXim component lookup",
            f"component_tsv: {component_path}",
            f"{'Case':<14s} {'Depth':>5s} {'Mode':>5s} {'Batch':>5s} {'Cycles':>8s} {'Traffic':>8s} {'Energy':>8s}",
            "-" * 62,
        ]
    )
    for row in component_rows:
        lines.append(
            f"{row['case']:<14s} {row['depth']:>5s} {row['mode']:>5s} {str(row['batch']):>5s} "
            f"{row['cycles_norm']:8.3f} {row['traffic_norm']:8.3f} {row['energy_norm']:8.3f}"
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "- FullP8-miss replays the same miss set but forces every miss to D8.",
            "- GraphBit-now uses the real per-node stop depth, without larger W tile reuse.",
            "- OriginalOrder-bN preserves node order; a mixed micro-batch executes to the maximum depth inside the batch.",
            "- RiskBucket-bN groups miss nodes by actual stop-depth bucket before batching, so Wloads and Wscale are replayed from the real trace.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    print(txt_path.read_text())
    print(f"[GraphBitTraceReplay] wrote {tsv_path}")
    print(f"[GraphBitTraceReplay] wrote {component_path}")
    print(f"[GraphBitTraceReplay] wrote {json_path}")


if __name__ == "__main__":
    main()
