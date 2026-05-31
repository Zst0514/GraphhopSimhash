#!/usr/bin/env python3
"""Replay Graph-Bit node traces using actual-M ONNXim components.

Unlike `replay_graphbit_trace_scheduler.py`, this script does not use
ws_b32/ws_b64 memory scaling.  Each scheduled micro-batch is mapped to an
ONNXim component whose GEMM M dimension equals the real batch length or the
nearest available padded length.  This makes W reuse a consequence of the
simulated GEMM shape, not a manual Wscale multiplier.
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
            else:
                rows.append(obj)
    if not rows:
        raise SystemExit(f"No rows in trace: {path}")
    return meta, rows


def depth_key(row: dict[str, Any]) -> str:
    bucket = str(row.get("depth_bucket", "")).lower()
    if bucket in DEPTH_KEYS:
        return bucket
    depth = int(row.get("stop_depth", row.get("action_bit", 8)) or 8)
    return KEY_BY_DEPTH.get(depth, "p8")


def depth_value(row: dict[str, Any]) -> int:
    return DEPTH_BY_KEY[depth_key(row)]


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def traffic(row: dict[str, Any]) -> float:
    return float(row.get("dram_read_requests", 0.0) or 0.0) + float(
        row.get("dram_write_requests", 0.0) or 0.0
    )


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ComponentTable:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: dict[tuple[int, str], dict[str, Any]] = {}
        self.batches: list[int] = []
        self._load()

    def _load(self) -> None:
        for path in sorted(self.root.glob("m*_p*/aggregate.json")):
            name = path.parent.name
            # m32_p6
            try:
                m_part, depth = name.split("_", 1)
                batch = int(m_part[1:])
            except ValueError:
                continue
            row = load_json(path)["encoder"]
            self.rows[(batch, depth)] = row
            if batch not in self.batches:
                self.batches.append(batch)
        self.batches.sort()
        if not self.rows:
            raise SystemExit(f"No actual-batch components found under {self.root}")

    def pick_batch(self, size: int) -> int:
        for batch in self.batches:
            if batch >= size:
                return batch
        return self.batches[-1]

    def row(self, size: int, depth: str) -> tuple[int, dict[str, Any]]:
        batch = self.pick_batch(size)
        key = (batch, depth)
        if key not in self.rows:
            raise SystemExit(f"Missing component for M={batch}, depth={depth}")
        return batch, self.rows[key]

    def p8_baseline(self, batch: int) -> dict[str, Any]:
        if (batch, "p8") not in self.rows:
            batch = self.pick_batch(batch)
        return self.rows[(batch, "p8")]


def schedule_full_p8(rows: list[dict[str, Any]], batch_size: int) -> list[tuple[int, str]]:
    return [(len(block), "p8") for block in chunked(rows, batch_size)]


def schedule_original(rows: list[dict[str, Any]], batch_size: int) -> list[tuple[int, str]]:
    schedule: list[tuple[int, str]] = []
    for block in chunked(rows, batch_size):
        max_depth = max(depth_value(row) for row in block)
        schedule.append((len(block), KEY_BY_DEPTH.get(max_depth, "p8")))
    return schedule


def schedule_bucket(rows: list[dict[str, Any]], batch_size: int) -> list[tuple[int, str]]:
    buckets = {key: [] for key in DEPTH_KEYS}
    for row in rows:
        buckets[depth_key(row)].append(row)
    schedule: list[tuple[int, str]] = []
    for key in DEPTH_KEYS:
        for block in chunked(buckets[key], batch_size):
            schedule.append((len(block), key))
    return schedule


def compose(
    *,
    schedule: list[tuple[int, str]],
    components: ComponentTable,
    norm_cycles: float,
    norm_traffic: float,
    miss_count: int,
) -> dict[str, Any]:
    cycles = 0.0
    traff = 0.0
    energy = 0.0
    depth_sum = 0.0
    padded = 0
    wloads = 0
    hist = {key: 0 for key in DEPTH_KEYS}
    batch_hist: dict[int, int] = {}
    for size, depth in schedule:
        picked_batch, row = components.row(size, depth)
        cycles += f(row, "cycles")
        this_traffic = traffic(row)
        traff += this_traffic
        energy += 0.5 * f(row, "cycles") + 0.5 * this_traffic
        depth_sum += DEPTH_BY_KEY[depth] * size
        hist[depth] += size
        padded += picked_batch
        wloads += 1
        batch_hist[picked_batch] = batch_hist.get(picked_batch, 0) + 1
    return {
        "cycles_abs": cycles,
        "traffic_abs": traff,
        "energy_abs": energy,
        "cycles": cycles / norm_cycles if norm_cycles else 0.0,
        "traffic": traff / norm_traffic if norm_traffic else 0.0,
        "energy": energy / (0.5 * norm_cycles + 0.5 * norm_traffic)
        if norm_cycles or norm_traffic
        else 0.0,
        "avg_depth": depth_sum / miss_count if miss_count else 0.0,
        "depth_hist": hist,
        "wloads": wloads,
        "tail_util": miss_count / padded if padded else 1.0,
        "batch_hist": batch_hist,
    }


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_hist(hist: dict[str, int], denom: int) -> str:
    if denom <= 0:
        return "-"
    return ",".join(
        f"{key.upper()}:{100.0 * count / denom:.1f}%"
        for key, count in hist.items()
        if count
    )


def fmt_batch_hist(hist: dict[int, int]) -> str:
    return ",".join(f"M{batch}:{count}" for batch, count in sorted(hist.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-batch", type=int, default=16)
    parser.add_argument("--bucket-batches", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--drop-percent", type=float, default=0.0)
    parser.add_argument("--fullp8-drop-percent", type=float, default=0.0)
    args = parser.parse_args()

    meta, rows = load_trace(args.trace)
    miss_rows = [row for row in rows if bool(row.get("is_miss", False))]
    total_nodes = len(rows)
    miss_count = len(miss_rows)
    reuse_count = total_nodes - miss_count
    components = ComponentTable(args.components_root)

    all_full_schedule = schedule_full_p8(rows, args.baseline_batch)
    norm = compose(
        schedule=all_full_schedule,
        components=components,
        norm_cycles=1.0,
        norm_traffic=1.0,
        miss_count=total_nodes,
    )
    norm_cycles = norm["cycles_abs"]
    norm_traffic = norm["traffic_abs"]

    outputs: list[dict[str, Any]] = []

    def add(method: str, schedule: list[tuple[int, str]], drop: float) -> None:
        row = compose(
            schedule=schedule,
            components=components,
            norm_cycles=norm_cycles,
            norm_traffic=norm_traffic,
            miss_count=miss_count,
        )
        row.update(
            {
                "method": method,
                "reuse": reuse_count / total_nodes if total_nodes else 0.0,
                "miss": miss_count / total_nodes if total_nodes else 0.0,
                "drop": drop,
            }
        )
        outputs.append(row)

    add(
        "FullP8-miss",
        schedule_full_p8(miss_rows, args.baseline_batch),
        args.fullp8_drop_percent,
    )
    add(
        "GraphBit-original",
        schedule_original(miss_rows, args.baseline_batch),
        args.drop_percent,
    )
    for batch in args.bucket_batches:
        add(
            f"FullP8-bucket-b{batch}",
            schedule_full_p8(miss_rows, batch),
            args.fullp8_drop_percent,
        )
        add(
            f"RiskBucket-b{batch}",
            schedule_bucket(miss_rows, batch),
            args.drop_percent,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{meta.get('dataset','trace')}_seed{meta.get('seed','x')}_{meta.get('config','config')}"
    tsv_path = args.output_dir / f"{prefix}_actual_batch_replay.tsv"
    txt_path = args.output_dir / f"{prefix}_actual_batch_replay.txt"
    json_path = args.output_dir / f"{prefix}_actual_batch_replay.json"

    payload = {
        "meta": meta,
        "components_root": str(args.components_root),
        "normalization": {
            "all_nodes_fullp8_batch": args.baseline_batch,
            "cycles_abs": norm_cycles,
            "traffic_abs": norm_traffic,
        },
        "rows": outputs,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fields = [
        "method",
        "reuse",
        "miss",
        "cycles",
        "traffic",
        "energy",
        "drop",
        "avg_depth",
        "wloads",
        "tail_util",
        "depth_hist",
        "batch_hist",
        "cycles_abs",
        "traffic_abs",
    ]
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in outputs:
            out = dict(row)
            out["depth_hist"] = fmt_hist(row["depth_hist"], miss_count)
            out["batch_hist"] = fmt_batch_hist(row["batch_hist"])
            for key in ("reuse", "miss", "cycles", "traffic", "energy", "avg_depth", "tail_util"):
                out[key] = f"{float(out[key]):.6f}"
            out["drop"] = f"{float(out['drop']):.3f}"
            writer.writerow({key: out[key] for key in fields})

    lines = [
        "Graph-Bit actual-batch ONNXim trace replay",
        f"trace: {args.trace}",
        f"components: {args.components_root}",
        f"normalization: all nodes FullP8, batch={args.baseline_batch}, cycles={norm_cycles:.0f}, traffic={norm_traffic:.0f}",
        f"nodes={total_nodes} | reuse={fmt_pct(reuse_count/total_nodes)} | miss={fmt_pct(miss_count/total_nodes)}",
        "",
        (
            f"{'Method':<20s} {'Reuse':>7s} {'Miss':>7s} {'Cycles':>8s} "
            f"{'Traffic':>8s} {'Energy':>8s} {'Drop':>7s} {'AvgD':>6s} "
            f"{'Wloads':>7s} {'Tail':>7s} {'Hist(miss)':<24s} {'BatchHist':<24s}"
        ),
        "-" * 140,
    ]
    for row in outputs:
        lines.append(
            f"{row['method']:<20s} {fmt_pct(row['reuse']):>7s} {fmt_pct(row['miss']):>7s} "
            f"{row['cycles']:8.3f} {row['traffic']:8.3f} {row['energy']:8.3f} "
            f"{row['drop']:6.2f}% {row['avg_depth']:6.2f} {row['wloads']:7d} "
            f"{fmt_pct(row['tail_util']):>7s} {fmt_hist(row['depth_hist'], miss_count):<24s} "
            f"{fmt_batch_hist(row['batch_hist']):<24s}"
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "- No memory_scale or Wscale is applied in this replay.",
            "- W reuse is represented by actual ONNXim GEMM M sizes: M16/M32/M64.",
            "- FullP8-bucket isolates larger batch W reuse while keeping every miss at P8.",
            "- RiskBucket uses the real stop-depth trace and maps each bucket batch to an actual-M component.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(txt_path.read_text(encoding="utf-8"))
    print(f"[GraphBitActualReplay] wrote {tsv_path}")
    print(f"[GraphBitActualReplay] wrote {json_path}")


if __name__ == "__main__":
    main()
