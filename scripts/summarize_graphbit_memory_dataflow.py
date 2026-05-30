#!/usr/bin/env python3
"""Summarize Graph-Bit memory-dataflow variants.

The goal is to separate three effects:

1. bit-serial compute reduction,
2. activation bit-plane packing/read reduction,
3. FFN intermediate output bypass, which reduces some output writes and the
   following FFN-down input reads.

This is a hardware/dataflow summary for Cora h8_54_T40.  It does not change the
validated algorithmic drop numbers.
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
            if (
                row.get("dataset") == "cora"
                and row.get("heads") == "h8"
                and row.get("T") == "40"
                and row.get("budget") == "balanced"
                and row.get("config") == config
            ):
                return row
    raise SystemExit(f"Missing Cora h8 T40 balanced config={config} in {path}")


def load_agg(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing aggregate: {path}")
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def enc(agg: dict[str, Any]) -> dict[str, Any]:
    return agg["encoder"]


def metric(agg: dict[str, Any], key: str) -> float:
    return float(enc(agg).get(key, 0.0) or 0.0)


def reqs(agg: dict[str, Any]) -> float:
    return metric(agg, "dram_read_requests") + metric(agg, "dram_write_requests")


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def bitcomp_ratio(agg: dict[str, Any]) -> float:
    raw = metric(agg, "graphbit_raw_compute_cycles")
    if raw <= 0:
        return 1.0
    return metric(agg, "graphbit_effective_compute_cycles") / raw


def depth(agg: dict[str, Any], fallback: float = 8.0) -> float:
    value = enc(agg).get("graphbit_avg_depth")
    return fallback if value is None else float(value)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def weighted(weights: dict[str, float], aggs: dict[str, dict[str, Any]], key: str) -> float:
    return sum(weights[name] * metric(aggs[name], key) for name in weights)


def weighted_reqs(weights: dict[str, float], aggs: dict[str, dict[str, Any]]) -> float:
    return sum(weights[name] * reqs(aggs[name]) for name in weights)


def weighted_bitcomp(weights: dict[str, float], aggs: dict[str, dict[str, Any]]) -> float:
    return sum(weights[name] * bitcomp_ratio(aggs[name]) for name in weights)


def weighted_depth(weights: dict[str, float], aggs: dict[str, dict[str, Any]]) -> float:
    return sum(weights[name] * depth(aggs[name]) for name in weights)


def ffn_bypass_savings(workspace: Path, layers: int) -> dict[str, float]:
    """Return traffic saved by keeping FFN intermediate on chip.

    We approximate a fused FFN as saving:
      - ffn_up/gate output writes, and
      - ffn_down input reads.

    Both are multiplied by count_per_layer and number of layers to match the
    aggregate encoder scale.
    """

    rows = read_rows(workspace / "summary.tsv")
    saved_read = 0.0
    saved_write = 0.0
    for row in rows:
        count = float(row["count_per_layer"])
        if row["name"] == "ffn_down":
            saved_read += float(row["mem_read_input_actual"]) * count * layers
        if row["name"] == "ffn_up":
            saved_write += float(row["mem_write_output"]) * count * layers
    return {"read": saved_read, "write": saved_write, "total": saved_read + saved_write}


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "graphbit_predictor_free" / "cora_h8_54_T40" / "memory_dataflow",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    degree_row = load_summary_row(args.summary, "Deg")
    full_row = load_summary_row(args.summary, "FullP8")
    miss_ratio = parse_percent(degree_row["P8"]) + parse_percent(degree_row["P6"]) + parse_percent(degree_row["P4"])
    weights = {
        "high": parse_percent(degree_row["P8"]) / miss_ratio,
        "mid": parse_percent(degree_row["P6"]) / miss_ratio,
        "low": parse_percent(degree_row["P4"]) / miss_ratio,
    }

    p8_ws = args.microbench_dir / f"microbench_s{args.seq_len}_internal_p8"
    mid_ws = args.microbench_dir / f"microbench_s{args.seq_len}_internal_bound_mid_min6_t0p02"
    low_ws = args.microbench_dir / f"microbench_s{args.seq_len}_internal_bound_low_min4_t0p04"
    full = load_agg(p8_ws / "aggregate.json")
    early_aggs = {
        "high": full,
        "mid": load_agg(mid_ws / "aggregate.json"),
        "low": load_agg(low_ws / "aggregate.json"),
    }
    no_pack_aggs = {
        # Same bitcomp/depth as early-stop, but memory traffic is modeled as
        # full P8.  This isolates compute-only savings from activation packing.
        "high": full,
        "mid": full,
        "low": full,
    }

    base_traffic = reqs(full)
    base_cycles = metric(full, "cycles")
    base_input = metric(full, "mem_read_input_actual")
    base_weight = metric(full, "mem_read_weight")
    base_output = metric(full, "mem_write_output")

    layers = int(load_agg(p8_ws / "aggregate.json").get("layers", 32))
    bypass = {
        "high": ffn_bypass_savings(p8_ws, layers),
        "mid": ffn_bypass_savings(mid_ws, layers),
        "low": ffn_bypass_savings(low_ws, layers),
    }
    weighted_bypass = sum(weights[name] * bypass[name]["total"] for name in weights)

    rows = []

    def add_row(
        name: str,
        aggs: dict[str, dict[str, Any]],
        *,
        pack: bool,
        bypass_enable: bool,
        drop: float,
    ) -> None:
        traffic = weighted_reqs(weights, aggs)
        if bypass_enable:
            traffic = max(0.0, traffic - weighted_bypass)
        rows.append(
            {
                "method": name,
                "avg_depth": weighted_depth(weights, early_aggs if "EarlyStop" in name else aggs),
                "bitcomp": weighted_bitcomp(weights, early_aggs if "EarlyStop" in name else aggs),
                "cycles": safe_div(weighted(weights, aggs, "cycles"), base_cycles),
                "act_read": safe_div(weighted(weights, early_aggs if pack else no_pack_aggs, "mem_read_input_actual"), base_input),
                "weight_read": safe_div(weighted(weights, aggs, "mem_read_weight"), base_weight),
                "output_write": safe_div(weighted(weights, aggs, "mem_write_output"), base_output),
                "traffic": safe_div(traffic, base_traffic),
                "drop": drop,
                "note": "",
            }
        )

    full_drop = float(full_row["drop"].rstrip("%"))
    degree_drop = float(degree_row["drop"].rstrip("%"))
    add_row(
        "FullP8 miss",
        {"high": full, "mid": full, "low": full},
        pack=False,
        bypass_enable=False,
        drop=full_drop,
    )
    add_row("EarlyStop compute-only", no_pack_aggs, pack=False, bypass_enable=False, drop=degree_drop)
    add_row("EarlyStop + ActPack", early_aggs, pack=True, bypass_enable=False, drop=degree_drop)
    add_row("EarlyStop + ActPack + FFNBypass", early_aggs, pack=True, bypass_enable=True, drop=degree_drop)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "memory_dataflow.tsv"
    txt_path = args.output_dir / "memory_dataflow.txt"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Cora h8_54_T40 Graph-Bit memory dataflow summary",
        f"Miss ratio={fmt_pct(miss_ratio)} | Degree drop anchor={degree_row['drop']} | FullP8 drop={full_row['drop']}",
        "",
        "Method                         AvgD BitComp Cycles ActRd WeightRd OutWr Traffic Drop",
        "----------------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<30} {row['avg_depth']:>4.2f} "
            f"{row['bitcomp']:>7.3f} {row['cycles']:>6.3f} {row['act_read']:>5.3f} "
            f"{row['weight_read']:>8.3f} {row['output_write']:>5.3f} "
            f"{row['traffic']:>7.3f} {row['drop']:>4.2f}%"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- compute-only shows the arithmetic benefit if activation memory is still stored as A8.",
            "- ActPack is the current Graph-Bit bit-plane packed activation path.",
            "- FFNBypass is a dataflow upper bound: keep FFN intermediate on chip to avoid ffn_up output write and ffn_down input read.",
            "- FFNBypass does not change accuracy by itself; it is an exact dataflow optimization if SRAM capacity is sufficient.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[GraphBitMemoryDataflow] wrote {tsv_path}")
    print(f"[GraphBitMemoryDataflow] wrote {txt_path}")
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
