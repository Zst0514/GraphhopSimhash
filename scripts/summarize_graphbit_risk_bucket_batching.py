#!/usr/bin/env python3
"""Summarize Graph-Bit risk-bucket batching effects.

This is a hardware/dataflow summary.  It keeps the already validated Cora
`h8_54_T40` front-end fixed and asks a narrower NPU question:

    If miss nodes have different Graph-Bit depths, what happens when they are
    randomly mixed inside the same bit-serial micro-batch versus grouped by
    degree/risk bucket?

The model assumes a shared bit-plane controller per micro-batch.  A randomly
mixed batch must execute to the maximum depth present in that batch, while a
degree-bucketed schedule can execute each risk bucket with its own depth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def depth(agg: dict[str, Any], fallback: float = 8.0) -> float:
    value = enc(agg).get("graphbit_avg_depth")
    return fallback if value is None else float(value)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def weighted_aggs(weights: dict[str, float], aggs: dict[str, dict[str, Any]], key: str) -> float:
    return sum(weights[name] * metric(agg, key) for name, agg in aggs.items())


def weighted_reqs(weights: dict[str, float], aggs: dict[str, dict[str, Any]]) -> float:
    return sum(weights[name] * reqs(agg) for name, agg in aggs.items())


def random_max_weights(class_probs: dict[str, float], batch_size: int) -> dict[str, float]:
    """Expected max-depth class for a randomly mixed batch.

    Class order is high > mid > low.  With a shared bit-plane controller, the
    whole batch executes to the highest-risk class present in that batch.
    """

    p_high = class_probs["high"]
    p_mid = class_probs["mid"]
    p_low = class_probs["low"]
    no_high = (1.0 - p_high) ** batch_size
    all_low = p_low**batch_size
    high = 1.0 - no_high
    mid = max(0.0, no_high - all_low)
    low = all_low
    total = high + mid + low
    return {"high": high / total, "mid": mid / total, "low": low / total}


def bucket_padding_overhead(class_probs: dict[str, float], batch_size: int, miss_nodes: int) -> float:
    """Padding slots added by bucketed execution relative to useful miss nodes."""

    useful = max(1, miss_nodes)
    padded = 0
    for prob in class_probs.values():
        count = int(round(prob * useful))
        if count == 0:
            continue
        padded += math.ceil(count / batch_size) * batch_size
    return padded / useful


def make_row(
    *,
    method: str,
    assignment: str,
    schedule: str,
    mode: str,
    batch_size: int,
    class_probs: dict[str, float],
    exec_weights: dict[str, float],
    aggs: dict[str, dict[str, Any]],
    base: dict[str, Any],
    drop: float,
    miss_nodes: int,
) -> dict[str, Any]:
    useful_depth = sum(class_probs[name] * depth(aggs[name]) for name in class_probs)
    exec_depth = sum(exec_weights[name] * depth(aggs[name]) for name in exec_weights)
    bit_util = safe_div(useful_depth, exec_depth)
    base_cycles = metric(base, "cycles")
    base_input = metric(base, "mem_read_input_actual")
    base_weight = metric(base, "mem_read_weight")
    base_output = metric(base, "mem_write_output")
    base_traffic = reqs(base)

    bitcomp = sum(exec_weights[name] * bitcomp_ratio(aggs[name]) for name in exec_weights)
    cycles = weighted_aggs(exec_weights, aggs, "cycles")
    act_read = weighted_aggs(exec_weights, aggs, "mem_read_input_actual")
    act_original = weighted_aggs(exec_weights, aggs, "mem_read_input_original")
    weight_read = weighted_aggs(exec_weights, aggs, "mem_read_weight")
    out_write = weighted_aggs(exec_weights, aggs, "mem_write_output")
    traffic = weighted_reqs(exec_weights, aggs)

    padding = (
        bucket_padding_overhead(class_probs, batch_size, miss_nodes)
        if schedule == "risk_bucket"
        else 1.0
    )

    return {
        "method": method,
        "assignment": assignment,
        "schedule": schedule,
        "mode": mode,
        "batch": batch_size,
        "useful_depth": useful_depth,
        "executed_depth": exec_depth,
        "bit_util": bit_util,
        "wasted_bitplanes": 1.0 - bit_util,
        "cycles_norm": safe_div(cycles, base_cycles),
        "bitcomp_norm": bitcomp,
        "act_read_norm": safe_div(act_read, base_input),
        "act_save": 1.0 - safe_div(act_read, act_original),
        "weight_read_norm": safe_div(weight_read, base_weight),
        "output_write_norm": safe_div(out_write, base_output),
        "traffic_norm": safe_div(traffic, base_traffic),
        "bucket_padding": padding,
        "drop": drop,
    }


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
        default=root / "output" / "graphbit_predictor_free" / "cora_h8_54_T40" / "risk_bucket_batching",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--node-count", type=int, default=2708)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()

    full_row = load_summary_row(args.summary, "FullP8")
    degree_row = load_summary_row(args.summary, "Deg")
    random_row = load_summary_row(args.summary, "Rand")

    miss_ratio = parse_percent(degree_row["P8"]) + parse_percent(degree_row["P6"]) + parse_percent(degree_row["P4"])
    class_probs = {
        "high": parse_percent(degree_row["P8"]) / miss_ratio,
        "mid": parse_percent(degree_row["P6"]) / miss_ratio,
        "low": parse_percent(degree_row["P4"]) / miss_ratio,
    }
    miss_nodes = int(round(args.node_count * miss_ratio))

    full = load_agg(args.microbench_dir / f"microbench_s{args.seq_len}_internal_p8" / "aggregate.json")
    static_aggs = {
        "high": full,
        "mid": load_agg(args.microbench_dir / f"microbench_s{args.seq_len}_internal_p6" / "aggregate.json"),
        "low": load_agg(args.microbench_dir / f"microbench_s{args.seq_len}_internal_p4" / "aggregate.json"),
    }
    early_aggs = {
        "high": full,
        "mid": load_agg(
            args.microbench_dir / f"microbench_s{args.seq_len}_internal_bound_mid_min6_t0p02" / "aggregate.json"
        ),
        "low": load_agg(
            args.microbench_dir / f"microbench_s{args.seq_len}_internal_bound_low_min4_t0p04" / "aggregate.json"
        ),
    }

    rows = []
    for batch_size in args.batch_sizes:
        random_exec_static = random_max_weights(class_probs, batch_size)
        random_exec_early = random_max_weights(class_probs, batch_size)
        rows.append(
            make_row(
                method="RandomOrder Static",
                assignment="degree",
                schedule="random_mixed",
                mode="static",
                batch_size=batch_size,
                class_probs=class_probs,
                exec_weights=random_exec_static,
                aggs=static_aggs,
                base=full,
                drop=float(degree_row["drop"].rstrip("%")),
                miss_nodes=miss_nodes,
            )
        )
        rows.append(
            make_row(
                method="DegreeBucket Static",
                assignment="degree",
                schedule="risk_bucket",
                mode="static",
                batch_size=batch_size,
                class_probs=class_probs,
                exec_weights=class_probs,
                aggs=static_aggs,
                base=full,
                drop=float(degree_row["drop"].rstrip("%")),
                miss_nodes=miss_nodes,
            )
        )
        rows.append(
            make_row(
                method="RandomRisk Bucket",
                assignment="random",
                schedule="risk_bucket",
                mode="static",
                batch_size=batch_size,
                class_probs=class_probs,
                exec_weights=class_probs,
                aggs=static_aggs,
                base=full,
                drop=float(random_row["drop"].rstrip("%")),
                miss_nodes=miss_nodes,
            )
        )
        rows.append(
            make_row(
                method="RandomOrder EarlyStop",
                assignment="degree",
                schedule="random_mixed",
                mode="earlystop",
                batch_size=batch_size,
                class_probs=class_probs,
                exec_weights=random_exec_early,
                aggs=early_aggs,
                base=full,
                drop=float(degree_row["drop"].rstrip("%")),
                miss_nodes=miss_nodes,
            )
        )
        rows.append(
            make_row(
                method="DegreeBucket EarlyStop",
                assignment="degree",
                schedule="risk_bucket",
                mode="earlystop",
                batch_size=batch_size,
                class_probs=class_probs,
                exec_weights=class_probs,
                aggs=early_aggs,
                base=full,
                drop=float(degree_row["drop"].rstrip("%")),
                miss_nodes=miss_nodes,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "risk_bucket_batching.tsv"
    txt_path = args.output_dir / "risk_bucket_batching.txt"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    main_batch = max(args.batch_sizes)
    main_rows = [row for row in rows if row["batch"] == main_batch]
    lines = [
        "Cora h8_54_T40 Graph-Bit risk-bucket batching summary",
        f"Miss nodes={miss_nodes} | class mix among misses: high={fmt_pct(class_probs['high'])}, "
        f"mid={fmt_pct(class_probs['mid'])}, low={fmt_pct(class_probs['low'])}",
        "",
        f"Main table at micro-batch size {main_batch}:",
        "Method                   Assign  Sched        Mode       UsefulD ExecD  Util  Waste  Cycles BitComp ActRd Traffic Drop",
        "-----------------------------------------------------------------------------------------------------------------------",
    ]
    for row in main_rows:
        lines.append(
            f"{row['method']:<24} {row['assignment']:<7} {row['schedule']:<12} {row['mode']:<9} "
            f"{row['useful_depth']:>7.2f} {row['executed_depth']:>5.2f} "
            f"{fmt_pct(row['bit_util']):>6} {fmt_pct(row['wasted_bitplanes']):>6} "
            f"{row['cycles_norm']:>7.3f} {row['bitcomp_norm']:>7.3f} "
            f"{row['act_read_norm']:>5.3f} {row['traffic_norm']:>7.3f} "
            f"{row['drop']:>4.2f}%"
        )

    lines.extend(
        [
            "",
            "Batch-size sweep:",
            "Batch Method                   ExecD  Util  Cycles ActRd Traffic",
            "---------------------------------------------------------------",
        ]
    )
    for row in rows:
        if row["method"] not in {"RandomOrder Static", "DegreeBucket Static", "DegreeBucket EarlyStop"}:
            continue
        lines.append(
            f"{row['batch']:>5} {row['method']:<24} {row['executed_depth']:>5.2f} "
            f"{fmt_pct(row['bit_util']):>6} {row['cycles_norm']:>7.3f} "
            f"{row['act_read_norm']:>5.3f} {row['traffic_norm']:>7.3f}"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- RandomOrder keeps the same Degree assignment but mixes risk classes inside a micro-batch.",
            "- With a shared bit-plane controller, the mixed batch runs to the maximum depth in that batch.",
            "- DegreeBucket groups high/mid/low risk misses, so each bucket can use its own bit-depth or early-stop tolerance.",
            "- RandomRisk Bucket has the same hardware cost as DegreeBucket Static but uses random risk assignment; its higher drop isolates why the graph proxy matters.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[GraphBitRiskBucket] wrote {tsv_path}")
    print(f"[GraphBitRiskBucket] wrote {txt_path}")
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
