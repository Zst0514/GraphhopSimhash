#!/usr/bin/env python3
"""Build the Cora Graph-Bit predictor-free main table.

Input:
  1. residual_precision_depth TSV from summarize_residual_graphbit.py
  2. ONNXim LLaMA GEMM microbenchmark aggregate.json

Output:
  A compact table with static precision-depth rows and, when available, true
  runtime-bound rows from residual_precision_depth:

    FullP8-miss
    Random static P8/P6/P5/P4
    Degree/TSER static P8/P6/P5/P4
    Degree/TSER runtime-bound min_depth+tolerance

The runtime-bound rows use the actual embedding pool selected by the runner
after graph risk sets min_depth/tolerance and the predictor-free bound decides
the final depth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONFIG_ALIASES = {
    "RandomDepthBudget": "Rand",
    "DegreeDepthBudget": "Deg",
    "TSERDepthBudget": "TSER",
    "ContextDepthBudget": "Ctx",
    "LowUniqueDepthBudget": "Uniq",
    "DegreeBound": "DegBound",
    "DegBoundNode": "DegBoundNode",
    "DegreeBoundNode": "DegBoundNode",
    "TSERBound": "TSERBound",
    "TSERBoundNode": "TSERBoundNode",
    "ContextBound": "CtxBound",
    "CtxBoundNode": "CtxBoundNode",
    "ContextBoundNode": "CtxBoundNode",
    "LowUniqueBound": "UniqBound",
    "LowUniqueBoundNode": "UniqBoundNode",
}


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_config(name: str) -> str:
    return CONFIG_ALIASES.get(str(name), str(name))


def parse_percent(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    number = float(text)
    return number / 100.0 if abs(number) > 1.0 else number


def parse_float(value: Any, default: float | None = 0.0) -> float | None:
    text = str(value or "").strip()
    if not text:
        return default
    if text.endswith("%"):
        return float(text[:-1])
    return float(text)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_row(rows: list[dict[str, str]], args: argparse.Namespace, config: str) -> dict[str, str]:
    matches = []
    for row in rows:
        if args.dataset and row.get("dataset") != args.dataset:
            continue
        if args.heads and row.get("heads") != args.heads:
            continue
        if args.threshold is not None and str(row.get("T", "")) != str(args.threshold):
            continue
        if args.budget and row.get("budget") != args.budget:
            continue
        if args.runs is not None and str(row.get("runs", "")) != str(args.runs):
            continue
        if normalize_config(row.get("config", "")) != normalize_config(config):
            continue
        matches.append(row)
    if not matches:
        raise SystemExit(f"No row found for config={config} in {args.residual_summary}")
    return matches[0]


def maybe_find_row(rows: list[dict[str, str]], args: argparse.Namespace, config: str) -> dict[str, str] | None:
    try:
        return find_row(rows, args, config)
    except SystemExit:
        return None


def ratios_from_row(row: dict[str, str]) -> dict[str, float]:
    direct = parse_percent(row.get("direct"))
    residual = parse_percent(row.get("residual"))
    reuse = parse_percent(row.get("reuse")) or (direct + residual)
    return {
        "reuse": reuse,
        "direct": direct,
        "residual": residual,
        "miss": max(0.0, 1.0 - reuse),
        "p8": parse_percent(row.get("P8")),
        "p7": parse_percent(row.get("P7")),
        "p6": parse_percent(row.get("P6")),
        "p5": parse_percent(row.get("P5")),
        "p4": parse_percent(row.get("P4")),
    }


def depth_factor(ratios: dict[str, float], bits: dict[str, float]) -> float:
    return (
        ratios.get("p8", 0.0) * bits["p8"] / 8.0
        + ratios.get("p7", 0.0) * bits["p7"] / 8.0
        + ratios.get("p6", 0.0) * bits["p6"] / 8.0
        + ratios.get("p5", 0.0) * bits["p5"] / 8.0
        + ratios.get("p4", 0.0) * bits["p4"] / 8.0
    )


def miss_avg_depth(ratios: dict[str, float], bits: dict[str, float]) -> float:
    miss = max(
        1e-12,
        ratios.get("p8", 0.0)
        + ratios.get("p7", 0.0)
        + ratios.get("p6", 0.0)
        + ratios.get("p5", 0.0)
        + ratios.get("p4", 0.0),
    )
    return (
        ratios.get("p8", 0.0) * bits["p8"]
        + ratios.get("p7", 0.0) * bits["p7"]
        + ratios.get("p6", 0.0) * bits["p6"]
        + ratios.get("p5", 0.0) * bits["p5"]
        + ratios.get("p4", 0.0) * bits["p4"]
    ) / miss


def traffic_factor(
    ratios: dict[str, float],
    bits: dict[str, float],
    weight_share: float,
    activation_share: float,
    output_share: float,
    cache_read_cost: float,
    residual_traffic_cost: float,
) -> float:
    miss = (
        ratios.get("p8", 0.0)
        + ratios.get("p7", 0.0)
        + ratios.get("p6", 0.0)
        + ratios.get("p5", 0.0)
        + ratios.get("p4", 0.0)
    )
    act_factor = depth_factor(ratios, bits)
    direct = ratios.get("direct", 0.0)
    residual = ratios.get("residual", 0.0)
    return (
        weight_share * miss
        + (activation_share + output_share) * act_factor
        + cache_read_cost * direct
        + (cache_read_cost + residual_traffic_cost) * residual
    )


def method_row(
    method: str,
    source_row: dict[str, str],
    args: argparse.Namespace,
    aggregate: dict[str, Any] | None,
    bounded: bool,
) -> dict[str, Any]:
    ratios = ratios_from_row(source_row)
    fixed_bits = {"p8": 8.0, "p7": 7.0, "p6": 6.0, "p5": 5.0, "p4": 4.0}
    bits = dict(fixed_bits)
    if bounded:
        bits["p6"] = max(1.0, bits["p6"] - args.bounded_save_p6)
        bits["p5"] = max(1.0, bits["p5"] - args.bounded_save_p5)
        bits["p4"] = max(1.0, bits["p4"] - args.bounded_save_p4)

    compute_norm = (
        depth_factor(ratios, bits)
        + ratios.get("direct", 0.0) * args.cache_read_compute_cost
        + ratios.get("residual", 0.0) * args.residual_compute_cost
    )
    traffic_norm = traffic_factor(
        ratios,
        bits,
        args.weight_traffic_share,
        args.activation_traffic_share,
        args.output_traffic_share,
        args.cache_read_traffic_cost,
        args.residual_traffic_cost,
    )
    energy = args.compute_energy_weight * compute_norm + args.traffic_energy_weight * traffic_norm

    baseline_cycles = None
    baseline_mem = None
    if aggregate is not None:
        encoder = aggregate["encoder"]
        baseline_cycles = float(encoder["cycles"])
        baseline_mem = float(encoder.get("dram_read_requests", 0.0)) + float(encoder.get("dram_write_requests", 0.0))

    return {
        "method": method,
        "config": normalize_config(source_row.get("config", "")),
        "reuse": ratios["reuse"],
        "direct": ratios["direct"],
        "residual": ratios["residual"],
        "p8": ratios["p8"],
        "p7": ratios["p7"],
        "p6": ratios["p6"],
        "p5": ratios["p5"],
        "p4": ratios["p4"],
        "algo_cost": parse_float(source_row.get("cost")),
        "acc": parse_float(source_row.get("acc")),
        "drop": parse_float(source_row.get("drop")),
        "finalerr": parse_float(source_row.get("finalerr")),
        "avg_miss_depth": miss_avg_depth(ratios, bits),
        "saved_bitplanes": 8.0 - miss_avg_depth(ratios, bits),
        "cycles_norm": compute_norm,
        "traffic_norm": traffic_norm,
        "energy_proxy": energy,
        "cycles": None if baseline_cycles is None else baseline_cycles * compute_norm,
        "mem_requests": None if baseline_mem is None else baseline_mem * traffic_norm,
        "drop_source": "static_proxy",
    }


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "config",
        "reuse",
        "direct",
        "residual",
        "p8",
        "p7",
        "p6",
        "p5",
        "p4",
        "algo_cost",
        "acc",
        "drop",
        "finalerr",
        "avg_miss_depth",
        "saved_bitplanes",
        "cycles_norm",
        "traffic_norm",
        "energy_proxy",
        "cycles",
        "mem_requests",
        "drop_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_compact(rows: list[dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    title_dataset = str(args.dataset).upper()
    if args.soft_support >= args.hard_support:
        residual_support_text = f">={args.soft_support}"
    elif args.soft_support + 1 == args.hard_support:
        residual_support_text = f"=={args.soft_support}"
    else:
        residual_support_text = f"={args.soft_support}..{args.hard_support - 1}"
    lines = [
        f"{title_dataset} Graph-Bit predictor-free main table",
        (
            f"Fixed front-end: {args.frontend_id}, R=2, "
            f"hard>={args.hard_support}, residual support{residual_support_text}."
        ),
        "Runtime-bound rows are evaluated with the embedding depth selected by min_depth+tolerance bound.",
        "",
        "Method                         Reuse   P8     P7     P6     P5     P4     AvgBit Saved  Cycles Traffic Energy Drop",
        "-------------------------------------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<30} "
            f"{fmt_pct(row['reuse']):>6} "
            f"{fmt_pct(row['p8']):>6} "
            f"{fmt_pct(row['p7']):>6} "
            f"{fmt_pct(row['p6']):>6} "
            f"{fmt_pct(row['p5']):>6} "
            f"{fmt_pct(row['p4']):>6} "
            f"{row['avg_miss_depth']:>6.2f} "
            f"{row['saved_bitplanes']:>5.2f} "
            f"{fmt_num(row['cycles_norm']):>7} "
            f"{fmt_num(row['traffic_norm']):>7} "
            f"{fmt_num(row['energy_proxy']):>6} "
            f"{fmt_num(row['drop'], 2):>5}%"
        )
    path.write_text("\n".join(lines) + "\n")


def write_workload(rows: list[dict[str, Any]], args: argparse.Namespace, path: Path) -> None:
    profiles = []
    for row in rows:
        profiles.append(
            {
                "id": row["method"].lower().replace(" ", "_").replace("/", "_"),
                "dataset": args.dataset,
                "model": "llama2_7b",
                "route": {
                    "frontend": args.frontend_id,
                    "method": row["method"],
                    "config": row["config"],
                    "budget": args.budget,
                    "threshold": args.threshold,
                    "heads": args.heads,
                },
                "ratios": {
                    "reuse": row["reuse"],
                    "direct": row["direct"],
                    "residual": row["residual"],
                    "p8": row["p8"],
                    "p7": row["p7"],
                    "p6": row["p6"],
                    "p5": row["p5"],
                    "p4": row["p4"],
                },
                "hardware": {
                    "avg_miss_depth": row["avg_miss_depth"],
                    "saved_bitplanes": row["saved_bitplanes"],
                    "cycles_norm": row["cycles_norm"],
                    "traffic_norm": row["traffic_norm"],
                    "energy_proxy": row["energy_proxy"],
                },
                "metrics": {
                    "acc": row["acc"],
                    "drop_percent": row["drop"],
                    "finalerr": row["finalerr"],
                },
            }
        )
    payload = {
        "schema": "graphbit_predictor_free_main.v1",
        "description": "Ratios are fractions of all graph nodes. Bounded row estimates predictor-free bit-plane early stop.",
        "profiles": profiles,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-summary", type=Path, required=True)
    parser.add_argument("--microbench", type=Path, default=root / "output" / "onnxim_graphbit" / "microbench_s64" / "aggregate.json")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "graphbit_predictor_free" / "cora_h8_54_T40")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--heads", default="h8")
    parser.add_argument("--threshold", type=int, default=40)
    parser.add_argument("--budget", default="balanced")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--frontend-id", default="h8_54_T40")
    parser.add_argument("--hard-support", type=int, default=5)
    parser.add_argument("--soft-support", type=int, default=4)
    parser.add_argument("--bounded-save-p6", type=float, default=0.50)
    parser.add_argument("--bounded-save-p5", type=float, default=0.35)
    parser.add_argument("--bounded-save-p4", type=float, default=0.25)
    parser.add_argument("--weight-traffic-share", type=float, default=0.65)
    parser.add_argument("--activation-traffic-share", type=float, default=0.25)
    parser.add_argument("--output-traffic-share", type=float, default=0.10)
    parser.add_argument("--cache-read-compute-cost", type=float, default=0.001)
    parser.add_argument("--cache-read-traffic-cost", type=float, default=0.003)
    parser.add_argument("--residual-compute-cost", type=float, default=0.005)
    parser.add_argument("--residual-traffic-cost", type=float, default=0.005)
    parser.add_argument("--compute-energy-weight", type=float, default=0.55)
    parser.add_argument("--traffic-energy-weight", type=float, default=0.45)
    args = parser.parse_args()

    rows = read_rows(args.residual_summary)
    aggregate = json.loads(args.microbench.read_text()) if args.microbench.exists() else None
    if aggregate is None:
        print(f"[GraphBitPF] microbench not found, writing normalized metrics only: {args.microbench}")

    full = find_row(rows, args, "FullP8")
    rand = maybe_find_row(rows, args, "Rand")
    deg = maybe_find_row(rows, args, "Deg")
    tser = maybe_find_row(rows, args, "TSER")
    deg_bound = maybe_find_row(rows, args, "DegBoundNode") or maybe_find_row(rows, args, "DegBound")
    tser_bound = maybe_find_row(rows, args, "TSERBoundNode") or maybe_find_row(rows, args, "TSERBound")

    out_rows = [method_row("FullP8-miss", full, args, aggregate, bounded=False)]
    include_static_budget_rows = not str(args.budget).startswith("node_")
    if include_static_budget_rows and rand is not None:
        out_rows.append(method_row("Random static P8/P6/P5/P4", rand, args, aggregate, bounded=False))
    if include_static_budget_rows and deg is not None:
        out_rows.append(method_row("Degree static P8/P6/P5/P4", deg, args, aggregate, bounded=False))
    if include_static_budget_rows and tser is not None:
        out_rows.append(method_row("TSER static P8/P6/P5/P4", tser, args, aggregate, bounded=False))
    if deg_bound is not None:
        out_rows.append(method_row("Degree runtime-bound", deg_bound, args, aggregate, bounded=False))
    elif include_static_budget_rows and deg is not None:
        out_rows.append(method_row("Degree synthetic EarlyStop", deg, args, aggregate, bounded=True))
    if tser_bound is not None:
        out_rows.append(method_row("TSER runtime-bound", tser_bound, args, aggregate, bounded=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "predictor_free_main.tsv"
    txt_path = args.output_dir / "predictor_free_main.txt"
    json_path = args.output_dir / "predictor_free_workload.json"
    write_tsv(out_rows, tsv_path)
    write_compact(out_rows, txt_path, args)
    write_workload(out_rows, args, json_path)

    print(f"[GraphBitPF] wrote {tsv_path}")
    print(f"[GraphBitPF] wrote {txt_path}")
    print(f"[GraphBitPF] wrote {json_path}")
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
