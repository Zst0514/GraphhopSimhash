#!/usr/bin/env python3
"""Combine Graph-Bit workload profiles with ONNXim GEMM baseline.

This script converts algorithm-level routing ratios into hardware-facing
metrics.  ONNXim supplies the Full-P8 encoder baseline cycles/requests; the
Graph-Bit model scales the remaining encoder work by bit-plane depth.
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
    "PredictorDepthBudget": "Pred",
    "OracleDamageBudget": "Oracle",
}


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_config(name: str) -> str:
    return CONFIG_ALIASES.get(name, name)


def load_profiles(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if "profiles" in payload:
        return payload["profiles"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported workload profile format: {path}")


def depth_factor(ratios: dict[str, float], bits: dict[str, float]) -> float:
    return (
        ratios.get("p8", 0.0) * bits["p8"] / 8.0
        + ratios.get("p6", 0.0) * bits["p6"] / 8.0
        + ratios.get("p5", 0.0) * bits["p5"] / 8.0
        + ratios.get("p4", 0.0) * bits["p4"] / 8.0
    )


def traffic_factor(
    ratios: dict[str, float],
    bits: dict[str, float],
    weight_share: float,
    activation_share: float,
    output_share: float,
    cache_read_cost: float,
    residual_traffic_cost: float,
) -> float:
    miss = ratios.get("p8", 0.0) + ratios.get("p6", 0.0) + ratios.get("p5", 0.0) + ratios.get("p4", 0.0)
    act_factor = depth_factor(ratios, bits)
    reuse = ratios.get("direct", 0.0)
    residual = ratios.get("residual", 0.0)
    return (
        weight_share * miss
        + (activation_share + output_share) * act_factor
        + cache_read_cost * reuse
        + (cache_read_cost + residual_traffic_cost) * residual
    )


def summarize_profile(profile: dict[str, Any], aggregate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ratios = profile["ratios"]
    fixed_bits = {"p8": 8.0, "p6": 6.0, "p5": 5.0, "p4": 4.0}
    bounded_bits = {
        "p8": 8.0,
        "p6": max(1.0, 6.0 - args.bounded_save_p6),
        "p5": max(1.0, 5.0 - args.bounded_save_p5),
        "p4": max(1.0, 4.0 - args.bounded_save_p4),
    }
    baseline_cycles = float(aggregate["encoder"]["cycles"])
    baseline_read = float(aggregate["encoder"].get("dram_read_requests", 0.0))
    baseline_write = float(aggregate["encoder"].get("dram_write_requests", 0.0))
    baseline_req = baseline_read + baseline_write

    fixed_compute = depth_factor(ratios, fixed_bits)
    bounded_compute = depth_factor(ratios, bounded_bits)
    residual = ratios.get("residual", 0.0)
    direct = ratios.get("direct", 0.0)
    residual_compute = residual * args.residual_compute_cost
    direct_compute = direct * args.cache_read_compute_cost

    fixed_cycles_norm = fixed_compute + residual_compute + direct_compute
    bounded_cycles_norm = bounded_compute + residual_compute + direct_compute
    fixed_traffic_norm = traffic_factor(
        ratios,
        fixed_bits,
        args.weight_traffic_share,
        args.activation_traffic_share,
        args.output_traffic_share,
        args.cache_read_traffic_cost,
        args.residual_traffic_cost,
    )
    bounded_traffic_norm = traffic_factor(
        ratios,
        bounded_bits,
        args.weight_traffic_share,
        args.activation_traffic_share,
        args.output_traffic_share,
        args.cache_read_traffic_cost,
        args.residual_traffic_cost,
    )
    fixed_energy = args.compute_energy_weight * fixed_cycles_norm + args.traffic_energy_weight * fixed_traffic_norm
    bounded_energy = args.compute_energy_weight * bounded_cycles_norm + args.traffic_energy_weight * bounded_traffic_norm

    route = profile.get("route", {})
    metrics = profile.get("metrics", {})
    return {
        "profile": profile.get("id", ""),
        "dataset": profile.get("dataset", ""),
        "config": normalize_config(route.get("config", "")),
        "heads": route.get("heads", ""),
        "T": route.get("threshold", ""),
        "budget": route.get("budget", ""),
        "reuse": ratios.get("reuse", 0.0),
        "direct": ratios.get("direct", 0.0),
        "residual": residual,
        "p8": ratios.get("p8", 0.0),
        "p6": ratios.get("p6", 0.0),
        "p5": ratios.get("p5", 0.0),
        "p4": ratios.get("p4", 0.0),
        "acc": metrics.get("acc"),
        "drop_percent": metrics.get("drop_percent"),
        "algo_cost": metrics.get("cost"),
        "fixed_cycles_norm": fixed_cycles_norm,
        "bounded_cycles_norm": bounded_cycles_norm,
        "fixed_traffic_norm": fixed_traffic_norm,
        "bounded_traffic_norm": bounded_traffic_norm,
        "fixed_energy_proxy": fixed_energy,
        "bounded_energy_proxy": bounded_energy,
        "fixed_cycles": baseline_cycles * fixed_cycles_norm,
        "bounded_cycles": baseline_cycles * bounded_cycles_norm,
        "fixed_mem_requests": baseline_req * fixed_traffic_norm,
        "bounded_mem_requests": baseline_req * bounded_traffic_norm,
    }


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_compact(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "dataset",
        "config",
        "heads",
        "T",
        "budget",
        "reuse",
        "drop",
        "cycles",
        "traffic",
        "energy",
        "bounded",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["dataset"]),
                    str(row["config"]),
                    str(row["heads"]),
                    str(row["T"]),
                    str(row["budget"]),
                    f"{100 * row['reuse']:.1f}%",
                    "" if row["drop_percent"] is None else f"{row['drop_percent']:.2f}%",
                    f"{row['fixed_cycles_norm']:.3f}",
                    f"{row['fixed_traffic_norm']:.3f}",
                    f"{row['fixed_energy_proxy']:.3f}",
                    f"{row['bounded_cycles_norm']:.3f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--microbench",
        type=Path,
        default=root / "output" / "onnxim_graphbit" / "microbench_s128" / "aggregate.json",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "onnxim_graphbit" / "summary")
    parser.add_argument("--weight-traffic-share", type=float, default=0.65)
    parser.add_argument("--activation-traffic-share", type=float, default=0.25)
    parser.add_argument("--output-traffic-share", type=float, default=0.10)
    parser.add_argument("--cache-read-compute-cost", type=float, default=0.001)
    parser.add_argument("--cache-read-traffic-cost", type=float, default=0.003)
    parser.add_argument("--residual-compute-cost", type=float, default=0.005)
    parser.add_argument("--residual-traffic-cost", type=float, default=0.005)
    parser.add_argument("--compute-energy-weight", type=float, default=0.55)
    parser.add_argument("--traffic-energy-weight", type=float, default=0.45)
    parser.add_argument("--bounded-save-p6", type=float, default=0.0)
    parser.add_argument("--bounded-save-p5", type=float, default=0.0)
    parser.add_argument("--bounded-save-p4", type=float, default=0.0)
    args = parser.parse_args()

    profiles = load_profiles(args.workload)
    aggregate = json.loads(args.microbench.read_text())
    rows = [summarize_profile(profile, aggregate, args) for profile in profiles]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.workload.stem}_hardware.tsv"
    compact_path = args.output_dir / f"{args.workload.stem}_compact.txt"
    write_tsv(rows, summary_path)
    write_compact(rows, compact_path)

    print(f"[GraphBitHardware] wrote {summary_path}")
    print(f"[GraphBitHardware] wrote {compact_path}")
    for row in rows:
        print(
            "[GraphBitHardware] "
            f"{row['profile']} | cycles={row['fixed_cycles_norm']:.3f} "
            f"traffic={row['fixed_traffic_norm']:.3f} "
            f"energy={row['fixed_energy_proxy']:.3f} "
            f"bounded_cycles={row['bounded_cycles_norm']:.3f}"
        )


if __name__ == "__main__":
    main()
