#!/usr/bin/env python3
"""Detailed frontend-path timing comparison against a HEAT-style bit-serial PE.

This simulator compares two different hardware paths under the same local GFM
workload trace:

1. GRACE TSER40 + fixed W4BFPA4 miss-node encoder.
   Compute cycles come from the existing BFP array traces.

2. HEAT-style topology-aware W8A10/W4A2 bit-serial frontend.
   Compute cycles are derived from HEAT Sec. 5.2.1: 32 PEs, each with a 32x32
   1-bit systolic array, and sequential W-bit x A-bit bitmap GEMMs.

The simulator also models streamed weight loading, activation loading,
intermediate output writes, final embedding writes/reads, and CAM query cycles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent
DEFAULT_REUSE_TSV = OFA_ROOT / "output" / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"
DEFAULT_HEAT_PROXY_TSV = OFA_ROOT / "output" / "heat_style_bitserial_quant" / "summary.tsv"

TASKS = {
    "CN": ("cora", 2708),
    "CL": ("cora", 2708),
    "PN": ("pubmed", 19717),
    "PL": ("pubmed", 19717),
    "AR": ("arxiv", 169343),
    "WK": ("wikics", 11701),
}

TRACE_DIRS = {
    "cora": OFA_ROOT / "output" / "e2e_time_breakdown_40reuse" / "array_cora_graphstress20",
    "pubmed": OFA_ROOT / "output" / "e2e_time_breakdown_40reuse" / "array_pubmed_graphstress20",
    "arxiv": OFA_ROOT / "output" / "e2e_time_breakdown_40reuse" / "array_arxiv_graphstress10",
    "wikics": OFA_ROOT / "output" / "e2e_time_breakdown_40reuse" / "array_wikics_graphstress20",
}

HEAT_OVERLAP = {"CN", "CL", "PN", "PL", "AR"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def gbps_to_cycles(num_bytes: float, bandwidth_gbs: float, clock_mhz: float) -> float:
    if bandwidth_gbs <= 0:
        return 0.0
    seconds = float(num_bytes) / (float(bandwidth_gbs) * 1.0e9)
    return seconds * float(clock_mhz) * 1.0e6


def ceil_div(a: float, b: float) -> int:
    if a <= 0:
        return 0
    return int(math.ceil(float(a) / float(b)))


def pct(x: float) -> str:
    if math.isnan(float(x)):
        return "-"
    return f"{100.0 * float(x):.2f}%"


def num(x: float, digits: int = 4) -> str:
    if math.isnan(float(x)):
        return "-"
    return f"{float(x):.{digits}f}"


def read_reuse(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_tsv(path):
        task = str(row.get("task", "")).strip()
        if not task:
            continue
        reuse = float(row.get("anchor_reuse", row.get("reuse", 40.0))) / 100.0
        drop = float(row.get("target_anchor_drop", row.get("drop", 0.0))) / 100.0
        out[task] = {"reuse": reuse, "drop": drop}
    return out


def read_heat_proxy(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for row in read_tsv(path):
        if row.get("low_tag") != "W4BFPA4_B256":
            continue
        if row.get("policy") != "HEATTopDegree10":
            continue
        task = str(row.get("task", "")).strip()
        if task:
            out[task] = float(row.get("drop", "nan"))
    return out


def load_modules(dataset: str) -> list[dict[str, Any]]:
    path = TRACE_DIRS[dataset] / "module_array_trace.tsv"
    rows = read_tsv(path)
    modules: list[dict[str, Any]] = []
    for row in rows:
        modules.append(
            {
                "module": row["module"],
                "kind": row["kind"],
                "calls": int(float(row["calls"])),
                "token_rows": float(row["token_rows"]),
                "in_features": int(float(row["in_features"])),
                "out_features": int(float(row["out_features"])),
                "full_bfpa4_cycles": float(row["full_bfpa4_cycles"]),
            }
        )
    return modules


def heat_systolic_cycles(
    *,
    rows: float,
    in_features: int,
    out_features: int,
    bitplanes: int,
    pes: int,
    sa_dim: int,
    utilization: float,
    su_cycles: int,
    mode: str,
) -> float:
    if rows <= 0:
        return 0.0
    cells = float(pes) * float(sa_dim) * float(sa_dim)
    if mode == "throughput":
        return (
            float(rows)
            * float(in_features)
            * float(out_features)
            * float(bitplanes)
            / (cells * float(utilization))
        )
    if mode != "systolic_tile":
        raise ValueError(f"unknown heat_compute_mode={mode}")
    tiles_m = ceil_div(rows, sa_dim)
    tiles_n = ceil_div(out_features, sa_dim)
    tiles_k = ceil_div(in_features, sa_dim)
    output_tile_groups = ceil_div(tiles_m * tiles_n, pes)

    # For each K tile and bit-plane pair, a 32x32 systolic array takes roughly
    # K_tile cycles plus fill/drain.  The final K tile may be short.
    full_k_tiles = int(in_features) // int(sa_dim)
    last_k = int(in_features) % int(sa_dim)
    k_cycle_sum = full_k_tiles * int(sa_dim)
    if last_k:
        k_cycle_sum += last_k
    fill = max(0, 2 * int(sa_dim) - 2) + int(su_cycles)
    per_bitplane = k_cycle_sum + tiles_k * fill
    return output_tile_groups * int(bitplanes) * per_bitplane / float(utilization)


def module_weight_bytes(elements: float, bits: float, calls: int) -> float:
    return float(elements) * float(bits) * float(calls) / 8.0


def module_activation_bytes(rows: float, features: int, bits: float) -> float:
    return float(rows) * float(features) * float(bits) / 8.0


def module_bfp_exponent_bytes(rows: float, features: int, block_size: int, exponent_bits: int) -> float:
    blocks = ceil_div(float(rows) * float(features), int(block_size))
    return float(blocks) * float(exponent_bits) / 8.0


def infer_node_batch(nodes: int, modules: list[dict[str, Any]]) -> float:
    calls = max(1, int(modules[0]["calls"]))
    return max(1.0, float(nodes) / float(calls))


def simulate_dataset_task(
    *,
    task: str,
    dataset: str,
    nodes: int,
    reuse: float,
    drop: float,
    heat_proxy_drop: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    modules = load_modules(dataset)
    node_batch = infer_node_batch(nodes, modules)
    miss = 1.0 - float(reuse)
    key = float(args.heat_alpha)
    low = 1.0 - key

    rows: list[dict[str, Any]] = []

    def blank(policy: str) -> dict[str, Any]:
        return {
            "task": task,
            "dataset": dataset,
            "policy": policy,
            "nodes": nodes,
            "reuse": 0.0,
            "drop": float("nan"),
            "heat_proxy_drop": float("nan"),
            "compute_cycles": 0.0,
            "weight_bytes": 0.0,
            "activation_bytes": 0.0,
            "bfp_exponent_bytes": 0.0,
            "intermediate_output_bytes": 0.0,
            "final_embedding_write_bytes": 0.0,
            "reuse_cache_read_bytes": 0.0,
            "cam_cycles": 0.0,
            "weight_load_cycles": 0.0,
            "activation_load_cycles": 0.0,
            "output_load_cycles": 0.0,
            "embedding_io_cycles": 0.0,
            "memory_cycles": 0.0,
            "total_no_overlap_cycles": 0.0,
            "total_overlap_cycles": 0.0,
            "speedup_no_overlap": float("nan"),
            "speedup_overlap": float("nan"),
        }

    baseline = blank("NoReuse+W4BFPA4")
    grace = blank("TSER40+W4BFPA4")
    heat = blank("HEAT-style W8A10/W4A2")
    grace["reuse"] = reuse
    grace["drop"] = drop
    heat["heat_proxy_drop"] = heat_proxy_drop

    for mod in modules:
        calls = int(mod["calls"])
        token_rows = float(mod["token_rows"])
        in_features = int(mod["in_features"])
        out_features = int(mod["out_features"])
        weight_elements = float(in_features) * float(out_features)

        miss_calls = ceil_div(float(nodes) * miss, node_batch)
        high_calls = ceil_div(float(nodes) * key, node_batch)
        low_calls = ceil_div(float(nodes) * low, node_batch)

        baseline["compute_cycles"] += float(mod["full_bfpa4_cycles"])
        grace["compute_cycles"] += float(mod["full_bfpa4_cycles"]) * miss

        high_rows = token_rows * key
        low_rows = token_rows * low
        heat["compute_cycles"] += heat_systolic_cycles(
            rows=high_rows,
            in_features=in_features,
            out_features=out_features,
            bitplanes=int(args.heat_hi_weight_bits) * int(args.heat_hi_activation_bits),
            pes=int(args.heat_pes),
            sa_dim=int(args.heat_sa_dim),
            utilization=float(args.heat_utilization),
            su_cycles=int(args.heat_su_cycles),
            mode=str(args.heat_compute_mode),
        )
        heat["compute_cycles"] += heat_systolic_cycles(
            rows=low_rows,
            in_features=in_features,
            out_features=out_features,
            bitplanes=int(args.heat_lo_weight_bits) * int(args.heat_lo_activation_bits),
            pes=int(args.heat_pes),
            sa_dim=int(args.heat_sa_dim),
            utilization=float(args.heat_utilization),
            su_cycles=int(args.heat_su_cycles),
            mode=str(args.heat_compute_mode),
        )

        baseline["weight_bytes"] += module_weight_bytes(weight_elements, args.local_weight_bits, calls)
        grace["weight_bytes"] += module_weight_bytes(weight_elements, args.local_weight_bits, miss_calls)
        if args.heat_weight_mode == "dual":
            heat["weight_bytes"] += module_weight_bytes(weight_elements, args.heat_hi_weight_bits, high_calls)
            heat["weight_bytes"] += module_weight_bytes(weight_elements, args.heat_lo_weight_bits, low_calls)
        elif args.heat_weight_mode == "shared_w8":
            heat["weight_bytes"] += module_weight_bytes(weight_elements, args.heat_hi_weight_bits, high_calls + low_calls)
        elif args.heat_weight_mode == "mixed_batch_dual":
            heat["weight_bytes"] += module_weight_bytes(
                weight_elements,
                args.heat_hi_weight_bits + args.heat_lo_weight_bits,
                calls,
            )
        else:
            raise ValueError(f"unknown heat_weight_mode={args.heat_weight_mode}")

        baseline["activation_bytes"] += module_activation_bytes(token_rows, in_features, args.local_activation_bits)
        baseline["bfp_exponent_bytes"] += module_bfp_exponent_bytes(
            token_rows,
            in_features,
            args.bfp_block_size,
            args.bfp_exponent_bits,
        )
        grace["activation_bytes"] += module_activation_bytes(token_rows * miss, in_features, args.local_activation_bits)
        grace["bfp_exponent_bytes"] += module_bfp_exponent_bytes(
            token_rows * miss,
            in_features,
            args.bfp_block_size,
            args.bfp_exponent_bits,
        )
        heat["activation_bytes"] += module_activation_bytes(high_rows, in_features, args.heat_hi_activation_bits)
        heat["activation_bytes"] += module_activation_bytes(low_rows, in_features, args.heat_lo_activation_bits)

        baseline["intermediate_output_bytes"] += module_activation_bytes(token_rows, out_features, args.internal_output_bits)
        grace["intermediate_output_bytes"] += module_activation_bytes(token_rows * miss, out_features, args.internal_output_bits)
        heat["intermediate_output_bytes"] += module_activation_bytes(token_rows, out_features, args.internal_output_bits)

    baseline["final_embedding_write_bytes"] = module_activation_bytes(nodes, args.embedding_dim, args.embedding_bits)
    grace["final_embedding_write_bytes"] = module_activation_bytes(float(nodes) * miss, args.embedding_dim, args.embedding_bits)
    grace["reuse_cache_read_bytes"] = module_activation_bytes(float(nodes) * reuse, args.embedding_dim, args.embedding_bits)
    heat["final_embedding_write_bytes"] = module_activation_bytes(nodes, args.embedding_dim, args.embedding_bits)

    grace["cam_cycles"] = float(nodes) * (
        float(args.cam_search_cycles) + float(args.cam_select_cycles) + miss * float(args.cam_miss_update_cycles)
    )

    for row in (baseline, grace, heat):
        row["activation_bytes_total"] = row["activation_bytes"] + row["bfp_exponent_bytes"]
        row["output_bytes_total"] = row["intermediate_output_bytes"]
        row["embedding_io_bytes"] = row["final_embedding_write_bytes"] + row["reuse_cache_read_bytes"]
        row["weight_load_cycles"] = gbps_to_cycles(row["weight_bytes"], args.weight_bw_gbs, args.clock_mhz)
        row["activation_load_cycles"] = gbps_to_cycles(
            row["activation_bytes_total"],
            args.activation_bw_gbs,
            args.clock_mhz,
        )
        row["output_load_cycles"] = gbps_to_cycles(row["output_bytes_total"], args.activation_bw_gbs, args.clock_mhz)
        row["embedding_io_cycles"] = gbps_to_cycles(row["embedding_io_bytes"], args.embedding_bw_gbs, args.clock_mhz)
        row["memory_cycles"] = (
            row["weight_load_cycles"]
            + row["activation_load_cycles"]
            + row["output_load_cycles"]
        )
        row["total_no_overlap_cycles"] = (
            row["compute_cycles"]
            + row["memory_cycles"]
            + row["embedding_io_cycles"]
            + row["cam_cycles"]
        )
        row["total_overlap_cycles"] = (
            max(row["compute_cycles"], row["memory_cycles"])
            + row["embedding_io_cycles"]
            + row["cam_cycles"]
        )

    for row in (grace, heat):
        row["compute_norm"] = row["compute_cycles"] / baseline["compute_cycles"]
        row["weight_norm"] = row["weight_load_cycles"] / max(1.0, baseline["weight_load_cycles"])
        row["activation_norm"] = row["activation_load_cycles"] / max(1.0, baseline["activation_load_cycles"])
        row["output_norm"] = row["output_load_cycles"] / max(1.0, baseline["output_load_cycles"])
        row["total_no_overlap_norm"] = row["total_no_overlap_cycles"] / baseline["total_no_overlap_cycles"]
        row["total_overlap_norm"] = row["total_overlap_cycles"] / baseline["total_overlap_cycles"]
        row["speedup_no_overlap"] = baseline["total_no_overlap_cycles"] / row["total_no_overlap_cycles"]
        row["speedup_overlap"] = baseline["total_overlap_cycles"] / row["total_overlap_cycles"]
    baseline["compute_norm"] = 1.0
    baseline["weight_norm"] = 1.0
    baseline["activation_norm"] = 1.0
    baseline["output_norm"] = 1.0
    baseline["total_no_overlap_norm"] = 1.0
    baseline["total_overlap_norm"] = 1.0
    baseline["speedup_no_overlap"] = 1.0
    baseline["speedup_overlap"] = 1.0
    return [baseline, heat, grace]


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope, tasks in (("AVG_HEAT5", HEAT_OVERLAP), ("AVG6", set(TASKS))):
        for policy in ("HEAT-style W8A10/W4A2", "TSER40+W4BFPA4"):
            vals = [row for row in rows if row["task"] in tasks and row["policy"] == policy]
            if not vals:
                continue
            item: dict[str, Any] = {"scope": scope, "policy": policy}
            for key in (
                "reuse",
                "drop",
                "heat_proxy_drop",
                "compute_norm",
                "weight_norm",
                "activation_norm",
                "output_norm",
                "total_no_overlap_norm",
                "total_overlap_norm",
                "speedup_no_overlap",
                "speedup_overlap",
            ):
                nums = [float(row[key]) for row in vals if not math.isnan(float(row[key]))]
                item[key] = sum(nums) / len(nums) if nums else float("nan")
            out.append(item)
    return out


def render_report(rows: list[dict[str, Any]], agg: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Detailed Frontend Path Timing vs HEAT-Style Bit-Serial",
        "",
        "## What Is Simulated",
        "",
        "- GRACE uses the existing local BFP array trace for the fixed `W4BFPA4` encoder path.",
        "- HEAT-style uses a separate bit-serial PE model: `32` PEs by default, each with a `32x32` 1-bit systolic array.",
        "- HEAT key vertices execute `W8A10`; non-key vertices execute `W4A2`.",
        "- Weight loading, activation loading, BFP exponent loading, intermediate output writes, final embedding IO, and CAM query cycles are all accounted for.",
        "",
        "## Configuration",
        "",
        f"- Clock: `{args.clock_mhz} MHz`.",
        f"- Weight bandwidth: `{args.weight_bw_gbs} GB/s`; activation/output bandwidth: `{args.activation_bw_gbs} GB/s`; embedding bandwidth: `{args.embedding_bw_gbs} GB/s`.",
        f"- HEAT weight-load mode: `{args.heat_weight_mode}`.",
        f"- HEAT compute mode: `{args.heat_compute_mode}`.",
        f"- HEAT PE: `{args.heat_pes}` PEs, `{args.heat_sa_dim}x{args.heat_sa_dim}` cells/PE, utilization `{args.heat_utilization}`.",
        f"- GRACE CAM query: search `{args.cam_search_cycles}` + select `{args.cam_select_cycles}` + miss update `{args.cam_miss_update_cycles}` cycles.",
        "",
        "## Aggregate Speedup",
        "",
        "| Scope | Policy | Reuse | Drop | HEAT Proxy Drop | Compute Norm | Weight Load Norm | Activation Load Norm | Output Norm | Total Norm, No Overlap | Speedup, No Overlap | Total Norm, Overlap | Speedup, Overlap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in agg:
        lines.append(
            f"| {row['scope']} | {row['policy']} | {pct(row['reuse'])} | "
            f"{pct(row['drop'])} | {pct(row['heat_proxy_drop'])} | "
            f"{num(row['compute_norm'])}x | {num(row['weight_norm'])}x | "
            f"{num(row['activation_norm'])}x | {num(row['output_norm'])}x | "
            f"{num(row['total_no_overlap_norm'])}x | {num(row['speedup_no_overlap'], 2)}x | "
            f"{num(row['total_overlap_norm'])}x | {num(row['speedup_overlap'], 2)}x |"
        )

    lines.extend(
        [
            "",
            "## Per-Task Frontend Path",
            "",
            "| Task | Policy | Reuse | Compute Norm | Weight Norm | Act Norm | Output Norm | Total Norm, Overlap | Speedup, Overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["policy"] == "NoReuse+W4BFPA4":
            continue
        lines.append(
            f"| {row['task']} | {row['policy']} | {pct(row['reuse'])} | "
            f"{num(row['compute_norm'])}x | {num(row['weight_norm'])}x | "
            f"{num(row['activation_norm'])}x | {num(row['output_norm'])}x | "
            f"{num(row['total_overlap_norm'])}x | {num(row['speedup_overlap'], 2)}x |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the correct speedup comparison for the local fixed-W4BFPA4 path: GRACE and HEAT-style are simulated through different hardware datapaths.",
            "- HEAT-style is not represented by a single average bit-plane ratio. Its W8A10 key path serializes both weight and activation bits, while GRACE's W4BFPA4 array uses the measured local BFP trace.",
            "- TSER40 reduces the number of encoder invocations. Therefore compute, weight streaming, activation loading, and intermediate output writes all shrink with the miss-node stream after compaction.",
            "- The overlap model assumes compute can overlap with module weight/activation/output movement; the no-overlap model is a conservative upper bound.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse_tsv", default=str(DEFAULT_REUSE_TSV))
    parser.add_argument("--heat_proxy_tsv", default=str(DEFAULT_HEAT_PROXY_TSV))
    parser.add_argument("--clock_mhz", type=float, default=500.0)
    parser.add_argument("--weight_bw_gbs", type=float, default=25.6)
    parser.add_argument("--activation_bw_gbs", type=float, default=1024.0)
    parser.add_argument("--embedding_bw_gbs", type=float, default=25.6)
    parser.add_argument("--local_weight_bits", type=int, default=4)
    parser.add_argument("--local_activation_bits", type=int, default=4)
    parser.add_argument("--bfp_block_size", type=int, default=256)
    parser.add_argument("--bfp_exponent_bits", type=int, default=8)
    parser.add_argument("--internal_output_bits", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=4096)
    parser.add_argument("--embedding_bits", type=int, default=16)
    parser.add_argument("--heat_alpha", type=float, default=0.1)
    parser.add_argument("--heat_hi_weight_bits", type=int, default=8)
    parser.add_argument("--heat_hi_activation_bits", type=int, default=10)
    parser.add_argument("--heat_lo_weight_bits", type=int, default=4)
    parser.add_argument("--heat_lo_activation_bits", type=int, default=2)
    parser.add_argument("--heat_pes", type=int, default=32)
    parser.add_argument("--heat_sa_dim", type=int, default=32)
    parser.add_argument("--heat_utilization", type=float, default=0.85)
    parser.add_argument("--heat_su_cycles", type=int, default=0)
    parser.add_argument(
        "--heat_compute_mode",
        choices=["throughput", "systolic_tile"],
        default="throughput",
    )
    parser.add_argument(
        "--heat_weight_mode",
        choices=["dual", "shared_w8", "mixed_batch_dual"],
        default="dual",
    )
    parser.add_argument("--cam_search_cycles", type=float, default=1.0)
    parser.add_argument("--cam_select_cycles", type=float, default=1.0)
    parser.add_argument("--cam_miss_update_cycles", type=float, default=1.0)
    parser.add_argument(
        "--output_dir",
        default=str(OFA_ROOT / "output" / "heat_frontend_path_timing"),
    )
    parser.add_argument(
        "--repo_report",
        default=str(REPO_ROOT / "HEAT" / "results" / "FRONTEND_PATH_TIMING_DETAILED.md"),
    )
    args = parser.parse_args()

    reuse = read_reuse(Path(args.reuse_tsv))
    heat_proxy = read_heat_proxy(Path(args.heat_proxy_tsv))
    rows: list[dict[str, Any]] = []
    for task, (dataset, nodes) in TASKS.items():
        info = reuse.get(task, {"reuse": 0.40, "drop": float("nan")})
        rows.extend(
            simulate_dataset_task(
                task=task,
                dataset=dataset,
                nodes=nodes,
                reuse=float(info["reuse"]),
                drop=float(info["drop"]),
                heat_proxy_drop=float(heat_proxy.get(task, float("nan"))),
                args=args,
            )
        )

    agg = aggregate(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "frontend_path_rows.tsv", rows)
    write_tsv(output_dir / "frontend_path_aggregate.tsv", agg)
    (output_dir / "frontend_path_timing.json").write_text(
        json.dumps({"config": vars(args), "rows": rows, "aggregate": agg}, indent=2),
        encoding="utf-8",
    )
    report = render_report(rows, agg, args)
    (output_dir / "FRONTEND_PATH_TIMING_DETAILED.md").write_text(report, encoding="utf-8")
    repo_report = Path(args.repo_report)
    repo_report.parent.mkdir(parents=True, exist_ok=True)
    repo_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
