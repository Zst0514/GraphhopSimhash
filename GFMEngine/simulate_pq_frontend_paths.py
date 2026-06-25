#!/usr/bin/env python3
"""Path-level frontend timing model for a GFMEngine-style PQ baseline.

This script compares three online frontend paths on the local GraphHopSimhash
workload traces:

1. NoReuse + fixed W4BFPA4 array.
2. TSER40 + fixed W4BFPA4 miss-node array.
3. GFMEngine-style PQ MatMul.

GFMEngine does not use the HEAT W8A10/W4A2 bit-serial split.  Its ASPDAC'25
paper replaces weight-involving Transformer GEMMs with online centroid search
and activation-book lookup.  The offline codebook/activation-book construction
is not charged, following the paper's theoretical analysis.
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

GFMENGINE_LLAMA2_7B_ACCURACY_LOSS = {
    # ASPDAC'25 GFMEngine Table 3, original minus PQ-based MatMul accuracy.
    "CN": -0.0086,  # CR-Node
    "CL": 0.0051,  # CR-Link
    "PN": 0.0137,  # PM-Node
    "PL": 0.0016,  # PM-Link
    "AR": -0.0009,  # AX
    "WK": 0.0031,  # WK
}


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


def seconds_from_cycles(cycles: float, clock_mhz: float) -> float:
    return float(cycles) / (float(clock_mhz) * 1.0e6)


def seconds_from_bytes(num_bytes: float, bandwidth_gbs: float) -> float:
    if bandwidth_gbs <= 0:
        return 0.0
    return float(num_bytes) / (float(bandwidth_gbs) * 1.0e9)


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


def load_modules(dataset: str) -> list[dict[str, Any]]:
    path = TRACE_DIRS[dataset] / "module_array_trace.tsv"
    modules: list[dict[str, Any]] = []
    for row in read_tsv(path):
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


def activation_bytes(rows: float, features: int, bits: float) -> float:
    return float(rows) * float(features) * float(bits) / 8.0


def weight_bytes(elements: float, bits: float, calls: int) -> float:
    return float(elements) * float(bits) * float(calls) / 8.0


def bfp_exponent_bytes(rows: float, features: int, block_size: int, exponent_bits: int) -> float:
    blocks = ceil_div(float(rows) * float(features), int(block_size))
    return float(blocks) * float(exponent_bits) / 8.0


def gfm_codebook_bytes(*, calls: int, in_features: int, args: argparse.Namespace) -> float:
    bytes_per_codebook = float(args.pq_centroids) * float(in_features) * float(args.pq_centroid_bits) / 8.0
    if args.codebook_load_mode == "resident":
        return 0.0
    if args.codebook_load_mode == "per_module":
        return bytes_per_codebook
    if args.codebook_load_mode == "per_call":
        return bytes_per_codebook * float(calls)
    if args.codebook_load_mode == "per_row":
        # Upper bound when centroid vectors cannot be reused across token rows.
        return bytes_per_codebook * float(calls) * float(args.rows_per_call_for_per_row_codebook)
    raise ValueError(f"unknown codebook_load_mode={args.codebook_load_mode}")


def blank_row(task: str, dataset: str, nodes: int, policy: str) -> dict[str, Any]:
    return {
        "task": task,
        "dataset": dataset,
        "policy": policy,
        "nodes": nodes,
        "reuse": 0.0,
        "drop": float("nan"),
        "gfmengine_accuracy_loss": float("nan"),
        "compute_seconds": 0.0,
        "search_compute_seconds": 0.0,
        "rowmax_seconds": 0.0,
        "query_add_seconds": 0.0,
        "index_unit_seconds": 0.0,
        "attention_seconds": 0.0,
        "weight_bytes": 0.0,
        "activation_bytes": 0.0,
        "bfp_exponent_bytes": 0.0,
        "codebook_bytes": 0.0,
        "activation_book_bytes_raw": 0.0,
        "activation_book_bytes_after_iu": 0.0,
        "index_bytes_raw": 0.0,
        "index_bytes_after_iu": 0.0,
        "intermediate_output_bytes": 0.0,
        "final_embedding_write_bytes": 0.0,
        "reuse_cache_read_bytes": 0.0,
        "search_memory_seconds": 0.0,
        "query_memory_seconds": 0.0,
        "weight_memory_seconds": 0.0,
        "activation_memory_seconds": 0.0,
        "output_memory_seconds": 0.0,
        "embedding_io_seconds": 0.0,
        "cam_seconds": 0.0,
        "memory_seconds": 0.0,
        "total_no_overlap_seconds": 0.0,
        "total_overlap_seconds": 0.0,
        "compute_norm": float("nan"),
        "memory_norm": float("nan"),
        "output_norm": float("nan"),
        "embedding_norm": float("nan"),
        "total_no_overlap_norm": float("nan"),
        "total_overlap_norm": float("nan"),
        "speedup_no_overlap": float("nan"),
        "speedup_overlap": float("nan"),
    }


def infer_transformer_shape(modules: list[dict[str, Any]], nodes: int) -> tuple[int, int, float]:
    q_modules = [mod for mod in modules if mod["kind"] == "q_proj"]
    layers = len(q_modules) if q_modules else max(1, len(modules) // 7)
    hidden = int(q_modules[0]["in_features"]) if q_modules else int(modules[0]["in_features"])
    token_rows = float(q_modules[0]["token_rows"]) if q_modules else float(modules[0]["token_rows"])
    avg_seq_len = token_rows / max(1.0, float(nodes))
    return layers, hidden, avg_seq_len


def simulate_task(
    *,
    task: str,
    dataset: str,
    nodes: int,
    reuse: float,
    drop: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    modules = load_modules(dataset)
    miss = 1.0 - float(reuse)
    gfm_macs_per_cycle = float(args.gfm_pes) * float(args.gfm_sa_rows) * float(args.gfm_sa_cols)
    gfm_at_lanes = float(args.gfm_pes) * float(args.gfm_adder_trees_per_pe) * float(args.gfm_adder_tree_width)
    layers, hidden, avg_seq_len = infer_transformer_shape(modules, nodes)

    baseline = blank_row(task, dataset, nodes, "NoReuse+W4BFPA4")
    tser = blank_row(task, dataset, nodes, "TSER40+W4BFPA4")
    gfm = blank_row(task, dataset, nodes, "GFMEngine-PQ")
    tser["reuse"] = reuse
    tser["drop"] = drop
    gfm["gfmengine_accuracy_loss"] = GFMENGINE_LLAMA2_7B_ACCURACY_LOSS.get(task, float("nan"))

    for mod in modules:
        calls = int(mod["calls"])
        token_rows = float(mod["token_rows"])
        in_features = int(mod["in_features"])
        out_features = int(mod["out_features"])
        weight_elements = float(in_features) * float(out_features)
        miss_calls = ceil_div(float(calls) * miss, 1.0)

        baseline["compute_seconds"] += seconds_from_cycles(mod["full_bfpa4_cycles"], args.local_clock_mhz)
        tser["compute_seconds"] += seconds_from_cycles(
            float(mod["full_bfpa4_cycles"]) * miss,
            args.local_clock_mhz,
        )

        baseline["weight_bytes"] += weight_bytes(weight_elements, args.local_weight_bits, calls)
        tser["weight_bytes"] += weight_bytes(weight_elements, args.local_weight_bits, miss_calls)
        baseline["activation_bytes"] += activation_bytes(token_rows, in_features, args.local_activation_bits)
        baseline["bfp_exponent_bytes"] += bfp_exponent_bytes(
            token_rows,
            in_features,
            args.bfp_block_size,
            args.bfp_exponent_bits,
        )
        tser["activation_bytes"] += activation_bytes(token_rows * miss, in_features, args.local_activation_bits)
        tser["bfp_exponent_bytes"] += bfp_exponent_bytes(
            token_rows * miss,
            in_features,
            args.bfp_block_size,
            args.bfp_exponent_bits,
        )
        baseline["intermediate_output_bytes"] += activation_bytes(token_rows, out_features, args.internal_output_bits)
        tser["intermediate_output_bytes"] += activation_bytes(token_rows * miss, out_features, args.internal_output_bits)

        search_ops = token_rows * float(args.pq_centroids) * float(in_features)
        rowmax_ops = token_rows * float(args.pq_centroids)
        query_add_ops = token_rows * float(args.pq_subvectors) * float(out_features)
        index_ops = token_rows * float(args.pq_subvectors)

        gfm_search_cycles = search_ops / (gfm_macs_per_cycle * float(args.gfm_sa_utilization))
        gfm_rowmax_cycles = rowmax_ops / (gfm_at_lanes * float(args.gfm_at_utilization))
        gfm_query_add_cycles = query_add_ops / (gfm_at_lanes * float(args.gfm_at_utilization))
        gfm_index_cycles = index_ops / float(args.gfm_index_unit_indices_per_cycle)

        gfm["search_compute_seconds"] += seconds_from_cycles(gfm_search_cycles, args.gfm_clock_mhz)
        gfm["rowmax_seconds"] += seconds_from_cycles(gfm_rowmax_cycles, args.gfm_clock_mhz)
        gfm["query_add_seconds"] += seconds_from_cycles(gfm_query_add_cycles, args.gfm_clock_mhz)
        gfm["index_unit_seconds"] += seconds_from_cycles(gfm_index_cycles, args.gfm_clock_mhz)

        gfm["activation_bytes"] += activation_bytes(token_rows, in_features, args.pq_input_bits)
        gfm["codebook_bytes"] += gfm_codebook_bytes(calls=calls, in_features=in_features, args=args)
        raw_book = activation_bytes(token_rows * float(args.pq_subvectors), out_features, args.pq_activation_book_bits)
        raw_index = token_rows * float(args.pq_subvectors) * float(args.pq_index_bits) / 8.0
        gfm["activation_book_bytes_raw"] += raw_book
        gfm["index_bytes_raw"] += raw_index
        gfm["activation_book_bytes_after_iu"] += raw_book * (1.0 - float(args.iu_memory_reduction))
        gfm["index_bytes_after_iu"] += raw_index * (1.0 - float(args.iu_memory_reduction))
        gfm["intermediate_output_bytes"] += activation_bytes(token_rows, out_features, args.internal_output_bits)

    if args.include_attention_residual:
        attention_macs = (
            float(args.attention_ops_multiplier)
            * float(layers)
            * float(nodes)
            * float(avg_seq_len)
            * float(avg_seq_len)
            * float(hidden)
        )
        local_attention_cycles = (
            attention_macs
            * float(args.local_attention_bits)
            * float(args.local_attention_bits)
            / (float(args.local_bit_macs_per_cycle) * float(args.local_attention_utilization))
        )
        gfm_attention_cycles = attention_macs / (gfm_macs_per_cycle * float(args.gfm_attention_utilization))
        baseline["attention_seconds"] = seconds_from_cycles(local_attention_cycles, args.local_clock_mhz)
        tser["attention_seconds"] = seconds_from_cycles(local_attention_cycles * miss, args.local_clock_mhz)
        gfm["attention_seconds"] = seconds_from_cycles(gfm_attention_cycles, args.gfm_clock_mhz)
        baseline["compute_seconds"] += baseline["attention_seconds"]
        tser["compute_seconds"] += tser["attention_seconds"]
        gfm["compute_seconds"] += gfm["attention_seconds"]
        for row in (baseline, tser, gfm):
            row["avg_seq_len"] = avg_seq_len
            row["attention_macs"] = attention_macs
            row["attention_layers"] = layers
            row["attention_hidden"] = hidden

    for row in (baseline, tser, gfm):
        row["activation_bytes_total"] = row["activation_bytes"] + row["bfp_exponent_bytes"]
        row["output_bytes_total"] = row["intermediate_output_bytes"]

    baseline["final_embedding_write_bytes"] = activation_bytes(nodes, args.embedding_dim, args.embedding_bits)
    tser["final_embedding_write_bytes"] = activation_bytes(float(nodes) * miss, args.embedding_dim, args.embedding_bits)
    tser["reuse_cache_read_bytes"] = activation_bytes(float(nodes) * reuse, args.embedding_dim, args.embedding_bits)
    gfm["final_embedding_write_bytes"] = activation_bytes(nodes, args.embedding_dim, args.embedding_bits)

    tser["cam_seconds"] = seconds_from_cycles(
        float(nodes)
        * (float(args.cam_search_cycles) + float(args.cam_select_cycles) + miss * float(args.cam_miss_update_cycles)),
        args.local_clock_mhz,
    )

    for row in (baseline, tser):
        row["weight_memory_seconds"] = seconds_from_bytes(row["weight_bytes"], args.local_weight_bw_gbs)
        row["activation_memory_seconds"] = seconds_from_bytes(
            row["activation_bytes_total"],
            args.local_activation_bw_gbs,
        )
        row["output_memory_seconds"] = seconds_from_bytes(row["output_bytes_total"], args.local_activation_bw_gbs)
        row["embedding_io_seconds"] = seconds_from_bytes(
            row["final_embedding_write_bytes"] + row["reuse_cache_read_bytes"],
            args.local_embedding_bw_gbs,
        )
        row["memory_seconds"] = (
            row["weight_memory_seconds"] + row["activation_memory_seconds"] + row["output_memory_seconds"]
        )
        row["total_no_overlap_seconds"] = (
            row["compute_seconds"] + row["memory_seconds"] + row["embedding_io_seconds"] + row["cam_seconds"]
        )
        row["total_overlap_seconds"] = (
            max(row["compute_seconds"], row["memory_seconds"]) + row["embedding_io_seconds"] + row["cam_seconds"]
        )

    gfm["compute_seconds"] = (
        gfm["search_compute_seconds"]
        + gfm["rowmax_seconds"]
        + gfm["query_add_seconds"]
        + gfm["index_unit_seconds"]
        + gfm["attention_seconds"]
    )
    gfm["search_memory_seconds"] = seconds_from_bytes(
        gfm["activation_bytes"] + gfm["codebook_bytes"],
        args.gfm_hbm_bw_gbs,
    )
    gfm["query_memory_seconds"] = seconds_from_bytes(
        gfm["activation_book_bytes_after_iu"] + gfm["index_bytes_after_iu"],
        args.gfm_hbm_bw_gbs,
    )
    gfm["output_memory_seconds"] = seconds_from_bytes(gfm["output_bytes_total"], args.gfm_global_buffer_bw_gbs)
    gfm["embedding_io_seconds"] = seconds_from_bytes(gfm["final_embedding_write_bytes"], args.gfm_hbm_bw_gbs)
    gfm["memory_seconds"] = gfm["search_memory_seconds"] + gfm["query_memory_seconds"] + gfm["output_memory_seconds"]
    search_compute = gfm["search_compute_seconds"] + gfm["rowmax_seconds"]
    query_compute = gfm["query_add_seconds"] + gfm["index_unit_seconds"]
    gfm["total_no_overlap_seconds"] = (
        gfm["compute_seconds"] + gfm["memory_seconds"] + gfm["embedding_io_seconds"]
    )
    gfm["total_overlap_seconds"] = (
        max(search_compute, gfm["search_memory_seconds"])
        + max(query_compute, gfm["query_memory_seconds"])
        + gfm["output_memory_seconds"]
        + gfm["embedding_io_seconds"]
    )

    for row in (baseline, tser, gfm):
        row["compute_norm"] = row["compute_seconds"] / baseline["compute_seconds"]
        row["memory_norm"] = row["memory_seconds"] / max(1.0e-30, baseline["memory_seconds"])
        row["output_norm"] = row["output_memory_seconds"] / max(1.0e-30, baseline["output_memory_seconds"])
        row["embedding_norm"] = row["embedding_io_seconds"] / max(1.0e-30, baseline["embedding_io_seconds"])
        row["total_no_overlap_norm"] = row["total_no_overlap_seconds"] / baseline["total_no_overlap_seconds"]
        row["total_overlap_norm"] = row["total_overlap_seconds"] / baseline["total_overlap_seconds"]
        row["speedup_no_overlap"] = baseline["total_no_overlap_seconds"] / row["total_no_overlap_seconds"]
        row["speedup_overlap"] = baseline["total_overlap_seconds"] / row["total_overlap_seconds"]

    return [baseline, gfm, tser]


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tasks = set(TASKS)
    for policy in ("GFMEngine-PQ", "TSER40+W4BFPA4"):
        vals = [row for row in rows if row["task"] in tasks and row["policy"] == policy]
        bases = [row for row in rows if row["task"] in tasks and row["policy"] == "NoReuse+W4BFPA4"]
        if not vals or not bases:
            continue
        item: dict[str, Any] = {"scope": "AVG6", "policy": policy}
        for key in ("reuse", "drop", "gfmengine_accuracy_loss"):
            nums = [float(row[key]) for row in vals if not math.isnan(float(row[key]))]
            item[key] = sum(nums) / len(nums) if nums else float("nan")
        for key in (
            "compute_seconds",
            "attention_seconds",
            "memory_seconds",
            "output_memory_seconds",
            "embedding_io_seconds",
            "total_no_overlap_seconds",
            "total_overlap_seconds",
        ):
            val_sum = sum(float(row[key]) for row in vals)
            base_sum = sum(float(row[key]) for row in bases)
            item[f"{key}_norm"] = val_sum / base_sum if base_sum > 0 else float("nan")
        item["speedup_no_overlap"] = 1.0 / item["total_no_overlap_seconds_norm"]
        item["speedup_overlap"] = 1.0 / item["total_overlap_seconds_norm"]
        out.append(item)
    return out


def render_report(rows: list[dict[str, Any]], agg: list[dict[str, Any]], args: argparse.Namespace) -> str:
    gfm_macs_per_cycle = (
        int(args.gfm_pes)
        * int(args.gfm_sa_rows)
        * int(args.gfm_sa_cols)
    )
    gfm_at_lanes = int(args.gfm_pes) * int(args.gfm_adder_trees_per_pe) * int(args.gfm_adder_tree_width)
    lines = [
        "# GFMEngine-Style PQ Frontend Path Timing",
        "",
        "## Important Correction",
        "",
        "- `N * [0.1 * T_bitserial(W8A10) + 0.9 * T_bitserial(W4A2)]` is a HEAT-style topology-aware bit-serial model, not GFMEngine.",
        "- GFMEngine's ASPDAC'25 path is PQ-based MatMul: online centroid search plus activation-book lookup. The paper ignores offline codebook/book construction, and this simulator follows that convention.",
        "- The baseline therefore charges every node/token for centroid search, activation-book/index traffic, IU cycles, adder-tree accumulation, intermediate output traffic, and final embedding writes. There is no TSER-style node skip in GFMEngine-PQ.",
        "",
        "## Configuration",
        "",
        f"- Local array: `W{args.local_weight_bits}BFPA{args.local_activation_bits}`, `{args.local_clock_mhz} MHz`.",
        f"- Local bandwidths: weight `{args.local_weight_bw_gbs} GB/s`, activation/output `{args.local_activation_bw_gbs} GB/s`, embedding `{args.local_embedding_bw_gbs} GB/s`.",
        f"- GFMEngine: `{args.gfm_clock_mhz} MHz`, `{args.gfm_pes}` PEs, each with one `{args.gfm_sa_rows}x{args.gfm_sa_cols}` SA and `{args.gfm_adder_trees_per_pe}` `{args.gfm_adder_tree_width}`-lane ATs.",
        f"- GFMEngine peak model: `{gfm_macs_per_cycle}` centroid-search MAC-equivalent ops/cycle and `{gfm_at_lanes}` AT lanes/cycle before utilization.",
        f"- PQ: `{args.pq_centroids}` centroids, `{args.pq_subvectors}` subvectors, input `{args.pq_input_bits}`b, centroid `{args.pq_centroid_bits}`b, activation-book `{args.pq_activation_book_bits}`b, index `{args.pq_index_bits}`b.",
        f"- GFMEngine HBM bandwidth `{args.gfm_hbm_bw_gbs} GB/s`; GB bandwidth `{args.gfm_global_buffer_bw_gbs} GB/s`; IU memory reduction `{pct(args.iu_memory_reduction)}`.",
        f"- Codebook load mode: `{args.codebook_load_mode}`.",
        f"- Attention residual: `{bool(args.include_attention_residual)}`; multiplier `{args.attention_ops_multiplier}` for `QK^T` and `AV`.",
        "",
        "## Aggregate Result",
        "",
        "| Scope | Policy | Reuse | Drop / PQ Loss | Compute Norm | Attention Norm | Memory Norm | Output Norm | Total Norm, No Overlap | Speedup, No Overlap | Total Norm, Pipelined | Speedup, Pipelined |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in agg:
        loss = row["drop"] if row["policy"].startswith("TSER") else row["gfmengine_accuracy_loss"]
        lines.append(
            f"| {row['scope']} | {row['policy']} | {pct(row['reuse'])} | {pct(loss)} | "
            f"{num(row['compute_seconds_norm'])}x | {num(row['attention_seconds_norm'])}x | "
            f"{num(row['memory_seconds_norm'])}x | "
            f"{num(row['output_memory_seconds_norm'])}x | "
            f"{num(row['total_no_overlap_seconds_norm'])}x | {num(row['speedup_no_overlap'], 2)}x | "
            f"{num(row['total_overlap_seconds_norm'])}x | {num(row['speedup_overlap'], 2)}x |"
        )

    lines.extend(
        [
            "",
            "## Per-Task Timing",
            "",
            "| Task | Policy | Reuse | Drop / PQ Loss | Compute Norm | Attention s | Memory Norm | Total Norm, Pipelined | Speedup, Pipelined |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["policy"] == "NoReuse+W4BFPA4":
            continue
        loss = row["drop"] if row["policy"].startswith("TSER") else row["gfmengine_accuracy_loss"]
        lines.append(
            f"| {row['task']} | {row['policy']} | {pct(row['reuse'])} | {pct(loss)} | "
            f"{num(row['compute_norm'])}x | {num(row['attention_seconds'], 3)} | "
            f"{num(row['memory_norm'])}x | "
            f"{num(row['total_overlap_norm'])}x | {num(row['speedup_overlap'], 2)}x |"
        )

    lines.extend(
        [
            "",
            "## What Dominates GFMEngine-PQ Here",
            "",
            "| Task | Search Compute (s) | Query/Add Compute (s) | Search Mem (s) | Query Mem After IU (s) | Raw Activation Book GB | IU-Reduced Activation Book GB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["policy"] != "GFMEngine-PQ":
            continue
        lines.append(
            f"| {row['task']} | {num(row['search_compute_seconds'], 3)} | "
            f"{num(row['query_add_seconds'] + row['index_unit_seconds'], 3)} | "
            f"{num(row['search_memory_seconds'], 3)} | {num(row['query_memory_seconds'], 3)} | "
            f"{num(row['activation_book_bytes_raw'] / 1.0e9, 2)} | "
            f"{num(row['activation_book_bytes_after_iu'] / 1.0e9, 2)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- GFMEngine-PQ removes full weight-GEMM MACs, but it still runs every token row. Its online cost is dominated by activation-book lookup and M-way accumulation when the activation book is modeled explicitly.",
            "- TSER40 gets speedup from reducing the miss stream: compute, weight loading, activation loading, intermediate outputs, and final writes all shrink by roughly the miss rate.",
            "- The GFMEngine paper reports cycle-level results from its own simulator, but it does not publish enough per-layer trace detail to reproduce exact cycles. This script is therefore a transparent path-level reconstruction using the paper's public parameters.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse_tsv", default=str(DEFAULT_REUSE_TSV))
    parser.add_argument("--local_clock_mhz", type=float, default=500.0)
    parser.add_argument("--local_weight_bw_gbs", type=float, default=25.6)
    parser.add_argument("--local_activation_bw_gbs", type=float, default=1024.0)
    parser.add_argument("--local_embedding_bw_gbs", type=float, default=25.6)
    parser.add_argument("--local_weight_bits", type=int, default=4)
    parser.add_argument("--local_activation_bits", type=int, default=4)
    parser.add_argument("--local_attention_bits", type=int, default=4)
    parser.add_argument("--local_attention_utilization", type=float, default=0.9)
    parser.add_argument("--local_bit_macs_per_cycle", type=float, default=16384.0)
    parser.add_argument("--bfp_block_size", type=int, default=256)
    parser.add_argument("--bfp_exponent_bits", type=int, default=8)
    parser.add_argument("--internal_output_bits", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=4096)
    parser.add_argument("--embedding_bits", type=int, default=16)
    parser.add_argument("--cam_search_cycles", type=float, default=1.0)
    parser.add_argument("--cam_select_cycles", type=float, default=1.0)
    parser.add_argument("--cam_miss_update_cycles", type=float, default=1.0)

    parser.add_argument("--gfm_clock_mhz", type=float, default=1000.0)
    parser.add_argument("--gfm_hbm_bw_gbs", type=float, default=256.0)
    parser.add_argument("--gfm_global_buffer_bw_gbs", type=float, default=1024.0)
    parser.add_argument("--gfm_pes", type=int, default=16)
    parser.add_argument("--gfm_sa_rows", type=int, default=4)
    parser.add_argument("--gfm_sa_cols", type=int, default=16)
    parser.add_argument("--gfm_adder_trees_per_pe", type=int, default=2)
    parser.add_argument("--gfm_adder_tree_width", type=int, default=8)
    parser.add_argument("--gfm_sa_utilization", type=float, default=0.85)
    parser.add_argument("--gfm_attention_utilization", type=float, default=0.85)
    parser.add_argument("--gfm_at_utilization", type=float, default=0.85)
    parser.add_argument("--gfm_index_unit_indices_per_cycle", type=float, default=1024.0)

    parser.add_argument("--pq_centroids", type=int, default=256)
    parser.add_argument("--pq_subvectors", type=int, default=16)
    parser.add_argument("--pq_input_bits", type=int, default=8)
    parser.add_argument("--pq_centroid_bits", type=int, default=8)
    parser.add_argument("--pq_activation_book_bits", type=int, default=8)
    parser.add_argument("--pq_index_bits", type=int, default=8)
    parser.add_argument("--iu_memory_reduction", type=float, default=0.30)
    parser.add_argument(
        "--include_attention_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add QK^T/AV attention GEMMs that PQ-based MatMul does not remove.",
    )
    parser.add_argument("--attention_ops_multiplier", type=float, default=2.0)
    parser.add_argument(
        "--codebook_load_mode",
        choices=["resident", "per_module", "per_call", "per_row"],
        default="per_call",
    )
    parser.add_argument(
        "--rows_per_call_for_per_row_codebook",
        type=float,
        default=1.0,
        help="Only used for the pessimistic per_row codebook mode.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(OFA_ROOT / "output" / "gfmengine_pq_frontend_path_timing"),
    )
    parser.add_argument(
        "--repo_report",
        default=str(REPO_ROOT / "GFMEngine" / "results" / "GFMENGINE_PQ_PATH_TIMING.md"),
    )
    args = parser.parse_args()

    reuse = read_reuse(Path(args.reuse_tsv))
    rows: list[dict[str, Any]] = []
    for task, (dataset, nodes) in TASKS.items():
        info = reuse.get(task, {"reuse": 0.40, "drop": float("nan")})
        rows.extend(
            simulate_task(
                task=task,
                dataset=dataset,
                nodes=nodes,
                reuse=float(info["reuse"]),
                drop=float(info["drop"]),
                args=args,
            )
        )

    agg = aggregate(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "pq_frontend_rows.tsv", rows)
    write_tsv(output_dir / "pq_frontend_aggregate.tsv", agg)
    (output_dir / "pq_frontend_timing.json").write_text(
        json.dumps({"config": vars(args), "rows": rows, "aggregate": agg}, indent=2),
        encoding="utf-8",
    )
    report = render_report(rows, agg, args)
    (output_dir / "GFMENGINE_PQ_PATH_TIMING.md").write_text(report, encoding="utf-8")
    repo_report = Path(args.repo_report)
    repo_report.parent.mkdir(parents=True, exist_ok=True)
    repo_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
