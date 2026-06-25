#!/usr/bin/env python3
"""Build a unified GRACE performance/activity trace.

The output is intended to be the bridge between cycle-level performance traces
and an activity-based analytical energy model.  It keeps measured ONNXim/CAM
fields separate from modeled counters by using the source_kind and notes
columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

TASKS = ("CN", "CL", "PN", "PL", "AR", "WK")
TASK_TO_DATASET = {
    "CN": "cora",
    "CL": "cora",
    "PN": "pubmed",
    "PL": "pubmed",
    "AR": "arxiv",
    "WK": "wikics",
}
TASK_NODES = {
    "CN": 2708,
    "CL": 2708,
    "PN": 19717,
    "PL": 19717,
    "AR": 169343,
    "WK": 11701,
}
TASK_EDGES = {
    "CN": 10556,
    "CL": 10556,
    "PN": 88648,
    "PL": 88648,
    "AR": 1166243,
    "WK": 216123,
}
ARRAY_TRACE_DIRS = {
    "cora": "e2e_time_breakdown_40reuse/array_cora_graphstress20",
    "pubmed": "e2e_time_breakdown_40reuse/array_pubmed_graphstress20",
    "arxiv": "e2e_time_breakdown_40reuse/array_arxiv_graphstress10",
    "wikics": "e2e_time_breakdown_40reuse/array_wikics_graphstress20",
}

TRACE_COLUMNS = [
    "task",
    "dataset",
    "policy",
    "component",
    "subcomponent",
    "module",
    "source_kind",
    "cycles",
    "time_s",
    "time_ns",
    "frequency_mhz",
    "nodes",
    "edges",
    "reuse_pct",
    "miss_pct",
    "refine_ratio",
    "effective_bits",
    "block_size",
    "token_rows",
    "calls",
    "in_features",
    "out_features",
    "total_blocks",
    "refined_blocks",
    "bit_mac_ops",
    "pe_active_cycles",
    "a_rf_reads",
    "w_rf_reads",
    "psum_reads",
    "psum_writes",
    "psum_updates",
    "sram_read_bytes",
    "sram_write_bytes",
    "fifo_reads",
    "fifo_writes",
    "hbm_read_reqs",
    "hbm_write_reqs",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "cam_lookups",
    "cam_active_rows",
    "cam_compared_bits",
    "cam_verified_rows",
    "cam_hits",
    "cam_misses",
    "cam_inserts",
    "cam_evictions",
    "cam_hot_reads",
    "notes",
]


@dataclass(frozen=True)
class OnnximConfig:
    npu_clock_mhz: float
    ndp_clock_mhz: float
    dram_req_bytes: int
    add_tree_latency: int
    exp_latency: int
    scalar_add_latency: int
    scalar_mul_latency: int


def find_output_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates: list[Path] = []
    env_value = os.environ.get("OFA_OUTPUT_ROOT")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            REPO_ROOT.parent / "Transformer" / "OFA" / "output",
            REPO_ROOT.parent / "output",
            Path("/home/zhangshangtong/Transformer/OFA/output"),
        ]
    )
    required = Path(ARRAY_TRACE_DIRS["cora"]) / "summary.json"
    for candidate in candidates:
        if (candidate / required).exists() and (candidate / "tser_reuse_drop_tradeoff_40pt_alignment.tsv").exists():
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise SystemExit(f"Cannot find OFA output root. Searched: {searched}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing tsv: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, 0) for col in columns})


def seconds_from_cycles(cycles: float, mhz: float) -> float:
    return cycles / (mhz * 1.0e6)


def cycles_from_seconds(seconds: float, mhz: float) -> float:
    return seconds * mhz * 1.0e6


def reqs_from_bytes(num_bytes: float, req_bytes: int) -> int:
    if num_bytes <= 0.0:
        return 0
    return int(math.ceil(num_bytes / float(req_bytes)))


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def load_reuse(output_root: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_tsv(output_root / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"):
        out[row["task"]] = {
            "reuse_pct": float(row["anchor_reuse"]),
            "raw_anchor_drop": float(row.get("raw_anchor_drop", 0.0) or 0.0),
            "target_anchor_drop": float(row.get("target_anchor_drop", 0.0) or 0.0),
        }
    missing = set(TASKS) - set(out)
    if missing:
        raise SystemExit(f"Reuse TSV is missing tasks: {sorted(missing)}")
    return out


def load_onnxim_config(path: Path, args: argparse.Namespace) -> OnnximConfig:
    payload = read_json(path)
    core = payload["core_config"]["core_0"]
    return OnnximConfig(
        npu_clock_mhz=float(args.npu_clock_mhz or payload.get("core_freq", 1000)),
        ndp_clock_mhz=float(args.ndp_clock_mhz),
        dram_req_bytes=int(args.dram_req_bytes or payload.get("dram_req_size", 32)),
        add_tree_latency=int(core.get("add_tree_latency", 1)),
        exp_latency=int(core.get("exp_latency", 1)),
        scalar_add_latency=int(core.get("scalar_add_latency", 1)),
        scalar_mul_latency=int(core.get("scalar_mul_latency", 1)),
    )


def adjusted_refine_ratio(observed_ratio: float, args: argparse.Namespace) -> float:
    if args.refine_ratio_mode == "observed":
        return observed_ratio
    return float(args.target_refine_ratio)


def adjusted_cycles(full4: float, observed_dynamic: float, observed_ratio: float, target_ratio: float) -> float:
    if observed_ratio <= 0.0:
        return observed_dynamic
    return full4 + target_ratio * ((observed_dynamic - full4) / observed_ratio)


def adjusted_bitmacs(
    observed_bitmacs: float,
    observed_cycles: float,
    target_cycles: float,
) -> float:
    if observed_cycles <= 0.0:
        return observed_bitmacs
    return observed_bitmacs * target_cycles / observed_cycles


def base_row(
    *,
    task: str,
    dataset: str,
    component: str,
    subcomponent: str,
    module: str,
    source_kind: str,
    cycles: float,
    frequency_mhz: float,
    reuse_pct: float,
    refine_ratio: float,
    effective_bits: float,
    block_size: int,
    notes: str,
) -> dict[str, Any]:
    time_s = seconds_from_cycles(cycles, frequency_mhz) if frequency_mhz > 0.0 else 0.0
    return {
        "task": task,
        "dataset": dataset,
        "policy": "TSER40_BFPLift",
        "component": component,
        "subcomponent": subcomponent,
        "module": module,
        "source_kind": source_kind,
        "cycles": cycles,
        "time_s": time_s,
        "time_ns": time_s * 1.0e9,
        "frequency_mhz": frequency_mhz,
        "nodes": TASK_NODES[task],
        "edges": TASK_EDGES[task],
        "reuse_pct": reuse_pct,
        "miss_pct": 100.0 - reuse_pct,
        "refine_ratio": refine_ratio,
        "effective_bits": effective_bits,
        "block_size": block_size,
        "notes": notes,
    }


def build_module_rows(
    *,
    task: str,
    dataset: str,
    output_root: Path,
    reuse_pct: float,
    cfg: OnnximConfig,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    trace_dir = output_root / ARRAY_TRACE_DIRS[dataset]
    summary = read_json(trace_dir / "summary.json")
    rows = read_tsv(trace_dir / "module_array_trace.tsv")
    miss_frac = 1.0 - reuse_pct / 100.0
    block_size = int(summary.get("block_size", args.block_size))
    base_bits = float(summary.get("base_mantissa", 4))
    refine_bits = float(summary.get("refine_mantissa", 6))
    out: list[dict[str, Any]] = []
    for raw in rows:
        observed_ratio = f(raw, "refined_ratio")
        ratio = adjusted_refine_ratio(observed_ratio, args)
        full4 = f(raw, "full_bfpa4_cycles")
        observed_dynamic = f(raw, "dynamic_cycles")
        target_single_array_cycles = adjusted_cycles(full4, observed_dynamic, observed_ratio, ratio)
        wall_cycles = target_single_array_cycles * miss_frac / float(args.npu_arrays)
        observed_bitmacs = f(raw, "dynamic_bit_macs")
        bitmacs = adjusted_bitmacs(observed_bitmacs, observed_dynamic, target_single_array_cycles) * miss_frac
        effective_bits = base_bits + (refine_bits - base_bits) * ratio
        token_rows = f(raw, "token_rows") * miss_frac
        calls = f(raw, "calls") * miss_frac
        in_features = i(raw, "in_features")
        out_features = i(raw, "out_features")
        total_blocks = f(raw, "total_blocks") * miss_frac
        refined_blocks = f(raw, "total_blocks") * ratio * miss_frac

        activation_read_bytes = token_rows * in_features * effective_bits / 8.0
        weight_read_bytes = calls * in_features * out_features * float(args.weight_bits) / 8.0
        output_write_bytes = token_rows * out_features * float(args.output_bits) / 8.0

        row = base_row(
            task=task,
            dataset=dataset,
            component="BFPArray",
            subcomponent=str(raw.get("kind", "")),
            module=str(raw.get("module", "")),
            source_kind="onnxim_module_trace_scaled",
            cycles=wall_cycles,
            frequency_mhz=cfg.npu_clock_mhz,
            reuse_pct=reuse_pct,
            refine_ratio=ratio,
            effective_bits=effective_bits,
            block_size=block_size,
            notes=(
                "cycles are ONNXim single-array cycles scaled by miss fraction "
                "and divided by npu_arrays; RF/psum counts are inferred from bit_mac_ops"
            ),
        )
        row.update(
            {
                "token_rows": token_rows,
                "calls": calls,
                "in_features": in_features,
                "out_features": out_features,
                "total_blocks": total_blocks,
                "refined_blocks": refined_blocks,
                "bit_mac_ops": bitmacs,
                "pe_active_cycles": target_single_array_cycles * miss_frac,
                "a_rf_reads": bitmacs,
                "w_rf_reads": bitmacs,
                "psum_reads": bitmacs,
                "psum_writes": bitmacs,
                "psum_updates": bitmacs,
                "sram_read_bytes": activation_read_bytes + weight_read_bytes,
                "sram_write_bytes": output_write_bytes,
                "hbm_read_bytes": activation_read_bytes + weight_read_bytes,
                "hbm_write_bytes": output_write_bytes,
                "hbm_read_reqs": reqs_from_bytes(activation_read_bytes + weight_read_bytes, cfg.dram_req_bytes),
                "hbm_write_reqs": reqs_from_bytes(output_write_bytes, cfg.dram_req_bytes),
            }
        )
        out.append(row)
    return out


def aggregate_rows(task_rows: list[dict[str, Any]], *, task: str, dataset: str, reuse_pct: float, cfg: OnnximConfig) -> dict[str, Any]:
    if not task_rows:
        raise SystemExit(f"No module rows for {task}/{dataset}")
    first = task_rows[0]
    sum_fields = [
        "cycles",
        "time_s",
        "time_ns",
        "token_rows",
        "calls",
        "total_blocks",
        "refined_blocks",
        "bit_mac_ops",
        "pe_active_cycles",
        "a_rf_reads",
        "w_rf_reads",
        "psum_reads",
        "psum_writes",
        "psum_updates",
        "sram_read_bytes",
        "sram_write_bytes",
        "hbm_read_reqs",
        "hbm_write_reqs",
        "hbm_read_bytes",
        "hbm_write_bytes",
    ]
    out = base_row(
        task=task,
        dataset=dataset,
        component="BFPArray",
        subcomponent="all_modules",
        module="LLaMA2-7B-MLP-linear",
        source_kind="onnxim_module_trace_aggregate",
        cycles=sum(f(row, "cycles") for row in task_rows),
        frequency_mhz=cfg.npu_clock_mhz,
        reuse_pct=reuse_pct,
        refine_ratio=f(first, "refine_ratio"),
        effective_bits=f(first, "effective_bits"),
        block_size=i(first, "block_size"),
        notes="aggregate of module_array_trace.tsv rows",
    )
    for field in sum_fields:
        out[field] = sum(f(row, field) for row in task_rows)
    out["time_s"] = seconds_from_cycles(out["cycles"], cfg.npu_clock_mhz)
    out["time_ns"] = out["time_s"] * 1.0e9
    return out


def build_loader_row(
    *,
    task: str,
    dataset: str,
    reuse_pct: float,
    module_rows: list[dict[str, Any]],
    cfg: OnnximConfig,
    npu_arrays: int,
) -> dict[str, Any]:
    block_size = i(module_rows[0], "block_size")
    reduction_levels = int(math.ceil(math.log2(block_size)))
    exp_cycles_per_block = reduction_levels * cfg.add_tree_latency + cfg.exp_latency
    pack_cycles_per_block = cfg.scalar_mul_latency + cfg.scalar_add_latency
    stress_cycles_per_block = cfg.scalar_mul_latency + cfg.scalar_add_latency
    queue_cycles_per_refined = 1.0
    total_blocks = sum(f(row, "total_blocks") for row in module_rows)
    refined_blocks = sum(f(row, "refined_blocks") for row in module_rows)
    single_array_cycles = (
        total_blocks * (exp_cycles_per_block + pack_cycles_per_block + stress_cycles_per_block)
        + refined_blocks * queue_cycles_per_refined
    )
    wall_cycles = single_array_cycles / max(1.0, float(npu_arrays))
    row = base_row(
        task=task,
        dataset=dataset,
        component="BFPLoaderControl",
        subcomponent="exp_pack_stress_queue",
        module="all_modules",
        source_kind="onnxim_latency_model",
        cycles=wall_cycles,
        frequency_mhz=cfg.npu_clock_mhz,
        reuse_pct=reuse_pct,
        refine_ratio=f(module_rows[0], "refine_ratio"),
        effective_bits=f(module_rows[0], "effective_bits"),
        block_size=block_size,
        notes=(
            f"per block cycles: exp={exp_cycles_per_block}, pack={pack_cycles_per_block}, "
            f"stress={stress_cycles_per_block}; queue per refined block=1"
        ),
    )
    row.update(
        {
            "total_blocks": total_blocks,
            "refined_blocks": refined_blocks,
            "pe_active_cycles": single_array_cycles,
            "sram_read_bytes": total_blocks * block_size * f(module_rows[0], "effective_bits") / 8.0,
            "fifo_writes": refined_blocks,
        }
    )
    return row


def average_active_rows(num_queries: int, inserts: float, capacity: int) -> float:
    if num_queries <= 0 or inserts <= 0.0:
        return 0.0
    miss_per_query = inserts / float(num_queries)
    if inserts <= capacity:
        return inserts / 2.0
    warmup_queries = capacity / max(miss_per_query, 1.0e-12)
    area = 0.5 * capacity * warmup_queries + capacity * max(0.0, num_queries - warmup_queries)
    return area / float(num_queries)


def build_cam_row(
    *,
    task: str,
    dataset: str,
    reuse_pct: float,
    cfg: OnnximConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    nodes = TASK_NODES[task]
    miss_frac = 1.0 - reuse_pct / 100.0
    hits = nodes * reuse_pct / 100.0
    misses = nodes - hits
    inserts = misses
    evictions = max(0.0, inserts - args.cam_entries)
    active_avg = average_active_rows(nodes, inserts, args.cam_entries)
    active_rows = active_avg * nodes
    compared_bits = active_rows * args.simhash_bits
    cycles = nodes * (args.cam_search_cycles + args.cam_select_cycles + miss_frac * args.cam_update_cycles)
    row = base_row(
        task=task,
        dataset=dataset,
        component="CAMFrontend",
        subcomponent="HD-CAM+LRU",
        module="simhash_directory",
        source_kind="cam_capacity_model",
        cycles=cycles,
        frequency_mhz=cfg.ndp_clock_mhz,
        reuse_pct=reuse_pct,
        refine_ratio=0.0,
        effective_bits=0.0,
        block_size=0,
        notes="CAM counts inferred from TSER reuse and LRU fill curve; replace with CAM_sim report when available",
    )
    hot_read_bytes = hits * args.embedding_dim * args.embedding_bits / 8.0
    directory_write_bytes = inserts * (args.simhash_bits + args.cam_metadata_bits) / 8.0
    row.update(
        {
            "sram_read_bytes": hot_read_bytes,
            "sram_write_bytes": directory_write_bytes,
            "cam_lookups": nodes,
            "cam_active_rows": active_rows,
            "cam_compared_bits": compared_bits,
            "cam_hits": hits,
            "cam_misses": misses,
            "cam_inserts": inserts,
            "cam_evictions": evictions,
            "cam_hot_reads": hits,
        }
    )
    return row


def build_embedding_io_row(
    *,
    task: str,
    dataset: str,
    reuse_pct: float,
    cfg: OnnximConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    nodes = TASK_NODES[task]
    reuse_frac = reuse_pct / 100.0
    miss_frac = 1.0 - reuse_frac
    emb_bytes = args.embedding_dim * args.embedding_bits / 8.0
    hit_read_bytes = nodes * reuse_frac * emb_bytes
    miss_write_bytes = nodes * miss_frac * emb_bytes
    seconds = hit_read_bytes / (args.ndp_local_dram_bw_gbs * 1.0e9)
    seconds += miss_write_bytes / (args.npu_to_ndp_bw_gbs * 1.0e9)
    row = base_row(
        task=task,
        dataset=dataset,
        component="NDPEmbeddingIO",
        subcomponent="hit_read+miss_write",
        module="embedding_store",
        source_kind="bandwidth_model",
        cycles=cycles_from_seconds(seconds, cfg.ndp_clock_mhz),
        frequency_mhz=cfg.ndp_clock_mhz,
        reuse_pct=reuse_pct,
        refine_ratio=0.0,
        effective_bits=0.0,
        block_size=0,
        notes="hit embedding reads use NDP-local DRAM BW; miss writes use NPU-to-NDP BW",
    )
    row.update(
        {
            "hbm_read_bytes": hit_read_bytes,
            "hbm_write_bytes": miss_write_bytes,
            "hbm_read_reqs": reqs_from_bytes(hit_read_bytes, cfg.dram_req_bytes),
            "hbm_write_reqs": reqs_from_bytes(miss_write_bytes, cfg.dram_req_bytes),
        }
    )
    return row


def build_graph_memory_row(
    *,
    task: str,
    dataset: str,
    reuse_pct: float,
    cfg: OnnximConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    nodes = TASK_NODES[task]
    edges = TASK_EDGES[task]
    emb_bytes = args.embedding_dim * args.embedding_bits / 8.0
    graph_index_bytes = (nodes + 1 + edges) * args.graph_index_bits / 8.0
    neighbor_embedding_bytes = edges * emb_bytes
    read_bytes = graph_index_bytes + neighbor_embedding_bytes
    seconds = read_bytes / (args.ndp_local_dram_bw_gbs * 1.0e9)
    row = base_row(
        task=task,
        dataset=dataset,
        component="NDPGraphMemory",
        subcomponent="index+neighbor_embeddings",
        module="graph_store",
        source_kind="bandwidth_model",
        cycles=cycles_from_seconds(seconds, cfg.ndp_clock_mhz),
        frequency_mhz=cfg.ndp_clock_mhz,
        reuse_pct=reuse_pct,
        refine_ratio=0.0,
        effective_bits=0.0,
        block_size=0,
        notes="graph index and neighbor embedding traffic derived from PyG edge_index counts",
    )
    row.update(
        {
            "hbm_read_bytes": read_bytes,
            "hbm_read_reqs": reqs_from_bytes(read_bytes, cfg.dram_req_bytes),
        }
    )
    return row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    *,
    path: Path,
    output_root: Path,
    activity_rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    by_task: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_component: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in activity_rows:
        task = str(row["task"])
        component = str(row["component"])
        by_task[task]["cycles"] += f(row, "cycles")
        by_task[task]["time_s"] += f(row, "time_s")
        by_task[task]["bit_mac_ops"] += f(row, "bit_mac_ops")
        by_task[task]["hbm_read_bytes"] += f(row, "hbm_read_bytes")
        by_task[task]["hbm_write_bytes"] += f(row, "hbm_write_bytes")
        by_task[task]["cam_lookups"] += f(row, "cam_lookups")
        by_component[component]["time_s"] += f(row, "time_s")
        by_component[component]["bit_mac_ops"] += f(row, "bit_mac_ops")
        by_component[component]["hbm_read_bytes"] += f(row, "hbm_read_bytes")
        by_component[component]["hbm_write_bytes"] += f(row, "hbm_write_bytes")
        by_component[component]["cam_lookups"] += f(row, "cam_lookups")

    lines = [
        "# GRACE Activity Trace",
        "",
        f"- Output root: `{output_root}`",
        f"- NPU arrays: `{args.npu_arrays}` x 128x128",
        f"- Refine ratio mode: `{args.refine_ratio_mode}`; target refine ratio: `{args.target_refine_ratio}`",
        "- `cycles` for BFPArray/BFPLoaderControl are wall cycles after dividing the single-array trace by NPU array count.",
        "- RF/psum counters are inferred from `bit_mac_ops`; replace them with RTL counters when available.",
        "",
        "## Files",
        "",
        f"- `grace_activity_trace.tsv`: task-level component trace ({len(activity_rows)} rows).",
        f"- `grace_module_activity_trace.tsv`: per-module BFP array trace ({len(module_rows)} rows).",
        f"- `grace_activity_trace.json`: machine-readable copy with metadata.",
        "",
        "## Task Totals",
        "",
        "| Task | Time (s) | Bit-MAC ops | HBM read GB | HBM write GB | CAM lookups |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        row = by_task[task]
        lines.append(
            f"| {task} | {row['time_s']:.6f} | {row['bit_mac_ops']:.4e} | "
            f"{row['hbm_read_bytes'] / 1.0e9:.3f} | {row['hbm_write_bytes'] / 1.0e9:.3f} | "
            f"{row['cam_lookups']:.0f} |"
        )
    lines.extend(["", "## Component Totals", "", "| Component | Time (s) | Bit-MAC ops | HBM read GB | HBM write GB | CAM lookups |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for component in sorted(by_component):
        row = by_component[component]
        lines.append(
            f"| {component} | {row['time_s']:.6f} | {row['bit_mac_ops']:.4e} | "
            f"{row['hbm_read_bytes'] / 1.0e9:.3f} | {row['hbm_write_bytes'] / 1.0e9:.3f} | "
            f"{row['cam_lookups']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- ONNXim module cycles and bit-MACs are trace-derived.",
            "- CAM activity is currently inferred from the TSER reuse point and capacity model.",
            "- HBM/SRAM/RF/psum counts are analytical estimates for energy modeling, not synthesis counters.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--onnxim-config",
        type=Path,
        default=REPO_ROOT / "ONNXim" / "configs" / "systolic_ws_128x128_c4_simple_noc_tpuv4.json",
    )
    parser.add_argument("--npu-arrays", type=int, default=16)
    parser.add_argument("--npu-clock-mhz", type=float, default=None)
    parser.add_argument("--ndp-clock-mhz", type=float, default=500.0)
    parser.add_argument("--dram-req-bytes", type=int, default=None)
    parser.add_argument("--refine-ratio-mode", choices=("target", "observed"), default="target")
    parser.add_argument("--target-refine-ratio", type=float, default=0.20)
    parser.add_argument("--weight-bits", type=float, default=4.0)
    parser.add_argument("--output-bits", type=float, default=16.0)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--embedding-bits", type=int, default=16)
    parser.add_argument("--graph-index-bits", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--simhash-bits", type=int, default=128)
    parser.add_argument("--cam-metadata-bits", type=int, default=64)
    parser.add_argument("--cam-entries", type=int, default=4096)
    parser.add_argument("--cam-search-cycles", type=float, default=1.0)
    parser.add_argument("--cam-select-cycles", type=float, default=1.0)
    parser.add_argument("--cam-update-cycles", type=float, default=1.0)
    parser.add_argument("--ndp-local-dram-bw-gbs", type=float, default=256.0)
    parser.add_argument("--npu-to-ndp-bw-gbs", type=float, default=64.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = find_output_root(args.output_root)
    output_dir = args.output_dir or (output_root / "grace_activity_trace")
    cfg = load_onnxim_config(args.onnxim_config, args)
    reuse = load_reuse(output_root)

    activity_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for task in TASKS:
        dataset = TASK_TO_DATASET[task]
        reuse_pct = reuse[task]["reuse_pct"]
        task_module_rows = build_module_rows(
            task=task,
            dataset=dataset,
            output_root=output_root,
            reuse_pct=reuse_pct,
            cfg=cfg,
            args=args,
        )
        module_rows.extend(task_module_rows)
        activity_rows.append(aggregate_rows(task_module_rows, task=task, dataset=dataset, reuse_pct=reuse_pct, cfg=cfg))
        activity_rows.append(
            build_loader_row(
                task=task,
                dataset=dataset,
                reuse_pct=reuse_pct,
                module_rows=task_module_rows,
                cfg=cfg,
                npu_arrays=args.npu_arrays,
            )
        )
        activity_rows.append(build_cam_row(task=task, dataset=dataset, reuse_pct=reuse_pct, cfg=cfg, args=args))
        activity_rows.append(build_embedding_io_row(task=task, dataset=dataset, reuse_pct=reuse_pct, cfg=cfg, args=args))
        activity_rows.append(build_graph_memory_row(task=task, dataset=dataset, reuse_pct=reuse_pct, cfg=cfg, args=args))

    write_tsv(output_dir / "grace_activity_trace.tsv", activity_rows, TRACE_COLUMNS)
    write_tsv(output_dir / "grace_module_activity_trace.tsv", module_rows, TRACE_COLUMNS)
    write_json(
        output_dir / "grace_activity_trace.json",
        {
            "schema": "grace_activity_trace.v1",
            "output_root": str(output_root),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "onnxim": cfg.__dict__,
            "activity_rows": activity_rows,
            "module_rows": module_rows,
        },
    )
    write_report(
        path=output_dir / "GRACE_ACTIVITY_TRACE.md",
        output_root=output_root,
        activity_rows=activity_rows,
        module_rows=module_rows,
        args=args,
    )
    print(f"[GRACEActivity] wrote {output_dir / 'grace_activity_trace.tsv'}")
    print(f"[GRACEActivity] wrote {output_dir / 'grace_module_activity_trace.tsv'}")
    print(f"[GRACEActivity] wrote {output_dir / 'GRACE_ACTIVITY_TRACE.md'}")


if __name__ == "__main__":
    main()
