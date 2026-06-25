#!/usr/bin/env python3
"""Measure offline preprocessing time for the six evaluation tasks.

This script times the graph-side preprocessing needed before online reuse:

1. build graph-context keys;
2. build multi-head SimHash code tables;
3. build node-side P/C/U risk fields with separate timing.

It intentionally excludes LLaMA embedding generation, AWQ/BFPA pool generation,
and online CAM query/reuse filtering.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from profile_candidate_discovery_ablation import (
    build_codes,
    graph_context_key,
    load_graph,
    load_key_features,
    make_projection,
    normalize_edges,
)


TASKS = {
    "CN": "cora",
    "CL": "cora",
    "PN": "pubmed",
    "PL": "pubmed",
    "AR": "arxiv",
    "WK": "wikics",
}


def quantize_4bit(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.numel() == 0:
        return x.to(torch.uint8)
    lo = float(x.min().item())
    hi = float(x.max().item())
    if hi <= lo:
        return torch.zeros_like(x, dtype=torch.uint8)
    y = (x - lo) / (hi - lo)
    return torch.clamp(torch.round(y * 15.0), 0, 15).to(torch.uint8)


def degree_from_edges(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    src = edge_index[0].long()
    deg = torch.bincount(src, minlength=n).float()
    return deg


def hamming_u16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    popcount = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    return popcount[np.bitwise_xor(a, b)]


def build_risk_fields(
    edge_index: torch.Tensor,
    self_codes: np.ndarray,
    graph_codes: np.ndarray,
    chunk_edges: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    risk_times: dict[str, float] = {}
    n = int(graph_codes.shape[1])

    t0 = time.perf_counter()
    deg = degree_from_edges(edge_index, n)
    denom = torch.log1p(deg.max().clamp_min(1.0))
    p_norm = torch.log1p(deg) / denom
    p_q = torch.clamp(torch.round(p_norm * 15.0), 0, 15).to(torch.uint8)
    risk_times["risk_propagation_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    src_np = edge_index[0].cpu().numpy().astype(np.int64, copy=False)
    dst_np = edge_index[1].cpu().numpy().astype(np.int64, copy=False)
    boundary_sum = np.zeros(n, dtype=np.float64)
    boundary_cnt = np.zeros(n, dtype=np.int64)
    for start in range(0, src_np.shape[0], chunk_edges):
        end = min(start + chunk_edges, src_np.shape[0])
        s = src_np[start:end]
        d = dst_np[start:end]
        dist_heads = []
        for h in range(graph_codes.shape[0]):
            dist_heads.append(hamming_u16(graph_codes[h, s], graph_codes[h, d]).astype(np.float32))
        dist = np.mean(dist_heads, axis=0) / 16.0
        np.add.at(boundary_sum, s, dist)
        np.add.at(boundary_cnt, s, 1)
    boundary = torch.from_numpy(boundary_sum / np.maximum(boundary_cnt, 1)).float()

    drift_heads = []
    for h in range(graph_codes.shape[0]):
        drift_heads.append(hamming_u16(self_codes[h], graph_codes[h]).astype(np.float32))
    drift = torch.from_numpy(np.mean(drift_heads, axis=0) / 16.0).float()
    c_q = quantize_4bit(torch.maximum(boundary, drift))
    risk_times["risk_context_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Rarity uses the first two graph-context heads as a compact bucket key.
    if graph_codes.shape[0] >= 2:
        bucket = graph_codes[0].astype(np.uint32) | (graph_codes[1].astype(np.uint32) << 16)
    else:
        bucket = graph_codes[0].astype(np.uint32)
    _uniq, inv, counts = np.unique(bucket, return_inverse=True, return_counts=True)
    bucket_count = torch.from_numpy(counts[inv].astype(np.float32))
    rarity = 1.0 / torch.log2(bucket_count + 2.0)
    u_score = rarity * (1.0 - p_norm.clamp(0.0, 1.0))
    u_q = quantize_4bit(u_score)
    risk_times["risk_uniqueness_s"] = time.perf_counter() - t0

    return (
        {
            "propagation_q": p_q,
            "graph_context_q": c_q,
            "uniqueness_q": u_q,
        },
        risk_times,
    )


def profile_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    times: dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    data = load_graph(dataset)
    n = int(getattr(data, "num_nodes", 0) or data.x.size(0))
    edge_index = normalize_edges(data.edge_index, n)
    cheap = load_key_features(dataset, data)[:n]
    cheap = torch.nn.functional.normalize(cheap, p=2, dim=1)
    times["load_graph_features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph_key = graph_context_key(cheap, edge_index, args.self_weight)
    times["graph_context_key_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    projections = make_projection(int(cheap.size(1)), args.bits, args.heads, args.seed)
    self_codes = build_codes(cheap, projections, args.feature_chunk)
    graph_codes = build_codes(graph_key, projections, args.feature_chunk)
    times["simhash_code_table_s"] = time.perf_counter() - t0

    risks, risk_times = build_risk_fields(edge_index, self_codes, graph_codes, args.edge_chunk)
    times.update(risk_times)

    times["method_preprocess_s"] = (
        times["graph_context_key_s"]
        + times["simhash_code_table_s"]
        + times["risk_propagation_s"]
        + times["risk_context_s"]
        + times["risk_uniqueness_s"]
    )
    times["wall_total_s"] = time.perf_counter() - t_total
    return {
        "dataset": dataset,
        "nodes": n,
        "edges": int(edge_index.size(1)),
        "feature_dim": int(cheap.size(1)),
        "heads": int(args.heads),
        "bits": int(args.bits),
        "risk_field_bytes": int(sum(v.numel() * v.element_size() for v in risks.values())),
        **times,
    }


def fmt_s(x: float) -> str:
    return f"{x:.3f}"


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_path = out_dir / "preprocess_time_unique_datasets.tsv"
    task_path = out_dir / "preprocess_time_six_tasks.tsv"
    json_path = out_dir / "preprocess_time_raw.json"
    md_path = out_dir / "preprocess_time_six_tasks.md"

    columns = [
        "dataset",
        "nodes",
        "edges",
        "feature_dim",
        "graph_context_key_s",
        "simhash_code_table_s",
        "risk_propagation_s",
        "risk_context_s",
        "risk_uniqueness_s",
        "method_preprocess_s",
        "load_graph_features_s",
        "wall_total_s",
        "risk_field_bytes",
    ]
    with unique_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in columns) + "\n")

    by_dataset = {row["dataset"]: row for row in rows}
    with task_path.open("w", encoding="utf-8") as f:
        f.write("task\t" + "\t".join(columns) + "\n")
        for task, dataset in TASKS.items():
            row = by_dataset[dataset]
            f.write(task + "\t" + "\t".join(str(row[c]) for c in columns) + "\n")

    payload = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
        },
        "unique_datasets": rows,
        "six_tasks": [{**by_dataset[dataset], "task": task} for task, dataset in TASKS.items()],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Six-Task Preprocessing Time",
        "",
        "Scope: graph-side offline preprocessing only. This excludes data loading, LLaMA/AWQ/BFPA pool generation, and online CAM query execution.",
        "",
        f"- SimHash heads: `{args.heads}`",
        f"- bits/head: `{args.bits}`",
        f"- graph-context key: `{args.self_weight:.2f} * self + {1.0 - args.self_weight:.2f} * neighbor_mean`",
        f"- measured on: `{platform.node()}`",
        f"- torch threads: `{torch.get_num_threads()}`",
        "",
        "## Six Evaluation Tasks",
        "",
        "| Task | Dataset | Nodes | Edges | Graph Key (s) | SimHash Table (s) | P Risk (s) | C Risk (s) | U Risk (s) | Method Total (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, dataset in TASKS.items():
        row = by_dataset[dataset]
        lines.append(
            f"| {task} | {dataset} | {row['nodes']} | {row['edges']} | "
            f"{fmt_s(row['graph_context_key_s'])} | {fmt_s(row['simhash_code_table_s'])} | "
            f"{fmt_s(row['risk_propagation_s'])} | {fmt_s(row['risk_context_s'])} | "
            f"{fmt_s(row['risk_uniqueness_s'])} | {fmt_s(row['method_preprocess_s'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- CN and CL share the same Cora preprocessing artifact.",
            "- PN and PL share the same PubMed preprocessing artifact.",
            "- `Method Total` excludes graph loading and cheap-feature loading; those are still kept in the raw JSON/TSV for reproducibility.",
            "- `SimHash Table` builds both self-only and graph-context multi-head signatures for profiling/reuse support.",
            "- `P Risk`, `C Risk`, and `U Risk` separately time propagation, graph-context, and low-degree uniqueness metadata construction.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "arxiv", "wikics"])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--self-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--feature-chunk", type=int, default=8192)
    parser.add_argument("--edge-chunk", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing_time_six_tasks"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        print(f"[profile] {dataset}", flush=True)
        rows.append(profile_dataset(dataset, args))
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
