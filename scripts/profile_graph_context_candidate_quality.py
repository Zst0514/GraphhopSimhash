#!/usr/bin/env python3
"""Compare self-only and graph-context SimHash retrieval keys.

For each sampled query node, this profiler searches a common anchor pool with
two binary keys:

* self: SimHash(normalize(f_self))
* graph_context: SimHash(normalize(0.5 f_self + 0.5 mean_neighbor(f_self)))

It then evaluates the selected anchor with offline-only quality metrics:
self cosine, one-hop context cosine, label agreement, and degree gap. These
metrics are for motivation/profiling only; labels are not used online.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.features import _compute_neighbor_mean  # noqa: E402
from GraphhopSimhash.scripts.profile_semantic_locality import (  # noqa: E402
    LLAMA_CACHE,
    ST_CACHE,
    load_embedding_tensor,
    load_edge_index,
    load_labels,
    make_projection,
    write_json,
)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.to(torch.float32), p=2, dim=1)


def node_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index.long()
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    valid_row = (row >= 0) & (row < num_nodes)
    valid_col = (col >= 0) & (col < num_nodes)
    deg.index_add_(0, row[valid_row].cpu(), torch.ones(int(valid_row.sum())))
    deg.index_add_(0, col[valid_col].cpu(), torch.ones(int(valid_col.sum())))
    return deg


def simhash_bits(features: torch.Tensor, proj: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, features.size(0), chunk_size):
        end = min(start + chunk_size, features.size(0))
        chunks.append((features[start:end] @ proj) >= 0)
    return torch.cat(chunks, dim=0)


def nearest_by_hamming(
    query_bits: torch.Tensor,
    anchor_bits: torch.Tensor,
    query_nodes: torch.Tensor,
    anchor_nodes: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_anchor = torch.empty(query_nodes.numel(), dtype=torch.long)
    best_dist = torch.empty(query_nodes.numel(), dtype=torch.float32)
    anchor_bits = anchor_bits.cpu()
    anchor_nodes = anchor_nodes.cpu()
    for start in range(0, query_nodes.numel(), chunk_size):
        end = min(start + chunk_size, query_nodes.numel())
        q_bits = query_bits[start:end].cpu()
        q_nodes = query_nodes[start:end].cpu()
        dist = torch.logical_xor(q_bits[:, None, :], anchor_bits[None, :, :]).sum(dim=2)
        same = q_nodes[:, None] == anchor_nodes[None, :]
        dist[same] = query_bits.size(1) + 1
        val, idx = dist.min(dim=1)
        best_anchor[start:end] = anchor_nodes[idx]
        best_dist[start:end] = val.to(torch.float32) / float(query_bits.size(1))
    return best_anchor, best_dist


def gather_quality(
    name: str,
    query_nodes: torch.Tensor,
    anchor_nodes: torch.Tensor,
    hamming: torch.Tensor,
    self_feat: torch.Tensor,
    context_feat: torch.Tensor,
    y: torch.Tensor | None,
    deg: torch.Tensor,
) -> dict[str, float | str]:
    q = query_nodes.long()
    a = anchor_nodes.long()
    self_sim = (self_feat[q] * self_feat[a]).sum(dim=1)
    ctx_sim = (context_feat[q] * context_feat[a]).sum(dim=1)
    degree_gap = (torch.log1p(deg[q]) - torch.log1p(deg[a])).abs()
    out: dict[str, float | str] = {
        "key": name,
        "self_sim": float(self_sim.mean()),
        "context_sim": float(ctx_sim.mean()),
        "hamming": float(hamming.mean()),
        "degree_gap": float(degree_gap.mean()),
    }
    if y is not None:
        yy = y.view(-1).cpu().long()
        out["label_hit"] = float((yy[q] == yy[a]).to(torch.float32).mean() * 100.0)
    else:
        out["label_hit"] = float("nan")
    return out


def profile_dataset(dataset: str, args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
    x = load_embedding_tensor(dataset, "llama")
    edge_index = load_edge_index(dataset)
    if edge_index.max().item() >= x.size(0):
        mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
        edge_index = edge_index[:, mask]
    y = load_labels(dataset)
    if y is not None and y.numel() != x.size(0):
        y = None

    self_feat = normalize(x)
    neighbor_mean = _compute_neighbor_mean(self_feat, edge_index)
    context_feat = normalize(neighbor_mean)
    graph_key = normalize(0.5 * self_feat + 0.5 * neighbor_mean)
    deg = node_degrees(edge_index, int(x.size(0)))

    num_nodes = int(x.size(0))
    num_queries = min(args.queries, num_nodes)
    num_anchors = min(args.anchors, num_nodes)
    query_nodes = torch.from_numpy(rng.choice(num_nodes, size=num_queries, replace=False)).long()
    anchor_nodes = torch.from_numpy(rng.choice(num_nodes, size=num_anchors, replace=False)).long()
    if num_anchors < 2:
        raise ValueError("Need at least two anchors")

    proj = make_projection(int(x.size(1)), args.hash_bits, args.seed + 777)
    self_bits_all = simhash_bits(self_feat, proj, args.feature_chunk)
    graph_bits_all = simhash_bits(graph_key, proj, args.feature_chunk)

    rows: list[dict[str, Any]] = []
    for key_name, bits_all in (("self_only", self_bits_all), ("graph_context", graph_bits_all)):
        anchors, hamming = nearest_by_hamming(
            bits_all[query_nodes],
            bits_all[anchor_nodes],
            query_nodes,
            anchor_nodes,
            args.search_chunk,
        )
        row = gather_quality(
            key_name,
            query_nodes,
            anchors,
            hamming,
            self_feat,
            context_feat,
            y,
            deg,
        )
        row.update(
            {
                "dataset": dataset,
                "queries": int(num_queries),
                "anchors": int(num_anchors),
                "nodes": int(num_nodes),
            }
        )
        rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "key",
        "queries",
        "anchors",
        "self_sim",
        "context_sim",
        "label_hit",
        "degree_gap",
        "hamming",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.6f}")
                else:
                    vals.append(str(val))
            f.write("\t".join(vals) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Graph-Context SimHash Candidate Quality",
        "",
        "This profiling compares two retrieval keys over the same sampled query and anchor pools:",
        "",
        "- `self_only`: SimHash of the node LLaMA embedding.",
        "- `graph_context`: SimHash of `0.5 * self + 0.5 * one-hop neighbor mean`.",
        "",
        "Labels are used only for offline profiling.",
        "",
        f"queries per dataset: `{args.queries}`",
        f"anchor pool per dataset: `{args.anchors}`",
        f"hash bits: `{args.hash_bits}`",
        "",
        "| Dataset | Key | SelfSim ↑ | CtxSim ↑ | LabelHit ↑ | DegreeGap ↓ | Ham ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['key']} | "
            f"{row['self_sim']:.4f} | {row['context_sim']:.4f} | "
            f"{row['label_hit']:.2f}% | {row['degree_gap']:.4f} | {row['hamming']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `SelfSim` measures text/embedding similarity between query and selected anchor.",
            "- `CtxSim` measures one-hop neighborhood semantic similarity.",
            "- `LabelHit` is an offline sanity check for class agreement.",
            "- `DegreeGap` checks whether selected anchors have similar propagation scale.",
            "",
            "A useful graph-context key should preserve comparable self similarity while improving context similarity or label agreement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "arxiv"])
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--anchors", type=int, default=8192)
    parser.add_argument("--hash-bits", type=int, default=128)
    parser.add_argument("--feature-chunk", type=int, default=4096)
    parser.add_argument("--search-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "graph_context_candidate_quality")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"[GraphContextQuality] dataset={dataset}")
        rows.extend(profile_dataset(dataset.lower(), args, rng))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arg_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    write_json(args.output_dir / "summary.json", {"rows": rows, "args": arg_payload})
    write_tsv(args.output_dir / "summary.tsv", rows)
    write_markdown(args.output_dir / "summary.md", rows, args)
    print(f"wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
