#!/usr/bin/env python3
"""Lightweight ogbn-products proxy for frontend scalability checks.

The dataset has no raw node text in the standard OGB release. This script uses
the built-in 100-d node features only as a scalability proxy for CAM pressure,
degree-risk distribution, and sampled hash-reuse behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = payload["stats"]
    degree = payload["degree"]
    reuse = payload["reuse_proxy"]
    lines = [
        "# ogbn-products Feature Proxy",
        "",
        "This is a feature-level scalability proxy. It does not use LLaMA/ST text embeddings.",
        "",
        "## Dataset",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Nodes | {stats['num_nodes']} |",
        f"| Directed edge records | {stats['num_edges']} |",
        f"| Feature dim | {stats['feature_dim']} |",
        f"| Classes | {stats['num_classes']} |",
        f"| Train nodes | {stats['split_sizes']['train']} |",
        f"| Valid nodes | {stats['split_sizes']['valid']} |",
        f"| Test nodes | {stats['split_sizes']['test']} |",
        "",
        "## Degree",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean incidence degree | {degree['mean']:.2f} |",
        f"| P50 | {degree['p50']:.1f} |",
        f"| P90 | {degree['p90']:.1f} |",
        f"| P95 | {degree['p95']:.1f} |",
        f"| P99 | {degree['p99']:.1f} |",
        f"| Max | {degree['max']} |",
        "",
        "## Sampled SimHash/CAM Proxy",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Query sample | {reuse['query_sample']} |",
        f"| Anchor sample | {reuse['anchor_sample']} |",
        f"| Heads x bits | {reuse['heads']} x {reuse['bits']} |",
        f"| Radius | {reuse['radius']} |",
        f"| Hard support threshold | {reuse['hard_support_threshold']} |",
        f"| Soft support threshold | {reuse['soft_support_threshold']} |",
        f"| Hard reuse | {reuse['hard_reuse_pct']:.2f}% |",
        f"| Soft reuse | {reuse['soft_reuse_pct']:.2f}% |",
        f"| Miss | {reuse['miss_pct']:.2f}% |",
        f"| Hard label agreement | {reuse['hard_label_agreement_pct']:.2f}% |",
        f"| Soft label agreement | {reuse['soft_label_agreement_pct']:.2f}% |",
        f"| Hard feature cosine | {reuse['hard_mean_cosine']:.4f} |",
        f"| Soft feature cosine | {reuse['soft_mean_cosine']:.4f} |",
        "",
        "## Interpretation",
        "",
        "- ogbn-products is suitable as a million-scale node-classification pressure test.",
        "- Standard OGB products does not provide raw product text, so LLaMA/ST encoder reuse is not directly comparable to Cora/PubMed/Arxiv/Wiki-CS without adding text metadata.",
        "- The sampled proxy is useful for CAM capacity, degree-risk distribution, and miss-node batching analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_products(root: Path):
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name="ogbn-products", root=str(root / "data"))
    data = dataset[0]
    split = dataset.get_idx_split()
    return data, split


def degree_stats(edge_index: torch.Tensor, num_nodes: int, chunk_edges: int) -> tuple[torch.Tensor, dict[str, Any]]:
    deg = torch.zeros(num_nodes, dtype=torch.long)
    for row in (0, 1):
        values = edge_index[row].cpu()
        for start in range(0, values.numel(), chunk_edges):
            chunk = values[start : start + chunk_edges]
            deg += torch.bincount(chunk, minlength=num_nodes)
    deg_f = deg.float()
    quantiles = torch.quantile(deg_f, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    stats = {
        "mean": float(deg_f.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "max": int(deg.max().item()),
    }
    return deg, stats


def pack_bits_to_uint16(bits: np.ndarray) -> np.ndarray:
    packed = np.zeros(bits.shape[0], dtype=np.uint16)
    for i in range(bits.shape[1]):
        packed |= (bits[:, i].astype(np.uint16) << i)
    return packed


def build_hash_codes(features: np.ndarray, projections: list[np.ndarray]) -> np.ndarray:
    heads = len(projections)
    codes = np.empty((heads, features.shape[0]), dtype=np.uint16)
    for h, proj in enumerate(projections):
        signs = (features @ proj) >= 0
        codes[h] = pack_bits_to_uint16(signs)
    return codes


def sampled_hash_reuse(
    x: torch.Tensor,
    y: torch.Tensor,
    split: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    train_idx = split["train"].cpu().numpy()
    query_pool = torch.cat([split["valid"], split["test"]]).cpu().numpy()
    anchor_idx = rng.choice(train_idx, size=min(args.anchor_sample, train_idx.size), replace=False)
    query_idx = rng.choice(query_pool, size=min(args.query_sample, query_pool.size), replace=False)

    x_np = x.float().cpu().numpy()
    x_np = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)
    anchor_feat = x_np[anchor_idx]
    query_feat = x_np[query_idx]

    projections = [
        rng.standard_normal((x_np.shape[1], args.bits), dtype=np.float32)
        for _ in range(args.heads)
    ]
    anchor_codes = build_hash_codes(anchor_feat, projections)
    query_codes = build_hash_codes(query_feat, projections)
    popcount = np.array([bin(i).count("1") for i in range(1 << args.bits)], dtype=np.uint8)

    best_support = np.zeros(query_feat.shape[0], dtype=np.uint8)
    best_anchor = np.zeros(query_feat.shape[0], dtype=np.int64)
    best_tiebreak = np.full(query_feat.shape[0], 10_000, dtype=np.int32)

    for start in range(0, query_feat.shape[0], args.query_chunk):
        end = min(start + args.query_chunk, query_feat.shape[0])
        support = np.zeros((end - start, anchor_feat.shape[0]), dtype=np.uint8)
        total_dist = np.zeros((end - start, anchor_feat.shape[0]), dtype=np.int16)
        for h in range(args.heads):
            xor = np.bitwise_xor(query_codes[h, start:end, None], anchor_codes[h, None, :])
            dist = popcount[xor]
            support += dist <= args.radius
            total_dist += dist.astype(np.int16)
        score = support.astype(np.int16) * 1000 - total_dist
        local = np.argmax(score, axis=1)
        rows = np.arange(end - start)
        best_support[start:end] = support[rows, local]
        best_tiebreak[start:end] = total_dist[rows, local]
        best_anchor[start:end] = local

    anchor_labels = y[anchor_idx].view(-1).cpu().numpy()
    query_labels = y[query_idx].view(-1).cpu().numpy()
    selected_labels = anchor_labels[best_anchor]
    selected_cos = np.sum(query_feat * anchor_feat[best_anchor], axis=1)

    hard = best_support >= args.hard_support_threshold
    soft = best_support >= args.soft_support_threshold
    miss = ~soft

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(values[mask]))

    label_match = selected_labels == query_labels
    return {
        "query_sample": int(query_feat.shape[0]),
        "anchor_sample": int(anchor_feat.shape[0]),
        "heads": args.heads,
        "bits": args.bits,
        "radius": args.radius,
        "hard_support_threshold": args.hard_support_threshold,
        "soft_support_threshold": args.soft_support_threshold,
        "hard_reuse_pct": float(np.mean(hard) * 100.0),
        "soft_reuse_pct": float(np.mean(soft) * 100.0),
        "miss_pct": float(np.mean(miss) * 100.0),
        "hard_label_agreement_pct": masked_mean(label_match.astype(np.float32) * 100.0, hard),
        "soft_label_agreement_pct": masked_mean(label_match.astype(np.float32) * 100.0, soft),
        "hard_mean_cosine": masked_mean(selected_cos, hard),
        "soft_mean_cosine": masked_mean(selected_cos, soft),
        "mean_best_support": float(np.mean(best_support)),
        "mean_best_hamming_sum": float(np.mean(best_tiebreak)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "ogbn_products_proxy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-sample", type=int, default=4096)
    parser.add_argument("--query-sample", type=int, default=10000)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--hard-support-threshold", type=int, default=5)
    parser.add_argument("--soft-support-threshold", type=int, default=3)
    parser.add_argument("--degree-chunk-edges", type=int, default=5_000_000)
    args = parser.parse_args()

    data, split = load_products(ROOT)
    deg, deg_stats = degree_stats(data.edge_index, int(data.num_nodes), args.degree_chunk_edges)
    reuse = sampled_hash_reuse(data.x, data.y, split, args)
    payload = {
        "stats": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "feature_dim": int(data.x.size(1)),
            "num_classes": int(data.y.max().item()) + 1,
            "split_sizes": {key: int(value.numel()) for key, value in split.items()},
        },
        "degree": deg_stats,
        "reuse_proxy": reuse,
    }
    write_json(args.output_dir / "summary.json", payload)
    write_markdown(args.output_dir / "summary.md", payload)
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'summary.md'}")
    print(
        "ogbn-products proxy | "
        f"hard={reuse['hard_reuse_pct']:.2f}% soft={reuse['soft_reuse_pct']:.2f}% "
        f"miss={reuse['miss_pct']:.2f}% p95deg={deg_stats['p95']:.1f}"
    )


if __name__ == "__main__":
    main()
