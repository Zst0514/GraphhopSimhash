#!/usr/bin/env python3
"""Profile candidate discovery quality for graph-aware encoder reuse.

This script isolates the lookup stage before TSER/residual decisions.  It
compares exact text caching, self-only SimHash, graph-context SimHash, and
multi-head graph-context SimHash over the same sampled query/anchor pools.

The lookup itself uses only cheap online keys.  LLaMA embeddings and labels are
used only after lookup to measure candidate quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
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


DATA_CACHE = {
    "cora": ROOT / "data" / "single_graph" / "Cora" / "cora.pt",
    "pubmed": ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt",
    "arxiv": ROOT / "cache_data" / "arxiv" / "ST" / "processed" / "geometric_data_processed.pt",
    "wikics": ROOT / "cache_data" / "wikics" / "ST" / "processed" / "geometric_data_processed.pt",
    "tape_products": ROOT / "cache_data" / "tape_products" / "ST" / "processed" / "geometric_data_processed.pt",
    "tape_arxiv23": ROOT / "cache_data" / "tape_arxiv23" / "ST" / "processed" / "geometric_data_processed.pt",
}

LLAMA_CACHE = {
    ds: ROOT / "cache_data" / f"{ds}_llama2_7b_oracle_W4BFPA8_B128.pt"
    for ds in DATA_CACHE
}

DISTILBERT_CACHE = {
    "cora": ROOT / "cache_data" / "cora_distilbert_l1.pt",
    "pubmed": ROOT / "cache_data" / "pubmed_distilbert_l1.pt",
    "arxiv": ROOT / "cache_data" / "arxiv_distilbert_l1.pt",
    "wikics": ROOT / "cache_data" / "wikics_distilbert_l1.pt",
}

DISPLAY = {
    "cora": "CR",
    "pubmed": "PB",
    "arxiv": "AR",
    "wikics": "WK",
    "tape_products": "PR",
    "tape_arxiv23": "A23",
}


def load_torch(path: Path) -> Any:
    obj = torch.load(path, map_location="cpu")
    return obj[0] if isinstance(obj, tuple) else obj


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("x", "embeddings", "features"):
            if key in obj and torch.is_tensor(obj[key]):
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"{path} does not contain a tensor")
    return obj.detach().to(torch.float32).cpu()


def load_graph(dataset: str) -> Any:
    path = DATA_CACHE[dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    data = load_torch(path)
    if not hasattr(data, "edge_index"):
        raise ValueError(f"{path} does not contain edge_index")
    return data


def load_key_features(dataset: str, data: Any) -> torch.Tensor:
    if dataset in DISTILBERT_CACHE and DISTILBERT_CACHE[dataset].exists():
        x = load_tensor(DISTILBERT_CACHE[dataset])
    elif hasattr(data, "x") and torch.is_tensor(data.x):
        x = data.x.detach().to(torch.float32).cpu()
    else:
        raise ValueError(f"No cheap key features available for {dataset}")
    x = x - x.mean(dim=0, keepdim=True)
    return F.normalize(x, p=2, dim=1)


def load_labels(data: Any, n: int) -> torch.Tensor | None:
    y = getattr(data, "y", None)
    if y is None or not torch.is_tensor(y):
        return None
    y = y.view(-1).detach().cpu().long()
    return y if y.numel() == n else None


def maybe_load_raw_texts(dataset: str, n: int) -> list[str] | None:
    try:
        from GraphhopSimhash.data import load_raw_texts

        texts = load_raw_texts(dataset)
    except Exception:
        return None
    if len(texts) != n:
        return None
    return [str(text) for text in texts]


def normalize_edges(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    src, dst = edge_index
    keep = (src >= 0) & (src < n) & (dst >= 0) & (dst < n) & (src != dst)
    return edge_index[:, keep]


def graph_context_key(features: torch.Tensor, edge_index: torch.Tensor, self_weight: float) -> torch.Tensor:
    neigh = _compute_neighbor_mean(features, edge_index)
    key = self_weight * features + (1.0 - self_weight) * neigh
    return F.normalize(key, p=2, dim=1)


def make_projection(dim: int, bits: int, heads: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    projections = []
    for _ in range(heads):
        proj = rng.standard_normal((dim, bits), dtype=np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
        projections.append(proj)
    return projections


def pack_uint16(signs: np.ndarray) -> np.ndarray:
    packed = np.zeros(signs.shape[0], dtype=np.uint16)
    for bit in range(signs.shape[1]):
        packed |= signs[:, bit].astype(np.uint16) << bit
    return packed


def build_codes(features: torch.Tensor, projections: list[np.ndarray], chunk: int) -> np.ndarray:
    x = features.detach().cpu().numpy().astype(np.float32, copy=False)
    codes = np.empty((len(projections), x.shape[0]), dtype=np.uint16)
    for h, proj in enumerate(projections):
        for start in range(0, x.shape[0], chunk):
            end = min(start + chunk, x.shape[0])
            signs = (x[start:end] @ proj) >= 0
            codes[h, start:end] = pack_uint16(signs)
    return codes


def sampled_indices(n: int, sample: int, rng: np.random.Generator) -> np.ndarray:
    size = min(int(sample), n)
    return rng.choice(n, size=size, replace=False)


def hash_lookup(
    query_codes: np.ndarray,
    anchor_codes: np.ndarray,
    query_nodes: np.ndarray,
    anchor_nodes: np.ndarray,
    radius: int,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heads = query_codes.shape[0]
    popcount = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    selected_anchor_pos = np.full(query_nodes.shape[0], -1, dtype=np.int64)
    selected_support = np.zeros(query_nodes.shape[0], dtype=np.int16)
    selected_distance = np.full(query_nodes.shape[0], 9999, dtype=np.float32)

    for start in range(0, query_nodes.shape[0], chunk):
        end = min(start + chunk, query_nodes.shape[0])
        rows = end - start
        support = np.zeros((rows, anchor_nodes.shape[0]), dtype=np.int16)
        total_dist = np.zeros((rows, anchor_nodes.shape[0]), dtype=np.int16)
        for h in range(heads):
            xor = np.bitwise_xor(query_codes[h, start:end, None], anchor_codes[h, None, :])
            dist = popcount[xor].astype(np.int16)
            support += dist <= radius
            total_dist += dist
        same = query_nodes[start:end, None] == anchor_nodes[None, :]
        score = support.astype(np.int32) * 1000 - total_dist.astype(np.int32)
        score[same] = -1_000_000
        local = np.argmax(score, axis=1)
        r = np.arange(rows)
        selected_anchor_pos[start:end] = local
        selected_support[start:end] = support[r, local]
        selected_distance[start:end] = total_dist[r, local].astype(np.float32) / float(heads)
    return selected_anchor_pos, selected_support, selected_distance


def exact_text_lookup(
    texts: list[str] | None,
    query_nodes: np.ndarray,
    anchor_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.full(query_nodes.shape[0], -1, dtype=np.int64)
    support = np.zeros(query_nodes.shape[0], dtype=np.int16)
    distance = np.zeros(query_nodes.shape[0], dtype=np.float32)
    if texts is None:
        return selected, support, distance

    table: dict[str, list[int]] = defaultdict(list)
    for pos, node in enumerate(anchor_nodes):
        table[texts[int(node)]].append(pos)

    for i, node in enumerate(query_nodes):
        candidates = table.get(texts[int(node)], [])
        for pos in candidates:
            if int(anchor_nodes[pos]) != int(node):
                selected[i] = pos
                support[i] = 8
                break
    return selected, support, distance


def random_anchor_lookup(
    query_nodes: np.ndarray,
    anchor_nodes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = rng.integers(0, anchor_nodes.shape[0], size=query_nodes.shape[0], dtype=np.int64)
    same = anchor_nodes[selected] == query_nodes
    while np.any(same):
        selected[same] = rng.integers(0, anchor_nodes.shape[0], size=int(np.sum(same)), dtype=np.int64)
        same = anchor_nodes[selected] == query_nodes
    return selected, np.ones(query_nodes.shape[0], dtype=np.int16), np.zeros(query_nodes.shape[0], dtype=np.float32)


def quality_metrics(
    target: torch.Tensor,
    labels: torch.Tensor | None,
    query_nodes: np.ndarray,
    anchor_nodes: np.ndarray,
    selected_anchor_pos: np.ndarray,
    support: np.ndarray,
    soft_support: int,
    hard_support: int,
) -> dict[str, float]:
    valid = selected_anchor_pos >= 0
    any_hit = valid & (support > 0)
    usable = valid & (support >= soft_support)
    strong = usable & (support >= hard_support)
    fuzzy = usable & (support < hard_support)

    out = {
        "any_hit_pct": float(any_hit.mean() * 100.0),
        "usable_pct": float(usable.mean() * 100.0),
        "strong_pct": float(strong.mean() * 100.0),
        "fuzzy_pct": float(fuzzy.mean() * 100.0),
        "mean_support": float(support[valid].mean()) if np.any(valid) else float("nan"),
    }

    def masked_quality(mask: np.ndarray) -> tuple[float, float]:
        if not np.any(mask):
            return float("nan"), float("nan")
        q = torch.from_numpy(query_nodes[mask]).long()
        a_nodes = anchor_nodes[selected_anchor_pos[mask]]
        a = torch.from_numpy(a_nodes).long()
        q_emb = F.normalize(target.index_select(0, q), p=2, dim=1)
        a_emb = F.normalize(target.index_select(0, a), p=2, dim=1)
        cosine = float((q_emb * a_emb).sum(dim=1).mean())
        if labels is None:
            label_hit = float("nan")
        else:
            label_hit = float((labels[q] == labels[a]).to(torch.float32).mean() * 100.0)
        return cosine, label_hit

    out["any_cosine"], out["any_label_hit"] = masked_quality(any_hit)
    out["usable_cosine"], out["usable_label_hit"] = masked_quality(usable)
    return out


def profile_dataset(dataset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_seed = int(hashlib.md5(dataset.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.seed + dataset_seed % 100000)
    data = load_graph(dataset)
    target = load_tensor(LLAMA_CACHE[dataset])
    n = int(target.size(0))
    edge_index = normalize_edges(data.edge_index, n)
    labels = load_labels(data, n)
    cheap = load_key_features(dataset, data)[:n]
    cheap = F.normalize(cheap, p=2, dim=1)
    graph_key = graph_context_key(cheap, edge_index, args.self_weight)
    texts = maybe_load_raw_texts(dataset, n)

    query_nodes = sampled_indices(n, args.query_sample, rng)
    anchor_nodes = sampled_indices(n, args.anchor_sample, rng)
    if anchor_nodes.shape[0] < 2:
        raise ValueError(f"Need at least two anchors for {dataset}")

    projections = make_projection(int(cheap.size(1)), args.bits, args.heads, args.seed + 17)
    self_codes = build_codes(cheap, projections, args.feature_chunk)
    graph_codes = build_codes(graph_key, projections, args.feature_chunk)

    rows: list[dict[str, Any]] = []

    method_specs = [
        ("Random anchor", None, "random", 1, 1, 2),
        ("Exact text cache", None, "exact", args.hard_support, args.soft_support, args.hard_support),
        ("Self-only SimHash", self_codes[:1], "hash", 1, 1, 2),
        ("Graph-context SimHash", graph_codes[:1], "hash", 1, 1, 2),
        ("Multi-head graph-context", graph_codes, "hash", args.soft_support, args.soft_support, args.hard_support),
    ]

    for method, codes, kind, usable_support, soft_support, hard_support in method_specs:
        if kind == "random":
            selected, support, dist = random_anchor_lookup(query_nodes, anchor_nodes, rng)
        elif kind == "exact":
            selected, support, dist = exact_text_lookup(texts, query_nodes, anchor_nodes)
        else:
            assert codes is not None
            selected, support, dist = hash_lookup(
                codes[:, query_nodes],
                codes[:, anchor_nodes],
                query_nodes,
                anchor_nodes,
                args.radius,
                args.search_chunk,
            )
        metrics = quality_metrics(
            target,
            labels,
            query_nodes,
            anchor_nodes,
            selected,
            support,
            usable_support,
            hard_support,
        )
        row: dict[str, Any] = {
            "dataset": dataset,
            "abbr": DISPLAY.get(dataset, dataset),
            "method": method,
            "nodes": n,
            "queries": int(query_nodes.shape[0]),
            "anchors": int(anchor_nodes.shape[0]),
            "bits": args.bits,
            "heads": graph_codes.shape[0] if method.startswith("Multi") else 1,
            "radius": args.radius,
            "soft_support": soft_support,
            "hard_support": hard_support,
            "mean_hamming": float(np.mean(dist[support > 0]) / float(args.bits)) if np.any(support > 0) else float("nan"),
            "raw_text_available": texts is not None,
        }
        row.update(metrics)
        if method != "Multi-head graph-context":
            row["strong_pct"] = float("nan")
            row["fuzzy_pct"] = float("nan")
        rows.append(row)

    return rows


def average_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    averaged = []
    for method, cur in grouped.items():
        out: dict[str, Any] = {"method": method, "datasets": len(cur)}
        for key in (
            "any_hit_pct",
            "usable_pct",
            "strong_pct",
            "fuzzy_pct",
            "any_cosine",
            "usable_cosine",
            "any_label_hit",
            "usable_label_hit",
            "mean_support",
        ):
            vals = [float(r[key]) for r in cur if not math.isnan(float(r[key]))]
            out[key] = float(np.mean(vals)) if vals else float("nan")
        averaged.append(out)
    return averaged


def fmt(value: Any, pct: bool = False, digits: int = 2) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val):
        return "-"
    suffix = "%" if pct else ""
    return f"{val:.{digits}f}{suffix}"


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "abbr",
        "method",
        "queries",
        "anchors",
        "any_hit_pct",
        "usable_pct",
        "strong_pct",
        "fuzzy_pct",
        "any_cosine",
        "usable_cosine",
        "any_label_hit",
        "usable_label_hit",
        "mean_support",
        "mean_hamming",
        "raw_text_available",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], avg: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Candidate Discovery Ablation",
        "",
        "This experiment isolates the SimHash/CAM candidate-discovery stage before TSER and residual repair.",
        "Lookup uses cheap online keys; LLaMA embeddings and labels are used only for offline quality measurement.",
        "",
        "## Setup",
        "",
        f"- query sample per dataset: `{args.query_sample}`",
        f"- anchor sample per dataset: `{args.anchor_sample}`",
        f"- SimHash heads: `{args.heads}`",
        f"- bits per head: `{args.bits}`",
        f"- Hamming radius: `{args.radius}`",
        f"- graph-context key: `{args.self_weight:.2f} * self + {1.0 - args.self_weight:.2f} * neighbor_mean`",
        f"- usable multi-head support: `support >= {args.soft_support}`",
        f"- strong multi-head support: `support >= {args.hard_support}`",
        "",
        "## Average Across Datasets",
        "",
        "| Method | AnyHit | Usable | Strong | Fuzzy | CandCos | LabelHit | MeanSupport |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in avg:
        lines.append(
            f"| {row['method']} | {fmt(row['any_hit_pct'], True)} | {fmt(row['usable_pct'], True)} | "
            f"{fmt(row['strong_pct'], True)} | {fmt(row['fuzzy_pct'], True)} | "
            f"{fmt(row['usable_cosine'], False, 4)} | {fmt(row['usable_label_hit'], True)} | "
            f"{fmt(row['mean_support'], False, 2)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Dataset Results",
            "",
            "| Dataset | Method | AnyHit | Usable | Strong | Fuzzy | CandCos | LabelHit | Support |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['abbr']} | {row['method']} | {fmt(row['any_hit_pct'], True)} | "
            f"{fmt(row['usable_pct'], True)} | {fmt(row['strong_pct'], True)} | "
            f"{fmt(row['fuzzy_pct'], True)} | {fmt(row['usable_cosine'], False, 4)} | "
            f"{fmt(row['usable_label_hit'], True)} | {fmt(row['mean_support'], False, 2)} |"
        )

    lines.extend(
        [
            "",
            "## Reading The Metrics",
            "",
            "- `AnyHit`: the lookup found at least one candidate in the sampled anchor pool.",
            "- `Usable`: candidate evidence reaches the method-specific usable threshold. For multi-head, this is `support >= soft_support`.",
            "- `Strong` and `Fuzzy`: the high-support and medium-support regions used by the frontend policy.",
            "- `CandCos`: cosine similarity between query and selected anchor in the LLaMA target embedding space.",
            "- `LabelHit`: offline label agreement sanity check; labels are not used by lookup.",
            "",
            "The key comparison is not only hit rate. Exact text caching has little coverage, single-head SimHash can find candidates but lacks repeated evidence, and multi-head graph-context SimHash exposes support structure that can feed TSER/residual decisions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cora", "pubmed", "arxiv", "wikics", "tape_products", "tape_arxiv23"],
    )
    parser.add_argument("--query-sample", type=int, default=5000)
    parser.add_argument("--anchor-sample", type=int, default=8192)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--soft-support", type=int, default=3)
    parser.add_argument("--hard-support", type=int, default=5)
    parser.add_argument("--self-weight", type=float, default=0.5)
    parser.add_argument("--feature-chunk", type=int, default=8192)
    parser.add_argument("--search-chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "candidate_discovery_ablation",
    )
    args = parser.parse_args()

    invalid = [ds for ds in args.datasets if ds not in DATA_CACHE]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}")
    if args.bits != 16:
        raise ValueError("This profiler currently packs each SimHash head into uint16; use --bits 16.")

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"[CandidateDiscovery] dataset={dataset}")
        rows.extend(profile_dataset(dataset, args))

    avg = average_by_method(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "candidate_discovery_by_dataset.tsv", rows)
    write_tsv(args.output_dir / "candidate_discovery_by_method.tsv", avg)
    write_json(args.output_dir / "candidate_discovery.json", {"rows": rows, "average": avg})
    write_markdown(args.output_dir / "CANDIDATE_DISCOVERY_ABLATION.md", rows, avg, args)
    write_markdown(REPO_ROOT / "docs" / "results" / "CANDIDATE_DISCOVERY_ABLATION.md", rows, avg, args)
    print(f"[Done] wrote {args.output_dir / 'CANDIDATE_DISCOVERY_ABLATION.md'}")


if __name__ == "__main__":
    main()
