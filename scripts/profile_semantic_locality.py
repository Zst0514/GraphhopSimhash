#!/usr/bin/env python3
"""Profile semantic locality for TAG/GFM bypass motivation.

The experiment compares graph-neighbor node pairs with random node pairs.
For each dataset it reports:

* embedding cosine similarity
* SimHash normalized Hamming distance

The goal is to quantify the raw bypass opportunity before applying the
SimHash/CAM/TSER/residual policy.
"""

from __future__ import annotations

import argparse
import json
import math
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


ST_CACHE = {
    "cora": ROOT / "cache_data" / "Cora" / "ST" / "processed" / "geometric_data_processed.pt",
    "pubmed": ROOT / "cache_data" / "Pubmed" / "ST" / "processed" / "geometric_data_processed.pt",
    "arxiv": ROOT / "cache_data" / "arxiv" / "ST" / "processed" / "geometric_data_processed.pt",
    "wikics": ROOT / "cache_data" / "wikics" / "ST" / "processed" / "geometric_data_processed.pt",
}

LLAMA_CACHE = {
    "cora": ROOT / "cache_data" / "cora_llama2_7b_oracle_W4A8.pt",
    "pubmed": ROOT / "cache_data" / "pubmed_llama2_7b_oracle_W4A8.pt",
    "arxiv": ROOT / "cache_data" / "arxiv_llama2_7b_oracle_W4A8.pt",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_torch_data(path: Path) -> Any:
    obj = torch.load(path, map_location="cpu")
    return obj[0] if isinstance(obj, tuple) else obj


def load_embedding_tensor(dataset: str, source: str) -> torch.Tensor:
    ds = dataset.lower()
    if ds == "products":
        data, _ = load_products()
        return data.x.detach().to(torch.float32).cpu()

    if source == "st":
        path = ST_CACHE.get(ds)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing ST cache for {dataset}: {path}")
        data = load_torch_data(path)
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError(f"{path} does not contain data.x")
        return x.detach().to(torch.float32).cpu()

    if source == "llama":
        path = LLAMA_CACHE.get(ds)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing LLaMA W4A8 cache for {dataset}: {path}")
        x = torch.load(path, map_location="cpu")
        if isinstance(x, dict):
            for key in ("x", "embeddings", "features"):
                if key in x and torch.is_tensor(x[key]):
                    x = x[key]
                    break
        if not torch.is_tensor(x):
            raise TypeError(f"{path} did not contain an embedding tensor")
        return x.detach().to(torch.float32).cpu()

    raise ValueError(f"Unsupported embedding source: {source}")


def load_edge_index(dataset: str) -> torch.Tensor:
    ds = dataset.lower()
    if ds == "products":
        data, _ = load_products()
        return data.edge_index.detach().cpu().long()

    if ds == "cora":
        path = ROOT / "data" / "single_graph" / "Cora" / "cora.pt"
    elif ds == "pubmed":
        path = ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt"
    elif ds == "arxiv":
        path = ST_CACHE["arxiv"]
    elif ds == "wikics":
        path = ST_CACHE["wikics"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    data = load_torch_data(path)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None and isinstance(data, dict):
        edge_index = data.get("edge_index")
    if edge_index is None:
        raise ValueError(f"Could not find edge_index for {dataset}")
    return edge_index.detach().cpu().long()


def load_labels(dataset: str) -> torch.Tensor | None:
    ds = dataset.lower()
    try:
        if ds == "products":
            data, _ = load_products()
            return data.y.view(-1).detach().cpu().long()
        if ds == "cora":
            data = load_torch_data(ROOT / "data" / "single_graph" / "Cora" / "cora.pt")
        elif ds == "pubmed":
            data = load_torch_data(ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt")
        elif ds == "arxiv":
            data = load_torch_data(ST_CACHE["arxiv"])
        elif ds == "wikics":
            data = load_torch_data(ST_CACHE["wikics"])
        else:
            return None
        y = getattr(data, "y", None)
        if y is None:
            return None
        return y.view(-1).detach().cpu().long()
    except Exception:
        return None


def load_products():
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name="ogbn-products", root=str(ROOT / "data"))
    return dataset[0], dataset.get_idx_split()


def sample_neighbor_pairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    sample_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    edge_index = edge_index.cpu().long()
    src, dst = edge_index[0], edge_index[1]
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes) & (src != dst)
    valid_idx = valid.nonzero(as_tuple=False).view(-1)
    if valid_idx.numel() == 0:
        raise ValueError("No valid non-self edges available")
    size = min(sample_pairs, valid_idx.numel())
    replace = valid_idx.numel() < sample_pairs
    chosen = rng.choice(valid_idx.numpy(), size=size, replace=replace)
    return src[chosen].numpy(), dst[chosen].numpy()


def sample_random_pairs(
    num_nodes: int,
    sample_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.integers(0, num_nodes, size=sample_pairs, dtype=np.int64)
    v = rng.integers(0, num_nodes, size=sample_pairs, dtype=np.int64)
    same = u == v
    while np.any(same):
        v[same] = rng.integers(0, num_nodes, size=int(np.sum(same)), dtype=np.int64)
        same = u == v
    return u, v


def sample_same_label_pairs(
    y: torch.Tensor | None,
    sample_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    if y is None:
        return None
    labels = y.cpu().numpy()
    label_to_nodes: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        nodes = np.where(labels == label)[0]
        if nodes.size >= 2:
            label_to_nodes[int(label)] = nodes
    if not label_to_nodes:
        return None
    label_keys = np.array(list(label_to_nodes.keys()), dtype=np.int64)
    counts = np.array([label_to_nodes[int(k)].size for k in label_keys], dtype=np.float64)
    probs = counts / counts.sum()

    u = np.empty(sample_pairs, dtype=np.int64)
    v = np.empty(sample_pairs, dtype=np.int64)
    chosen_labels = rng.choice(label_keys, size=sample_pairs, replace=True, p=probs)
    for i, label in enumerate(chosen_labels):
        nodes = label_to_nodes[int(label)]
        pair = rng.choice(nodes, size=2, replace=False)
        u[i], v[i] = pair[0], pair[1]
    return u, v


def make_projection(dim: int, bits: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    proj = torch.randn(dim, bits, generator=gen, dtype=torch.float32)
    proj = F.normalize(proj, p=2, dim=0)
    return proj


def pair_metrics(
    x: torch.Tensor,
    pairs: tuple[np.ndarray, np.ndarray],
    proj: torch.Tensor,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = pairs
    n = left.shape[0]
    cosine_chunks: list[torch.Tensor] = []
    hamming_chunks: list[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        li = torch.from_numpy(left[start:end]).long()
        ri = torch.from_numpy(right[start:end]).long()
        a = x.index_select(0, li).to(torch.float32)
        b = x.index_select(0, ri).to(torch.float32)
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        cosine_chunks.append((a * b).sum(dim=1).cpu())

        a_bits = (a @ proj) >= 0
        b_bits = (b @ proj) >= 0
        hamming_chunks.append(torch.logical_xor(a_bits, b_bits).sum(dim=1).to(torch.float32).cpu())

    cosine = torch.cat(cosine_chunks).numpy()
    hamming = torch.cat(hamming_chunks).numpy() / float(proj.size(1))
    return cosine, hamming


def summarize(values: np.ndarray, higher_is_better: bool) -> dict[str, float]:
    qs = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    out = {
        "mean": float(np.mean(values)),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "p95": float(qs[5]),
    }
    if higher_is_better:
        out.update(
            {
                "frac_ge_0.50": float(np.mean(values >= 0.50) * 100.0),
                "frac_ge_0.70": float(np.mean(values >= 0.70) * 100.0),
                "frac_ge_0.80": float(np.mean(values >= 0.80) * 100.0),
            }
        )
    else:
        out.update(
            {
                "frac_le_0.20": float(np.mean(values <= 0.20) * 100.0),
                "frac_le_0.25": float(np.mean(values <= 0.25) * 100.0),
                "frac_le_0.30": float(np.mean(values <= 0.30) * 100.0),
            }
        )
    return out


def cdf_rows(dataset: str, pair_type: str, metric: str, values: np.ndarray) -> list[dict[str, Any]]:
    if metric == "cosine":
        thresholds = np.linspace(-1.0, 1.0, 201)
    else:
        thresholds = np.linspace(0.0, 1.0, 201)
    sorted_values = np.sort(values)
    frac = np.searchsorted(sorted_values, thresholds, side="right") / float(values.size)
    return [
        {
            "dataset": dataset,
            "pair_type": pair_type,
            "metric": metric,
            "threshold": float(t),
            "cdf": float(f),
        }
        for t, f in zip(thresholds, frac)
    ]


def write_cdf_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("dataset\tpair_type\tmetric\tthreshold\tcdf\n")
        for row in rows:
            f.write(
                f"{row['dataset']}\t{row['pair_type']}\t{row['metric']}\t"
                f"{row['threshold']:.6f}\t{row['cdf']:.8f}\n"
            )


def maybe_plot(path: Path, cdf: list[dict[str, Any]], datasets: list[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    pair_types = ["neighbor", "random", "same_label"]
    colors = {"neighbor": "#1f77b4", "random": "#7f7f7f", "same_label": "#2ca02c"}
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, max(2.4 * n_rows, 3.2)), squeeze=False)
    for row_idx, ds in enumerate(datasets):
        for col_idx, metric in enumerate(("cosine", "hamming")):
            ax = axes[row_idx][col_idx]
            for pair_type in pair_types:
                rows = [
                    r
                    for r in cdf
                    if r["dataset"] == ds and r["metric"] == metric and r["pair_type"] == pair_type
                ]
                if not rows:
                    continue
                x_vals = [r["threshold"] for r in rows]
                y_vals = [r["cdf"] for r in rows]
                label = pair_type.replace("_", " ")
                ax.plot(x_vals, y_vals, label=label, color=colors[pair_type], linewidth=1.8)
            ax.set_title(f"{ds} {metric}")
            ax.set_ylabel("CDF")
            if metric == "cosine":
                ax.set_xlabel("cosine similarity")
            else:
                ax.set_xlabel("normalized Hamming distance")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Semantic Locality Profile",
        "",
        "This profiling result is for Motivation: it measures the raw locality opportunity before applying the final reuse policy.",
        "",
        "Pair types:",
        "",
        "```text",
        "neighbor: graph edge pairs",
        "random: uniformly sampled unrelated node pairs",
        "same_label: same-class pairs when labels are available",
        "```",
        "",
        f"Embedding source: `{payload['embedding_source']}`",
        f"SimHash bits: `{payload['hash_bits']}`",
        f"Sample pairs per type: `{payload['sample_pairs']}`",
        "",
        "## Summary",
        "",
        "| Dataset | Pair | Cos mean | Cos p50 | Cos >=0.50 | Cos >=0.70 | Ham mean | Ham <=0.25 | Ham <=0.30 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ds in payload["datasets"]:
        ds_summary = summary[ds]
        for pair_type in ("neighbor", "random", "same_label"):
            if pair_type not in ds_summary:
                continue
            item = ds_summary[pair_type]
            cos = item["cosine"]
            ham = item["hamming"]
            lines.append(
                f"| {ds} | {pair_type} | {cos['mean']:.4f} | {cos['p50']:.4f} | "
                f"{cos['frac_ge_0.50']:.1f}% | {cos['frac_ge_0.70']:.1f}% | "
                f"{ham['mean']:.4f} | {ham['frac_le_0.25']:.1f}% | {ham['frac_le_0.30']:.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Neighbor vs Random Gap",
            "",
            "| Dataset | Cos mean lift | Cos >=0.50 lift | Hamming mean reduction | Hamming <=0.30 lift |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for ds in payload["datasets"]:
        ds_summary = summary[ds]
        if "neighbor" not in ds_summary or "random" not in ds_summary:
            continue
        n_cos = ds_summary["neighbor"]["cosine"]
        r_cos = ds_summary["random"]["cosine"]
        n_ham = ds_summary["neighbor"]["hamming"]
        r_ham = ds_summary["random"]["hamming"]
        lines.append(
            f"| {ds} | {n_cos['mean'] - r_cos['mean']:.4f} | "
            f"{n_cos['frac_ge_0.50'] - r_cos['frac_ge_0.50']:.1f}% | "
            f"{r_ham['mean'] - n_ham['mean']:.4f} | "
            f"{n_ham['frac_le_0.30'] - r_ham['frac_le_0.30']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary JSON: `{payload['summary_json']}`",
            f"- CDF TSV: `{payload['cdf_tsv']}`",
            f"- CDF figure: `{payload['cdf_png']}`",
            "",
            "Interpretation:",
            "",
            "```text",
            "If neighbor pairs have higher cosine and lower SimHash Hamming distance than random pairs,",
            "the graph-text workload has semantic locality that a fuzzy bypass path can exploit.",
            "The final safe reuse rate should be reported separately from this raw opportunity profile.",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def profile_dataset(
    dataset: str,
    source: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = load_embedding_tensor(dataset, source)
    edge_index = load_edge_index(dataset)
    labels = load_labels(dataset)
    num_nodes = int(x.size(0))
    if edge_index.max().item() >= num_nodes:
        mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, mask]

    proj = make_projection(int(x.size(1)), args.hash_bits, args.seed + 101)
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "neighbor": sample_neighbor_pairs(edge_index, num_nodes, args.sample_pairs, rng),
        "random": sample_random_pairs(num_nodes, args.sample_pairs, rng),
    }
    same_label = sample_same_label_pairs(labels, args.sample_pairs, rng)
    if same_label is not None:
        pairs["same_label"] = same_label

    summary: dict[str, Any] = {
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.size(1)),
        "feature_dim": int(x.size(1)),
        "embedding_source": "products_proxy" if dataset.lower() == "products" else source,
    }
    cdf: list[dict[str, Any]] = []
    for pair_type, pair in pairs.items():
        cosine, hamming = pair_metrics(x, pair, proj, args.chunk_size)
        summary[pair_type] = {
            "sample_pairs": int(cosine.size),
            "cosine": summarize(cosine, higher_is_better=True),
            "hamming": summarize(hamming, higher_is_better=False),
        }
        cdf.extend(cdf_rows(dataset, pair_type, "cosine", cosine))
        cdf.extend(cdf_rows(dataset, pair_type, "hamming", hamming))
    return summary, cdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "arxiv"])
    parser.add_argument("--embedding-source", choices=["st", "llama"], default="st")
    parser.add_argument("--sample-pairs", type=int, default=50_000)
    parser.add_argument("--hash-bits", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "semantic_locality_profile")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir / args.embedding_source
    summary: dict[str, Any] = {}
    all_cdf: list[dict[str, Any]] = []
    completed: list[str] = []
    for dataset in args.datasets:
        ds = dataset.lower()
        source = args.embedding_source
        if ds == "products":
            source = "products_proxy"
        print(f"[SemanticLocality] dataset={ds} source={source}")
        dataset_summary, dataset_cdf = profile_dataset(ds, args.embedding_source, args, rng)
        summary[ds] = dataset_summary
        all_cdf.extend(dataset_cdf)
        completed.append(ds)

    cdf_tsv = output_dir / "cdf.tsv"
    summary_json = output_dir / "summary.json"
    cdf_png = output_dir / "cdf.png"
    summary_md = output_dir / "summary.md"
    payload = {
        "datasets": completed,
        "embedding_source": args.embedding_source,
        "sample_pairs": args.sample_pairs,
        "hash_bits": args.hash_bits,
        "summary": summary,
        "summary_json": str(summary_json),
        "cdf_tsv": str(cdf_tsv),
        "cdf_png": str(cdf_png),
    }
    write_cdf_tsv(cdf_tsv, all_cdf)
    write_json(summary_json, payload)
    maybe_plot(cdf_png, all_cdf, completed)
    write_markdown(summary_md, payload)

    print(f"wrote {summary_json}")
    print(f"wrote {cdf_tsv}")
    print(f"wrote {summary_md}")
    if cdf_png.exists():
        print(f"wrote {cdf_png}")


if __name__ == "__main__":
    main()
