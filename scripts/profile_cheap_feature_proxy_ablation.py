#!/usr/bin/env python3
"""Compare cheap semantic proxies for SimHash graph-context keys.

This profiler answers a narrow preprocessing question: whether the current
DistilBERT-L1 proxy can be replaced by a cheaper proxy without destroying
SimHash-CAM candidate quality.  It compares:

* cached DistilBERT-L1 features used by the current pipeline;
* a ModelScope TinyBERT ONNX model;
* a CPU TF-IDF bag-of-words proxy.

The LLaMA target embeddings and labels are used only for offline quality
measurement.  Online lookup uses only the cheap proxy keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
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
from GraphhopSimhash.real_quant import load_tensor_pool  # noqa: E402


DATA_CACHE = {
    "cora": ROOT / "data" / "single_graph" / "Cora" / "cora.pt",
    "pubmed": ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt",
    "arxiv": ROOT / "cache_data" / "arxiv" / "ST" / "processed" / "geometric_data_processed.pt",
    "wikics": ROOT / "cache_data" / "wikics" / "ST" / "processed" / "geometric_data_processed.pt",
}

DISTILBERT_CACHE = {
    "cora": ROOT / "cache_data" / "cora_distilbert_l1.pt",
    "pubmed": ROOT / "cache_data" / "pubmed_distilbert_l1.pt",
    "arxiv": ROOT / "cache_data" / "arxiv_distilbert_l1.pt",
    "wikics": ROOT / "cache_data" / "wikics_distilbert_l1.pt",
}

DISTILBERT_MEASURED_TIME = {
    "cora": 5.012,
    "pubmed": 11.134,
    "arxiv": 89.841,
    "wikics": 8.513,
}

LLAMA_CACHE = {
    ds: ROOT / "cache_data" / f"{ds}_llama2_7b_oracle_W4BFPA8_B128.pt"
    for ds in DATA_CACHE
}

DISPLAY = {
    "cora": "CN",
    "pubmed": "PN",
    "arxiv": "AR",
    "wikics": "WK",
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
    return obj.detach().float().cpu()


def load_graph(dataset: str) -> Any:
    data = load_torch(DATA_CACHE[dataset])
    if not hasattr(data, "edge_index"):
        raise ValueError(f"{DATA_CACHE[dataset]} has no edge_index")
    return data


def load_labels(data: Any, n: int) -> torch.Tensor | None:
    y = getattr(data, "y", None)
    if y is None or not torch.is_tensor(y):
        return None
    y = y.view(-1).detach().cpu().long()
    return y if y.numel() == n else None


def load_target(dataset: str) -> torch.Tensor:
    path = LLAMA_CACHE[dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    return load_tensor_pool(str(path), device="cpu").detach().float().cpu()


def normalize_edges(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    src, dst = edge_index
    keep = (src >= 0) & (src < n) & (dst >= 0) & (dst < n) & (src != dst)
    return edge_index[:, keep]


def load_raw_texts(ds_key: str) -> list[str]:
    ds_key = ds_key.lower()
    if ds_key == "cora":
        data = torch.load(ROOT / "data" / "single_graph" / "Cora" / "cora.pt", map_location="cpu")
        return [str(x) for x in data.raw_texts]
    if ds_key == "pubmed":
        data = torch.load(ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt", map_location="cpu")
        return [str(x) for x in data.raw_texts]
    if ds_key == "arxiv":
        import pandas as pd

        path = ROOT / "data" / "single_graph" / "arxiv"
        nodeidx2paperid = pd.read_csv(path / "nodeidx2paperid.csv.gz", index_col="node idx").sort_index()
        titleabs = pd.read_csv(
            path / "titleabs.tsv",
            sep="\t",
            names=["paper id", "title", "abstract"],
            index_col="paper id",
            on_bad_lines="skip",
            quoting=3,
        )
        titleabs = nodeidx2paperid.join(titleabs, on="paper id").fillna("")
        text = "feature node. paper title and abstract: " + titleabs["title"] + ". " + titleabs["abstract"]
        return text.astype(str).tolist()
    if ds_key == "wikics":
        import functools
        import json as json_lib

        path = ROOT / "data" / "single_graph" / "wikics" / "metadata.json"
        with path.open("r", encoding="utf-8") as f:
            raw_data = json_lib.load(f)
        texts = []
        for node in raw_data["nodes"]:
            content = functools.reduce(lambda x, y: x + " " + y, node["tokens"])
            texts.append(
                (
                    "feature node. wikipedia entry name: "
                    + node["title"]
                    + ". entry content: "
                    + content
                )
                .lower()
                .strip()
            )
        return texts
    raise ValueError(f"unsupported dataset: {ds_key}")


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    return F.normalize(x, p=2, dim=1)


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


def sample_indices(n: int, sample: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, size=min(n, sample), replace=False)


def sample_neighbor_pairs(edge_index: torch.Tensor, n: int, sample: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    src, dst = edge_index
    idx = np.arange(src.numel())
    chosen = rng.choice(idx, size=min(sample, idx.size), replace=idx.size < sample)
    return src[chosen].cpu().numpy(), dst[chosen].cpu().numpy()


def sample_random_pairs(n: int, sample: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.integers(0, n, size=sample, dtype=np.int64)
    v = rng.integers(0, n, size=sample, dtype=np.int64)
    same = u == v
    while np.any(same):
        v[same] = rng.integers(0, n, size=int(np.sum(same)), dtype=np.int64)
        same = u == v
    return u, v


def mean_hamming(codes: np.ndarray, pairs: tuple[np.ndarray, np.ndarray], bits: int) -> float:
    popcount = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    left, right = pairs
    total = np.zeros(left.shape[0], dtype=np.float32)
    for h in range(codes.shape[0]):
        total += popcount[np.bitwise_xor(codes[h, left], codes[h, right])].astype(np.float32)
    return float(np.mean(total / float(bits * codes.shape[0])))


def median_hamming(codes: np.ndarray, pairs: tuple[np.ndarray, np.ndarray], bits: int) -> float:
    popcount = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    left, right = pairs
    total = np.zeros(left.shape[0], dtype=np.float32)
    for h in range(codes.shape[0]):
        total += popcount[np.bitwise_xor(codes[h, left], codes[h, right])].astype(np.float32)
    return float(np.median(total / float(bits * codes.shape[0])))


def quality_metrics(
    target: torch.Tensor,
    labels: torch.Tensor | None,
    query_nodes: np.ndarray,
    anchor_nodes: np.ndarray,
    selected_anchor_pos: np.ndarray,
    support: np.ndarray,
    soft_support: int,
    hard_support: int,
    valid_cosine_threshold: float,
) -> dict[str, float]:
    valid = selected_anchor_pos >= 0
    usable = valid & (support >= soft_support)
    strong = usable & (support >= hard_support)
    fuzzy = usable & (support < hard_support)

    out = {
        "lookup_yield_pct": float(usable.mean() * 100.0),
        "strong_pct": float(strong.mean() * 100.0),
        "fuzzy_pct": float(fuzzy.mean() * 100.0),
        "mean_support": float(support[usable].mean()) if np.any(usable) else float("nan"),
    }
    if not np.any(usable):
        out.update(
            {
                "valid_anchor_pct": float("nan"),
                "valid_yield_pct": 0.0,
                "embedding_cosine": float("nan"),
                "label_agreement_pct": float("nan"),
            }
        )
        return out

    q = torch.from_numpy(query_nodes[usable]).long()
    a_nodes = anchor_nodes[selected_anchor_pos[usable]]
    a = torch.from_numpy(a_nodes).long()
    q_emb = F.normalize(target.index_select(0, q), p=2, dim=1)
    a_emb = F.normalize(target.index_select(0, a), p=2, dim=1)
    pair_cos = (q_emb * a_emb).sum(dim=1)
    out["valid_anchor_pct"] = float((pair_cos >= valid_cosine_threshold).float().mean() * 100.0)
    out["valid_yield_pct"] = out["lookup_yield_pct"] * out["valid_anchor_pct"] / 100.0
    out["embedding_cosine"] = float(pair_cos.mean())
    if labels is not None:
        out["label_agreement_pct"] = float((labels[q] == labels[a]).float().mean() * 100.0)
    else:
        out["label_agreement_pct"] = float("nan")
    return out


def distilbert_features(dataset: str) -> tuple[torch.Tensor, float, str]:
    path = DISTILBERT_CACHE[dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    return load_tensor(path), DISTILBERT_MEASURED_TIME.get(dataset, float("nan")), "RTX4090 cached measurement"


def tinybert_onnx_features(texts: list[str], model_dir: Path, max_length: int, batch_size: int) -> tuple[torch.Tensor, float, str]:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model_path = model_dir / "onnx" / "model_quantized.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = []
    t0 = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
        feed = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded["token_type_ids"].astype(np.int64),
        }
        last_hidden = session.run(["last_hidden_state"], feed)[0]
        outputs.append(torch.from_numpy(last_hidden[:, 0, :]).float())
    elapsed = time.perf_counter() - t0
    return torch.cat(outputs, dim=0), elapsed, "ModelScope TinyBERT ONNX CPU"


def bow_tfidf_features(texts: list[str], max_features: int) -> tuple[torch.Tensor, float, str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=(1, 1),
        min_df=1,
        dtype=np.float32,
    )
    t0 = time.perf_counter()
    x = vec.fit_transform(texts)
    dense = x.toarray().astype(np.float32, copy=False)
    elapsed = time.perf_counter() - t0
    return torch.from_numpy(dense), elapsed, "CPU TF-IDF BoW"


def evaluate_cora_30_drop(
    graph_key: torch.Tensor,
    args: argparse.Namespace,
    seed: int = 42,
) -> dict[str, float]:
    from types import SimpleNamespace

    old_cwd = os.getcwd()
    removed_paths = []
    for path in (str(REPO_ROOT), old_cwd, ""):
        while path in sys.path:
            sys.path.remove(path)
            removed_paths.append(path)
    os.chdir(str(ROOT))
    try:
        from GraphhopSimhash.data import load_run_state
        from GraphhopSimhash.runner import evaluate_gnn_embeddings, train_baseline_model
    finally:
        os.chdir(old_cwd)
        for path in reversed(removed_paths):
            sys.path.insert(0, path)

    run_args = SimpleNamespace(
        llm_name="ST",
        emb_dim=768,
        radius=2,
        max_test=None,
        standard_eval_baseline=False,
        hash_view="mix",
        hash_mix_weights=[0.30, 0.70, 0.0],
        sketch_bits=14,
        controller_seed=seed,
        run_seed=seed,
        score_rarity_bits=12,
        score_rarity_seed=12345,
        score_propagation_weight=3,
        score_graph_context_weight=1,
        score_low_unique_weight=1,
    )
    old_exec_cwd = os.getcwd()
    os.chdir(str(ROOT))
    try:
        _conf, data, _verify_features, device = load_run_state("cora", run_args, seed)
        reference = load_target("cora").to(device)
        data.x = reference
        model, base_acc, baseline_hidden, _logits = train_baseline_model(data, run_args, device)
        baseline_hidden = baseline_hidden.detach()

        projections = make_projection(int(graph_key.size(1)), args.bits, args.heads, args.seed + 17)
        codes = build_codes(graph_key, projections, args.feature_chunk)
        nodes = np.arange(int(graph_key.size(0)), dtype=np.int64)
        selected_pos, support, dist = hash_lookup(
            codes[:, nodes],
            codes[:, nodes],
            nodes,
            nodes,
            args.radius,
            args.search_chunk,
        )
        usable = support >= args.soft_support
        score = support.astype(np.float32) * 1000.0 - dist.astype(np.float32)
        score[~usable] = -1e9
        k = int(round(args.eval_reuse * len(nodes)))
        chosen = np.argsort(-score)[:k]
        chosen = chosen[score[chosen] > -1e8]
        reuse_hidden = baseline_hidden.clone()
        if chosen.size:
            idx = torch.from_numpy(chosen).long().to(device)
            anchors = torch.from_numpy(nodes[selected_pos[chosen]]).long().to(device)
            reuse_hidden[idx] = baseline_hidden[anchors]
        reuse_acc = float(evaluate_gnn_embeddings(model, data, reuse_hidden))
    finally:
        os.chdir(old_exec_cwd)
    return {
        "cora_base_acc": float(base_acc),
        "cora_reuse_acc": reuse_acc,
        "cora_drop_pct": float((base_acc - reuse_acc) * 100.0),
        "cora_reuse_pct": float(chosen.size / len(nodes) * 100.0),
    }


def profile_source(dataset: str, source: str, features: torch.Tensor, feature_time: float, note: str, args: argparse.Namespace) -> dict[str, Any]:
    data = load_graph(dataset)
    target = load_target(dataset)
    n = int(target.size(0))
    labels = load_labels(data, n)
    edge_index = normalize_edges(data.edge_index, n)
    cheap = normalize_features(features[:n])
    graph_key = graph_context_key(cheap, edge_index, args.self_weight)

    projections = make_projection(int(graph_key.size(1)), args.bits, args.heads, args.seed + 17)
    codes = build_codes(graph_key, projections, args.feature_chunk)

    rng = np.random.default_rng(args.seed + abs(hash(dataset + source)) % 100000)
    neighbor_pairs = sample_neighbor_pairs(edge_index, n, args.pair_sample, rng)
    random_pairs = sample_random_pairs(n, args.pair_sample, rng)
    neigh_med = median_hamming(codes, neighbor_pairs, args.bits)
    rand_med = median_hamming(codes, random_pairs, args.bits)
    neigh_mean = mean_hamming(codes, neighbor_pairs, args.bits)
    rand_mean = mean_hamming(codes, random_pairs, args.bits)

    query_nodes = sample_indices(n, args.query_sample, rng)
    anchor_nodes = sample_indices(n, args.anchor_sample, rng)
    selected, support, _dist = hash_lookup(
        codes[:, query_nodes],
        codes[:, anchor_nodes],
        query_nodes,
        anchor_nodes,
        args.radius,
        args.search_chunk,
    )
    q = quality_metrics(
        target,
        labels,
        query_nodes,
        anchor_nodes,
        selected,
        support,
        args.soft_support,
        args.hard_support,
        args.valid_cosine_threshold,
    )
    row: dict[str, Any] = {
        "dataset": dataset,
        "task": DISPLAY[dataset],
        "source": source,
        "nodes": n,
        "feature_time_s": float(feature_time),
        "source_note": note,
        "neighbor_ham_median": neigh_med,
        "random_ham_median": rand_med,
        "median_gap": rand_med - neigh_med,
        "neighbor_ham_mean": neigh_mean,
        "random_ham_mean": rand_mean,
        "mean_gap": rand_mean - neigh_mean,
    }
    row.update(q)
    if args.eval_cora_drop and dataset == "cora":
        row.update(evaluate_cora_30_drop(graph_key, args))
    return row


def fmt(x: Any, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if math.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.output_dir / "cheap_feature_proxy_ablation.tsv"
    md = args.output_dir / "cheap_feature_proxy_ablation.md"
    js = args.output_dir / "cheap_feature_proxy_ablation.json"
    cols = [
        "task",
        "dataset",
        "source",
        "nodes",
        "feature_time_s",
        "median_gap",
        "mean_gap",
        "lookup_yield_pct",
        "valid_anchor_pct",
        "valid_yield_pct",
        "embedding_cosine",
        "label_agreement_pct",
        "cora_reuse_pct",
        "cora_drop_pct",
        "source_note",
    ]
    with tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    js.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# Cheap Semantic Proxy Ablation",
        "",
        "This profiling compares inputs used to build graph-context SimHash keys.",
        "LLaMA embeddings and labels are used only to score candidate quality.",
        "",
        f"- SimHash: `{args.heads}` heads x `{args.bits}` bits, radius `{args.radius}`",
        f"- Graph key: `{args.self_weight:.2f} * self + {1.0 - args.self_weight:.2f} * neighbor_mean`",
        f"- TinyBERT source: `{args.tinybert_model_dir}`",
        f"- TinyBERT max length: `{args.tinybert_max_length}`",
        f"- BoW max features: `{args.bow_max_features}`",
        "",
        "| Task | Source | Feat. Time (s) | Median Gap | Lookup Yield | Valid Anchor | Valid Yield | Emb. Cos. | Label Agree. | Cora 30% Drop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['source']} | {fmt(row['feature_time_s'])} | "
            f"{fmt(row['median_gap'], 4)} | {fmt(row['lookup_yield_pct'], 2)}% | "
            f"{fmt(row['valid_anchor_pct'], 2)}% | {fmt(row['valid_yield_pct'], 2)}% | "
            f"{fmt(row['embedding_cosine'], 4)} | {fmt(row['label_agreement_pct'], 2)}% | "
            f"{fmt(row.get('cora_drop_pct', float('nan')), 2)}% |"
        )
    lines.extend(["", f"Raw TSV: `{tsv}`"])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "arxiv", "wikics"])
    parser.add_argument("--sources", nargs="+", default=["distilbert_l1", "tinybert_onnx", "bow_tfidf"])
    parser.add_argument("--tinybert-model-dir", type=Path, default=ROOT / "models" / "modelscope_TinyBERT_General_4L_312D_ONNX")
    parser.add_argument("--tinybert-max-length", type=int, default=64)
    parser.add_argument("--tinybert-batch-size", type=int, default=256)
    parser.add_argument("--bow-max-features", type=int, default=768)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--soft-support", type=int, default=3)
    parser.add_argument("--hard-support", type=int, default=5)
    parser.add_argument("--valid-cosine-threshold", type=float, default=0.8)
    parser.add_argument("--self-weight", type=float, default=0.5)
    parser.add_argument("--feature-chunk", type=int, default=8192)
    parser.add_argument("--search-chunk", type=int, default=256)
    parser.add_argument("--query-sample", type=int, default=5000)
    parser.add_argument("--anchor-sample", type=int, default=8192)
    parser.add_argument("--pair-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-cora-drop", action="store_true")
    parser.add_argument("--eval-reuse", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "cheap_feature_proxy_ablation")
    args = parser.parse_args()

    if args.bits != 16:
        raise ValueError("This script packs SimHash heads into uint16; use --bits 16.")

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"[Dataset] {dataset}", flush=True)
        texts: list[str] | None = None
        for source in args.sources:
            print(f"  [Source] {source}", flush=True)
            if source == "distilbert_l1":
                features, elapsed, note = distilbert_features(dataset)
            elif source == "tinybert_onnx":
                if texts is None:
                    texts = load_raw_texts(dataset)
                features, elapsed, note = tinybert_onnx_features(
                    texts,
                    args.tinybert_model_dir,
                    args.tinybert_max_length,
                    args.tinybert_batch_size,
                )
            elif source == "bow_tfidf":
                if texts is None:
                    texts = load_raw_texts(dataset)
                features, elapsed, note = bow_tfidf_features(texts, args.bow_max_features)
            else:
                raise ValueError(f"unknown source: {source}")
            rows.append(profile_source(dataset, source, features, elapsed, note, args))
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
