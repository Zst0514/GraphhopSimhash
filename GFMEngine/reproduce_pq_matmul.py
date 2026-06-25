#!/usr/bin/env python3
"""Algorithm-level reproduction of GFMEngine-style PQ MatMul.

The reproduced operation is:

    X @ W ~= sum_m activation_book[m, id_m(X_m)]

where every input row is split into M subvectors, each subvector is mapped to
the closest centroid, and each centroid's contribution to W is precomputed in
an activation book.

If torch is available, --pool_path can load a real GraphHopSimhash embedding
pool.  Without torch, the script runs a deterministic synthetic microbench and
still reports the exact online PQ operation and byte counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent


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


def load_pool(path: Path, rows: int, dims: int) -> tuple[np.ndarray, str]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required to load a .pt pool") from exc
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, (tuple, list)):
        obj = obj[0]
    if not hasattr(obj, "detach"):
        raise TypeError(f"{path} did not contain a tensor")
    x = obj.detach().cpu().float().numpy()
    x = x[:rows, :dims]
    return np.asarray(x, dtype=np.float32), "real_pool"


def synthetic_pool(rows: int, dims: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    base = rng.standard_normal((rows, dims), dtype=np.float32)
    # Add low-rank structure so PQ has non-trivial centroids instead of pure noise.
    rank = min(16, dims)
    left = rng.standard_normal((rows, rank), dtype=np.float32)
    right = rng.standard_normal((rank, dims), dtype=np.float32)
    x = 0.7 * base + 0.3 * (left @ right) / math.sqrt(rank)
    return np.asarray(x, dtype=np.float32), "synthetic_lowrank_gaussian"


def squared_distances(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(c * c, axis=1, keepdims=True).T
    return x2 + c2 - 2.0 * (x @ c.T)


def train_kmeans(
    x: np.ndarray,
    k: int,
    *,
    iters: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if x.shape[0] < k:
        reps = int(math.ceil(k / x.shape[0]))
        pool = np.tile(x, (reps, 1))[:k]
        return pool.astype(np.float32, copy=True)
    init = rng.choice(x.shape[0], size=k, replace=False)
    centroids = x[init].astype(np.float32, copy=True)
    for _ in range(iters):
        ids = np.argmin(squared_distances(x, centroids), axis=1)
        sums = np.zeros_like(centroids)
        counts = np.bincount(ids, minlength=k).astype(np.float32)
        np.add.at(sums, ids, x)
        empty = counts == 0
        counts[empty] = 1.0
        centroids = sums / counts[:, None]
        if np.any(empty):
            centroids[empty] = x[rng.choice(x.shape[0], size=int(np.sum(empty)), replace=True)]
    return centroids.astype(np.float32, copy=False)


def pq_matmul_trial(
    *,
    x: np.ndarray,
    rows_eval: int,
    out_features: int,
    subvectors: int,
    centroids: int,
    train_rows: int,
    iters: int,
    seed: int,
    activation_book_bits: int,
    index_bits: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 999)
    rows, dims = x.shape
    if dims % subvectors != 0:
        raise ValueError(f"dims={dims} must be divisible by subvectors={subvectors}")
    dsub = dims // subvectors
    train = x[: min(train_rows, rows)]
    eval_x = x[-min(rows_eval, rows) :]
    weight = rng.standard_normal((dims, out_features), dtype=np.float32) / math.sqrt(dims)

    approx = np.zeros((eval_x.shape[0], out_features), dtype=np.float32)
    ids_all = []
    for m in range(subvectors):
        lo = m * dsub
        hi = lo + dsub
        c = train_kmeans(train[:, lo:hi], centroids, iters=iters, seed=seed + m)
        ids = np.argmin(squared_distances(eval_x[:, lo:hi], c), axis=1)
        book = c @ weight[lo:hi, :]
        approx += book[ids]
        ids_all.append(ids.astype(np.uint16))

    exact = eval_x @ weight
    err = approx - exact
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sqrt(np.mean(exact * exact)))
    rel_rmse = rmse / max(1.0e-12, denom)
    dot = np.sum(approx * exact, axis=1)
    cos = dot / (np.linalg.norm(approx, axis=1) * np.linalg.norm(exact, axis=1) + 1.0e-12)

    n = float(eval_x.shape[0])
    dense_macs = n * float(dims) * float(out_features)
    search_ops = n * float(centroids) * float(dims)
    query_add_ops = n * float(subvectors) * float(out_features)
    activation_book_bytes = n * float(subvectors) * float(out_features) * float(activation_book_bits) / 8.0
    index_bytes = n * float(subvectors) * float(index_bits) / 8.0
    codebook_bytes = float(subvectors) * float(centroids) * float(dsub) * 8.0 / 8.0
    offline_book_bytes = float(subvectors) * float(centroids) * float(out_features) * float(activation_book_bits) / 8.0
    return {
        "rows_eval": int(eval_x.shape[0]),
        "in_features": int(dims),
        "out_features": int(out_features),
        "subvectors": int(subvectors),
        "centroids": int(centroids),
        "dsub": int(dsub),
        "relative_rmse": rel_rmse,
        "mean_cosine": float(np.mean(cos)),
        "p05_cosine": float(np.quantile(cos, 0.05)),
        "dense_macs": dense_macs,
        "search_ops": search_ops,
        "query_add_ops": query_add_ops,
        "online_compute_ops": search_ops + query_add_ops,
        "online_compute_vs_dense": (search_ops + query_add_ops) / dense_macs,
        "activation_book_bytes": activation_book_bytes,
        "index_bytes": index_bytes,
        "codebook_bytes": codebook_bytes,
        "offline_activation_book_bytes": offline_book_bytes,
        "activation_book_bytes_per_row": activation_book_bytes / max(1.0, n),
    }


def render_report(rows: list[dict[str, Any]], source: str, args: argparse.Namespace) -> str:
    lines = [
        "# GFMEngine PQ MatMul Reproduction",
        "",
        "## Scope",
        "",
        "This is an algorithm-level reproduction of GFMEngine's PQ-based MatMul, not a formula-only estimate.",
        "It trains PQ centroids, builds activation books, and evaluates `X @ W` through activation-book lookup and summation.",
        "",
        f"- Source: `{source}`.",
        f"- Rows generated/loaded: `{args.rows}`; eval rows: `{args.rows_eval}`.",
        f"- Input dim: `{args.in_features}`; output dim: `{args.out_features}`; centroids: `{args.centroids}`.",
        "",
        "## Results",
        "",
        "| M | dsub | Rel RMSE | Mean Cosine | Online Compute / Dense | Activation-Book Bytes / Row | Offline Book Size |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['subvectors']} | {row['dsub']} | {row['relative_rmse']:.4f} | "
            f"{row['mean_cosine']:.4f} | {row['online_compute_vs_dense']:.4f}x | "
            f"{row['activation_book_bytes_per_row'] / 1024.0:.2f} KiB | "
            f"{row['offline_activation_book_bytes'] / (1024.0 ** 2):.2f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Search compute is independent of `M` for fixed `K` and `D`: `rows * K * D`.",
            "- Activation-book traffic grows linearly with `M`: `rows * M * out_features * bits`.",
            "- This is why a small `M` can make GFMEngine-PQ look strong, while a realistic larger `M` can become memory-bound.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_path", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--rows_eval", type=int, default=512)
    parser.add_argument("--train_rows", type=int, default=1536)
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=4096)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--subvectors", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation_book_bits", type=int, default=8)
    parser.add_argument("--index_bits", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OFA_ROOT / "output" / "gfmengine_pq_reproduction",
    )
    parser.add_argument(
        "--repo_report",
        type=Path,
        default=REPO_ROOT / "GFMEngine" / "results" / "GFMENGINE_PQ_MATMUL_REPRO.md",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    source = "synthetic_lowrank_gaussian"
    if args.pool_path is not None:
        try:
            x, source = load_pool(args.pool_path, args.rows, args.in_features)
        except Exception as exc:
            x, source = synthetic_pool(args.rows, args.in_features, rng)
            source = f"{source}; pool_load_failed={type(exc).__name__}: {exc}"
    else:
        x, source = synthetic_pool(args.rows, args.in_features, rng)

    rows = [
        pq_matmul_trial(
            x=x,
            rows_eval=args.rows_eval,
            out_features=args.out_features,
            subvectors=m,
            centroids=args.centroids,
            train_rows=args.train_rows,
            iters=args.iters,
            seed=args.seed,
            activation_book_bits=args.activation_book_bits,
            index_bits=args.index_bits,
        )
        for m in args.subvectors
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "pq_matmul_repro.tsv", rows)
    (args.output_dir / "pq_matmul_repro.json").write_text(
        json.dumps({"source": source, "config": vars(args), "rows": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    report = render_report(rows, source, args)
    (args.output_dir / "GFMENGINE_PQ_MATMUL_REPRO.md").write_text(report, encoding="utf-8")
    args.repo_report.parent.mkdir(parents=True, exist_ok=True)
    args.repo_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
