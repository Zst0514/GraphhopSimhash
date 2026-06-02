#!/usr/bin/env python3
"""Validate whether graph-aware row ordering reduces BFP shared-exponent error.

The existing BFP activation wrapper quantizes blocks along the hidden
dimension of each row.  In that layout, node order cannot affect BFP error.
This script also simulates a tile layout where one BFP block spans multiple
token/node rows and a few hidden dimensions.  That is the layout where graph
or SimHash ordering can reduce shared-exponent pollution.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.features import _compute_neighbor_mean  # noqa: E402


def _dataset_graph_path(dataset: str) -> Path:
    ds = dataset.lower()
    if ds == "cora":
        return ROOT / "data" / "single_graph" / "Cora" / "cora.pt"
    if ds == "pubmed":
        return ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt"
    if ds == "arxiv":
        return ROOT / "data" / "ogbn_arxiv" / "processed" / "geometric_data_processed.pt"
    raise ValueError(f"unsupported dataset: {dataset}")


def _default_tensor_path(dataset: str, model_name: str) -> Path:
    return ROOT / "cache_data" / f"{dataset.lower()}_{model_name}_oracle_W4A8.pt"


def _default_cheap_path(dataset: str) -> Path:
    return ROOT / "cache_data" / f"{dataset.lower()}_distilbert_l1.pt"


def _load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("x", "embeddings", "features"):
            if key in obj and torch.is_tensor(obj[key]):
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"{path} did not contain a tensor")
    return obj.detach().to(torch.float32).cpu()


def _load_edge_index(dataset: str) -> torch.Tensor:
    data = torch.load(_dataset_graph_path(dataset), map_location="cpu")
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None and isinstance(data, dict):
        edge_index = data.get("edge_index")
    if edge_index is None:
        raise ValueError(f"could not find edge_index for {dataset}")
    return edge_index.detach().cpu().long()


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack boolean matrix [N, B] into int64 codes."""
    bits = bits.to(torch.int64)
    weights = (1 << torch.arange(bits.size(1), dtype=torch.int64)).view(1, -1)
    return (bits * weights).sum(dim=1)


def _simhash_order(features: torch.Tensor, bits: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    feat = F.normalize(features.to(torch.float32), p=2, dim=1)
    proj = torch.randn(feat.size(1), int(bits), generator=gen, dtype=torch.float32)
    codes = _pack_bits((feat @ proj) >= 0)
    return torch.argsort(codes, stable=True)


def _make_orders(
    dataset: str,
    x: torch.Tensor,
    cheap: torch.Tensor,
    edge_index: torch.Tensor,
    hash_bits: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    n = x.size(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    cheap = cheap[:n].to(torch.float32)
    context = F.normalize(0.5 * cheap + 0.5 * _compute_neighbor_mean(cheap, edge_index), p=2, dim=1)
    orders = {
        "original": torch.arange(n, dtype=torch.long),
        "random": torch.randperm(n, generator=gen),
        "activation_norm": torch.argsort(torch.linalg.vector_norm(x, ord=2, dim=1), stable=True),
        "simhash_bucket": _simhash_order(cheap, hash_bits, seed + 17),
        "graph_context_bucket": _simhash_order(context, hash_bits, seed + 29),
    }
    return orders


def _pad_2d(x: torch.Tensor, row_chunk: int, col_chunk: int) -> tuple[torch.Tensor, int, int]:
    n, d = x.shape
    row_pad = (row_chunk - (n % row_chunk)) % row_chunk
    col_pad = (col_chunk - (d % col_chunk)) % col_chunk
    if col_pad:
        x = F.pad(x, (0, col_pad))
    if row_pad:
        x = torch.cat([x, torch.zeros(row_pad, x.size(1), dtype=x.dtype)], dim=0)
    return x, row_pad, col_pad


def _bfp_quantize_ordered_tile(
    x: torch.Tensor,
    order: torch.Tensor,
    mantissa_bit: int,
    row_chunk: int,
    col_chunk: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BFP quantize with one exponent per row_chunk x col_chunk tile."""
    n, d = x.shape
    x_ord = x.index_select(0, order).contiguous()
    x_pad, row_pad, col_pad = _pad_2d(x_ord, row_chunk, col_chunk)
    n_pad, d_pad = x_pad.shape

    grouped = (
        x_pad.reshape(n_pad // row_chunk, row_chunk, d_pad // col_chunk, col_chunk)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(n_pad // row_chunk, d_pad // col_chunk, row_chunk * col_chunk)
    )

    q_min = -(2 ** (mantissa_bit - 1))
    q_max = (2 ** (mantissa_bit - 1)) - 1
    abs_grouped = grouped.detach().abs()
    abs_max = abs_grouped.amax(dim=-1, keepdim=True)
    safe_abs = abs_max.to(torch.float32).clamp_min(1e-30)
    exponent = torch.ceil(torch.log2(safe_abs / float(q_max))).clamp(min=-30.0, max=30.0)
    scale = torch.pow(torch.full_like(exponent, 2.0), exponent).to(dtype=grouped.dtype)
    quantized = torch.round(grouped / scale).clamp(q_min, q_max)
    out = (
        (quantized * scale)
        .reshape(n_pad // row_chunk, d_pad // col_chunk, row_chunk, col_chunk)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(n_pad, d_pad)
    )
    if row_pad:
        out = out[:-row_pad]
    if col_pad:
        out = out[:, :-col_pad]

    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), dtype=torch.long)
    restored = out.index_select(0, inv)[:n, :d].contiguous()

    # Shared-exponent pollution diagnostics.  These are block-level stats, so
    # they intentionally look at values before restoring the original order.
    eps = 1.0e-30
    log_abs = torch.log2(abs_grouped.clamp_min(eps))
    spread = (log_abs.amax(dim=-1) - log_abs.median(dim=-1).values).reshape(-1)
    abs_median = abs_grouped.median(dim=-1).values.clamp_min(eps).reshape(-1)
    outlier_ratio = (abs_max.squeeze(-1).reshape(-1) / abs_median).clamp_max(1.0e12)
    zero_ratio = (quantized == 0).to(torch.float32).mean()
    sat_ratio = (quantized.abs() >= q_max).to(torch.float32).mean()
    stats = {
        "spread_mean": float(spread.mean().item()),
        "spread_p90": float(torch.quantile(spread, 0.90).item()),
        "spread_p95": float(torch.quantile(spread, 0.95).item()),
        "outlier_ratio_p90": float(torch.quantile(outlier_ratio, 0.90).item()),
        "outlier_ratio_p95": float(torch.quantile(outlier_ratio, 0.95).item()),
        "zero_ratio": float(zero_ratio.item()),
        "sat_ratio": float(sat_ratio.item()),
    }
    return restored, stats


def _metrics(x: torch.Tensor, qx: torch.Tensor) -> dict[str, float]:
    diff = qx - x
    row_norm = torch.linalg.vector_norm(x, ord=2, dim=1).clamp_min(1e-12)
    row_err = torch.linalg.vector_norm(diff, ord=2, dim=1) / row_norm
    cosine = F.cosine_similarity(x, qx, dim=1, eps=1e-8)
    return {
        "rel_fro_err": float(torch.linalg.vector_norm(diff).item() / max(torch.linalg.vector_norm(x).item(), 1e-12)),
        "row_rel_err_mean": float(row_err.mean().item()),
        "row_rel_err_p90": float(torch.quantile(row_err, 0.90).item()),
        "cos_mean": float(cosine.mean().item()),
        "cos_p10": float(torch.quantile(cosine, 0.10).item()),
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _format_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("layout", 14),
        ("order", 22),
        ("rel_fro_err", 11),
        ("row_p90", 9),
        ("cos_mean", 9),
        ("spread_p95", 11),
        ("outlier_p95", 12),
        ("zero", 7),
        ("sat", 7),
    ]
    lines = [" ".join(name.rjust(width) for name, width in cols)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        values = {
            "layout": row["layout"],
            "order": row["order"],
            "rel_fro_err": f"{row['rel_fro_err']:.5f}",
            "row_p90": f"{row['row_rel_err_p90']:.5f}",
            "cos_mean": f"{row['cos_mean']:.5f}",
            "spread_p95": f"{row['spread_p95']:.2f}",
            "outlier_p95": f"{row['outlier_ratio_p95']:.1f}",
            "zero": f"{100.0 * row['zero_ratio']:.1f}%",
            "sat": f"{100.0 * row['sat_ratio']:.1f}%",
        }
        lines.append(" ".join(str(values[name]).rjust(width) for name, width in cols))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--model_name", default="llama2_7b")
    parser.add_argument("--tensor_path", default=None)
    parser.add_argument("--cheap_path", default=None)
    parser.add_argument("--mantissa_bits", type=int, default=4)
    parser.add_argument("--hash_bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_rows", type=int, default=0, help="0 means all rows")
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=["rowwise_1x128", "tile_16x8", "tile_32x4"],
        help="Layouts are name:RxC or built-ins rowwise_1x128/tile_16x8/tile_32x4.",
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    tensor_path = Path(args.tensor_path) if args.tensor_path else _default_tensor_path(args.dataset, args.model_name)
    cheap_path = Path(args.cheap_path) if args.cheap_path else _default_cheap_path(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / "graphbfp_shared_exponent" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    x = _load_tensor(tensor_path)
    cheap = _load_tensor(cheap_path)
    edge_index = _load_edge_index(args.dataset)
    n = x.size(0)
    if args.sample_rows and args.sample_rows < n:
        # Keep graph/order construction on full node set, then validate on a
        # deterministic prefix subset for quick large-dataset probing.
        keep = torch.arange(int(args.sample_rows), dtype=torch.long)
        x = x.index_select(0, keep)
        cheap = cheap.index_select(0, keep)
        edge_index = edge_index[:, (edge_index[0] < args.sample_rows) & (edge_index[1] < args.sample_rows)]
        n = x.size(0)

    orders = _make_orders(args.dataset, x, cheap, edge_index, args.hash_bits, args.seed)

    layout_specs: list[tuple[str, int, int]] = []
    builtins = {
        "rowwise_1x128": (1, 128),
        "tile_8x16": (8, 16),
        "tile_16x8": (16, 8),
        "tile_32x4": (32, 4),
    }
    for spec in args.layouts:
        if spec in builtins:
            r, c = builtins[spec]
            layout_specs.append((spec, r, c))
            continue
        if ":" in spec and "x" in spec:
            name, shape = spec.split(":", 1)
            r_s, c_s = shape.lower().split("x", 1)
            layout_specs.append((name, int(r_s), int(c_s)))
            continue
        raise ValueError(f"unknown layout spec: {spec}")

    rows: list[dict[str, Any]] = []
    for layout_name, row_chunk, col_chunk in layout_specs:
        block = int(row_chunk) * int(col_chunk)
        for order_name, order in orders.items():
            qx, stats = _bfp_quantize_ordered_tile(
                x,
                order,
                mantissa_bit=args.mantissa_bits,
                row_chunk=row_chunk,
                col_chunk=col_chunk,
            )
            row = {
                "dataset": args.dataset,
                "tensor": str(tensor_path),
                "mantissa_bits": args.mantissa_bits,
                "layout": layout_name,
                "row_chunk": row_chunk,
                "col_chunk": col_chunk,
                "block_values": block,
                "order": order_name,
                **_metrics(x, qx),
                **stats,
            }
            rows.append(row)

    _write_tsv(out_dir / "summary.tsv", rows)
    table = _format_table(rows)
    note = (
        f"BFP shared-exponent order validation | dataset={args.dataset} | "
        f"tensor={tensor_path} | mantissa=A{args.mantissa_bits}\n"
        f"Rows={x.size(0)} Hidden={x.size(1)}\n\n"
        f"{table}\n\n"
        "Interpretation:\n"
        "- rowwise_1x128 matches the current BFP pool layout; node order should not change it.\n"
        "- tile_* layouts share one exponent across multiple rows; ordering can reduce outlier pollution there.\n"
    )
    (out_dir / "summary.txt").write_text(note, encoding="utf-8")
    print(note)
    print(f"[Saved] {out_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
