#!/usr/bin/env python3
"""Evaluate downstream GNN accuracy for different BFP block layouts.

This is a fast embedding-level validation.  It starts from an existing
reference embedding pool, applies BFP packing in memory, then trains/evaluates
the same GNN backend used by the main GraphhopSimhash experiments.

It is meant to answer a narrow question before doing expensive full encoder
pool generation: do row-wise and cross-row BFP block layouts visibly differ in
downstream graph accuracy?
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.runner import evaluate_gnn_embeddings, train_baseline_model  # noqa: E402
from GraphhopSimhash.scripts.bfp_shared_exponent_order_validation import (  # noqa: E402
    _bfp_quantize_ordered_tile,
    _default_cheap_path,
    _default_tensor_path,
    _load_edge_index,
    _load_tensor,
    _make_orders,
    _metrics,
)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _format_summary(rows: list[dict[str, Any]]) -> str:
    headers = [
        ("Layout", 14),
        ("Order", 22),
        ("Acc", 8),
        ("Drop", 8),
        ("RelErr", 8),
        ("Cos", 8),
        ("Spread95", 9),
        ("Zero", 7),
    ]
    lines = [" ".join(name.rjust(width) for name, width in headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        values = {
            "Layout": row["layout"],
            "Order": row["order"],
            "Acc": f"{row['acc_mean']:.4f}",
            "Drop": f"{row['drop_mean']:.2%}",
            "RelErr": f"{row['rel_fro_err']:.4f}",
            "Cos": f"{row['cos_mean']:.4f}",
            "Spread95": f"{row['spread_p95']:.2f}",
            "Zero": f"{100.0 * row['zero_ratio']:.1f}%",
        }
        lines.append(" ".join(str(values[name]).rjust(width) for name, width in headers))
    return "\n".join(lines)


def _parse_layout(spec: str) -> tuple[str, int, int]:
    builtins = {
        "rowwise_1x128": (1, 128),
        "tile_8x16": (8, 16),
        "tile_16x8": (16, 8),
        "tile_32x4": (32, 4),
        "tile_64x2": (64, 2),
        "tile_128x1": (128, 1),
    }
    if spec in builtins:
        r, c = builtins[spec]
        return spec, r, c
    if ":" in spec and "x" in spec:
        name, shape = spec.split(":", 1)
        r_s, c_s = shape.lower().split("x", 1)
        return name, int(r_s), int(c_s)
    raise ValueError(f"unknown layout spec: {spec}")


def _make_run_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        llm_name=args.llm_name,
        emb_dim=args.emb_dim,
        radius=args.radius,
        max_test=args.max_test,
        standard_eval_baseline=args.standard_eval_baseline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--model_name", default="llama2_7b")
    parser.add_argument("--reference_path", default=None)
    parser.add_argument("--cheap_path", default=None)
    parser.add_argument("--mantissa_bits", type=int, default=4)
    parser.add_argument("--hash_bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--llm_name", default="ST")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--standard_eval_baseline", action="store_true")
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=["rowwise_1x128", "tile_16x8", "tile_32x4", "tile_128x1"],
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["original", "random", "activation_norm", "simhash_bucket", "graph_context_bucket"],
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    reference_path = Path(args.reference_path) if args.reference_path else _default_tensor_path(args.dataset, args.model_name)
    cheap_path = Path(args.cheap_path) if args.cheap_path else _default_cheap_path(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / "graphbfp_block_layout_accuracy" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_cpu = _load_tensor(reference_path)
    cheap = _load_tensor(cheap_path)
    edge_index = _load_edge_index(args.dataset)
    orders = _make_orders(args.dataset, reference_cpu, cheap, edge_index, args.hash_bits, args.seed)
    selected_orders = {name: orders[name] for name in args.orders}
    layout_specs = [_parse_layout(spec) for spec in args.layouts]

    quantized: dict[tuple[str, str], dict[str, Any]] = {}
    for layout_name, row_chunk, col_chunk in layout_specs:
        for order_name, order in selected_orders.items():
            qx, stats = _bfp_quantize_ordered_tile(
                reference_cpu,
                order,
                mantissa_bit=args.mantissa_bits,
                row_chunk=row_chunk,
                col_chunk=col_chunk,
            )
            quantized[(layout_name, order_name)] = {
                "emb": qx,
                "stats": stats,
                "metrics": _metrics(reference_cpu, qx),
                "row_chunk": row_chunk,
                "col_chunk": col_chunk,
            }

    seed_rows: list[dict[str, Any]] = []
    summary: dict[tuple[str, str], dict[str, Any]] = {
        key: {"acc": [], "drop": []}
        for key in quantized
    }
    baseline_accs: list[float] = []

    for run_idx in range(args.runs):
        seed = int(args.seed) + run_idx
        run_args = _make_run_args(args)
        _conf, data, _verify_features, device = load_run_state(args.dataset, run_args, seed)
        data.x = reference_cpu.to(device)
        model, baseline_acc, _ref_embs, _ref_logits = train_baseline_model(data, run_args, device)
        baseline_accs.append(float(baseline_acc))
        for (layout_name, order_name), item in quantized.items():
            emb = item["emb"].to(device)
            with torch.no_grad():
                hidden = model.encoder(emb)
            acc = evaluate_gnn_embeddings(model, data, hidden)
            drop = float(baseline_acc - acc)
            summary[(layout_name, order_name)]["acc"].append(float(acc))
            summary[(layout_name, order_name)]["drop"].append(drop)
            seed_rows.append(
                {
                    "seed": seed,
                    "baseline_acc": float(baseline_acc),
                    "layout": layout_name,
                    "order": order_name,
                    "acc": float(acc),
                    "drop": drop,
                    **item["metrics"],
                    **item["stats"],
                }
            )

    rows: list[dict[str, Any]] = []
    for (layout_name, order_name), item in quantized.items():
        accs = summary[(layout_name, order_name)]["acc"]
        drops = summary[(layout_name, order_name)]["drop"]
        rows.append(
            {
                "dataset": args.dataset,
                "reference": str(reference_path),
                "mantissa_bits": args.mantissa_bits,
                "baseline_acc_mean": float(np.mean(baseline_accs)),
                "layout": layout_name,
                "order": order_name,
                "row_chunk": item["row_chunk"],
                "col_chunk": item["col_chunk"],
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "drop_mean": float(np.mean(drops)),
                "drop_std": float(np.std(drops)),
                **item["metrics"],
                **item["stats"],
            }
        )

    _write_tsv(out_dir / "per_seed.tsv", seed_rows)
    _write_tsv(out_dir / "summary.tsv", rows)
    table = _format_summary(rows)
    note = (
        f"BFP block-layout downstream accuracy | dataset={args.dataset} | "
        f"reference={reference_path} | A{args.mantissa_bits} | runs={args.runs}\n"
        f"Baseline Acc mean: {float(np.mean(baseline_accs)):.4f}\n\n"
        f"{table}\n\n"
        "Note: this is embedding-level packed-proxy validation.  It compares final "
        "GNN accuracy under different BFP block layouts before generating full "
        "encoder activation-level pools.\n"
    )
    (out_dir / "summary.txt").write_text(note, encoding="utf-8")
    print(note)
    print(f"[Saved] {out_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
