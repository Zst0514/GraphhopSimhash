#!/usr/bin/env python3
"""Validate graph-aware BFP packing on real LLaMA activations.

This script hooks selected Linear modules, captures their input activations,
and compares BFP shared-exponent error under different node ordering policies.
It reports both the natural sequence-major row order and a token-position-major
order that models a scheduler grouping same-position token rows across nodes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_raw_texts  # noqa: E402
from GraphhopSimhash.generate_real_quant_pools import load_model_and_tokenizer  # noqa: E402
from GraphhopSimhash.scripts.bfp_shared_exponent_order_validation import (  # noqa: E402
    _bfp_quantize_ordered_tile,
    _default_cheap_path,
    _default_tensor_path,
    _load_edge_index,
    _load_tensor,
    _make_orders,
    _metrics,
)


def _module_selected(name: str, layer_ids: set[int], suffixes: tuple[str, ...]) -> bool:
    if not any(name.endswith("." + suffix) for suffix in suffixes):
        return False
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if not match:
        return False
    return int(match.group(1)) in layer_ids


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


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _format(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("row_mode", 20),
        ("layout", 13),
        ("order", 21),
        ("rel_err", 8),
        ("cos", 8),
        ("spread95", 9),
        ("out95", 8),
        ("zero", 7),
    ]
    lines = [" ".join(name.rjust(width) for name, width in cols)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        vals = {
            "row_mode": row["row_mode"],
            "layout": row["layout"],
            "order": row["order"],
            "rel_err": f"{row['rel_fro_err_mean']:.4f}",
            "cos": f"{row['cos_mean_mean']:.4f}",
            "spread95": f"{row['spread_p95_mean']:.2f}",
            "out95": f"{row['outlier_ratio_p95_mean']:.1f}",
            "zero": f"{100.0 * row['zero_ratio_mean']:.1f}%",
        }
        lines.append(" ".join(str(vals[name]).rjust(width) for name, width in cols))
    return "\n".join(lines)


def _ordered_sample_nodes(full_order: torch.Tensor, sample_nodes: torch.Tensor) -> torch.Tensor:
    pos = torch.empty(int(full_order.numel()), dtype=torch.long)
    pos[full_order.cpu()] = torch.arange(int(full_order.numel()), dtype=torch.long)
    rank = pos[sample_nodes.cpu()]
    return sample_nodes[torch.argsort(rank, stable=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--llm_name", default="llama2_7b")
    parser.add_argument("--model_config", default="fp16")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--reference_path", default=None)
    parser.add_argument("--cheap_path", default=None)
    parser.add_argument("--sample_nodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--mantissa_bits", type=int, default=4)
    parser.add_argument("--hash_bits", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument("--module_suffixes", nargs="+", default=["q_proj", "o_proj", "up_proj", "down_proj"])
    parser.add_argument("--layouts", nargs="+", default=["rowwise_1x128", "tile_16x8", "tile_32x4"])
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["original", "random", "activation_norm", "simhash_bucket", "graph_context_bucket"],
    )
    parser.add_argument("--row_modes", nargs="+", default=["sequence_major", "token_position_major"])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / "graphbfp_activation_order" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference_path) if args.reference_path else _default_tensor_path(args.dataset, args.llm_name)
    cheap_path = Path(args.cheap_path) if args.cheap_path else _default_cheap_path(args.dataset)
    reference = _load_tensor(reference_path)
    cheap = _load_tensor(cheap_path)
    edge_index = _load_edge_index(args.dataset)
    full_orders = _make_orders(args.dataset, reference, cheap, edge_index, args.hash_bits, args.seed)

    n = reference.size(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed))
    sample = torch.randperm(n, generator=gen)[: min(int(args.sample_nodes), n)]
    order_to_nodes = {name: _ordered_sample_nodes(full_orders[name], sample) for name in args.orders}

    texts = load_raw_texts(args.dataset)
    model, tokenizer, _tag = load_model_and_tokenizer(args.llm_name, args.model_config, args.cache_dir, force_cpu=False)
    target_model = model.model if hasattr(model, "model") and model.__class__.__name__.endswith("ForCausalLM") else model
    device = next(target_model.parameters()).device

    selected_layers = {int(x) for x in args.layers}
    suffixes = tuple(str(x) for x in args.module_suffixes)
    selected = [(name, module) for name, module in target_model.named_modules() if _module_selected(name, selected_layers, suffixes)]
    if not selected:
        raise RuntimeError("No selected modules matched; check --layers/--module_suffixes")
    layout_specs = [_parse_layout(spec) for spec in args.layouts]

    raw_rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    current_order_name: str | None = None
    current_mask: torch.Tensor | None = None

    def process_activation(module_name: str, x: torch.Tensor, mask: torch.Tensor) -> None:
        if x.ndim != 3:
            return
        bsz, seqlen, hidden = x.shape
        for row_mode in args.row_modes:
            if row_mode == "sequence_major":
                flat = x.detach().to(torch.float32).reshape(bsz * seqlen, hidden)
                valid = mask.reshape(-1).bool()
            elif row_mode == "token_position_major":
                flat = x.detach().to(torch.float32).transpose(0, 1).reshape(seqlen * bsz, hidden)
                valid = mask.transpose(0, 1).reshape(-1).bool()
            else:
                raise ValueError(f"unknown row mode: {row_mode}")
            flat = flat[valid.to(flat.device)].contiguous()
            if flat.size(0) <= 0:
                continue
            identity = torch.arange(flat.size(0), dtype=torch.long, device="cpu")
            flat_cpu = flat.cpu()
            for layout_name, row_chunk, col_chunk in layout_specs:
                qx, stats = _bfp_quantize_ordered_tile(
                    flat_cpu,
                    identity,
                    mantissa_bit=args.mantissa_bits,
                    row_chunk=row_chunk,
                    col_chunk=col_chunk,
                )
                metrics = _metrics(flat_cpu, qx)
                key = (row_mode, layout_name, str(current_order_name))
                for metric_name, value in {**metrics, **stats}.items():
                    aggregate[key][metric_name].append(float(value))
                raw_rows.append(
                    {
                        "order": current_order_name,
                        "module": module_name,
                        "row_mode": row_mode,
                        "layout": layout_name,
                        "rows": int(flat_cpu.size(0)),
                        "hidden": int(flat_cpu.size(1)),
                        **metrics,
                        **stats,
                    }
                )

    hooks = []
    for module_name, module in selected:
        def make_hook(name: str):
            def hook(_module, inputs, _outputs):
                if current_mask is None:
                    return
                process_activation(name, inputs[0], current_mask)
            return hook
        hooks.append(module.register_forward_hook(make_hook(module_name)))

    try:
        with torch.no_grad():
            for order_name, node_ids in order_to_nodes.items():
                current_order_name = order_name
                ordered_texts = [texts[int(idx)] for idx in node_ids.tolist()]
                for start in range(0, len(ordered_texts), int(args.batch_size)):
                    batch_texts = ordered_texts[start : start + int(args.batch_size)]
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=int(args.max_length),
                        return_tensors="pt",
                    )
                    current_mask = encoded["attention_mask"].to(device)
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    _ = model(**encoded)
    finally:
        for hook in hooks:
            hook.remove()

    summary_rows: list[dict[str, Any]] = []
    for (row_mode, layout_name, order_name), metric_map in sorted(aggregate.items()):
        row: dict[str, Any] = {
            "dataset": args.dataset,
            "mantissa_bits": int(args.mantissa_bits),
            "sample_nodes": int(sample.numel()),
            "layers": ",".join(str(x) for x in args.layers),
            "modules": ",".join(str(x) for x in args.module_suffixes),
            "row_mode": row_mode,
            "layout": layout_name,
            "order": order_name,
        }
        for metric_name, values in metric_map.items():
            stats = _summarize(values)
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
        summary_rows.append(row)

    _write_tsv(out_dir / "raw.tsv", raw_rows)
    _write_tsv(out_dir / "summary.tsv", summary_rows)
    table = _format(summary_rows)
    note = (
        f"BFP real-activation order validation | dataset={args.dataset} | "
        f"A{args.mantissa_bits} | nodes={int(sample.numel())} | max_length={args.max_length}\n"
        f"layers={args.layers} modules={args.module_suffixes}\n\n"
        f"{table}\n\n"
        "sequence_major is ordinary per-sequence flattening. token_position_major models "
        "a scheduler that places same-position token rows from ordered nodes together.\n"
    )
    (out_dir / "summary.txt").write_text(note, encoding="utf-8")
    print(note)
    print(f"[Saved] {out_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
