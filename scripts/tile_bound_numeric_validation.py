#!/usr/bin/env python3
"""Tile-level numeric validation for Graph-Bit predictor-free bounds.

This script samples real LLaMA activations and real Linear weight tiles, then
measures how much the omitted low activation bit-planes would contribute:

    delta(depth) = A_low(depth) @ W_tile.T

It compares the measured delta ratio against several predictor-free bound
forms and reports coverage/tightness plus runtime-vs-oracle stop-depth
agreement.  The goal is to validate the bound itself without re-running the
full encoder in a custom bit-serial implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

# Allow running as `python GraphhopSimhash/scripts/...` from OFA root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_raw_texts  # noqa: E402
from GraphhopSimhash.generate_real_quant_pools import load_model_and_tokenizer  # noqa: E402

BOUND_MODES = ("range", "tile_p95", "ratio_mean", "ratio_max", "exact_l1")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_degree_q(dataset: str) -> torch.Tensor:
    ds_key = dataset.lower()
    if ds_key == "cora":
        path = ROOT / "data" / "single_graph" / "Cora" / "cora.pt"
    elif ds_key == "pubmed":
        path = ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt"
    else:
        raise ValueError("degree_q helper currently supports cora/pubmed")
    data = torch.load(path, map_location="cpu")
    edge_index = data.edge_index.cpu()
    row, col = edge_index
    sym_row = torch.cat([row, col], dim=0)
    deg = torch.zeros(int(data.num_nodes), dtype=torch.float32)
    deg.index_add_(0, sym_row, torch.ones_like(sym_row, dtype=torch.float32))
    risk = torch.log1p(deg) / torch.log1p(deg.max().clamp(min=1.0))
    return torch.round(risk.clamp(0.0, 1.0) * 15.0).to(torch.float32)


def select_nodes(num_nodes: int, sample_nodes: int, seed: int, strategy: str) -> list[int]:
    sample_nodes = min(num_nodes, max(1, int(sample_nodes)))
    if strategy == "first":
        return list(range(sample_nodes))
    gen = random.Random(seed)
    return gen.sample(range(num_nodes), sample_nodes)


def module_selected(name: str, layer_ids: set[int], suffixes: tuple[str, ...]) -> bool:
    if not any(name.endswith("." + suffix) for suffix in suffixes):
        return False
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if not match:
        return False
    return int(match.group(1)) in layer_ids


def affine_qparams_per_row(x: torch.Tensor, bits: int = 8):
    qmin = 0.0
    qmax = float((1 << bits) - 1)
    min_val = x.amin(dim=-1, keepdim=True)
    max_val = x.amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_val - min_val, min=1.0e-8) / qmax
    zp = torch.round(qmin - min_val / scale).clamp(qmin, qmax)
    q = torch.round(x / scale + zp).clamp(qmin, qmax)
    return q, scale, zp


def dequant_from_q(q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    return (q - zp) * scale


def low_q_from_depth(q: torch.Tensor, depth: int, full_depth: int = 8) -> torch.Tensor:
    if depth >= full_depth:
        return torch.zeros_like(q)
    step = float(1 << (full_depth - depth))
    q_trunc = torch.floor(q / step) * step
    return q - q_trunc


def remaining_range_bound(depth: int, full_depth: int, tile_k: int, strength: float = 1.0, scale: float = 1.0) -> float:
    if depth >= full_depth:
        return 0.0
    omitted = (1 << (full_depth - depth)) - 1
    denom = (1 << full_depth) - 1
    return float(scale) * (float(omitted) / float(denom)) * math.sqrt(max(1, tile_k) / 128.0) * float(strength)


def remaining_ratio_bound(
    depth: int,
    full_depth: int,
    tile_k: int,
    weight_abs_mean: float,
    weight_abs_bound: float,
    scale: float = 1.0,
    safety_factor: float = 1.0,
    partial_norm_scale: float = 1.0,
    partial_norm_floor: float = 1.0e-6,
) -> float:
    if depth >= full_depth:
        return 0.0
    omitted = (1 << (full_depth - depth)) - 1
    denom = (1 << full_depth) - 1
    normalized_omitted = float(omitted) / float(denom)
    tile = float(max(1, tile_k))
    remaining = normalized_omitted * tile * max(1.0e-9, weight_abs_bound) * max(0.0, safety_factor)
    high_range = max(0.0, 1.0 - normalized_omitted)
    partial = high_range * tile * max(1.0e-9, weight_abs_mean) * max(0.0, partial_norm_scale)
    partial = max(partial, partial_norm_floor)
    k_scale = math.sqrt(max(1, tile_k) / 128.0)
    return float(scale) * remaining / max(1.0e-12, partial + remaining) * k_scale


def low_bit_budget(depth: int, full_depth: int = 8) -> float:
    if depth >= full_depth:
        return 0.0
    omitted = (1 << (full_depth - depth)) - 1
    denom = (1 << full_depth) - 1
    return float(omitted) / float(denom)


def tolerance_from_risk(risk_q: torch.Tensor, min_tol: float, max_tol: float, gamma: float, risk_max: float) -> torch.Tensor:
    risk_norm = torch.clamp(risk_q.float() / max(risk_max, 1.0e-12), 0.0, 1.0)
    return float(min_tol) + (float(max_tol) - float(min_tol)) * torch.pow(1.0 - risk_norm, float(gamma))


def nearest_depth(depth: int, available: list[int]) -> int:
    for bit in sorted(set(available)):
        if bit >= depth:
            return int(bit)
    return int(max(available))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "pubmed"])
    parser.add_argument("--llm_name", default="llama2_7b")
    parser.add_argument("--model_config", default="fp16")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--sample_nodes", type=int, default=8)
    parser.add_argument("--sample_strategy", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument(
        "--module_suffixes",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tiles_per_module", type=int, default=4)
    parser.add_argument("--rows_per_module", type=int, default=32)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--available_depths", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--min_depth", type=int, default=4)
    parser.add_argument("--min_tol", type=float, default=0.0)
    parser.add_argument("--max_tol", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--risk_max", type=float, default=15.0)
    parser.add_argument("--bound_scale", type=float, default=1.0)
    parser.add_argument(
        "--score_taus",
        type=float,
        nargs="+",
        default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
    )
    parser.add_argument("--score_alpha", type=float, default=1.0)
    parser.add_argument("--score_beta", type=float, default=1.0)
    parser.add_argument("--score_w_cap", type=float, default=2.0)
    parser.add_argument("--score_w_reference_quantile", type=float, default=0.90)
    parser.add_argument("--score_node_floor", type=float, default=0.0)
    parser.add_argument("--output_dir", default="output/graphbit_tile_bound_numeric/cora_quick")
    parser.add_argument("--write_samples", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = load_raw_texts(args.dataset)
    node_ids = select_nodes(len(texts), args.sample_nodes, args.seed, args.sample_strategy)
    selected_texts = [texts[idx] for idx in node_ids]
    degree_q = load_degree_q(args.dataset)

    print(
        f"[TileBound] dataset={args.dataset} nodes={len(node_ids)} max_length={args.max_length} "
        f"layers={args.layers} modules={args.module_suffixes}"
    )
    model, tokenizer, _tag = load_model_and_tokenizer(args.llm_name, args.model_config, args.cache_dir, force_cpu=False)
    target_model = model.model if hasattr(model, "model") and model.__class__.__name__.endswith("ForCausalLM") else model
    device = next(target_model.parameters()).device

    selected_layers = set(int(x) for x in args.layers)
    suffixes = tuple(str(x) for x in args.module_suffixes)
    selected_modules = [(name, module) for name, module in target_model.named_modules() if module_selected(name, selected_layers, suffixes)]
    if not selected_modules:
        raise RuntimeError("No selected modules matched; check --layers/--module_suffixes")
    print(f"[TileBound] selected_modules={len(selected_modules)}")

    sample_rows: list[dict[str, Any]] = []
    module_summaries: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    tile_score_records: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    current_node_ids: torch.Tensor | None = None
    current_mask: torch.Tensor | None = None

    def process_module(name: str, module: torch.nn.Module, x: torch.Tensor, node_ids_batch: torch.Tensor, mask: torch.Tensor) -> None:
        if x.ndim != 3:
            return
        bsz, seqlen, hidden = x.shape
        flat = x.detach().to(dtype=torch.float32).reshape(bsz * seqlen, hidden)
        valid = mask.reshape(-1).bool().to(flat.device)
        if int(valid.sum().item()) <= 0:
            return
        row_node_ids = node_ids_batch.to(flat.device).repeat_interleave(seqlen)[valid]
        flat = flat[valid]
        if flat.size(0) > args.rows_per_module:
            perm = torch.randperm(flat.size(0), device=flat.device)[: args.rows_per_module]
            flat = flat[perm]
            row_node_ids = row_node_ids[perm]
        if flat.numel() == 0:
            return

        weight = module.weight.detach().to(device=flat.device, dtype=torch.float32)
        n_dim, k_dim = int(weight.shape[0]), int(weight.shape[1])
        actual_tile_k = min(args.tile_k, k_dim)
        actual_tile_n = min(args.tile_n, n_dim)
        if hidden != k_dim:
            return
        layer_mean_abs = float(weight.abs().mean().item())
        q8, scale, zp = affine_qparams_per_row(flat, bits=8)
        risk_q = degree_q[row_node_ids.cpu()].to(device=flat.device)
        tolerances = tolerance_from_risk(risk_q, args.min_tol, args.max_tol, args.gamma, args.risk_max)

        possible_k = list(range(0, max(1, k_dim - actual_tile_k + 1), actual_tile_k))
        possible_n = list(range(0, max(1, n_dim - actual_tile_n + 1), actual_tile_n))
        if not possible_k:
            possible_k = [0]
        if not possible_n:
            possible_n = [0]
        tile_pairs = []
        for _ in range(max(1, args.tiles_per_module)):
            tile_pairs.append((rng.choice(possible_n), rng.choice(possible_k)))

        module_actuals_by_depth: dict[int, list[float]] = defaultdict(list)
        module_bounds_by_depth: dict[str, dict[int, list[float]]] = {mode: defaultdict(list) for mode in BOUND_MODES}
        module_decision_gap: dict[str, list[int]] = {mode: [] for mode in BOUND_MODES}
        module_coverage: dict[str, dict[int, list[float]]] = {mode: defaultdict(list) for mode in BOUND_MODES}

        for n0, k0 in tile_pairs:
            n1 = min(n0 + actual_tile_n, n_dim)
            k1 = min(k0 + actual_tile_k, k_dim)
            tile_k = k1 - k0
            if tile_k <= 0 or n1 <= n0:
                continue
            w_tile = weight[n0:n1, k0:k1]
            abs_w = w_tile.abs()
            row_l1 = abs_w.sum(dim=1)
            denom = max(layer_mean_abs * float(tile_k), 1.0e-12)
            strength_mean = float((row_l1.mean() / denom).item())
            strength_p95 = float((torch.quantile(row_l1, 0.95) / denom).item())
            strength_max = float((row_l1.max() / denom).item())
            weight_abs_mean = float(abs_w.mean().item())
            weight_abs_max = float(abs_w.max().item())

            q_tile = q8[:, k0:k1]
            scale_rows = scale
            zp_rows = zp
            a8_tile = dequant_from_q(q_tile, scale_rows, zp_rows)
            y_full = torch.matmul(a8_tile, w_tile.t())
            y_norm = torch.linalg.vector_norm(y_full, dim=1).clamp(min=1.0e-12)

            actual_by_depth: dict[int, torch.Tensor] = {}
            bounds_by_mode_depth: dict[str, dict[int, torch.Tensor]] = {mode: {} for mode in BOUND_MODES}
            for depth in sorted(set(args.depths)):
                q_low = low_q_from_depth(q_tile, depth, full_depth=8)
                a_low = q_low.to(dtype=torch.float32) * scale_rows
                y_low = torch.matmul(a_low, w_tile.t())
                ratio = torch.linalg.vector_norm(y_low, dim=1) / y_norm
                actual_by_depth[depth] = ratio
                omitted = (1 << (8 - depth)) - 1
                low_abs_max = float(omitted) * scale_rows.squeeze(-1)
                exact_l1_vec = torch.linalg.vector_norm(low_abs_max.unsqueeze(1) * row_l1.unsqueeze(0), dim=1) / y_norm
                b_range = remaining_range_bound(depth, 8, tile_k, strength=1.0, scale=args.bound_scale)
                b_tile = remaining_range_bound(depth, 8, tile_k, strength=strength_p95, scale=args.bound_scale)
                b_ratio_mean = remaining_ratio_bound(
                    depth,
                    8,
                    tile_k,
                    weight_abs_mean=weight_abs_mean,
                    weight_abs_bound=weight_abs_mean,
                    scale=args.bound_scale,
                )
                b_ratio_max = remaining_ratio_bound(
                    depth,
                    8,
                    tile_k,
                    weight_abs_mean=weight_abs_mean,
                    weight_abs_bound=weight_abs_max,
                    scale=args.bound_scale,
                )
                scalar_bounds = {
                    "range": b_range,
                    "tile_p95": b_tile,
                    "ratio_mean": b_ratio_mean,
                    "ratio_max": b_ratio_max,
                }
                for mode, bound in scalar_bounds.items():
                    bounds_by_mode_depth[mode][depth] = torch.full_like(ratio, float(bound))
                bounds_by_mode_depth["exact_l1"][depth] = exact_l1_vec

                actual_list = ratio.detach().cpu().tolist()
                module_actuals_by_depth[depth].extend(float(v) for v in actual_list)
                for mode in BOUND_MODES:
                    bound_list = bounds_by_mode_depth[mode][depth].detach().cpu().tolist()
                    module_bounds_by_depth[mode][depth].extend(float(v) for v in bound_list)
                    module_coverage[mode][depth].extend(
                        [1.0 if float(b) + 1e-12 >= float(a) else 0.0 for a, b in zip(actual_list, bound_list)]
                    )
                if args.write_samples:
                    for ridx, val in enumerate(actual_list):
                        sample_rows.append(
                            {
                                "module": name,
                                "node_id": int(row_node_ids[ridx].item()),
                                "degree_q": float(risk_q[ridx].item()),
                                "tolerance": float(tolerances[ridx].item()),
                                "n0": n0,
                                "k0": k0,
                                "depth": depth,
                                "actual_delta_ratio": float(val),
                                "bound_range": b_range,
                                "bound_tile_p95": b_tile,
                                "bound_ratio_mean": b_ratio_mean,
                                "bound_ratio_max": b_ratio_max,
                                "bound_exact_l1": float(exact_l1_vec[ridx].item()),
                                "strength_p95": strength_p95,
                            }
                        )

            actual_by_depth[8] = torch.zeros_like(y_norm)
            row_oracle_depths: list[int] = []
            for row_idx in range(flat.size(0)):
                tol = float(tolerances[row_idx].item())
                oracle_depth = 8
                for depth in sorted(args.available_depths):
                    if depth == 8:
                        actual_val = 0.0
                    elif depth in actual_by_depth:
                        actual_val = float(actual_by_depth[depth][row_idx].item())
                    else:
                        continue
                    if actual_val <= tol + 1e-12:
                        oracle_depth = depth
                        break
                row_oracle_depths.append(nearest_depth(oracle_depth, args.available_depths))
                record: dict[str, Any] = {
                    "module": name,
                    "node_id": int(row_node_ids[row_idx].item()),
                    "degree_q": float(risk_q[row_idx].item()),
                    "tolerance": tol,
                    "oracle_depth": int(row_oracle_depths[-1]),
                    "tile_k": int(tile_k),
                    "tile_n": int(n1 - n0),
                    "strength_mean": float(strength_mean),
                    "strength_p95": float(strength_p95),
                    "strength_max": float(strength_max),
                }
                for depth in sorted(set(args.depths)):
                    if depth in actual_by_depth:
                        record[f"actual_p{depth}"] = float(actual_by_depth[depth][row_idx].item())
                tile_score_records.append(record)

            for mode, depth_bounds in bounds_by_mode_depth.items():
                for row_idx in range(flat.size(0)):
                    tol = float(tolerances[row_idx].item())
                    oracle_depth = row_oracle_depths[row_idx]
                    runtime_depth = 8
                    for depth in sorted(args.available_depths):
                        if depth == 8:
                            bound_val = 0.0
                        elif depth in depth_bounds:
                            bound_tensor = depth_bounds[depth]
                            bound_val = float(bound_tensor[row_idx].item())
                        else:
                            continue
                        if depth < args.min_depth:
                            continue
                        if bound_val <= tol + 1e-12:
                            runtime_depth = depth
                            break
                    runtime_depth = nearest_depth(runtime_depth, args.available_depths)
                    module_decision_gap[mode].append(runtime_depth - oracle_depth)
                    decision_rows.append(
                        {
                            "module": name,
                            "mode": mode,
                            "node_id": int(row_node_ids[row_idx].item()),
                            "degree_q": float(risk_q[row_idx].item()),
                            "tolerance": tol,
                            "oracle_depth": int(oracle_depth),
                            "runtime_depth": int(runtime_depth),
                            "gap": int(runtime_depth - oracle_depth),
                        }
                    )

        for depth in sorted(set(args.depths)):
            vals = module_actuals_by_depth.get(depth, [])
            if not vals:
                continue
            for mode in BOUND_MODES:
                bounds = module_bounds_by_depth[mode][depth]
                cov = module_coverage[mode][depth]
                tight = [float(b) / max(float(a), 1.0e-12) for a, b in zip(vals, bounds)]
                depth_rows.append(
                    {
                        "module": name,
                        "mode": mode,
                        "depth": depth,
                        "samples": len(vals),
                        "actual_mean": summarize(vals)["mean"],
                        "actual_p90": summarize(vals)["p90"],
                        "actual_p95": summarize(vals)["p95"],
                        "bound_mean": summarize(bounds)["mean"],
                        "coverage": sum(cov) / max(1, len(cov)),
                        "tightness_p50": summarize(tight)["p50"],
                        "tightness_p90": summarize(tight)["p90"],
                    }
                )
        for mode, gaps in module_decision_gap.items():
            if not gaps:
                continue
            module_summaries.append(
                {
                    "module": name,
                    "mode": mode,
                    "decision_samples": len(gaps),
                    "exact_match": sum(1 for g in gaps if g == 0) / len(gaps),
                    "conservative": sum(1 for g in gaps if g > 0) / len(gaps),
                    "aggressive": sum(1 for g in gaps if g < 0) / len(gaps),
                    "gap_mean": sum(gaps) / len(gaps),
                    "gap_p90": quantile([float(g) for g in gaps], 0.90),
                }
            )

    hooks = []
    for mod_name, mod in selected_modules:
        def make_hook(name: str, module: torch.nn.Module):
            def hook(_module, inputs, _outputs):
                if current_node_ids is None or current_mask is None:
                    return
                process_module(name, module, inputs[0], current_node_ids, current_mask)
            return hook
        hooks.append(mod.register_forward_hook(make_hook(mod_name, mod)))

    try:
        with torch.no_grad():
            for start in range(0, len(selected_texts), args.batch_size):
                batch_texts = selected_texts[start : start + args.batch_size]
                batch_ids = node_ids[start : start + args.batch_size]
                tokens = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                current_node_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
                current_mask = tokens["attention_mask"].to(device)
                _ = target_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    output_hidden_states=False,
                    return_dict=True,
                )
    finally:
        for h in hooks:
            h.remove()

    # Global summaries.
    global_rows: list[dict[str, Any]] = []
    for mode in BOUND_MODES:
        for depth in sorted(set(args.depths)):
            rows = [r for r in depth_rows if r["mode"] == mode and r["depth"] == depth]
            total_samples = sum(int(r["samples"]) for r in rows)
            if total_samples == 0:
                continue
            weighted = lambda key: sum(float(r[key]) * int(r["samples"]) for r in rows) / total_samples
            global_rows.append(
                {
                    "mode": mode,
                    "depth": depth,
                    "samples": total_samples,
                    "actual_mean": weighted("actual_mean"),
                    "actual_p90_mean": weighted("actual_p90"),
                    "actual_p95_mean": weighted("actual_p95"),
                    "bound_mean": weighted("bound_mean"),
                    "coverage": weighted("coverage"),
                    "tightness_p50_mean": weighted("tightness_p50"),
                    "tightness_p90_mean": weighted("tightness_p90"),
                }
            )
    decision_global: list[dict[str, Any]] = []
    for mode in BOUND_MODES:
        rows = [r for r in decision_rows if r["mode"] == mode]
        if not rows:
            continue
        gaps = [int(r["gap"]) for r in rows]
        runtime_depths = [float(r["runtime_depth"]) for r in rows]
        oracle_depths = [float(r["oracle_depth"]) for r in rows]
        decision_global.append(
            {
                "mode": mode,
                "samples": len(rows),
                "runtime_avg_depth": sum(runtime_depths) / len(runtime_depths),
                "oracle_avg_depth": sum(oracle_depths) / len(oracle_depths),
                "exact_match": sum(1 for g in gaps if g == 0) / len(gaps),
                "conservative": sum(1 for g in gaps if g > 0) / len(gaps),
                "aggressive": sum(1 for g in gaps if g < 0) / len(gaps),
                "gap_mean": sum(gaps) / len(gaps),
                "gap_p90": quantile([float(g) for g in gaps], 0.90),
            }
        )

    score_rows: list[dict[str, Any]] = []
    if tile_score_records:
        ref_values = [float(r["strength_p95"]) for r in tile_score_records]
        reference_strength = max(1.0e-12, quantile(ref_values, args.score_w_reference_quantile))
        for tau in sorted(set(float(x) for x in args.score_taus)):
            runtime_depths: list[int] = []
            oracle_depths: list[int] = []
            gaps: list[int] = []
            actual_at_stop: list[float] = []
            score_at_stop: list[float] = []
            depth_hist: dict[int, int] = defaultdict(int)
            for record in tile_score_records:
                node_norm = float(record["degree_q"]) / max(float(args.risk_max), 1.0e-12)
                node_norm = max(float(args.score_node_floor), min(1.0, node_norm))
                w_norm = float(record["strength_p95"]) / reference_strength
                w_norm = max(0.0, min(float(args.score_w_cap), w_norm))
                runtime_depth = 8
                runtime_score = 0.0
                # Choose the lowest depth whose joint risk score is within the budget.
                for depth in sorted(args.available_depths):
                    score = (
                        math.pow(node_norm, float(args.score_alpha))
                        * math.pow(w_norm, float(args.score_beta))
                        * low_bit_budget(depth, full_depth=8)
                    )
                    if score <= tau + 1e-12:
                        runtime_depth = int(depth)
                        runtime_score = float(score)
                        break
                runtime_depth = nearest_depth(runtime_depth, args.available_depths)
                oracle_depth = int(record["oracle_depth"])
                actual = 0.0 if runtime_depth == 8 else float(record.get(f"actual_p{runtime_depth}", 0.0))
                runtime_depths.append(runtime_depth)
                oracle_depths.append(oracle_depth)
                gaps.append(runtime_depth - oracle_depth)
                actual_at_stop.append(actual)
                score_at_stop.append(runtime_score)
                depth_hist[runtime_depth] += 1
            total = max(1, len(runtime_depths))
            hist_str = ",".join(f"P{d}:{depth_hist.get(d, 0) / total * 100:.1f}%" for d in sorted(args.available_depths))
            score_rows.append(
                {
                    "policy": "tile_score_v2",
                    "tau": tau,
                    "samples": total,
                    "reference_strength_q": args.score_w_reference_quantile,
                    "reference_strength": reference_strength,
                    "alpha": args.score_alpha,
                    "beta": args.score_beta,
                    "w_cap": args.score_w_cap,
                    "node_floor": args.score_node_floor,
                    "runtime_avg_depth": sum(runtime_depths) / total,
                    "oracle_avg_depth": sum(oracle_depths) / total,
                    "exact_match": sum(1 for g in gaps if g == 0) / total,
                    "conservative": sum(1 for g in gaps if g > 0) / total,
                    "aggressive": sum(1 for g in gaps if g < 0) / total,
                    "gap_mean": sum(gaps) / total,
                    "gap_p90": quantile([float(g) for g in gaps], 0.90),
                    "actual_at_stop_mean": summarize(actual_at_stop)["mean"],
                    "actual_at_stop_p90": summarize(actual_at_stop)["p90"],
                    "actual_at_stop_p95": summarize(actual_at_stop)["p95"],
                    "score_at_stop_mean": summarize(score_at_stop)["mean"],
                    "depth_hist": hist_str,
                }
            )

    write_tsv(out_dir / "depth_summary.tsv", depth_rows)
    write_tsv(out_dir / "global_depth_summary.tsv", global_rows)
    write_tsv(out_dir / "decision_summary.tsv", module_summaries)
    write_tsv(out_dir / "global_decision_summary.tsv", decision_global)
    write_tsv(out_dir / "tile_score_v2_summary.tsv", score_rows)
    if args.write_samples:
        write_tsv(out_dir / "samples.tsv", sample_rows)
    manifest = {
        "dataset": args.dataset,
        "llm_name": args.llm_name,
        "model_config": args.model_config,
        "nodes": node_ids,
        "sample_nodes": len(node_ids),
        "max_length": args.max_length,
        "layers": args.layers,
        "module_suffixes": args.module_suffixes,
        "tile_k": args.tile_k,
        "tile_n": args.tile_n,
        "tiles_per_module": args.tiles_per_module,
        "rows_per_module": args.rows_per_module,
        "depths": args.depths,
        "tolerance": {
            "min_tol": args.min_tol,
            "max_tol": args.max_tol,
            "gamma": args.gamma,
            "risk_max": args.risk_max,
        },
        "tile_score_v2": {
            "score_taus": args.score_taus,
            "score_alpha": args.score_alpha,
            "score_beta": args.score_beta,
            "score_w_cap": args.score_w_cap,
            "score_w_reference_quantile": args.score_w_reference_quantile,
            "score_node_floor": args.score_node_floor,
        },
        "outputs": {
            "global_depth_summary": str(out_dir / "global_depth_summary.tsv"),
            "global_decision_summary": str(out_dir / "global_decision_summary.tsv"),
            "tile_score_v2_summary": str(out_dir / "tile_score_v2_summary.tsv"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[TileBound] wrote {out_dir / 'global_depth_summary.tsv'}")
    print(f"[TileBound] wrote {out_dir / 'global_decision_summary.tsv'}")
    for row in global_rows:
        if row["mode"] in ("range", "tile_p95", "exact_l1"):
            print(
                f"[TileBound] {row['mode']} P{row['depth']} actual_mean={row['actual_mean']:.5f} "
                f"bound={row['bound_mean']:.5f} coverage={row['coverage']:.3f}"
            )
    for row in decision_global:
        print(
            f"[TileBound] decision {row['mode']} runtime_avg={row['runtime_avg_depth']:.2f} "
            f"oracle_avg={row['oracle_avg_depth']:.2f} exact={row['exact_match']:.3f} "
            f"cons={row['conservative']:.3f} aggr={row['aggressive']:.3f}"
        )
    for row in score_rows:
        print(
            f"[TileScoreV2] tau={row['tau']:.5f} runtime_avg={row['runtime_avg_depth']:.2f} "
            f"oracle_avg={row['oracle_avg_depth']:.2f} aggr={row['aggressive']:.3f} "
            f"actual_p90={row['actual_at_stop_p90']:.5f} hist={row['depth_hist']}"
        )


if __name__ == "__main__":
    main()
