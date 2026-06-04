#!/usr/bin/env python3
"""Diagnose graph-aware BFP stress from real LLaMA activation blocks.

The embedding-level proxy in ``diagnose_graphbfp_signal_quality.py`` measures
BFP stress from final node embeddings.  This script moves the stress signal to
the actual LLaMA encoder path: it hooks selected Linear modules, records their
input activations, computes BFP block stress per sampled node, and then checks
whether that activation-level signal predicts BFPA4 damage or BFPA4->BFPA6
rescue.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_raw_texts, load_run_state  # noqa: E402
from GraphhopSimhash.generate_real_quant_pools import bfp_fake_quantize, load_model_and_tokenizer  # noqa: E402
from GraphhopSimhash.real_quant import build_real_quant_scores, default_pool_path, load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import (  # noqa: E402
    evaluate_gnn_embeddings,
    forward_gnn_logits,
    node_logit_margin,
    train_baseline_model,
)
from GraphhopSimhash.scripts.diagnose_graphbfp_signal_quality import (  # noqa: E402
    _auc_score,
    _mean_rows,
    _normalize,
    _pearson,
    _spearman,
    _topk_overlap,
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


def _make_run_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        llm_name=args.llm_name,
        emb_dim=args.emb_dim,
        radius=args.radius,
        max_test=args.max_test,
        standard_eval_baseline=args.standard_eval_baseline,
        hash_view=args.hash_view,
        hash_mix_weights=args.hash_mix_weights,
        sketch_bits=args.sketch_bits,
        controller_seed=seed,
        run_seed=seed,
        score_rarity_bits=args.score_rarity_bits,
        score_rarity_seed=args.score_rarity_seed,
        score_propagation_weight=args.score_propagation_weight,
        score_graph_context_weight=args.score_graph_context_weight,
        score_low_unique_weight=args.score_low_unique_weight,
    )


def _module_selected(name: str, layer_ids: set[int], suffixes: tuple[str, ...]) -> bool:
    if not any(name.endswith("." + suffix) for suffix in suffixes):
        return False
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return bool(match and int(match.group(1)) in layer_ids)


def _cosine_error(ref: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(ref, other, dim=1)).clamp_min(0.0)


def _select_nodes(
    n: int,
    risk: torch.Tensor,
    sample_nodes: int,
    seed: int,
    mode: str,
) -> torch.Tensor:
    sample_nodes = min(int(sample_nodes), int(n))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if mode == "random":
        return torch.randperm(n, generator=gen)[:sample_nodes]

    random_k = max(1, sample_nodes // 2)
    tail_k = max(1, (sample_nodes - random_k) // 2)
    high_k = max(0, sample_nodes - random_k - tail_k)
    pieces = [
        torch.randperm(n, generator=gen)[:random_k],
        torch.argsort(risk.detach().cpu(), descending=False)[:tail_k],
        torch.argsort(risk.detach().cpu(), descending=True)[:high_k],
    ]
    merged: list[int] = []
    seen: set[int] = set()
    for item in torch.cat(pieces).tolist():
        idx = int(item)
        if idx not in seen:
            merged.append(idx)
            seen.add(idx)
    if len(merged) < sample_nodes:
        for item in torch.randperm(n, generator=gen).tolist():
            idx = int(item)
            if idx not in seen:
                merged.append(idx)
                seen.add(idx)
            if len(merged) >= sample_nodes:
                break
    return torch.tensor(merged[:sample_nodes], dtype=torch.long)


def _activation_block_stats(
    x: torch.Tensor,
    block_size: int,
    base_mantissa: int,
    refine_mantissa: int,
) -> dict[str, float]:
    """Return BFP stress stats for one node's valid token rows."""
    x = x.detach().to(torch.float32)
    if x.numel() == 0:
        return {
            "act_stress_mean": 0.0,
            "act_stress_p90": 0.0,
            "act_outlier_mean": 0.0,
            "act_outlier_p90": 0.0,
            "act_zero_pressure": 0.0,
            "act_bfpa4_err": 0.0,
            "act_bfpa6_err": 0.0,
            "act_rescue": 0.0,
        }

    hidden = int(x.size(-1))
    pad = (int(block_size) - (hidden % int(block_size))) % int(block_size)
    x_work = F.pad(x, (0, pad)) if pad else x
    grouped = x_work.abs().reshape(-1, int(block_size))
    eps = 1.0e-12
    max_abs = grouped.amax(dim=-1)
    median_abs = grouped.median(dim=-1).values.clamp_min(eps)
    mean_abs = grouped.mean(dim=-1).clamp_min(eps)
    spread = torch.log2((max_abs / median_abs).clamp_min(1.0))
    outlier = torch.log2((max_abs / mean_abs).clamp_min(1.0))
    zero_pressure = (grouped < (max_abs.unsqueeze(-1).clamp_min(eps) / 16.0)).to(torch.float32).mean(dim=-1)

    q_base = bfp_fake_quantize(x, mantissa_bit=int(base_mantissa), block_size=int(block_size), dim=-1)
    q_refine = bfp_fake_quantize(x, mantissa_bit=int(refine_mantissa), block_size=int(block_size), dim=-1)
    denom = torch.linalg.vector_norm(x).clamp_min(eps)
    base_err = torch.linalg.vector_norm(q_base.to(torch.float32) - x) / denom
    refine_err = torch.linalg.vector_norm(q_refine.to(torch.float32) - x) / denom
    return {
        "act_stress_mean": float(spread.mean().item()),
        "act_stress_p90": float(torch.quantile(spread, 0.90).item()),
        "act_outlier_mean": float(outlier.mean().item()),
        "act_outlier_p90": float(torch.quantile(outlier, 0.90).item()),
        "act_zero_pressure": float(zero_pressure.mean().item()),
        "act_bfpa4_err": float(base_err.item()),
        "act_bfpa6_err": float(refine_err.item()),
        "act_rescue": float((base_err - refine_err).item()),
    }


def _collect_activation_trace(
    args: argparse.Namespace,
    sample_nodes: torch.Tensor,
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    texts = load_raw_texts(args.dataset)
    model, tokenizer, _tag = load_model_and_tokenizer(args.model_name, args.model_config, args.cache_dir, force_cpu=False)
    target_model = model.model if hasattr(model, "model") and model.__class__.__name__.endswith("ForCausalLM") else model
    target_model.eval()
    device = next(target_model.parameters()).device
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    layer_ids = {int(x) for x in args.layers}
    suffixes = tuple(str(x) for x in args.module_suffixes)
    selected = [(name, module) for name, module in target_model.named_modules() if _module_selected(name, layer_ids, suffixes)]
    if not selected:
        raise RuntimeError("No LLaMA modules matched --layers/--module_suffixes")

    node_accum: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    raw_rows: list[dict[str, Any]] = []
    current_nodes: torch.Tensor | None = None
    current_mask: torch.Tensor | None = None

    def process_activation(module_name: str, x: torch.Tensor, mask: torch.Tensor, node_ids: torch.Tensor) -> None:
        if x.ndim != 3:
            return
        x = x.detach()
        mask = mask.detach().bool()
        for batch_idx, node_id in enumerate(node_ids.tolist()):
            valid = mask[batch_idx]
            if int(valid.sum().item()) <= 0:
                continue
            stats = _activation_block_stats(
                x[batch_idx, valid, :],
                block_size=args.block_size,
                base_mantissa=args.base_mantissa,
                refine_mantissa=args.refine_mantissa,
            )
            nid = int(node_id)
            for key, value in stats.items():
                node_accum[nid][key].append(float(value))
            raw_rows.append(
                {
                    "node_id": nid,
                    "module": module_name,
                    "tokens": int(valid.sum().item()),
                    **stats,
                }
            )

    hooks = []
    for module_name, module in selected:
        def make_hook(name: str):
            def hook(_module, inputs, _outputs):
                if current_mask is None or current_nodes is None:
                    return
                process_activation(name, inputs[0], current_mask, current_nodes)

            return hook

        hooks.append(module.register_forward_hook(make_hook(module_name)))

    node_list = sample_nodes.detach().cpu().tolist()
    try:
        with torch.no_grad():
            for start in range(0, len(node_list), int(args.batch_size)):
                ids = torch.tensor(node_list[start : start + int(args.batch_size)], dtype=torch.long)
                batch_texts = [texts[int(idx)] for idx in ids.tolist()]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                    return_tensors="pt",
                )
                current_nodes = ids
                current_mask = encoded["attention_mask"].to(device)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                _ = model(**encoded)
    finally:
        for hook in hooks:
            hook.remove()

    _write_tsv(out_dir / "activation_module_trace.tsv", raw_rows)
    node_rows: list[dict[str, Any]] = []
    metrics = [
        "act_stress_mean",
        "act_stress_p90",
        "act_outlier_mean",
        "act_outlier_p90",
        "act_zero_pressure",
        "act_bfpa4_err",
        "act_bfpa6_err",
        "act_rescue",
    ]
    n_total = int(max(node_list) + 1) if node_list else 0
    # The returned tensors are sized later by caller; here we keep sparse rows.
    sparse: dict[str, dict[int, float]] = {key: {} for key in metrics}
    for nid in node_list:
        acc = node_accum[int(nid)]
        row: dict[str, Any] = {"node_id": int(nid), "modules_seen": len(acc.get("act_stress_mean", []))}
        for key in metrics:
            values = acc.get(key, [])
            value = float(np.mean(values)) if values else 0.0
            row[key] = value
            sparse[key][int(nid)] = value
        node_rows.append(row)
    _write_tsv(out_dir / "activation_node_trace.tsv", node_rows)

    return {"sample_nodes": sample_nodes, "sparse": sparse, "n_total_hint": torch.tensor(n_total)}


def _dense_from_sparse(sparse: dict[int, float], n: int, sample_nodes: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = torch.zeros(n, dtype=torch.float32, device=device)
    for node_id in sample_nodes.tolist():
        out[int(node_id)] = float(sparse.get(int(node_id), 0.0))
    return out


def _build_signals(
    args: argparse.Namespace,
    graph_scores: dict[str, torch.Tensor],
    act: dict[str, torch.Tensor],
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    degree = _normalize(graph_scores["propagation_q"].to(device))
    tser = _normalize(graph_scores["sensitivity_q"].to(device))
    stress = _normalize(act[args.activation_metric].to(device))
    rescue = _normalize(act["act_rescue"].to(device))
    zero = _normalize(act["act_zero_pressure"].to(device))
    stress_mix = _normalize((1.0 - args.zero_weight) * stress + args.zero_weight * zero)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 3433)
    random = torch.rand(degree.numel(), generator=gen).to(device)
    eps = 1.0e-6
    return {
        "Random": random,
        "ActStress": stress_mix,
        "ActRescue": rescue,
        "Degree": degree,
        "TSER": tser,
        "DegreeXActStress": degree.clamp_min(eps).pow(args.graph_power) * stress_mix.clamp_min(eps).pow(args.stress_power),
        "TSERXActStress": tser.clamp_min(eps).pow(args.graph_power) * stress_mix.clamp_min(eps).pow(args.stress_power),
        "DegreePlusActStress": _normalize(args.graph_weight * degree + args.stress_weight * stress_mix),
        "TSERPlusActStress": _normalize(args.graph_weight * tser + args.stress_weight * stress_mix),
    }


def _mix_by_sample_score(
    base_hidden: torch.Tensor,
    refine_hidden: torch.Tensor,
    score: torch.Tensor,
    sample_mask: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_idx = torch.nonzero(sample_mask, as_tuple=False).flatten()
    k = max(0, min(int(sample_idx.numel()), int(round(float(ratio) * int(sample_idx.numel())))))
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    if k > 0:
        local = sample_idx[torch.argsort(score[sample_idx], descending=True)[:k]]
        mask[local] = True
    mixed = base_hidden.clone()
    mixed[mask] = refine_hidden[mask]
    return mixed, mask


def _format_signal_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("signal", 20),
        ("act_err", 8),
        ("hidden", 8),
        ("gain", 8),
        ("margin", 8),
        ("dmg", 8),
        ("top25", 8),
    ]
    lines = [" ".join(name.rjust(width) for name, width in cols)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        vals = {
            "signal": row["signal"],
            "act_err": f"{row['spearman_act_err']:.3f}",
            "hidden": f"{row['spearman_hidden_err']:.3f}",
            "gain": f"{row['spearman_hidden_gain']:.3f}",
            "margin": f"{row['spearman_margin_drop']:.3f}",
            "dmg": f"{row['auc_damage']:.3f}",
            "top25": f"{row['top25_overlap_gain']:.3f}",
        }
        lines.append(" ".join(str(vals[name]).rjust(width) for name, width in cols))
    return "\n".join(lines)


def _format_oracle_table(rows: list[dict[str, Any]]) -> str:
    cols = [("ratio", 7), ("policy", 22), ("drop", 8), ("gain", 8), ("acc", 8)]
    lines = [" ".join(name.rjust(width) for name, width in cols)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        vals = {
            "ratio": f"{row['ratio']:.0%}",
            "policy": row["policy"],
            "drop": f"{row['drop']:.2%}",
            "gain": f"{row['gain_vs_base']:.2%}",
            "acc": f"{row['acc']:.4f}",
        }
        lines.append(" ".join(str(vals[name]).rjust(width) for name, width in cols))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--model_name", default="llama2_7b")
    parser.add_argument("--model_config", default="fp16")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--base_tag", default="W4BFPA4_B128")
    parser.add_argument("--refine_tag", default="W4BFPA6_B128")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--base_mantissa", type=int, default=4)
    parser.add_argument("--refine_mantissa", type=int, default=6)
    parser.add_argument("--sample_nodes", type=int, default=128)
    parser.add_argument("--sample_mode", choices=["random", "stratified"], default="stratified")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument("--module_suffixes", nargs="+", default=["q_proj", "o_proj", "up_proj", "down_proj"])
    parser.add_argument(
        "--signals",
        nargs="+",
        default=[
            "Random",
            "ActStress",
            "ActRescue",
            "Degree",
            "TSER",
            "DegreeXActStress",
            "TSERXActStress",
            "DegreePlusActStress",
            "TSERPlusActStress",
        ],
    )
    parser.add_argument("--activation_metric", default="act_outlier_p90")
    parser.add_argument("--zero_weight", type=float, default=0.0)
    parser.add_argument("--graph_power", type=float, default=1.0)
    parser.add_argument("--stress_power", type=float, default=1.0)
    parser.add_argument("--graph_weight", type=float, default=0.6)
    parser.add_argument("--stress_weight", type=float, default=0.4)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.10, 0.20, 0.25, 0.30, 0.40])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm_name", default="ST")
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--standard_eval_baseline", action="store_true")
    parser.add_argument("--hash_view", default="mix")
    parser.add_argument("--hash_mix_weights", nargs=3, type=float, default=[0.30, 0.70, 0.0])
    parser.add_argument("--sketch_bits", type=int, default=14)
    parser.add_argument("--score_rarity_bits", type=int, default=12)
    parser.add_argument("--score_rarity_seed", type=int, default=12345)
    parser.add_argument("--score_propagation_weight", type=float, default=3.0)
    parser.add_argument("--score_graph_context_weight", type=float, default=1.0)
    parser.add_argument("--score_low_unique_weight", type=float, default=1.0)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / "graphbfp_activation_stress" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = load_tensor_pool(default_pool_path(args.dataset, args.model_name, args.reference_tag), device)
    base = load_tensor_pool(default_pool_path(args.dataset, args.model_name, args.base_tag), device)
    refine = load_tensor_pool(default_pool_path(args.dataset, args.model_name, args.refine_tag), device)
    if ref.shape != base.shape or ref.shape != refine.shape:
        raise ValueError("reference/base/refine pools must have identical shape")

    seed0 = int(args.seed)
    run_args0 = _make_run_args(args, seed0)
    _conf0, data0, verify_features0, run_device0 = load_run_state(args.dataset, run_args0, seed0)
    scores0 = build_real_quant_scores(verify_features0, data0, run_args0, run_device0)
    risk0 = scores0["sensitivity_q"].detach().cpu()
    sample_nodes = _select_nodes(ref.size(0), risk0, args.sample_nodes, seed0, args.sample_mode)
    sample_mask_cpu = torch.zeros(ref.size(0), dtype=torch.bool)
    sample_mask_cpu[sample_nodes] = True

    print(
        f"[ActivationTrace] dataset={args.dataset} nodes={int(sample_nodes.numel())} "
        f"layers={args.layers} modules={args.module_suffixes} max_length={args.max_length}"
    )
    trace = _collect_activation_trace(args, sample_nodes, out_dir)
    sparse = trace["sparse"]

    signal_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    baseline_accs: list[float] = []
    base_accs: list[float] = []
    refine_accs: list[float] = []

    for run_idx in range(int(args.runs)):
        seed = int(args.seed) + run_idx
        run_args = _make_run_args(args, seed)
        _conf, data, verify_features, run_device = load_run_state(args.dataset, run_args, seed)
        ref_dev = ref.to(run_device)
        base_dev = base.to(run_device)
        refine_dev = refine.to(run_device)
        sample_mask = sample_mask_cpu.to(run_device)

        data.x = ref_dev
        model, baseline_acc, ref_hidden, ref_logits = train_baseline_model(data, run_args, run_device)
        with torch.no_grad():
            base_hidden = model.encoder(base_dev)
            refine_hidden = model.encoder(refine_dev)
            base_logits = forward_gnn_logits(model, data, base_hidden)
            refine_logits = forward_gnn_logits(model, data, refine_hidden)

        base_acc = evaluate_gnn_embeddings(model, data, base_hidden)
        refine_acc = evaluate_gnn_embeddings(model, data, refine_hidden)
        baseline_accs.append(float(baseline_acc))
        base_accs.append(float(base_acc))
        refine_accs.append(float(refine_acc))

        act = {key: _dense_from_sparse(value, ref.size(0), sample_nodes, run_device) for key, value in sparse.items()}
        scores = build_real_quant_scores(verify_features, data, run_args, run_device)
        signals = _build_signals(args, scores, act, seed, run_device)

        with torch.no_grad():
            act_err = act["act_bfpa4_err"]
            act_gain = act["act_rescue"]
            hidden_err = _cosine_error(ref_hidden, base_hidden)
            hidden_refine_err = _cosine_error(ref_hidden, refine_hidden)
            hidden_gain = hidden_err - hidden_refine_err
            ref_margin = node_logit_margin(ref_logits)
            base_margin = node_logit_margin(base_logits)
            margin_drop = ref_margin - base_margin
            ref_pred = ref_logits.argmax(dim=1)
            base_pred = base_logits.argmax(dim=1)
            flip = base_pred != ref_pred
            damage = (ref_pred == data.y) & (base_pred != data.y)
            ref_loss = F.cross_entropy(ref_logits, data.y, reduction="none")
            base_loss = F.cross_entropy(base_logits, data.y, reduction="none")
            loss_increase = base_loss - ref_loss

        mask = sample_mask
        for name in args.signals:
            if name not in signals:
                raise ValueError(f"unknown signal {name}; choices={sorted(signals)}")
            sig = signals[name].detach()
            signal_rows.append(
                {
                    "seed": seed,
                    "signal": name,
                    "sample_nodes": int(mask.sum().item()),
                    "pearson_act_err": _pearson(sig[mask], act_err[mask]),
                    "spearman_act_err": _spearman(sig[mask], act_err[mask]),
                    "spearman_act_gain": _spearman(sig[mask], act_gain[mask]),
                    "spearman_hidden_err": _spearman(sig[mask], hidden_err[mask]),
                    "spearman_hidden_gain": _spearman(sig[mask], hidden_gain[mask]),
                    "spearman_margin_drop": _spearman(sig[mask], margin_drop[mask]),
                    "spearman_loss_increase": _spearman(sig[mask], loss_increase[mask]),
                    "auc_flip": _auc_score(sig[mask], flip[mask]),
                    "auc_damage": _auc_score(sig[mask], damage[mask]),
                    "top10_overlap_gain": _topk_overlap(sig[mask], hidden_gain[mask], 0.10),
                    "top25_overlap_gain": _topk_overlap(sig[mask], hidden_gain[mask], 0.25),
                    "top40_overlap_gain": _topk_overlap(sig[mask], hidden_gain[mask], 0.40),
                }
            )

        oracle_scores = dict(signals)
        oracle_scores["OracleHiddenGain"] = hidden_gain
        oracle_scores["OracleLossIncrease"] = loss_increase
        for ratio in args.ratios:
            for name, sig in oracle_scores.items():
                mixed_hidden, route_mask = _mix_by_sample_score(base_hidden, refine_hidden, sig, sample_mask, ratio)
                acc = evaluate_gnn_embeddings(model, data, mixed_hidden)
                oracle_rows.append(
                    {
                        "seed": seed,
                        "ratio": float(ratio),
                        "policy": name,
                        "sample_refined": int(route_mask.sum().item()),
                        "baseline_acc": float(baseline_acc),
                        "base_acc": float(base_acc),
                        "acc": float(acc),
                        "drop": float(baseline_acc - acc),
                        "gain_vs_base": float(acc - base_acc),
                    }
                )

    signal_summary = _mean_rows(
        signal_rows,
        [
            "sample_nodes",
            "pearson_act_err",
            "spearman_act_err",
            "spearman_act_gain",
            "spearman_hidden_err",
            "spearman_hidden_gain",
            "spearman_margin_drop",
            "spearman_loss_increase",
            "auc_flip",
            "auc_damage",
            "top10_overlap_gain",
            "top25_overlap_gain",
            "top40_overlap_gain",
        ],
        ["signal"],
    )
    oracle_summary = _mean_rows(
        oracle_rows,
        ["sample_refined", "baseline_acc", "base_acc", "acc", "drop", "gain_vs_base"],
        ["ratio", "policy"],
    )
    _write_tsv(out_dir / "activation_signal_per_seed.tsv", signal_rows)
    _write_tsv(out_dir / "activation_signal_summary.tsv", signal_summary)
    _write_tsv(out_dir / "activation_sample_routing_per_seed.tsv", oracle_rows)
    _write_tsv(out_dir / "activation_sample_routing.tsv", oracle_summary)

    best_by_ratio = []
    for ratio in sorted({float(row["ratio"]) for row in oracle_summary}):
        candidates = [row for row in oracle_summary if math.isclose(float(row["ratio"]), ratio)]
        best_by_ratio.append(min(candidates, key=lambda row: float(row["drop"])))

    note = "\n".join(
        [
            f"Graph-BFP activation-stress diagnosis | dataset={args.dataset} | runs={args.runs}",
            f"reference={args.reference_tag} | base={args.base_tag} | refine={args.refine_tag}",
            f"sample_nodes={int(sample_nodes.numel())} | max_length={args.max_length} | model_config={args.model_config}",
            f"layers={args.layers} | modules={args.module_suffixes}",
            f"Baseline Acc: {float(np.mean(baseline_accs)):.4f}",
            f"All {args.base_tag}: Acc={float(np.mean(base_accs)):.4f}, Drop={float(np.mean(np.array(baseline_accs) - np.array(base_accs))):.2%}",
            f"All {args.refine_tag}: Acc={float(np.mean(refine_accs)):.4f}, Drop={float(np.mean(np.array(baseline_accs) - np.array(refine_accs))):.2%}",
            "",
            "Activation signal quality on sampled nodes:",
            _format_signal_table(signal_summary),
            "",
            "Best sample-node routing policy per ratio:",
            _format_oracle_table(best_by_ratio),
        ]
    )
    (out_dir / "summary.txt").write_text(note + "\n", encoding="utf-8")
    print(note)
    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    main()
