#!/usr/bin/env python3
"""Diagnose graph-risk and BFP-stress signals for BFP refinement.

This script separates three questions:

1. Is there enough oracle rescue space from BFPA4 -> BFPA6?
2. Do graph-risk / BFP-stress signals predict numerical error or downstream damage?
3. Does graph-risk x BFP-stress isolate a more fragile node bucket?
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.real_quant import build_real_quant_scores, default_pool_path, load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import (  # noqa: E402
    evaluate_gnn_embeddings,
    forward_gnn_logits,
    node_logit_margin,
    train_baseline_model,
)


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


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(torch.float32)
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return torch.zeros_like(x)
    lo = torch.quantile(x[finite], 0.01)
    hi = torch.quantile(x[finite], 0.99)
    denom = (hi - lo).clamp_min(1.0e-12)
    return torch.clamp((x - lo) / denom, 0.0, 1.0)


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().flatten().to(torch.float32)
    y = y.detach().flatten().to(torch.float32)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).mean()).clamp_min(1.0e-12) * torch.sqrt((y * y).mean()).clamp_min(1.0e-12)
    return float(((x * y).mean() / denom).item())


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearson(_rank(x.detach().flatten()), _rank(y.detach().flatten()))


def _auc_score(score: torch.Tensor, label: torch.Tensor) -> float:
    score = score.detach().flatten().to(torch.float32)
    label = label.detach().flatten().to(torch.bool)
    finite = torch.isfinite(score)
    score = score[finite]
    label = label[finite]
    pos = int(label.sum().item())
    neg = int((~label).sum().item())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _rank(score) + 1.0
    pos_rank_sum = ranks[label].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc.item())


def _topk_overlap(score: torch.Tensor, oracle: torch.Tensor, ratio: float) -> float:
    n = int(score.numel())
    k = max(1, min(n, int(round(float(ratio) * n))))
    s_idx = torch.argsort(score, descending=True)[:k]
    o_idx = torch.argsort(oracle, descending=True)[:k]
    mask = torch.zeros(n, dtype=torch.bool, device=score.device)
    mask[s_idx] = True
    return float(mask[o_idx].float().mean().item())


def _bfp_row_stress(x: torch.Tensor, block_size: int) -> dict[str, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(f"expected [N, D] tensor, got {tuple(x.shape)}")
    n, d = x.shape
    pad = (int(block_size) - (d % int(block_size))) % int(block_size)
    if pad:
        x = F.pad(x, (0, pad))
    grouped = x.detach().abs().to(torch.float32).reshape(n, -1, int(block_size))
    eps = 1.0e-12
    max_abs = grouped.amax(dim=-1)
    median_abs = grouped.median(dim=-1).values.clamp_min(eps)
    mean_abs = grouped.mean(dim=-1).clamp_min(eps)
    spread = torch.log2((max_abs / median_abs).clamp_min(1.0))
    outlier = torch.log2((max_abs / mean_abs).clamp_min(1.0))
    zero_pressure = (grouped < (max_abs.unsqueeze(-1).clamp_min(eps) / 16.0)).to(torch.float32).mean(dim=-1)
    return {
        "stress_mean": spread.mean(dim=1),
        "stress_p90": torch.quantile(spread, 0.90, dim=1),
        "outlier_mean": outlier.mean(dim=1),
        "outlier_p90": torch.quantile(outlier, 0.90, dim=1),
        "zero_pressure": zero_pressure.mean(dim=1),
    }


def _build_signals(
    scores: dict[str, torch.Tensor],
    stress: dict[str, torch.Tensor],
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    degree = _normalize(scores["propagation_q"].to(device))
    tser = _normalize(scores["sensitivity_q"].to(device))
    stress_main = _normalize(stress[args.stress_metric].to(device))
    zero = _normalize(stress["zero_pressure"].to(device))
    stress_mix = _normalize((1.0 - args.zero_weight) * stress_main + args.zero_weight * zero)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 1777)
    random = torch.rand(degree.numel(), generator=gen).to(device)

    eps = 1.0e-6
    return {
        "Random": random,
        "Stress": stress_mix,
        "Degree": degree,
        "TSER": tser,
        "DegreeXStress": torch.pow(degree.clamp_min(eps), args.graph_power)
        * torch.pow(stress_mix.clamp_min(eps), args.stress_power),
        "TSERXStress": torch.pow(tser.clamp_min(eps), args.graph_power)
        * torch.pow(stress_mix.clamp_min(eps), args.stress_power),
        "DegreePlusStress": _normalize(args.graph_weight * degree + args.stress_weight * stress_mix),
        "TSERPlusStress": _normalize(args.graph_weight * tser + args.stress_weight * stress_mix),
    }


def _mix_by_score(base_hidden: torch.Tensor, refine_hidden: torch.Tensor, score: torch.Tensor, ratio: float):
    n = int(score.numel())
    k = max(0, min(n, int(round(float(ratio) * n))))
    mask = torch.zeros(n, dtype=torch.bool, device=score.device)
    if k > 0:
        mask[torch.argsort(score, descending=True)[:k]] = True
    mixed = base_hidden.clone()
    mixed[mask] = refine_hidden[mask]
    return mixed, mask


def _cosine_error(ref: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(ref, other, dim=1)).clamp_min(0.0)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _mean_rows(rows: list[dict[str, Any]], keys: list[str], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for group, items in sorted(grouped.items(), key=lambda item: item[0]):
        row = {k: group[i] for i, k in enumerate(group_keys)}
        for key in keys:
            vals = [float(item[key]) for item in items if not math.isnan(float(item[key]))]
            row[key] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out


def _format_signal_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("signal", 16),
        ("err", 7),
        ("gain", 7),
        ("margin", 7),
        ("flip", 7),
        ("dmg", 7),
        ("top25", 7),
    ]
    lines = [" ".join(name.rjust(width) for name, width in cols)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        vals = {
            "signal": row["signal"],
            "err": f"{row['spearman_hidden_err']:.3f}",
            "gain": f"{row['spearman_hidden_gain']:.3f}",
            "margin": f"{row['spearman_margin_drop']:.3f}",
            "flip": f"{row['auc_flip']:.3f}",
            "dmg": f"{row['auc_damage']:.3f}",
            "top25": f"{row['top25_overlap_gain']:.3f}",
        }
        lines.append(" ".join(str(vals[name]).rjust(width) for name, width in cols))
    return "\n".join(lines)


def _format_oracle_table(rows: list[dict[str, Any]]) -> str:
    cols = [("ratio", 7), ("policy", 16), ("drop", 8), ("gain", 8), ("acc", 8)]
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
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--base_tag", default="W4BFPA4_B128")
    parser.add_argument("--refine_tag", default="W4BFPA6_B128")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--stress_metric", default="outlier_p90", choices=["stress_mean", "stress_p90", "outlier_mean", "outlier_p90"])
    parser.add_argument("--zero_weight", type=float, default=0.0)
    parser.add_argument("--graph_power", type=float, default=1.0)
    parser.add_argument("--stress_power", type=float, default=1.0)
    parser.add_argument("--graph_weight", type=float, default=0.6)
    parser.add_argument("--stress_weight", type=float, default=0.4)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.10, 0.20, 0.25, 0.30, 0.40])
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["Random", "Stress", "Degree", "TSER", "DegreeXStress", "TSERXStress", "DegreePlusStress", "TSERPlusStress"],
    )
    parser.add_argument("--bucket_graph_signal", default="TSER", choices=["Degree", "TSER"])
    parser.add_argument("--bucket_quantile", type=float, default=0.50)
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
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / "graphbfp_signal_diagnosis" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = load_tensor_pool(default_pool_path(args.dataset, args.model_name, args.reference_tag), device)
    base = load_tensor_pool(default_pool_path(args.dataset, args.model_name, args.base_tag), device)
    refine = load_tensor_pool(default_pool_path(args.dataset, args.model_name, args.refine_tag), device)
    if ref.shape != base.shape or ref.shape != refine.shape:
        raise ValueError("reference/base/refine pools must have identical shape")

    stress_cpu = _bfp_row_stress(ref.detach().cpu(), args.block_size)
    stress = {key: value.to(device) for key, value in stress_cpu.items()}

    signal_rows = []
    oracle_rows = []
    bucket_rows = []
    baseline_accs = []
    base_accs = []
    refine_accs = []

    for run_idx in range(args.runs):
        seed = int(args.seed) + run_idx
        run_args = _make_run_args(args, seed)
        _conf, data, verify_features, run_device = load_run_state(args.dataset, run_args, seed)
        ref_dev = ref.to(run_device)
        base_dev = base.to(run_device)
        refine_dev = refine.to(run_device)
        stress_dev = {key: value.to(run_device) for key, value in stress.items()}

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

        scores = build_real_quant_scores(verify_features, data, run_args, run_device)
        signals = _build_signals(scores, stress_dev, args, seed, run_device)

        with torch.no_grad():
            raw_err = _cosine_error(ref_dev, base_dev)
            raw_refine_err = _cosine_error(ref_dev, refine_dev)
            raw_gain = raw_err - raw_refine_err
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

        for name in args.signals:
            if name not in signals:
                raise ValueError(f"unknown signal {name}; choices={sorted(signals)}")
            sig = signals[name].detach()
            signal_rows.append(
                {
                    "seed": seed,
                    "signal": name,
                    "pearson_raw_err": _pearson(sig, raw_err),
                    "spearman_raw_err": _spearman(sig, raw_err),
                    "spearman_raw_gain": _spearman(sig, raw_gain),
                    "spearman_hidden_err": _spearman(sig, hidden_err),
                    "spearman_hidden_gain": _spearman(sig, hidden_gain),
                    "spearman_margin_drop": _spearman(sig, margin_drop),
                    "spearman_loss_increase": _spearman(sig, loss_increase),
                    "auc_flip": _auc_score(sig, flip),
                    "auc_damage": _auc_score(sig, damage),
                    "top10_overlap_gain": _topk_overlap(sig, hidden_gain, 0.10),
                    "top25_overlap_gain": _topk_overlap(sig, hidden_gain, 0.25),
                    "top40_overlap_gain": _topk_overlap(sig, hidden_gain, 0.40),
                }
            )

        oracle_scores = dict(signals)
        oracle_scores["OracleHiddenGain"] = hidden_gain
        oracle_scores["OracleLossIncrease"] = loss_increase
        for ratio in args.ratios:
            for name, sig in oracle_scores.items():
                mixed_hidden, _mask = _mix_by_score(base_hidden, refine_hidden, sig, ratio)
                acc = evaluate_gnn_embeddings(model, data, mixed_hidden)
                oracle_rows.append(
                    {
                        "seed": seed,
                        "ratio": float(ratio),
                        "policy": name,
                        "baseline_acc": float(baseline_acc),
                        "base_acc": float(base_acc),
                        "acc": float(acc),
                        "drop": float(baseline_acc - acc),
                        "gain_vs_base": float(acc - base_acc),
                    }
                )

        graph_sig = signals[args.bucket_graph_signal]
        stress_sig = _normalize(stress_dev[args.stress_metric])
        graph_thr = torch.quantile(graph_sig, float(args.bucket_quantile))
        stress_thr = torch.quantile(stress_sig, float(args.bucket_quantile))
        graph_high = graph_sig >= graph_thr
        stress_high = stress_sig >= stress_thr
        bucket_defs = {
            "lowG_lowS": (~graph_high) & (~stress_high),
            "lowG_highS": (~graph_high) & stress_high,
            "highG_lowS": graph_high & (~stress_high),
            "highG_highS": graph_high & stress_high,
        }
        n = max(1, int(graph_sig.numel()))
        for bucket_name, mask in bucket_defs.items():
            if int(mask.sum().item()) == 0:
                continue
            bucket_rows.append(
                {
                    "seed": seed,
                    "graph_signal": args.bucket_graph_signal,
                    "bucket": bucket_name,
                    "node_frac": float(mask.float().mean().item()),
                    "raw_err": float(raw_err[mask].mean().item()),
                    "hidden_err": float(hidden_err[mask].mean().item()),
                    "hidden_gain": float(hidden_gain[mask].mean().item()),
                    "margin_drop": float(margin_drop[mask].mean().item()),
                    "loss_increase": float(loss_increase[mask].mean().item()),
                    "flip_rate": float(flip[mask].float().mean().item()),
                    "damage_rate": float(damage[mask].float().mean().item()),
                    "nodes": int(mask.sum().item()),
                    "total_nodes": n,
                }
            )

    signal_summary = _mean_rows(
        signal_rows,
        [
            "pearson_raw_err",
            "spearman_raw_err",
            "spearman_raw_gain",
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
        ["baseline_acc", "base_acc", "acc", "drop", "gain_vs_base"],
        ["ratio", "policy"],
    )
    bucket_summary = _mean_rows(
        bucket_rows,
        ["node_frac", "raw_err", "hidden_err", "hidden_gain", "margin_drop", "loss_increase", "flip_rate", "damage_rate"],
        ["graph_signal", "bucket"],
    )

    _write_tsv(out_dir / "signal_summary.tsv", signal_summary)
    _write_tsv(out_dir / "oracle_routing.tsv", oracle_summary)
    _write_tsv(out_dir / "bucket_summary.tsv", bucket_summary)
    _write_tsv(out_dir / "signal_per_seed.tsv", signal_rows)
    _write_tsv(out_dir / "oracle_per_seed.tsv", oracle_rows)
    _write_tsv(out_dir / "bucket_per_seed.tsv", bucket_rows)

    best_by_ratio = []
    for ratio in sorted({float(row["ratio"]) for row in oracle_summary}):
        candidates = [row for row in oracle_summary if math.isclose(float(row["ratio"]), ratio)]
        best_by_ratio.append(min(candidates, key=lambda row: float(row["drop"])))

    note = "\n".join(
        [
            f"Graph-BFP signal diagnosis | dataset={args.dataset} | runs={args.runs}",
            f"reference={args.reference_tag} | base={args.base_tag} | refine={args.refine_tag}",
            f"Baseline Acc: {float(np.mean(baseline_accs)):.4f}",
            f"All {args.base_tag}: Acc={float(np.mean(base_accs)):.4f}, Drop={float(np.mean(np.array(baseline_accs) - np.array(base_accs))):.2%}",
            f"All {args.refine_tag}: Acc={float(np.mean(refine_accs)):.4f}, Drop={float(np.mean(np.array(baseline_accs) - np.array(refine_accs))):.2%}",
            "",
            "Signal quality summary:",
            _format_signal_table(signal_summary),
            "",
            "Best routing policy per ratio:",
            _format_oracle_table(best_by_ratio),
            "",
            f"2x2 bucket summary uses graph_signal={args.bucket_graph_signal}, stress={args.stress_metric}",
        ]
    )
    (out_dir / "summary.txt").write_text(note + "\n", encoding="utf-8")
    print(note)
    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    main()
