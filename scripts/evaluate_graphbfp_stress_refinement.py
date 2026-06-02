#!/usr/bin/env python3
"""Evaluate graph-risk x BFP-stress refinement for BFPA4/BFPA6 pools.

This is a fast pool-level validation for the Graph-BFP idea:

    default path:      W4BFPA4_B128
    refined path:      W4BFPA6_B128 for selected nodes
    reference path:    W4BFPA8_B128

The selector does not use oracle BFPA4/BFPA6 error.  It combines deployable
graph-risk proxies with a BFP shared-exponent stress proxy computed from the
reference feature blocks.  The stress proxy estimates whether a node has
large within-block dynamic-range pressure under BFP shared exponents.
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
from GraphhopSimhash.runner import evaluate_gnn_embeddings, train_baseline_model  # noqa: E402


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
    x = x.to(torch.float32)
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return torch.zeros_like(x, dtype=torch.float32)
    lo = torch.quantile(x[finite], 0.01)
    hi = torch.quantile(x[finite], 0.99)
    denom = (hi - lo).clamp_min(1e-12)
    return torch.clamp((x - lo) / denom, 0.0, 1.0)


def _bfp_row_stress(x: torch.Tensor, block_size: int) -> dict[str, torch.Tensor]:
    """Estimate per-node shared-exponent stress from row-wise BFP blocks.

    A BFP block shares one exponent.  If max_abs is much larger than typical
    values in the same block, small values waste mantissa range.  We summarize
    that pressure with log2(max_abs / median_abs) and a zero-pressure proxy.
    """
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
    # For BFP A4, values far below the block max tend to quantize to zero.
    zero_pressure = (grouped < (max_abs.unsqueeze(-1).clamp_min(eps) / 16.0)).to(torch.float32).mean(dim=-1)
    return {
        "stress_mean": spread.mean(dim=1),
        "stress_p90": torch.quantile(spread, 0.90, dim=1),
        "outlier_mean": outlier.mean(dim=1),
        "outlier_p90": torch.quantile(outlier, 0.90, dim=1),
        "zero_pressure": zero_pressure.mean(dim=1),
    }


def _priority_scores(
    scores: dict[str, torch.Tensor],
    stress: dict[str, torch.Tensor],
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    degree = _normalize(scores["propagation_q"].to(device))
    tser = _normalize(scores["sensitivity_q"].to(device))
    low_unique = _normalize(scores["low_degree_unique_q"].to(device))
    stress_main = _normalize(stress[args.stress_metric].to(device))
    zero = _normalize(stress["zero_pressure"].to(device))
    stress_mix = _normalize((1.0 - args.zero_weight) * stress_main + args.zero_weight * zero)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 8191)
    random = torch.rand(degree.numel(), generator=gen).to(device)

    eps = 1.0e-6
    def prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.pow(a.clamp_min(eps), args.graph_power) * torch.pow(b.clamp_min(eps), args.stress_power)

    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _normalize(args.graph_weight * a + args.stress_weight * b)

    return {
        "Random": random,
        "Stress": stress_mix,
        "Degree": degree,
        "TSER": tser,
        "LowUnique": low_unique,
        "DegreeXStress": prod(degree, stress_mix),
        "TSERXStress": prod(tser, stress_mix),
        "DegreePlusStress": add(degree, stress_mix),
        "TSERPlusStress": add(tser, stress_mix),
    }


def _mix_hidden(
    base_hidden: torch.Tensor,
    refine_hidden: torch.Tensor,
    priority: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(priority.numel())
    k = int(round(float(ratio) * n))
    k = max(0, min(k, n))
    mask = torch.zeros(n, dtype=torch.bool, device=priority.device)
    if k > 0:
        order = torch.argsort(priority, descending=True)
        mask[order[:k]] = True
    mixed = base_hidden.clone()
    if bool(mask.any()):
        mixed[mask] = refine_hidden[mask]
    return mixed, mask


def _cost_for_ratio(ratio: float, args: argparse.Namespace) -> float:
    fixed = float(args.fixed_cost)
    scale = float(args.cost_scale)
    ref_bits = float(args.reference_bits)
    base = scale * (fixed + (1.0 - fixed) * (float(args.base_bits) / ref_bits))
    refine = scale * (fixed + (1.0 - fixed) * (float(args.refine_bits) / ref_bits))
    return (1.0 - ratio) * base + ratio * refine


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
        ("ratio", 6),
        ("policy", 18),
        ("cost", 7),
        ("acc", 8),
        ("drop", 8),
        ("gain", 8),
        ("lift", 7),
    ]
    lines = [" ".join(name.rjust(width) for name, width in cols)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        vals = {
            "ratio": f"{row['ratio']:.0%}",
            "policy": row["policy"],
            "cost": f"{row['cost']:.3f}",
            "acc": f"{row['acc_mean']:.4f}",
            "drop": f"{row['drop_mean']:.2%}",
            "gain": f"{row['gain_vs_base_drop']:.2%}",
            "lift": f"{row['lift_rate']:.1%}",
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
    parser.add_argument("--reference_bits", type=int, default=8)
    parser.add_argument("--base_bits", type=int, default=4)
    parser.add_argument("--refine_bits", type=int, default=6)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--stress_metric", default="stress_p90", choices=["stress_mean", "stress_p90", "outlier_mean", "outlier_p90"])
    parser.add_argument("--zero_weight", type=float, default=0.25)
    parser.add_argument("--graph_power", type=float, default=1.0)
    parser.add_argument("--stress_power", type=float, default=1.0)
    parser.add_argument("--graph_weight", type=float, default=0.7)
    parser.add_argument("--stress_weight", type=float, default=0.3)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "Random",
            "Stress",
            "Degree",
            "TSER",
            "DegreeXStress",
            "TSERXStress",
            "DegreePlusStress",
            "TSERPlusStress",
        ],
    )
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
    parser.add_argument("--cost_scale", type=float, default=0.50)
    parser.add_argument("--fixed_cost", type=float, default=0.15)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "output" / "graphbfp_stress_refinement" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_path = default_pool_path(args.dataset, args.model_name, args.reference_tag)
    base_path = default_pool_path(args.dataset, args.model_name, args.base_tag)
    refine_path = default_pool_path(args.dataset, args.model_name, args.refine_tag)
    reference_raw = load_tensor_pool(ref_path, device)
    base_raw = load_tensor_pool(base_path, device)
    refine_raw = load_tensor_pool(refine_path, device)
    if reference_raw.shape != base_raw.shape or reference_raw.shape != refine_raw.shape:
        raise ValueError("reference/base/refine pools must have identical shape")

    stress_cpu = _bfp_row_stress(reference_raw.detach().cpu(), args.block_size)
    stress = {key: value.to(device) for key, value in stress_cpu.items()}

    rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    baseline_accs: list[float] = []
    base_accs: list[float] = []
    refine_accs: list[float] = []

    for run_idx in range(args.runs):
        seed = int(args.seed) + run_idx
        run_args = _make_run_args(args, seed)
        _conf, data, verify_features, run_device = load_run_state(args.dataset, run_args, seed)
        # Keep all pools on the experiment device selected by load_run_state.
        ref = reference_raw.to(run_device)
        base = base_raw.to(run_device)
        refine = refine_raw.to(run_device)
        stress_dev = {key: value.to(run_device) for key, value in stress.items()}

        data.x = ref
        model, baseline_acc, baseline_hidden, _baseline_logits = train_baseline_model(data, run_args, run_device)
        baseline_accs.append(float(baseline_acc))
        with torch.no_grad():
            base_hidden = model.encoder(base)
            refine_hidden = model.encoder(refine)
        base_acc = evaluate_gnn_embeddings(model, data, base_hidden)
        refine_acc = evaluate_gnn_embeddings(model, data, refine_hidden)
        base_accs.append(float(base_acc))
        refine_accs.append(float(refine_acc))

        scores = build_real_quant_scores(verify_features, data, run_args, run_device)
        priorities = _priority_scores(scores, stress_dev, args, seed, run_device)
        for ratio in args.ratios:
            for policy_name in args.policies:
                if policy_name not in priorities:
                    raise ValueError(f"unknown policy {policy_name}; choices={sorted(priorities)}")
                mixed_hidden, mask = _mix_hidden(base_hidden, refine_hidden, priorities[policy_name], ratio)
                acc = evaluate_gnn_embeddings(model, data, mixed_hidden)
                drop = float(baseline_acc - acc)
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "ratio": float(ratio),
                        "policy": policy_name,
                        "baseline_acc": float(baseline_acc),
                        "base_acc": float(base_acc),
                        "refine_acc": float(refine_acc),
                        "acc": float(acc),
                        "drop": drop,
                        "lift_rate": float(mask.float().mean().item()),
                        "cost": _cost_for_ratio(float(ratio), args),
                    }
                )

    groups: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in per_seed_rows:
        groups.setdefault((float(row["ratio"]), str(row["policy"])), []).append(row)

    base_drop_mean = float(np.mean([b - a for b, a in zip(baseline_accs, base_accs)]))
    for (ratio, policy_name), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        accs = [float(row["acc"]) for row in group]
        drops = [float(row["drop"]) for row in group]
        rows.append(
            {
                "dataset": args.dataset,
                "reference_tag": args.reference_tag,
                "base_tag": args.base_tag,
                "refine_tag": args.refine_tag,
                "ratio": float(ratio),
                "policy": policy_name,
                "cost": _cost_for_ratio(float(ratio), args),
                "lift_rate": float(np.mean([row["lift_rate"] for row in group])),
                "baseline_acc_mean": float(np.mean(baseline_accs)),
                "all_base_acc_mean": float(np.mean(base_accs)),
                "all_base_drop_mean": base_drop_mean,
                "all_refine_acc_mean": float(np.mean(refine_accs)),
                "all_refine_drop_mean": float(np.mean([b - a for b, a in zip(baseline_accs, refine_accs)])),
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "drop_mean": float(np.mean(drops)),
                "drop_std": float(np.std(drops)),
                "gain_vs_base_drop": float(base_drop_mean - np.mean(drops)),
            }
        )

    _write_tsv(out_dir / "per_seed.tsv", per_seed_rows)
    _write_tsv(out_dir / "summary.tsv", rows)

    stress_stats = {
        key: {
            "mean": float(value.mean().item()),
            "p90": float(torch.quantile(value, 0.90).item()),
            "p95": float(torch.quantile(value, 0.95).item()),
        }
        for key, value in stress_cpu.items()
    }
    best_by_ratio = []
    for ratio in sorted(set(float(row["ratio"]) for row in rows)):
        candidates = [row for row in rows if math.isclose(float(row["ratio"]), ratio)]
        best_by_ratio.append(min(candidates, key=lambda row: row["drop_mean"]))

    note_lines = [
        f"Graph-BFP stress refinement | dataset={args.dataset} | runs={args.runs}",
        f"reference={args.reference_tag} | base={args.base_tag} | refine={args.refine_tag}",
        f"Baseline Acc: {float(np.mean(baseline_accs)):.4f}",
        f"All {args.base_tag}: Acc={float(np.mean(base_accs)):.4f}, Drop={base_drop_mean:.2%}",
        f"All {args.refine_tag}: Acc={float(np.mean(refine_accs)):.4f}, "
        f"Drop={float(np.mean([b - a for b, a in zip(baseline_accs, refine_accs)])):.2%}",
        "",
        "Best policy per refine ratio:",
        _format_table(best_by_ratio),
        "",
        "Stress stats:",
    ]
    for key, stat in stress_stats.items():
        note_lines.append(f"  {key}: mean={stat['mean']:.4f}, p90={stat['p90']:.4f}, p95={stat['p95']:.4f}")
    note = "\n".join(note_lines) + "\n"
    (out_dir / "summary.txt").write_text(note, encoding="utf-8")
    print(note)
    print(f"[Saved] {out_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
