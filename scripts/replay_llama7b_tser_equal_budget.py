#!/usr/bin/env python3
"""Replay TSER policies under an exact reuse budget.

This is a stricter companion to threshold replay.  It fixes the exported
SimHash/CAM candidate trace and accepts the lowest-risk candidates until a
target node fraction is reached.  This removes threshold discretization when
comparing graph-risk components at the same reuse budget.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.real_quant import load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import evaluate_gnn_embeddings, train_baseline_model  # noqa: E402
from GraphhopSimhash.scripts.replay_llama7b_tser_from_trace import (  # noqa: E402
    DATASET_LABELS,
    TraceBundle,
    discover_traces,
    load_trace,
)


POLICIES = {
    "candidate_only": ("Candidate only", None),
    "degree_only": ("P only", (3, 0, 0)),
    "degree_context": ("P+C", (3, 1, 0)),
    "degree_unique": ("P+U", (3, 0, 1)),
    "tser": ("Full TSER", (3, 1, 1)),
}


@dataclass(frozen=True)
class BudgetRow:
    dataset: str
    run: int
    seed: int
    policy: str
    target_reuse: float
    reuse: float
    acc: float
    drop: float
    avg_hidden_err: float
    trace: str


def make_eval_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
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


def policy_score(trace: TraceBundle, weights: tuple[int, int, int] | None) -> torch.Tensor:
    if weights is None:
        return trace.score_error_q.float()
    sensitivity = (
        int(weights[0]) * trace.propagation_q
        + int(weights[1]) * trace.graph_context_q
        + int(weights[2]) * trace.low_unique_q
    )
    return (sensitivity * trace.score_error_q).float()


def equal_budget_mask(
    trace: TraceBundle,
    weights: tuple[int, int, int] | None,
    target_reuse: float,
    soft_support: int,
) -> torch.Tensor:
    valid = (trace.source_id >= 0) & (trace.support >= int(soft_support))
    n = int(trace.node_id.numel())
    k = min(int(round(float(target_reuse) * n)), int(valid.sum().item()))
    mask = torch.zeros(n, dtype=torch.bool)
    if k <= 0:
        return mask
    score = policy_score(trace, weights)
    # Deterministic tie break: prefer lower node id for equal score.
    node_eps = torch.arange(n, dtype=torch.float32) / max(float(n), 1.0) * 1e-6
    rank_score = score + node_eps
    rank_score[~valid] = float("inf")
    chosen = torch.topk(-rank_score, k=k, largest=True).indices
    mask[chosen] = True
    return mask


def evaluate_trace(
    trace: TraceBundle,
    reference: torch.Tensor,
    target_reuse: float,
    soft_support: int,
    device: torch.device,
) -> list[BudgetRow]:
    run_args = make_eval_args(trace.seed)
    _conf, data, _verify_features, run_device = load_run_state(trace.dataset, run_args, trace.seed)
    if device.type != "cpu":
        run_device = device
    ref_dev = reference.to(run_device)
    data.x = ref_dev
    model, baseline_acc, baseline_hidden, _logits = train_baseline_model(data, run_args, run_device)
    baseline_hidden = baseline_hidden.detach()

    rows: list[BudgetRow] = []
    source_id = trace.source_id.to(run_device)
    for policy, (_label, weights) in POLICIES.items():
        mask_cpu = equal_budget_mask(trace, weights, target_reuse, soft_support)
        mask = mask_cpu.to(run_device)
        accepted_idx = torch.nonzero(mask, as_tuple=False).flatten()
        hidden = baseline_hidden.clone()
        if accepted_idx.numel() > 0:
            hidden[accepted_idx] = baseline_hidden[source_id[accepted_idx]]
        acc = evaluate_gnn_embeddings(model, data, hidden)
        cos = F.cosine_similarity(hidden, baseline_hidden, dim=1)
        avg_err = float((1.0 - cos).mean().item())
        rows.append(
            BudgetRow(
                dataset=trace.dataset,
                run=trace.run,
                seed=trace.seed,
                policy=policy,
                target_reuse=target_reuse,
                reuse=float(mask.float().mean().item()),
                acc=float(acc),
                drop=float(baseline_acc - acc),
                avg_hidden_err=avg_err,
                trace=str(trace.path),
            )
        )
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def mean_rows(rows: list[BudgetRow]) -> list[BudgetRow]:
    groups: dict[tuple[str, str, float], list[BudgetRow]] = defaultdict(list)
    for row in rows:
        groups[(row.dataset, row.policy, row.target_reuse)].append(row)
    out: list[BudgetRow] = []
    for (dataset, policy, target), vals in groups.items():
        out.append(
            BudgetRow(
                dataset=dataset,
                run=0,
                seed=0,
                policy=policy,
                target_reuse=target,
                reuse=sum(v.reuse for v in vals) / len(vals),
                acc=sum(v.acc for v in vals) / len(vals),
                drop=sum(v.drop for v in vals) / len(vals),
                avg_hidden_err=sum(v.avg_hidden_err for v in vals) / len(vals),
                trace=";".join(v.trace for v in vals),
            )
        )
    return out


def write_tsv(path: Path, rows: list[BudgetRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dataset", "policy", "target_reuse", "reuse", "drop", "acc", "avg_hidden_err"])
        for row in sorted(mean_rows(rows), key=lambda r: (DATASET_LABELS.get(r.dataset, r.dataset), r.policy)):
            writer.writerow(
                [
                    DATASET_LABELS.get(row.dataset, row.dataset),
                    POLICIES[row.policy][0],
                    pct(row.target_reuse),
                    pct(row.reuse),
                    pct(row.drop),
                    f"{row.acc:.4f}",
                    f"{row.avg_hidden_err:.5f}",
                ]
            )


def write_markdown(path: Path, rows: list[BudgetRow]) -> None:
    avg = sorted(mean_rows(rows), key=lambda r: (DATASET_LABELS.get(r.dataset, r.dataset), r.policy))
    lines = [
        "# TSER Equal-Budget Replay",
        "",
        "This result fixes the SimHash/CAM candidate trace and accepts the same node fraction for each risk policy.",
        "It compares graph-risk components without threshold discretization.",
        "",
        "| Dataset | Policy | Target Reuse | Actual Reuse | Drop | AvgHiddenErr |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in avg:
        lines.append(
            f"| {DATASET_LABELS.get(row.dataset, row.dataset)} | {POLICIES[row.policy][0]} | "
            f"{pct(row.target_reuse)} | {pct(row.reuse)} | {pct(row.drop)} | {row.avg_hidden_err:.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", required=True, type=Path)
    parser.add_argument("--trace_tag_contains", default="")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--target_reuse", type=float, default=0.40)
    parser.add_argument("--soft_support", type=int, default=3)
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    trace_paths = discover_traces(args.trace_dir, args.datasets, args.trace_tag_contains)
    if not trace_paths:
        raise SystemExit(f"No trace TSVs found for {args.datasets} in {args.trace_dir}")

    pools: dict[str, torch.Tensor] = {}
    rows: list[BudgetRow] = []
    for path in trace_paths:
        trace = load_trace(path)
        if trace.dataset not in pools:
            pool_path = ROOT / "cache_data" / f"{trace.dataset}_llama2_7b_oracle_{args.reference_tag}.pt"
            if not pool_path.exists():
                raise FileNotFoundError(pool_path)
            pools[trace.dataset] = load_tensor_pool(str(pool_path), device="cpu").float()
        print(f"[EqualBudget] {trace.dataset} run={trace.run} seed={trace.seed} trace={path}")
        rows.extend(evaluate_trace(trace, pools[trace.dataset], args.target_reuse, args.soft_support, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "equal_budget_replay.tsv", rows)
    write_markdown(args.output_dir / "equal_budget_replay.md", rows)
    print(args.output_dir / "equal_budget_replay.md")


if __name__ == "__main__":
    main()
