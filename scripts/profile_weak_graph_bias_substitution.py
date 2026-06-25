#!/usr/bin/env python3
"""Profile a weak graph-biased 30% substitution baseline for Motivation.

This is intentionally not the final TSER/residual policy.  For each node, it
finds the most similar one-hop graph neighbor in the reference LLaMA embedding
space, then replaces exactly a fixed fraction of nodes with those neighbor
anchors.  The purpose is to test whether a lightweight local-topology prior can
improve over pure text-distance substitution in Motivation Table I.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.real_quant import load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import evaluate_gnn_embeddings, train_baseline_model  # noqa: E402
from GraphhopSimhash.scripts.profile_topology_risk_sensitivity import (  # noqa: E402
    eval_link,
    split_edges,
    train_link_predictor,
)


TASKS = {
    "CN": ("cora", "node", "Acc."),
    "CL": ("cora", "link", "AUC"),
    "PN": ("pubmed", "node", "Acc."),
    "PL": ("pubmed", "link", "AUC"),
    "AR": ("arxiv", "node", "Acc."),
    "WK": ("wikics", "node", "Acc."),
}


TEXT_ONLY_TABLE_DROP = {
    "CN": 0.0164,
    "CL": 0.0201,
    "PN": 0.0237,
    "PL": 0.0257,
    "AR": 0.0211,
    "WK": 0.0159,
}


@dataclass
class Row:
    task: str
    dataset: str
    metric: str
    run: int
    seed: int
    policy: str
    reuse_rate: float
    base_metric: float
    reuse_metric: float
    drop: float
    mean_anchor_cos: float


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


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def load_reference(dataset: str, tag: str) -> torch.Tensor:
    path = ROOT / "cache_data" / f"{dataset}_llama2_7b_oracle_{tag}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_tensor_pool(str(path), device="cpu").float()


def best_one_hop_anchor(
    reference: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best one-hop anchor and cosine score for every node."""
    num_nodes = int(reference.size(0))
    ref = F.normalize(reference.to(device), p=2, dim=1)
    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if src.size == 0:
        return torch.full((num_nodes,), -1, dtype=torch.long), torch.full((num_nodes,), -2.0)

    scores_parts = []
    src_t_all = torch.from_numpy(src).to(device)
    dst_t_all = torch.from_numpy(dst).to(device)
    with torch.no_grad():
        for start in range(0, src.size, chunk_size):
            end = min(start + chunk_size, src.size)
            s = src_t_all[start:end]
            d = dst_t_all[start:end]
            scores_parts.append((ref[s] * ref[d]).sum(dim=1).detach().cpu().numpy())
    scores = np.concatenate(scores_parts, axis=0)

    # For each source node, pick highest cosine; tie-break by lower dst ID.
    order = np.lexsort((dst, -scores, src))
    src_sorted = src[order]
    dst_sorted = dst[order]
    score_sorted = scores[order]
    unique_src, first_idx = np.unique(src_sorted, return_index=True)

    anchor = np.full(num_nodes, -1, dtype=np.int64)
    best_score = np.full(num_nodes, -2.0, dtype=np.float32)
    anchor[unique_src] = dst_sorted[first_idx]
    best_score[unique_src] = score_sorted[first_idx].astype(np.float32)
    return torch.from_numpy(anchor), torch.from_numpy(best_score)


def select_top_fraction(score: torch.Tensor, valid: torch.Tensor, fraction: float) -> torch.Tensor:
    n = int(score.numel())
    k = min(int(round(float(fraction) * n)), int(valid.sum().item()))
    mask = torch.zeros(n, dtype=torch.bool)
    if k <= 0:
        return mask
    rank = score.float().clone()
    rank[~valid] = -float("inf")
    chosen = torch.topk(rank, k=k, largest=True).indices
    mask[chosen] = True
    return mask


def evaluate_node(model, data, hidden: torch.Tensor) -> float:
    return float(evaluate_gnn_embeddings(model, data, hidden))


def run_one(
    task: str,
    run: int,
    seed: int,
    target_reuse: float,
    reference_tag: str,
    device: torch.device,
    link_epochs: int,
    anchor_chunk_size: int,
) -> Row:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dataset, task_type, metric = TASKS[task]
    run_args = make_eval_args(seed)
    _conf, data, _verify_features, run_device = load_run_state(dataset, run_args, seed)
    if device.type != "cpu":
        run_device = device
    reference = load_reference(dataset, reference_tag).to(run_device)
    data.x = reference
    model, base_node_acc, baseline_hidden, _logits = train_baseline_model(data, run_args, run_device)
    baseline_hidden = baseline_hidden.detach()

    anchor_cpu, score_cpu = best_one_hop_anchor(
        reference.detach().cpu(),
        data.edge_index,
        run_device,
        anchor_chunk_size,
    )
    valid = anchor_cpu >= 0
    selected_cpu = select_top_fraction(score_cpu, valid, target_reuse)
    selected = selected_cpu.to(run_device)
    anchors = anchor_cpu.to(run_device)

    reuse_hidden = baseline_hidden.clone()
    accepted_idx = torch.nonzero(selected, as_tuple=False).flatten()
    if accepted_idx.numel() > 0:
        reuse_hidden[accepted_idx] = baseline_hidden[anchors[accepted_idx]]

    if task_type == "node":
        base_metric = float(base_node_acc)
        reuse_metric = evaluate_node(model, data, reuse_hidden)
    else:
        train_edges, val_edges, test_edges = split_edges(data.edge_index, data.num_nodes, seed)
        link_model = train_link_predictor(baseline_hidden, train_edges, val_edges, seed, link_epochs)
        pos_test, neg_test = test_edges
        base_metric, _base_ap = eval_link(link_model, baseline_hidden, pos_test.to(run_device), neg_test.to(run_device))
        reuse_metric, _reuse_ap = eval_link(link_model, reuse_hidden, pos_test.to(run_device), neg_test.to(run_device))

    mean_cos = float(score_cpu[selected_cpu].mean().item()) if selected_cpu.any() else float("nan")
    row = Row(
        task=task,
        dataset=dataset,
        metric=metric,
        run=run,
        seed=seed,
        policy="GraphBias-1hop",
        reuse_rate=float(selected_cpu.float().mean().item()),
        base_metric=base_metric,
        reuse_metric=float(reuse_metric),
        drop=float(base_metric - reuse_metric),
        mean_anchor_cos=mean_cos,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def summarize(rows: list[Row]) -> list[Row]:
    out: list[Row] = []
    for task in sorted({row.task for row in rows}, key=lambda x: list(TASKS).index(x)):
        vals = [row for row in rows if row.task == task]
        first = vals[0]
        out.append(
            Row(
                task=task,
                dataset=first.dataset,
                metric=first.metric,
                run=0,
                seed=0,
                policy=first.policy,
                reuse_rate=float(np.mean([v.reuse_rate for v in vals])),
                base_metric=float(np.mean([v.base_metric for v in vals])),
                reuse_metric=float(np.mean([v.reuse_metric for v in vals])),
                drop=float(np.mean([v.drop for v in vals])),
                mean_anchor_cos=float(np.mean([v.mean_anchor_cos for v in vals])),
            )
        )
    return out


def write_outputs(rows: list[Row], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "weak_graph_bias_30_raw.tsv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["task", "dataset", "metric", "run", "seed", "policy", "reuse_rate", "base_metric", "reuse_metric", "drop", "mean_anchor_cos"])
        for row in rows:
            writer.writerow([row.task, row.dataset, row.metric, row.run, row.seed, row.policy, row.reuse_rate, row.base_metric, row.reuse_metric, row.drop, row.mean_anchor_cos])

    summary = summarize(rows)
    summary_path = output_dir / "weak_graph_bias_30_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["task", "metric", "base", "reuse", "drop", "llm_bypass", "mean_anchor_cos", "hash_only_drop", "delta_vs_hash_only"])
        for row in summary:
            text_drop = TEXT_ONLY_TABLE_DROP.get(row.task, float("nan"))
            writer.writerow(
                [
                    row.task,
                    row.metric,
                    f"{row.base_metric:.4f}",
                    f"{row.reuse_metric:.4f}",
                    pct(row.drop),
                    pct(row.reuse_rate),
                    f"{row.mean_anchor_cos:.4f}",
                    pct(text_drop) if text_drop == text_drop else "-",
                    pct(text_drop - row.drop) if text_drop == text_drop else "-",
                ]
            )

    md_path = output_dir / "weak_graph_bias_30_summary.md"
    lines = [
        "# Weak Graph-Biased Substitution at 30% Bypass",
        "",
        "This is a Motivation-only baseline. It does not use TSER, P/C/U risk scoring, or residual repair.",
        "Each node selects its most similar one-hop graph neighbor as the anchor; the top 30% most similar nodes are substituted.",
        "",
        "| Task | Metric | Base | Reuse | Drop | LLM Bypass | Anchor Cos. | Hash-only Drop | Delta vs Hash-only |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        text_drop = TEXT_ONLY_TABLE_DROP.get(row.task, float("nan"))
        lines.append(
            f"| {row.task} | {row.metric} | {row.base_metric * 100:.2f}% | {row.reuse_metric * 100:.2f}% | "
            f"{pct(row.drop)} | {pct(row.reuse_rate)} | {row.mean_anchor_cos:.4f} | "
            f"{pct(text_drop) if text_drop == text_drop else '-'} | {pct(text_drop - row.drop) if text_drop == text_drop else '-'} |"
        )
    lines.append("")
    lines.append(f"Raw TSV: `{raw_path}`")
    lines.append(f"Summary TSV: `{summary_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {raw_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), choices=list(TASKS.keys()))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_reuse", type=float, default=0.30)
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--link_epochs", type=int, default=300)
    parser.add_argument("--anchor_chunk_size", type=int, default=25_000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "output" / "motivation_weak_graph_bias_30")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows: list[Row] = []
    for task in args.tasks:
        for run in range(1, args.runs + 1):
            seed = int(args.seed) + run - 1
            print(f"[Run] task={task} run={run}/{args.runs} seed={seed}", flush=True)
            row = run_one(
                task,
                run,
                seed,
                args.target_reuse,
                args.reference_tag,
                device,
                args.link_epochs,
                args.anchor_chunk_size,
            )
            rows.append(row)
            print(
                f"[Result] {task} reuse={pct(row.reuse_rate)} "
                f"{row.metric}={row.base_metric:.4f}->{row.reuse_metric:.4f} "
                f"drop={pct(row.drop)} cos={row.mean_anchor_cos:.4f}",
                flush=True,
            )
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
