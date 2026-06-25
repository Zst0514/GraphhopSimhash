#!/usr/bin/env python3
"""Profile how graph-risk groups amplify the same anchor-reuse error.

The experiment fixes the SimHash/CAM candidate discovery frontend, then
perturbs a fixed fraction of nodes by replacing their encoder hidden states with
the discovered anchor hidden states.  It compares high-risk and low-risk node
sets under the same replacement budget, isolating whether graph position changes
the downstream damage of a candidate approximation.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.cli import build_parser, validate_args  # noqa: E402
from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.runner import (  # noqa: E402
    build_controller,
    build_route_bundle,
    evaluate_gnn_embeddings,
    load_residual_target_features,
    make_run_args,
    train_baseline_model,
)


DATASET_LABELS = {
    "cora": "CN",
    "pubmed": "PN",
    "arxiv": "AR",
    "wikics": "WK",
}

LINK_DATASET_LABELS = {
    "cora": "CL",
    "pubmed": "PL",
}


RISK_FIELDS = {
    "propagation": ("P", "propagation_q"),
    "context": ("C", "graph_context_q"),
    "unique": ("U", "low_degree_unique_q"),
}


@dataclass
class ProfileRow:
    dataset: str
    task: str
    run: int
    seed: int
    group: str
    risk: str
    replaced: int
    replaced_rate: float
    baseline_acc: float
    perturbed_acc: float
    drop: float
    mean_risk: float
    mean_support: float
    mean_hamming: float
    mean_anchor_cos: float
    label_agree: float


def metric_label(dataset: str, task: str) -> str:
    if task == "link":
        return LINK_DATASET_LABELS.get(dataset, f"{dataset}-link")
    return DATASET_LABELS.get(dataset, dataset)


def quiet(_msg: str) -> None:
    return None


def parse_datasets(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def make_profile_args(seed: int, dataset: str) -> SimpleNamespace:
    args_list = [
        "--datasets",
        dataset,
        "--runs",
        "1",
        "--seed",
        str(seed),
        "--experiment_suite",
        "residual_reuse",
        "--learned_hash_epochs",
        "10",
        "--learned_hash_dim",
        "128",
        "--hash_heads_per_route",
        "8",
        "--hamming_only_acceptor",
        "--disable_structure_check",
        "--allow_hub_fuzzy",
        "--allow_rare_fuzzy",
        "--score_pair_confidence_discount",
        "1",
        "--radius",
        "2",
        "--main_hash_head_bits",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "--real_quant_model_name",
        "llama2_7b",
        "--real_quant_fp_tag",
        "W4BFPA8_B128",
        "--residual_embedding_source",
        "real_quant_fp",
        "--residual_fit_profile",
        "llama",
        # Keep the score gate enabled only to materialize node_risk_scores.
        # The threshold/guards are made permissive so candidate discovery is
        # not filtered by TSER in this profiling experiment.
        "--enable_score_gate",
        "--score_reuse_threshold",
        "1000000",
        "--score_hub_threshold",
        "15",
        "--score_rare_threshold",
        "15",
        "--score_propagation_weight",
        "3",
        "--score_graph_context_weight",
        "1",
        "--score_low_unique_weight",
        "1",
    ]
    parser = build_parser()
    parsed = parser.parse_args(args_list)
    validate_args(parser, parsed)
    return parsed


def select_by_score(valid: torch.Tensor, score: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    if idx.numel() == 0 or k <= 0:
        return idx[:0]
    k = min(k, int(idx.numel()))
    local_score = score[idx].float()
    order = torch.argsort(local_score, descending=largest, stable=True)
    return idx[order[:k]]


def select_random(valid: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    idx = torch.nonzero(valid, as_tuple=False).flatten().cpu()
    if idx.numel() == 0 or k <= 0:
        return idx[:0]
    k = min(k, int(idx.numel()))
    gen = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(idx.numel(), generator=gen)
    return idx[order[:k]]


def _take_by_seed(idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    if idx.numel() == 0 or k <= 0:
        return idx[:0]
    k = min(k, int(idx.numel()))
    gen = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(idx.numel(), generator=gen)
    return idx[order[:k]]


def allocate_matched_counts(capacities: list[tuple[tuple[int, int], int]], k: int) -> dict[tuple[int, int], int]:
    total = sum(cap for _bucket, cap in capacities)
    if total <= 0 or k <= 0:
        return {}
    k = min(k, total)
    alloc: dict[tuple[int, int], int] = {}
    remainders: list[tuple[float, tuple[int, int]]] = []
    used = 0
    for bucket, cap in capacities:
        raw = float(k) * float(cap) / float(total)
        count = min(cap, int(math.floor(raw)))
        alloc[bucket] = count
        used += count
        remainders.append((raw - count, bucket))
    for _rem, bucket in sorted(remainders, reverse=True):
        if used >= k:
            break
        cap = dict(capacities)[bucket]
        if alloc[bucket] < cap:
            alloc[bucket] += 1
            used += 1
    return alloc


def select_matched_high_low(
    *,
    valid: torch.Tensor,
    risk_score: torch.Tensor,
    support: torch.Tensor,
    hamming: torch.Tensor,
    k: int,
    pool_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select high/low risk nodes with matched (support, hamming) buckets."""
    valid_cpu = valid.detach().cpu()
    risk_cpu = risk_score.detach().cpu().float()
    support_cpu = support.detach().cpu().long()
    hamming_cpu = hamming.detach().cpu().long()
    idx = torch.nonzero(valid_cpu, as_tuple=False).flatten()
    if idx.numel() == 0 or k <= 0:
        return idx[:0], idx[:0]

    pool_size = max(k, int(math.ceil(float(pool_frac) * float(idx.numel()))))
    pool_size = min(pool_size, int(idx.numel()))
    order_high = torch.argsort(risk_cpu[idx], descending=True, stable=True)
    order_low = torch.argsort(risk_cpu[idx], descending=False, stable=True)
    high_pool = idx[order_high[:pool_size]]
    low_pool = idx[order_low[:pool_size]]

    high_by_bucket: dict[tuple[int, int], torch.Tensor] = {}
    low_by_bucket: dict[tuple[int, int], torch.Tensor] = {}
    buckets = set()
    for pool, table in ((high_pool, high_by_bucket), (low_pool, low_by_bucket)):
        for node in pool.tolist():
            bucket = (int(support_cpu[node].item()), int(hamming_cpu[node].item()))
            buckets.add(bucket)
        for bucket in list(buckets):
            mask = (support_cpu[pool] == bucket[0]) & (hamming_cpu[pool] == bucket[1])
            bucket_idx = pool[mask]
            if bucket_idx.numel() > 0:
                table[bucket] = bucket_idx

    capacities = []
    for bucket in sorted(set(high_by_bucket) & set(low_by_bucket)):
        cap = min(int(high_by_bucket[bucket].numel()), int(low_by_bucket[bucket].numel()))
        if cap > 0:
            capacities.append((bucket, cap))
    alloc = allocate_matched_counts(capacities, k)
    high_parts = []
    low_parts = []
    for bucket, count in sorted(alloc.items()):
        high_parts.append(_take_by_seed(high_by_bucket[bucket], count, seed + bucket[0] * 1009 + bucket[1]))
        low_parts.append(_take_by_seed(low_by_bucket[bucket], count, seed + 17 + bucket[0] * 1009 + bucket[1]))
    if not high_parts:
        return idx[:0], idx[:0]
    return torch.cat(high_parts, dim=0), torch.cat(low_parts, dim=0)


def make_unique_edges(edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index.detach().cpu()
    mask = row != col
    row = row[mask]
    col = col[mask]
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    edges = torch.stack([lo, hi], dim=1)
    return torch.unique(edges, dim=0)


def sample_negative_edges(num_nodes: int, positives: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    pos_set = {tuple(map(int, edge)) for edge in positives.tolist()}
    neg = []
    seen = set()
    while len(neg) < count:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key in pos_set or key in seen:
            continue
        seen.add(key)
        neg.append(key)
    return torch.tensor(neg, dtype=torch.long)


def split_edges(edge_index: torch.Tensor, num_nodes: int, seed: int):
    positives = make_unique_edges(edge_index)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(positives.size(0), generator=gen)
    positives = positives[order]
    n_edges = positives.size(0)
    n_train = int(0.80 * n_edges)
    n_val = int(0.10 * n_edges)
    pos_train = positives[:n_train]
    pos_val = positives[n_train : n_train + n_val]
    pos_test = positives[n_train + n_val :]
    neg_all = sample_negative_edges(num_nodes, positives, n_edges, seed + 991)
    neg_train = neg_all[:n_train]
    neg_val = neg_all[n_train : n_train + n_val]
    neg_test = neg_all[n_train + n_val :]
    return (pos_train, neg_train), (pos_val, neg_val), (pos_test, neg_test)


class LinkPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        src = z[edges[:, 0]]
        dst = z[edges[:, 1]]
        feat = torch.cat([torch.abs(src - dst), src * dst], dim=1)
        return self.net(feat).view(-1)


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def eval_link(model: LinkPredictor, z: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> tuple[float, float]:
    edges = torch.cat([pos, neg], dim=0).to(z.device)
    labels = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))], dim=0).to(z.device)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(z, edges)).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    return auc_score(y, scores), float("nan")


def train_link_predictor(
    z: torch.Tensor,
    train_edges,
    val_edges,
    seed: int,
    epochs: int,
) -> LinkPredictor:
    torch.manual_seed(seed)
    model = LinkPredictor(z.size(1)).to(z.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_train, neg_train = (item.to(z.device) for item in train_edges)
    pos_val, neg_val = (item.to(z.device) for item in val_edges)
    train_edges_all = torch.cat([pos_train, neg_train], dim=0)
    train_labels = torch.cat(
        [torch.ones(pos_train.size(0)), torch.zeros(neg_train.size(0))],
        dim=0,
    ).to(z.device)
    best_state = None
    best_val = -1.0
    for _epoch in range(int(epochs)):
        model.train()
        opt.zero_grad()
        logits = model(z, train_edges_all)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward()
        opt.step()
        val_auc, _ = eval_link(model, z, pos_val, neg_val)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_hidden_score(
    *,
    hidden: torch.Tensor,
    downstream_task: str,
    model,
    data,
    link_model: Optional[LinkPredictor] = None,
    pos_test: Optional[torch.Tensor] = None,
    neg_test: Optional[torch.Tensor] = None,
) -> float:
    if downstream_task == "node":
        return float(evaluate_gnn_embeddings(model, data, hidden))
    if downstream_task == "link":
        if link_model is None or pos_test is None or neg_test is None:
            raise RuntimeError("link task requires link_model, pos_test, and neg_test")
        auc, _ap = eval_link(link_model, hidden, pos_test, neg_test)
        return float(auc)
    raise ValueError(f"unknown downstream task: {downstream_task}")


def evaluate_group(
    *,
    dataset: str,
    downstream_task: str,
    run: int,
    seed: int,
    group: str,
    risk_name: str,
    selected_cpu: torch.Tensor,
    source_ids: torch.Tensor,
    support: torch.Tensor,
    hamming: torch.Tensor,
    risk_score: torch.Tensor,
    baseline_hidden: torch.Tensor,
    baseline_score: float,
    model,
    data,
    link_model,
    pos_test,
    neg_test,
    perturbation: str,
    perturb_scale: float,
) -> ProfileRow:
    selected = selected_cpu.to(baseline_hidden.device)
    hidden = baseline_hidden.clone()
    if selected.numel() > 0:
        if perturbation == "anchor":
            src = source_ids[selected]
            hidden[selected] = baseline_hidden[src]
        elif perturbation == "noise":
            gen = torch.Generator(device=baseline_hidden.device).manual_seed(seed + selected.numel() * 17)
            noise = torch.randn(
                baseline_hidden[selected].shape,
                generator=gen,
                device=baseline_hidden.device,
                dtype=baseline_hidden.dtype,
            )
            noise = F.normalize(noise, p=2, dim=1)
            row_norm = baseline_hidden[selected].norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            hidden[selected] = baseline_hidden[selected] + float(perturb_scale) * row_norm * noise
        elif perturbation == "zero":
            hidden[selected] = 0.0
        else:
            raise ValueError(f"unknown perturbation: {perturbation}")
    score = evaluate_hidden_score(
        hidden=hidden,
        downstream_task=downstream_task,
        model=model,
        data=data,
        link_model=link_model,
        pos_test=pos_test,
        neg_test=neg_test,
    )
    drop = baseline_score - score

    if selected.numel() > 0:
        if perturbation == "anchor":
            src = source_ids[selected]
            anchor_cos = F.cosine_similarity(baseline_hidden[selected], baseline_hidden[src], dim=1)
            label_agree = (data.y[selected] == data.y[src]).float()
            label_agree_value = float(label_agree.mean().item())
        else:
            anchor_cos = F.cosine_similarity(baseline_hidden[selected], hidden[selected], dim=1)
            label_agree_value = float("nan")
        if perturbation == "anchor":
            mean_hamming = hamming[selected].float()
            mean_hamming = mean_hamming[mean_hamming >= 0]
            hamming_value = float(mean_hamming.mean().item()) if mean_hamming.numel() else float("nan")
            support_value = float(support[selected].float().mean().item())
        else:
            hamming_value = float("nan")
            support_value = float("nan")
        row = ProfileRow(
            dataset=dataset,
            task=downstream_task,
            run=run,
            seed=seed,
            group=group,
            risk=risk_name,
            replaced=int(selected.numel()),
            replaced_rate=float(selected.numel() / baseline_hidden.size(0)),
            baseline_acc=float(baseline_score),
            perturbed_acc=float(score),
            drop=float(drop),
            mean_risk=float(risk_score[selected].float().mean().item()),
            mean_support=support_value,
            mean_hamming=hamming_value,
            mean_anchor_cos=float(anchor_cos.mean().item()),
            label_agree=label_agree_value,
        )
    else:
        row = ProfileRow(
            dataset=dataset,
            task=downstream_task,
            run=run,
            seed=seed,
            group=group,
            risk=risk_name,
            replaced=0,
            replaced_rate=0.0,
            baseline_acc=float(baseline_score),
            perturbed_acc=float(score),
            drop=float(drop),
            mean_risk=float("nan"),
            mean_support=float("nan"),
            mean_hamming=float("nan"),
            mean_anchor_cos=float("nan"),
            label_agree=float("nan"),
        )
    return row


def run_one(
    dataset: str,
    seed: int,
    run: int,
    replace_frac: float,
    min_support: int,
    perturbation: str,
    perturb_scale: float,
    matched_quality: bool,
    risk_pool_frac: float,
    downstream_task: str,
    link_epochs: int,
) -> list[ProfileRow]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = make_profile_args(seed, dataset)
    run_args = make_run_args(args, seed)
    _conf, data, verify_features, device = load_run_state(dataset, run_args, seed)
    target_features = load_residual_target_features(dataset, data, run_args, device, quiet)
    data.x = target_features
    model, baseline_acc, baseline_hidden, oracle_logits = train_baseline_model(data, run_args, device)
    link_model = None
    pos_test = None
    neg_test = None
    baseline_score = float(baseline_acc)
    if downstream_task == "link":
        train_edges, val_edges, test_edges = split_edges(data.edge_index, data.num_nodes, seed)
        link_model = train_link_predictor(baseline_hidden, train_edges, val_edges, seed, link_epochs)
        pos_test, neg_test = test_edges
        baseline_score, _ap = eval_link(
            link_model,
            baseline_hidden,
            pos_test.to(device),
            neg_test.to(device),
        )
    route_bundle = build_route_bundle(verify_features, data, baseline_hidden, oracle_logits, run_args, quiet, device)
    controller = build_controller(
        data,
        verify_features,
        route_bundle,
        {"name": "CandidateTrace", "overrides": {}},
        run_args,
        device,
    )
    _direct_features, _hits = controller.query_full_batch(
        route_bundle["hash_route_features"],
        verify_features,
        target_features,
    )
    trace = controller.last_query_trace
    if trace is None or controller.node_risk_scores is None:
        raise RuntimeError("Controller did not produce candidate trace or node risk scores")

    source_ids = trace["source_ids"].to(device)
    support = trace["winning_base_table_hit_counts"].to(device)
    hamming = trace["best_dists"].to(device)
    if perturbation == "anchor":
        valid = (source_ids >= 0) & (source_ids != torch.arange(source_ids.numel(), device=device)) & (support >= min_support)
    else:
        valid = torch.ones(source_ids.numel(), dtype=torch.bool, device=device)
    k = max(1, int(round(float(replace_frac) * source_ids.numel())))

    rows: list[ProfileRow] = []
    random_idx = select_random(valid.detach().cpu(), k, seed + 1009)
    rows.append(
        evaluate_group(
            dataset=dataset,
            downstream_task=downstream_task,
            run=run,
            seed=seed,
            group="Random",
            risk_name="random",
            selected_cpu=random_idx,
            source_ids=source_ids,
            support=support,
            hamming=hamming,
            risk_score=torch.zeros_like(support, dtype=torch.float32),
            baseline_hidden=baseline_hidden,
            baseline_score=baseline_score,
            model=model,
            data=data,
            link_model=link_model,
            pos_test=pos_test.to(device) if pos_test is not None else None,
            neg_test=neg_test.to(device) if neg_test is not None else None,
            perturbation=perturbation,
            perturb_scale=perturb_scale,
        )
    )

    for risk_name, (short_name, field) in RISK_FIELDS.items():
        score = controller.node_risk_scores[field].to(device)
        if perturbation == "anchor" and matched_quality:
            high_idx, low_idx = select_matched_high_low(
                valid=valid,
                risk_score=score,
                support=support,
                hamming=hamming,
                k=k,
                pool_frac=risk_pool_frac,
                seed=seed + 37,
            )
        else:
            high_idx = select_by_score(valid, score, k, largest=True).detach().cpu()
            low_idx = select_by_score(valid, score, k, largest=False).detach().cpu()
        rows.append(
            evaluate_group(
                dataset=dataset,
                downstream_task=downstream_task,
                run=run,
                seed=seed,
                group=f"High-{short_name}",
                risk_name=risk_name,
                selected_cpu=high_idx,
                source_ids=source_ids,
                support=support,
                hamming=hamming,
                risk_score=score,
                baseline_hidden=baseline_hidden,
                baseline_score=baseline_score,
                model=model,
                data=data,
                link_model=link_model,
                pos_test=pos_test.to(device) if pos_test is not None else None,
                neg_test=neg_test.to(device) if neg_test is not None else None,
                perturbation=perturbation,
                perturb_scale=perturb_scale,
            )
        )
        rows.append(
            evaluate_group(
                dataset=dataset,
                downstream_task=downstream_task,
                run=run,
                seed=seed,
                group=f"Low-{short_name}",
                risk_name=risk_name,
                selected_cpu=low_idx,
                source_ids=source_ids,
                support=support,
                hamming=hamming,
                risk_score=score,
                baseline_hidden=baseline_hidden,
                baseline_score=baseline_score,
                model=model,
                data=data,
                link_model=link_model,
                pos_test=pos_test.to(device) if pos_test is not None else None,
                neg_test=neg_test.to(device) if neg_test is not None else None,
                perturbation=perturbation,
                perturb_scale=perturb_scale,
            )
        )
    return rows


def write_tsv(path: Path, rows: list[ProfileRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProfileRow.__dataclass_fields__.keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize(rows: list[ProfileRow]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[ProfileRow]] = {}
    for row in rows:
        grouped.setdefault((metric_label(row.dataset, row.task), row.group), []).append(row)
    summary = []
    for (dataset, group), items in sorted(grouped.items()):
        def mean(attr: str) -> float:
            vals = [float(getattr(item, attr)) for item in items]
            vals = [v for v in vals if not math.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        summary.append(
            {
                "dataset": dataset,
                "group": group,
                "replaced": f"{mean('replaced_rate') * 100:.2f}%",
                "drop": f"{mean('drop') * 100:.2f}%",
                "anchor_cos": f"{mean('mean_anchor_cos'):.4f}",
                "support": f"{mean('mean_support'):.2f}",
                "hamming": f"{mean('mean_hamming'):.2f}",
                "label_agree": f"{mean('label_agree') * 100:.2f}%",
            }
        )
    return summary


def write_markdown(path: Path, rows: list[ProfileRow]) -> None:
    summary = summarize(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Topology Risk Sensitivity Profiling",
        "",
        "This profiling fixes SimHash/CAM candidate discovery and replaces the same node budget with discovered anchor hidden states. It compares high-risk and low-risk node groups under the same replacement budget.",
        "",
        "| Dataset | Group | Replaced | Drop | Anchor Cos. | Support | Ham. | Label Agree. |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['group']} | {row['replaced']} | {row['drop']} | "
            f"{row['anchor_cos']} | {row['support']} | {row['hamming']} | {row['label_agree']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: High-P, High-C, and High-U rows quantify the damage caused by replacing nodes with high propagation, graph-context, and low-degree uniqueness risk. The Low-* rows use the same replacement budget and the same CAM candidate source.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="cora pubmed")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace_frac", type=float, default=0.10)
    parser.add_argument("--min_support", type=int, default=3)
    parser.add_argument("--perturbation", choices=["anchor", "noise", "zero"], default="noise")
    parser.add_argument("--perturb_scale", type=float, default=0.35)
    parser.add_argument("--matched_quality", action="store_true")
    parser.add_argument("--risk_pool_frac", type=float, default=0.35)
    parser.add_argument("--downstream_task", choices=["node", "link"], default="node")
    parser.add_argument("--link_epochs", type=int, default=300)
    parser.add_argument("--output_dir", type=Path, default=Path("output/topology_risk_sensitivity"))
    args = parser.parse_args()

    all_rows: list[ProfileRow] = []
    for dataset in parse_datasets(args.datasets):
        for run in range(args.runs):
            seed = int(args.seed) + run
            print(f"[Run] dataset={dataset} run={run} seed={seed}")
            rows = run_one(
                dataset,
                seed,
                run,
                args.replace_frac,
                args.min_support,
                args.perturbation,
                args.perturb_scale,
                args.matched_quality,
                args.risk_pool_frac,
                args.downstream_task,
                args.link_epochs,
            )
            all_rows.extend(rows)
            write_tsv(args.output_dir / "topology_risk_sensitivity_raw.tsv", all_rows)
            write_markdown(args.output_dir / "topology_risk_sensitivity_summary.md", all_rows)
    print(f"[Saved] {args.output_dir / 'topology_risk_sensitivity_summary.md'}")


if __name__ == "__main__":
    main()
