#!/usr/bin/env python3
"""Replay TSER policies from exported SimHash/CAM candidate traces.

This script is for TSER component ablation.  It fixes the candidate-discovery
frontend by reading a previously exported per-node trace, then replays different
graph-risk policies and score thresholds without querying CAM again.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
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


DATASET_LABELS = {
    "cora": "CN",
    "pubmed": "PN",
    "arxiv": "AR",
    "wikics": "WK",
    "tape_products": "PR",
    "tape_arxiv23": "TA23",
}

POLICIES = {
    "hash_only": ("Hash only", (0, 0, 0)),
    "degree_only": ("P only", (3, 0, 0)),
    "degree_context": ("P+C", (3, 1, 0)),
    "degree_unique": ("P+U", (3, 0, 1)),
    "tser": ("Full TSER", (3, 1, 1)),
}

POLICY_ORDER = {name: idx for idx, name in enumerate(POLICIES)}


@dataclass(frozen=True)
class TraceBundle:
    dataset: str
    run: int
    seed: int
    path: Path
    node_id: torch.Tensor
    source_id: torch.Tensor
    support: torch.Tensor
    min_dist: torch.Tensor
    score_error_q: torch.Tensor
    propagation_q: torch.Tensor
    graph_context_q: torch.Tensor
    low_unique_q: torch.Tensor


@dataclass(frozen=True)
class ReplayRow:
    dataset: str
    run: int
    seed: int
    policy: str
    threshold: float
    reuse: float
    acc: float
    drop: float
    avg_hidden_err: float
    trace: str


def _as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _fallback_error(dist: int, support: int) -> int:
    if dist <= 0:
        penalty = 1
    elif dist == 1:
        penalty = 2
    elif dist == 2:
        penalty = 4
    else:
        penalty = max(4, 2 * dist)
    if support >= 5:
        discount = 2
    elif support >= 3:
        discount = 1
    else:
        discount = 0
    return max(1, penalty - discount)


def load_trace(path: Path) -> TraceBundle:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows.extend(reader)
    if not rows:
        raise ValueError(f"empty trace: {path}")

    dataset = rows[0].get("dataset", "")
    run = _as_int(rows[0], "run", 1)
    seed = _as_int(rows[0], "seed", 42)

    def tensor(key: str, default: int = 0) -> torch.Tensor:
        return torch.tensor([_as_int(row, key, default) for row in rows], dtype=torch.long)

    node_id = tensor("node_id", -1)
    source_id = tensor("source_id", -1)
    support = tensor("support", 0)
    min_dist = tensor("min_dist", -1)
    score_error_q = tensor("score_error_q", 0)
    if int(score_error_q.max().item()) <= 0:
        score_error_q = torch.tensor(
            [_fallback_error(int(d), int(s)) for d, s in zip(min_dist.tolist(), support.tolist())],
            dtype=torch.long,
        )

    return TraceBundle(
        dataset=dataset,
        run=run,
        seed=seed,
        path=path,
        node_id=node_id,
        source_id=source_id,
        support=support,
        min_dist=min_dist,
        score_error_q=score_error_q,
        propagation_q=tensor("propagation_q", 0),
        graph_context_q=tensor("graph_context_q", 0),
        low_unique_q=tensor("low_unique_q", 0),
    )


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


def accepted_mask(trace: TraceBundle, weights: tuple[int, int, int], threshold: float, soft_support: int) -> torch.Tensor:
    valid = (trace.source_id >= 0) & (trace.support >= int(soft_support))
    sensitivity = (
        int(weights[0]) * trace.propagation_q
        + int(weights[1]) * trace.graph_context_q
        + int(weights[2]) * trace.low_unique_q
    )
    risk = sensitivity * trace.score_error_q
    if weights == (0, 0, 0):
        return valid
    return valid & (risk <= float(threshold))


def evaluate_trace(
    trace: TraceBundle,
    reference: torch.Tensor,
    thresholds: list[float],
    soft_support: int,
    device: torch.device,
) -> list[ReplayRow]:
    run_args = make_eval_args(trace.seed)
    _conf, data, _verify_features, run_device = load_run_state(trace.dataset, run_args, trace.seed)
    if device.type != "cpu":
        run_device = device
    ref_dev = reference.to(run_device)
    data.x = ref_dev
    model, baseline_acc, baseline_hidden, _logits = train_baseline_model(data, run_args, run_device)
    baseline_hidden = baseline_hidden.detach()

    rows: list[ReplayRow] = []
    n = int(reference.size(0))
    node_id = trace.node_id.to(run_device)
    source_id = trace.source_id.to(run_device)
    if node_id.numel() != n:
        raise ValueError(f"trace node count {node_id.numel()} != reference rows {n}: {trace.path}")

    for policy, (_label, weights) in POLICIES.items():
        policy_thresholds = [math.inf] if weights == (0, 0, 0) else thresholds
        for threshold in policy_thresholds:
            mask_cpu = accepted_mask(trace, weights, threshold, soft_support)
            mask = mask_cpu.to(run_device)
            accepted_idx = torch.nonzero(mask, as_tuple=False).flatten()
            hidden = baseline_hidden.clone()
            if accepted_idx.numel() > 0:
                hidden[accepted_idx] = baseline_hidden[source_id[accepted_idx]]
            acc = evaluate_gnn_embeddings(model, data, hidden)
            cos = F.cosine_similarity(hidden, baseline_hidden, dim=1)
            avg_err = float((1.0 - cos).mean().item())
            rows.append(
                ReplayRow(
                    dataset=trace.dataset,
                    run=trace.run,
                    seed=trace.seed,
                    policy=policy,
                    threshold=float(threshold),
                    reuse=float(mask.float().mean().item()),
                    acc=float(acc),
                    drop=float(baseline_acc - acc),
                    avg_hidden_err=avg_err,
                    trace=str(trace.path),
                )
            )
    return rows


def pct(value: float) -> str:
    if math.isinf(value) or math.isnan(value):
        return "-"
    return f"{value * 100:.2f}%"


def write_tsv(path: Path, rows: list[ReplayRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dataset", "run", "seed", "policy", "T", "reuse", "drop", "acc", "avg_hidden_err", "trace"])
        for r in sorted(rows, key=lambda x: (x.dataset, x.run, POLICY_ORDER.get(x.policy, 99), x.threshold)):
            label = DATASET_LABELS.get(r.dataset, r.dataset)
            writer.writerow(
                [
                    label,
                    r.run,
                    r.seed,
                    POLICIES[r.policy][0],
                    "all" if math.isinf(r.threshold) else f"{r.threshold:g}",
                    pct(r.reuse),
                    pct(r.drop),
                    f"{r.acc:.4f}",
                    f"{r.avg_hidden_err:.5f}",
                    r.trace,
                ]
            )


def mean_rows(rows: list[ReplayRow]) -> list[ReplayRow]:
    groups: dict[tuple[str, str, float], list[ReplayRow]] = defaultdict(list)
    for r in rows:
        groups[(r.dataset, r.policy, r.threshold)].append(r)
    out: list[ReplayRow] = []
    for (dataset, policy, threshold), vals in groups.items():
        out.append(
            ReplayRow(
                dataset=dataset,
                run=0,
                seed=0,
                policy=policy,
                threshold=threshold,
                reuse=sum(v.reuse for v in vals) / len(vals),
                acc=sum(v.acc for v in vals) / len(vals),
                drop=sum(v.drop for v in vals) / len(vals),
                avg_hidden_err=sum(v.avg_hidden_err for v in vals) / len(vals),
                trace=";".join(v.trace for v in vals),
            )
        )
    return out


def write_closest(path: Path, rows: list[ReplayRow], targets: list[float]) -> None:
    avg = mean_rows(rows)
    groups: dict[tuple[str, str], list[ReplayRow]] = defaultdict(list)
    for r in avg:
        groups[(r.dataset, r.policy)].append(r)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["target_reuse", "dataset", "policy", "T", "reuse", "drop", "acc", "avg_hidden_err"])
        for target in targets:
            for dataset in sorted({r.dataset for r in avg}, key=lambda x: DATASET_LABELS.get(x, x)):
                for policy in POLICIES:
                    candidates = groups.get((dataset, policy), [])
                    if not candidates:
                        continue
                    best = min(candidates, key=lambda r: (abs(r.reuse - target), r.drop))
                    writer.writerow(
                        [
                            pct(target),
                            DATASET_LABELS.get(dataset, dataset),
                            POLICIES[policy][0],
                            "all" if math.isinf(best.threshold) else f"{best.threshold:g}",
                            pct(best.reuse),
                            pct(best.drop),
                            f"{best.acc:.4f}",
                            f"{best.avg_hidden_err:.5f}",
                        ]
                    )


def write_markdown(path: Path, rows: list[ReplayRow], targets: list[float]) -> None:
    avg = mean_rows(rows)
    lines = [
        "# TSER Trace Replay Ablation",
        "",
        "This result replays fixed SimHash/CAM candidate traces.  CAM lookup is not rerun",
        "when changing TSER risk terms or thresholds; only the P/C/U score and accept",
        "decision are recomputed from the stored trace.",
        "",
        "## Averaged Frontier",
        "",
        "| Dataset | Policy | T | Reuse | Drop | AvgHiddenErr |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(avg, key=lambda x: (DATASET_LABELS.get(x.dataset, x.dataset), POLICY_ORDER.get(x.policy, 99), x.threshold)):
        lines.append(
            f"| {DATASET_LABELS.get(r.dataset, r.dataset)} | {POLICIES[r.policy][0]} | "
            f"{'all' if math.isinf(r.threshold) else f'{r.threshold:g}'} | "
            f"{pct(r.reuse)} | {pct(r.drop)} | {r.avg_hidden_err:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Closest Points",
            "",
            "| Target | Dataset | Policy | T | Reuse | Drop | AvgHiddenErr |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    groups: dict[tuple[str, str], list[ReplayRow]] = defaultdict(list)
    for r in avg:
        groups[(r.dataset, r.policy)].append(r)
    for target in targets:
        for dataset in sorted({r.dataset for r in avg}, key=lambda x: DATASET_LABELS.get(x, x)):
            for policy in POLICIES:
                candidates = groups.get((dataset, policy), [])
                if not candidates:
                    continue
                best = min(candidates, key=lambda r: (abs(r.reuse - target), r.drop))
                lines.append(
                    f"| {pct(target)} | {DATASET_LABELS.get(dataset, dataset)} | {POLICIES[policy][0]} | "
                    f"{'all' if math.isinf(best.threshold) else f'{best.threshold:g}'} | "
                    f"{pct(best.reuse)} | {pct(best.drop)} | {best.avg_hidden_err:.5f} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover_traces(trace_dir: Path, datasets: list[str], tag_contains: str = "") -> list[Path]:
    wanted = set(datasets)
    paths = []
    for path in sorted(trace_dir.glob("*_reuse_decisions.tsv")):
        if tag_contains and tag_contains not in path.name:
            continue
        # Dataset names may contain underscores, so read the first line rather
        # than relying on filename parsing.
        with path.open(encoding="utf-8") as f:
            _header = f.readline()
            first = f.readline()
        if not first:
            continue
        dataset = first.split("\t", 1)[0]
        if dataset in wanted:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", required=True, type=Path)
    parser.add_argument("--trace_tag_contains", default="")
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[16, 20, 24, 28, 31, 35, 40, 45, 50])
    parser.add_argument("--targets", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--soft_support", type=int, default=3)
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    trace_paths = discover_traces(args.trace_dir, args.datasets, args.trace_tag_contains)
    if not trace_paths:
        raise SystemExit(f"No trace TSVs found for {args.datasets} in {args.trace_dir}")

    by_dataset: dict[str, torch.Tensor] = {}
    all_rows: list[ReplayRow] = []
    for path in trace_paths:
        trace = load_trace(path)
        if trace.dataset not in by_dataset:
            pool_path = ROOT / "cache_data" / f"{trace.dataset}_llama2_7b_oracle_{args.reference_tag}.pt"
            if not pool_path.exists():
                raise FileNotFoundError(pool_path)
            by_dataset[trace.dataset] = load_tensor_pool(str(pool_path), device="cpu").float()
        print(f"[Replay] {trace.dataset} run={trace.run} seed={trace.seed} trace={path}")
        all_rows.extend(
            evaluate_trace(
                trace,
                by_dataset[trace.dataset],
                thresholds=args.thresholds,
                soft_support=args.soft_support,
                device=device,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "trace_replay_runs.tsv", all_rows)
    write_tsv(args.output_dir / "trace_replay_frontier.tsv", mean_rows(all_rows))
    write_closest(args.output_dir / "trace_replay_closest.tsv", all_rows, args.targets)
    write_markdown(args.output_dir / "trace_replay_summary.md", all_rows, args.targets)
    print(args.output_dir / "trace_replay_summary.md")


if __name__ == "__main__":
    main()
