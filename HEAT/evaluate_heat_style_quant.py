#!/usr/bin/env python3
"""Evaluate a HEAT-style topology-aware bit-serial quantization baseline.

This is a mechanism-level baseline, not a full HEAT reproduction.  It models
HEAT Sec. 5.2.1's bit-serial cost exactly from the published bit widths, then
uses local LLaMA2-7B BFPA embedding pools as an accuracy proxy for high/low
vertex precision routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

OFA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(OFA_ROOT)
if str(OFA_ROOT) not in sys.path:
    sys.path.insert(0, str(OFA_ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.real_quant import default_pool_path, load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import evaluate_gnn_embeddings, make_run_args, train_baseline_model  # noqa: E402
from GraphhopSimhash.scripts.profile_topology_risk_sensitivity import (  # noqa: E402
    eval_link,
    make_profile_args,
    split_edges,
    train_link_predictor,
)


TASKS: dict[str, tuple[str, str, str]] = {
    "CN": ("cora", "node", "Acc"),
    "CL": ("cora", "link", "AUC"),
    "PN": ("pubmed", "node", "Acc"),
    "PL": ("pubmed", "link", "AUC"),
    "AR": ("arxiv", "node", "Acc"),
    "WK": ("wikics", "node", "Acc"),
}

HEAT_OVERLAP_TASKS = {"CN", "CL", "PN", "PL", "AR"}


@dataclass
class RawRow:
    task: str
    dataset: str
    metric: str
    run: int
    seed: int
    low_tag: str
    policy: str
    alpha: float
    key_nodes: int
    key_rate: float
    low_nodes: int
    high_tag: str
    base_score: float
    score: float
    drop: float
    heat_avg_bitplanes: float
    heat_norm_vs_int8xint8: float
    heat_speedup_vs_int8xint8: float
    heat_norm_vs_w4a8: float
    heat_speedup_vs_w4a8: float
    heat_norm_vs_all_w8a10: float
    heat_speedup_vs_all_w8a10: float
    proxy_avg_bitplanes: float
    proxy_norm_vs_high_tag: float
    proxy_speedup_vs_high_tag: float


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%" if not math.isnan(float(value)) else "-"


def _num(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if not math.isnan(float(value)) else "-"


def _require_pool(dataset: str, model_name: str, tag: str) -> Path:
    path = OFA_ROOT / default_pool_path(dataset, model_name, tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding pool: {path}")
    return path


def parse_proxy_bitplanes(tag: str) -> int:
    """Parse local W4BFPAk/W4Ak tags into a weight_bits * activation_bits proxy."""
    raw = str(tag).upper()
    weight_bits = 4 if raw.startswith("W4") else 8
    act_bits = None
    for marker in ("BFPA", "A"):
        pos = raw.find(marker)
        if pos < 0:
            continue
        start = pos + len(marker)
        digits = []
        for char in raw[start:]:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            act_bits = int("".join(digits))
            break
    if act_bits is None:
        raise ValueError(f"Could not infer activation bits from tag={tag!r}")
    return int(weight_bits * act_bits)


def degree_scores(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """HEAT-style accumulated degree: traverse edges and count both endpoints."""
    deg = torch.zeros(int(num_nodes), dtype=torch.float32)
    src, dst = edge_index.detach().cpu()
    valid_src = src[(src >= 0) & (src < num_nodes)].to(torch.long)
    valid_dst = dst[(dst >= 0) & (dst < num_nodes)].to(torch.long)
    deg.scatter_add_(0, valid_src, torch.ones_like(valid_src, dtype=torch.float32))
    deg.scatter_add_(0, valid_dst, torch.ones_like(valid_dst, dtype=torch.float32))
    return deg


def top_degree_mask(edge_index: torch.Tensor, num_nodes: int, alpha: float) -> torch.Tensor:
    key_count = max(1, int(round(float(alpha) * int(num_nodes)))) if alpha > 0.0 else 0
    key_count = min(key_count, int(num_nodes))
    mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    if key_count <= 0:
        return mask
    deg = degree_scores(edge_index, int(num_nodes))
    order = torch.argsort(deg, descending=True, stable=True)
    mask[order[:key_count]] = True
    return mask


def random_key_mask(num_nodes: int, key_count: int, seed: int) -> torch.Tensor:
    mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    if key_count <= 0:
        return mask
    gen = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(int(num_nodes), generator=gen)
    mask[order[: min(key_count, int(num_nodes))]] = True
    return mask


def bitserial_costs(
    key_rate: float,
    *,
    hi_activation_bits: int,
    hi_weight_bits: int,
    lo_activation_bits: int,
    lo_weight_bits: int,
    int8_reference_bitplanes: int,
    w4a8_reference_bitplanes: int,
) -> dict[str, float]:
    hi = float(hi_activation_bits * hi_weight_bits)
    lo = float(lo_activation_bits * lo_weight_bits)
    avg = float(key_rate) * hi + (1.0 - float(key_rate)) * lo
    return {
        "heat_avg_bitplanes": avg,
        "heat_norm_vs_int8xint8": avg / float(int8_reference_bitplanes),
        "heat_speedup_vs_int8xint8": float(int8_reference_bitplanes) / avg,
        "heat_norm_vs_w4a8": avg / float(w4a8_reference_bitplanes),
        "heat_speedup_vs_w4a8": float(w4a8_reference_bitplanes) / avg,
        "heat_norm_vs_all_w8a10": avg / hi,
        "heat_speedup_vs_all_w8a10": hi / avg,
    }


def evaluate_score(
    *,
    task_type: str,
    model: torch.nn.Module,
    data: Any,
    features: torch.Tensor,
    link_model: torch.nn.Module | None,
    pos_test: torch.Tensor | None,
    neg_test: torch.Tensor | None,
) -> float:
    with torch.no_grad():
        hidden = model.encoder(features)
    if task_type == "link":
        if link_model is None or pos_test is None or neg_test is None:
            raise RuntimeError("link_model/pos_test/neg_test are required for link evaluation")
        score, _ = eval_link(link_model, hidden, pos_test.to(hidden.device), neg_test.to(hidden.device))
        return float(score)
    return float(evaluate_gnn_embeddings(model, data, hidden))


def evaluate_task(
    *,
    task: str,
    runs: int,
    seed: int,
    alpha: float,
    model_name: str,
    high_tag: str,
    low_tags: list[str],
    link_epochs: int,
    hi_activation_bits: int,
    hi_weight_bits: int,
    lo_activation_bits: int,
    lo_weight_bits: int,
    int8_reference_bitplanes: int,
    w4a8_reference_bitplanes: int,
) -> list[RawRow]:
    dataset, task_type, metric = TASKS[task]
    rows: list[RawRow] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    high_path = _require_pool(dataset, model_name, high_tag)
    low_paths = {tag: _require_pool(dataset, model_name, tag) for tag in low_tags}
    high_proxy_bitplanes = parse_proxy_bitplanes(high_tag)

    for run_idx in range(int(runs)):
        run_seed = int(seed) + run_idx
        args = make_profile_args(run_seed, dataset)
        run_args = make_run_args(args, run_seed)
        _conf, data, _verify_features, run_device = load_run_state(dataset, run_args, run_seed)
        if device.type == "cuda":
            run_device = device

        high_features = load_tensor_pool(str(high_path), run_device)
        data.x = high_features
        model, base_node_acc, baseline_hidden, _logits = train_baseline_model(data, run_args, run_device)

        link_model = None
        pos_test = None
        neg_test = None
        if task_type == "link":
            train_edges, val_edges, test_edges = split_edges(data.edge_index, data.num_nodes, run_seed)
            link_model = train_link_predictor(baseline_hidden, train_edges, val_edges, run_seed, link_epochs)
            pos_test, neg_test = test_edges
            base_score, _ = eval_link(
                link_model,
                baseline_hidden,
                pos_test.to(run_device),
                neg_test.to(run_device),
            )
        else:
            base_score = float(base_node_acc)

        top_mask_cpu = top_degree_mask(data.edge_index, int(data.num_nodes), alpha)
        key_count = int(top_mask_cpu.sum().item())
        key_rate = float(key_count) / float(int(data.num_nodes))
        random_mask_cpu = random_key_mask(int(data.num_nodes), key_count, run_seed + 1729)
        costs = bitserial_costs(
            key_rate,
            hi_activation_bits=hi_activation_bits,
            hi_weight_bits=hi_weight_bits,
            lo_activation_bits=lo_activation_bits,
            lo_weight_bits=lo_weight_bits,
            int8_reference_bitplanes=int8_reference_bitplanes,
            w4a8_reference_bitplanes=w4a8_reference_bitplanes,
        )

        for low_tag, low_path in low_paths.items():
            low_features = load_tensor_pool(str(low_path), run_device)
            low_proxy_bitplanes = parse_proxy_bitplanes(low_tag)

            policies = [
                ("AllLow", torch.zeros(int(data.num_nodes), dtype=torch.bool)),
                ("RandomKey10", random_mask_cpu),
                ("HEATTopDegree10", top_mask_cpu),
            ]
            for policy_name, key_mask_cpu in policies:
                actual_key_count = int(key_mask_cpu.sum().item())
                actual_key_rate = float(actual_key_count) / float(int(data.num_nodes))
                if actual_key_count == 0:
                    target_features = low_features
                else:
                    key_mask = key_mask_cpu.to(run_device)
                    target_features = low_features.clone()
                    target_features[key_mask] = high_features[key_mask]
                score = evaluate_score(
                    task_type=task_type,
                    model=model,
                    data=data,
                    features=target_features,
                    link_model=link_model,
                    pos_test=pos_test,
                    neg_test=neg_test,
                )
                if target_features is not low_features:
                    del target_features

                policy_costs = bitserial_costs(
                    actual_key_rate,
                    hi_activation_bits=hi_activation_bits,
                    hi_weight_bits=hi_weight_bits,
                    lo_activation_bits=lo_activation_bits,
                    lo_weight_bits=lo_weight_bits,
                    int8_reference_bitplanes=int8_reference_bitplanes,
                    w4a8_reference_bitplanes=w4a8_reference_bitplanes,
                )
                proxy_avg = (
                    actual_key_rate * float(high_proxy_bitplanes)
                    + (1.0 - actual_key_rate) * float(low_proxy_bitplanes)
                )
                rows.append(
                    RawRow(
                        task=task,
                        dataset=dataset,
                        metric=metric,
                        run=run_idx,
                        seed=run_seed,
                        low_tag=low_tag,
                        policy=policy_name,
                        alpha=float(alpha),
                        key_nodes=actual_key_count,
                        key_rate=actual_key_rate,
                        low_nodes=int(data.num_nodes) - actual_key_count,
                        high_tag=high_tag,
                        base_score=float(base_score),
                        score=float(score),
                        drop=float(base_score - score),
                        proxy_avg_bitplanes=proxy_avg,
                        proxy_norm_vs_high_tag=proxy_avg / float(high_proxy_bitplanes),
                        proxy_speedup_vs_high_tag=float(high_proxy_bitplanes) / proxy_avg,
                        **policy_costs,
                    )
                )
            del low_features
        del high_features, baseline_hidden, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def summarize_rows(rows: list[RawRow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[tuple[str, str, str], list[RawRow]] = {}
    for row in rows:
        by_group.setdefault((row.task, row.low_tag, row.policy), []).append(row)

    summary: list[dict[str, Any]] = []
    for (task, low_tag, policy), vals in sorted(
        by_group.items(),
        key=lambda item: (list(TASKS).index(item[0][0]), item[0][1], item[0][2]),
    ):
        summary.append(
            {
                "task": task,
                "dataset": vals[0].dataset,
                "metric": vals[0].metric,
                "low_tag": low_tag,
                "policy": policy,
                "runs": len(vals),
                "key_rate": _mean([v.key_rate for v in vals]),
                "base_score": _mean([v.base_score for v in vals]),
                "score": _mean([v.score for v in vals]),
                "drop": _mean([v.drop for v in vals]),
                "heat_avg_bitplanes": _mean([v.heat_avg_bitplanes for v in vals]),
                "heat_norm_vs_int8xint8": _mean([v.heat_norm_vs_int8xint8 for v in vals]),
                "heat_speedup_vs_int8xint8": _mean([v.heat_speedup_vs_int8xint8 for v in vals]),
                "heat_norm_vs_w4a8": _mean([v.heat_norm_vs_w4a8 for v in vals]),
                "heat_speedup_vs_w4a8": _mean([v.heat_speedup_vs_w4a8 for v in vals]),
                "heat_norm_vs_all_w8a10": _mean([v.heat_norm_vs_all_w8a10 for v in vals]),
                "heat_speedup_vs_all_w8a10": _mean([v.heat_speedup_vs_all_w8a10 for v in vals]),
                "proxy_avg_bitplanes": _mean([v.proxy_avg_bitplanes for v in vals]),
                "proxy_norm_vs_high_tag": _mean([v.proxy_norm_vs_high_tag for v in vals]),
                "proxy_speedup_vs_high_tag": _mean([v.proxy_speedup_vs_high_tag for v in vals]),
            }
        )

    aggregate: list[dict[str, Any]] = []
    for low_tag in sorted({row.low_tag for row in rows}):
        for policy in ["AllLow", "RandomKey10", "HEATTopDegree10"]:
            vals6 = [row for row in rows if row.low_tag == low_tag and row.policy == policy]
            vals5 = [row for row in vals6 if row.task in HEAT_OVERLAP_TASKS]
            for label, vals in [("AVG_HEAT5", vals5), ("AVG6", vals6)]:
                if not vals:
                    continue
                aggregate.append(
                    {
                        "task": label,
                        "dataset": "mixed",
                        "metric": "mixed",
                        "low_tag": low_tag,
                        "policy": policy,
                        "runs": len(vals),
                        "key_rate": _mean([v.key_rate for v in vals]),
                        "base_score": float("nan"),
                        "score": float("nan"),
                        "drop": _mean([v.drop for v in vals]),
                        "heat_avg_bitplanes": _mean([v.heat_avg_bitplanes for v in vals]),
                        "heat_norm_vs_int8xint8": _mean([v.heat_norm_vs_int8xint8 for v in vals]),
                        "heat_speedup_vs_int8xint8": _mean([v.heat_speedup_vs_int8xint8 for v in vals]),
                        "heat_norm_vs_w4a8": _mean([v.heat_norm_vs_w4a8 for v in vals]),
                        "heat_speedup_vs_w4a8": _mean([v.heat_speedup_vs_w4a8 for v in vals]),
                        "heat_norm_vs_all_w8a10": _mean([v.heat_norm_vs_all_w8a10 for v in vals]),
                        "heat_speedup_vs_all_w8a10": _mean([v.heat_speedup_vs_all_w8a10 for v in vals]),
                        "proxy_avg_bitplanes": _mean([v.proxy_avg_bitplanes for v in vals]),
                        "proxy_norm_vs_high_tag": _mean([v.proxy_norm_vs_high_tag for v in vals]),
                        "proxy_speedup_vs_high_tag": _mean([v.proxy_speedup_vs_high_tag for v in vals]),
                    }
                )
    return summary, aggregate


def render_markdown(
    *,
    args: argparse.Namespace,
    summary: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    raw_rows: list[RawRow],
) -> str:
    lines = [
        "# HEAT-Style Bit-Serial Quantization Evaluation",
        "",
        "## Scope",
        "",
        "This is a mechanism-level HEAT-style baseline, not a full HEAT simulator reproduction.",
        "It evaluates HEAT's topology-aware high/low vertex precision routing and the Sec. 5.2.1 bit-serial cost model.",
        "",
        "## Configuration",
        "",
        f"- Tasks: `{', '.join(args.tasks)}`.",
        f"- Runs: `{args.runs}` with seed base `{args.seed}`.",
        f"- Key vertex fraction alpha: `{args.alpha}`.",
        f"- HEAT-style key precision: `{args.hi_activation_bits}`-bit activation/token x `{args.hi_weight_bits}`-bit weight.",
        f"- HEAT-style non-key precision: `{args.lo_activation_bits}`-bit activation/token x `{args.lo_weight_bits}`-bit weight.",
        f"- Accuracy proxy high pool: `{args.high_tag}`.",
        f"- Accuracy proxy low pools: `{', '.join(args.low_tags)}`.",
        "",
        "The exact HEAT bit-serial reduction is reported from the published bit widths.",
        "Task-level drop is a local proxy based on existing LLaMA2-7B BFPA embedding pools.",
        "",
        "## Aggregate Result",
        "",
        "| Scope | Low Proxy | Policy | Key Rate | Drop | HEAT Norm vs INT8xINT8 | HEAT Speedup vs INT8xINT8 | HEAT Norm vs W4A8 | Proxy Norm vs High Pool |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['task']} | `{row['low_tag']}` | {row['policy']} | "
            f"{_pct(row['key_rate'])} | {_pct(row['drop'])} | "
            f"{_num(row['heat_norm_vs_int8xint8'], 4)}x | "
            f"{_num(row['heat_speedup_vs_int8xint8'], 2)}x | "
            f"{_num(row['heat_norm_vs_w4a8'], 4)}x | "
            f"{_num(row['proxy_norm_vs_high_tag'], 4)}x |"
        )

    lines.extend(
        [
            "",
            "## Per-Task Drop",
            "",
            "| Task | Low Proxy | Policy | Base | Score | Drop | Key Rate | HEAT Avg Bit-planes | HEAT Speedup vs INT8xINT8 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['task']} | `{row['low_tag']}` | {row['policy']} | "
            f"{_num(row['base_score'], 4)} | {_num(row['score'], 4)} | "
            f"{_pct(row['drop'])} | {_pct(row['key_rate'])} | "
            f"{_num(row['heat_avg_bitplanes'], 2)} | "
            f"{_num(row['heat_speedup_vs_int8xint8'], 2)}x |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `AllLow` uses the low proxy pool for every node.",
            "- `RandomKey10` protects the same number of nodes as HEAT but chooses them randomly.",
            "- `HEATTopDegree10` follows HEAT's topology rule and protects the top-degree `alpha` fraction.",
            "- HEAT bit-serial cost is independent from the proxy pool tags: with `alpha=0.1`, Fig. 6 gives about `15.2` average bit-plane GEMMs per multiply, or `0.2375x` of INT8xINT8.",
            "- The proxy drop should be read as the effect of applying HEAT-style vertex routing to this repository's LLaMA2-7B BFPA pools, not as HEAT's official SentenceBERT W8A10/W4A2 accuracy.",
            "",
            "## Raw Outputs",
            "",
            f"- Raw rows: `{args.output_dir}/raw.tsv`",
            f"- Summary rows: `{args.output_dir}/summary.tsv`",
            f"- Aggregate rows: `{args.output_dir}/aggregate.tsv`",
            f"- JSON: `{args.output_dir}/summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--model_name", default="llama2_7b")
    parser.add_argument("--high_tag", default="W4BFPA8_B128")
    parser.add_argument("--low_tags", nargs="+", default=["W4BFPA4_B256", "W4BFPA3_B256"])
    parser.add_argument("--link_epochs", type=int, default=300)
    parser.add_argument("--hi_activation_bits", type=int, default=10)
    parser.add_argument("--hi_weight_bits", type=int, default=8)
    parser.add_argument("--lo_activation_bits", type=int, default=2)
    parser.add_argument("--lo_weight_bits", type=int, default=4)
    parser.add_argument("--int8_reference_bitplanes", type=int, default=64)
    parser.add_argument("--w4a8_reference_bitplanes", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        default=str(OFA_ROOT / "output" / "heat_style_bitserial_quant"),
    )
    parser.add_argument(
        "--repo_report",
        default=str(REPO_ROOT / "HEAT" / "results" / "HEAT_STYLE_BITSERIAL_QUANT.md"),
    )
    args = parser.parse_args()

    if not (0.0 <= float(args.alpha) <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[RawRow] = []
    for task in args.tasks:
        print(f"[Task] {task} | runs={args.runs} | low_tags={','.join(args.low_tags)}")
        task_rows = evaluate_task(
            task=task,
            runs=args.runs,
            seed=args.seed,
            alpha=args.alpha,
            model_name=args.model_name,
            high_tag=args.high_tag,
            low_tags=args.low_tags,
            link_epochs=args.link_epochs,
            hi_activation_bits=args.hi_activation_bits,
            hi_weight_bits=args.hi_weight_bits,
            lo_activation_bits=args.lo_activation_bits,
            lo_weight_bits=args.lo_weight_bits,
            int8_reference_bitplanes=args.int8_reference_bitplanes,
            w4a8_reference_bitplanes=args.w4a8_reference_bitplanes,
        )
        all_rows.extend(task_rows)
        _write_tsv(output_dir / "raw.tsv", [asdict(row) for row in all_rows])
        summary, aggregate = summarize_rows(all_rows)
        _write_tsv(output_dir / "summary.tsv", summary)
        _write_tsv(output_dir / "aggregate.tsv", aggregate)
        report = render_markdown(args=args, summary=summary, aggregate=aggregate, raw_rows=all_rows)
        (output_dir / "HEAT_STYLE_BITSERIAL_QUANT.md").write_text(report, encoding="utf-8")
        print(report.split("## Per-Task Drop", maxsplit=1)[0])

    summary, aggregate = summarize_rows(all_rows)
    payload = {
        "config": vars(args),
        "summary": summary,
        "aggregate": aggregate,
        "raw": [asdict(row) for row in all_rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = render_markdown(args=args, summary=summary, aggregate=aggregate, raw_rows=all_rows)
    report_path = output_dir / "HEAT_STYLE_BITSERIAL_QUANT.md"
    report_path.write_text(report, encoding="utf-8")

    repo_report = Path(args.repo_report)
    repo_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, repo_report)
    print(f"[Done] report={report_path}")
    print(f"[Done] repo_report={repo_report}")


if __name__ == "__main__":
    main()
