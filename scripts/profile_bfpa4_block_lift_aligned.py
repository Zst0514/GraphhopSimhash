#!/usr/bin/env python3
"""Aligned BFPA precision and block-lift profile for the motivation table.

For each seed, this trains the downstream model once on the reference pool and
then evaluates BFPA3/4/5/6 plus random and stress-guided block lift with the
same trained model.  This keeps all columns on the same validation split and
removes run-to-run baseline drift across independently launched profile scripts.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

OFA_ROOT = Path(__file__).resolve().parents[2]
if str(OFA_ROOT) not in sys.path:
    sys.path.insert(0, str(OFA_ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.real_quant import default_pool_path, load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import evaluate_gnn_embeddings, make_run_args, train_baseline_model  # noqa: E402
from GraphhopSimhash.scripts.profile_bfpa_precision_tasks import TASKS  # noqa: E402
from GraphhopSimhash.scripts.profile_topology_risk_sensitivity import (  # noqa: E402
    eval_link,
    make_profile_args,
    split_edges,
    train_link_predictor,
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


def _pool(dataset: str, model: str, tag: str, device: torch.device) -> torch.Tensor:
    path = OFA_ROOT / default_pool_path(dataset, model, tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing pool: {path}")
    return load_tensor_pool(str(path), device)


def _mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _fmt_score(x: float) -> str:
    return f"{x:.4f}" if not math.isnan(float(x)) else "-"


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%" if not math.isnan(float(x)) else "-"


def _score(task_type: str, gnn_model, data, hidden, link_model, pos_test, neg_test) -> float:
    if task_type == "node":
        return float(evaluate_gnn_embeddings(gnn_model, data, hidden))
    auc, _ = eval_link(link_model, hidden, pos_test, neg_test)
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["CN", "CL", "PN", "PL", "AR", "WK"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="llama2_7b")
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--base_tag", default="W4BFPA4_B256")
    parser.add_argument("--rand_tag", default="W4BlockBFPA4to6_B256_random20")
    parser.add_argument("--stress_tag", default="W4BlockBFPA4to6_B256_oracle20")
    parser.add_argument("--bfpa3_tag", default="W4BFPA3_B256")
    parser.add_argument("--bfpa5_tag", default="W4BFPA5_B256")
    parser.add_argument("--bfpa6_tag", default="W4BFPA6_B256")
    parser.add_argument("--link_epochs", type=int, default=300)
    parser.add_argument("--output_dir", default="output/motivation_bfpa4_block_lift_aligned")
    args = parser.parse_args()

    tags = {
        "BFPA6": args.bfpa6_tag,
        "BFPA5": args.bfpa5_tag,
        "BFPA4": args.base_tag,
        "BFPA3": args.bfpa3_tag,
        "Rand20": args.rand_tag,
        "Stress20": args.stress_tag,
    }
    rows: list[dict[str, Any]] = []

    for task_name in args.tasks:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task {task_name}; choices={sorted(TASKS)}")
        dataset, task_type = TASKS[task_name]
        print(f"[Task] {task_name} dataset={dataset} type={task_type}")
        for run_idx in range(int(args.runs)):
            seed = int(args.seed) + run_idx
            run_args = make_run_args(make_profile_args(seed, dataset), seed)
            _conf, data, _verify_features, device = load_run_state(dataset, run_args, seed)
            ref = _pool(dataset, args.model_name, args.reference_tag, device)
            data.x = ref
            gnn_model, node_base, ref_hidden, _ = train_baseline_model(data, run_args, device)

            link_model = None
            pos_test = None
            neg_test = None
            if task_type == "link":
                train_edges, val_edges, test_edges = split_edges(data.edge_index, data.num_nodes, seed)
                link_model = train_link_predictor(ref_hidden, train_edges, val_edges, seed, args.link_epochs)
                pos_test, neg_test = (item.to(device) for item in test_edges)
                base_score, _ = eval_link(link_model, ref_hidden, pos_test, neg_test)
                metric = "AUC"
            else:
                base_score = float(node_base)
                metric = "Acc"

            for policy, tag in tags.items():
                features = _pool(dataset, args.model_name, tag, device)
                with torch.no_grad():
                    hidden = gnn_model.encoder(features)
                score = _score(task_type, gnn_model, data, hidden, link_model, pos_test, neg_test)
                rows.append(
                    {
                        "task": task_name,
                        "dataset": dataset,
                        "metric": metric,
                        "run": run_idx,
                        "seed": seed,
                        "policy": policy,
                        "tag": tag,
                        "reference_score": float(base_score),
                        "score": float(score),
                        "drop": float(base_score - score),
                    }
                )

    out_dir = Path(args.output_dir)
    _write_tsv(out_dir / "raw.tsv", rows)

    summary: list[dict[str, str]] = []
    for task_name in args.tasks:
        task_rows = [r for r in rows if r["task"] == task_name]
        if not task_rows:
            continue
        item = {
            "Task": task_name,
            "Ref": _fmt_score(_mean([float(r["reference_score"]) for r in task_rows if r["policy"] == "BFPA4"])),
        }
        for policy in ["BFPA6", "BFPA5", "BFPA4", "BFPA3", "Rand20", "Stress20"]:
            sub = [r for r in task_rows if r["policy"] == policy]
            item[f"{policy} Drop"] = _fmt_pct(_mean([float(r["drop"]) for r in sub]))
        summary.append(item)
    _write_tsv(out_dir / "summary.tsv", summary)

    lines = [
        "# Aligned BFPA Precision and Block-Lift Profile",
        "",
        "| Task | Ref. | BFPA6 | BFPA5 | BFPA4 | BFPA3 | Rand20 | Stress20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['Task']} | {row['Ref']} | {row['BFPA6 Drop']} | {row['BFPA5 Drop']} | "
            f"{row['BFPA4 Drop']} | {row['BFPA3 Drop']} | {row['Rand20 Drop']} | "
            f"{row['Stress20 Drop']} |"
        )
    lines.append("")
    lines.append("LaTeX rows:")
    lines.append("")
    lines.append("```tex")
    for row in summary:
        lines.append(
            f"{row['Task']} & {row['Ref']} & {row['BFPA6 Drop']} & {row['BFPA5 Drop']} & "
            f"{row['BFPA4 Drop']} & {row['BFPA3 Drop']} & {row['Rand20 Drop']} & "
            f"{row['Stress20 Drop']} \\\\"
        )
    lines.append("```")
    markdown = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
