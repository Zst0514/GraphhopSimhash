#!/usr/bin/env python3
"""Profile BFPA precision boundaries for node and link TAG tasks.

The script consumes existing LLaMA2-7B embedding pools:

    W4BFPA8_B128 as the accuracy-preserving reference
    W4BFPA{6,5,4,3}_B256 as target activation formats

Node tasks report accuracy.  Link tasks train a sampled link predictor on the
reference encoder hidden states and report AUC.
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
from GraphhopSimhash.scripts.profile_topology_risk_sensitivity import (  # noqa: E402
    eval_link,
    make_profile_args,
    split_edges,
    train_link_predictor,
)


TASKS = {
    "CN": ("cora", "node"),
    "CL": ("cora", "link"),
    "PN": ("pubmed", "node"),
    "PL": ("pubmed", "link"),
    "AR": ("arxiv", "node"),
    "WK": ("wikics", "node"),
}


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _fmt_score(x: float) -> str:
    return f"{x:.4f}" if not math.isnan(float(x)) else "-"


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%" if not math.isnan(float(x)) else "-"


def _require_pool(dataset: str, model: str, tag: str) -> Path:
    path = OFA_ROOT / default_pool_path(dataset, model, tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing pool: {path}")
    return path


def evaluate_task(
    *,
    task_name: str,
    dataset: str,
    task_type: str,
    bits: list[int],
    runs: int,
    seed: int,
    model_name: str,
    reference_tag: str,
    target_block: int,
    link_epochs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_idx in range(int(runs)):
        run_seed = int(seed) + run_idx
        args = make_profile_args(run_seed, dataset)
        run_args = make_run_args(args, run_seed)
        _conf, data, _verify_features, run_device = load_run_state(dataset, run_args, run_seed)

        ref_path = _require_pool(dataset, model_name, reference_tag)
        ref_features = load_tensor_pool(str(ref_path), run_device)
        data.x = ref_features
        gnn_model, node_base_acc, ref_hidden, _logits = train_baseline_model(data, run_args, run_device)

        link_model = None
        pos_test = None
        neg_test = None
        if task_type == "link":
            train_edges, val_edges, test_edges = split_edges(data.edge_index, data.num_nodes, run_seed)
            link_model = train_link_predictor(ref_hidden, train_edges, val_edges, run_seed, link_epochs)
            pos_test, neg_test = (item.to(run_device) for item in test_edges)
            base_score, _ = eval_link(link_model, ref_hidden, pos_test, neg_test)
            metric = "AUC"
        else:
            base_score = float(node_base_acc)
            metric = "Acc"

        for bit in bits:
            tag = f"W4BFPA{bit}_B{target_block}"
            target_path = _require_pool(dataset, model_name, tag)
            target_features = load_tensor_pool(str(target_path), run_device)
            with torch.no_grad():
                target_hidden = gnn_model.encoder(target_features)
            if task_type == "link":
                assert link_model is not None and pos_test is not None and neg_test is not None
                target_score, _ = eval_link(link_model, target_hidden, pos_test, neg_test)
            else:
                target_score = float(evaluate_gnn_embeddings(gnn_model, data, target_hidden))
            rows.append(
                {
                    "task": task_name,
                    "dataset": dataset,
                    "metric": metric,
                    "run": run_idx,
                    "seed": run_seed,
                    "reference_tag": reference_tag,
                    "target_tag": tag,
                    "bit": bit,
                    "base_score": float(base_score),
                    "target_score": float(target_score),
                    "drop": float(base_score - target_score),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], bits: list[int]) -> tuple[list[dict[str, str]], str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task"]), []).append(row)

    summary: list[dict[str, str]] = []
    for task in [t for t in TASKS if t in grouped]:
        task_rows = grouped[task]
        base = _mean([float(r["base_score"]) for r in task_rows if int(r["bit"]) == bits[0]])
        out = {"Task": task, "Base": _fmt_score(base)}
        for bit in bits:
            drop = _mean([float(r["drop"]) for r in task_rows if int(r["bit"]) == bit])
            out[f"BFPA{bit}"] = _fmt_pct(drop)
        summary.append(out)

    lines = [
        "# BFPA Precision Boundary for CN/CL/PN/PL/AR/WK",
        "",
        "Reference is `W4BFPA8_B128`; target pools are `W4BFPA{6,5,4,3}_B256`.",
        "Node tasks report accuracy; link tasks report sampled link AUC.",
        "",
        "| Task | Ref. Score | BFPA6 | BFPA5 | BFPA4 | BFPA3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['Task']} | {row['Base']} | {row.get('BFPA6','-')} | "
            f"{row.get('BFPA5','-')} | {row.get('BFPA4','-')} | {row.get('BFPA3','-')} |"
        )
    lines.append("")
    lines.append("LaTeX rows:")
    lines.append("")
    lines.append("```tex")
    for row in summary:
        lines.append(
            f"{row['Task']} & {row['Base']} & {row.get('BFPA6','-')} & "
            f"{row.get('BFPA5','-')} & {row.get('BFPA4','-')} & {row.get('BFPA3','-')} \\\\"
        )
    lines.append("```")
    return summary, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--bits", nargs="+", type=int, default=[6, 5, 4, 3])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="llama2_7b")
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--target_block", type=int, default=256)
    parser.add_argument("--link_epochs", type=int, default=300)
    parser.add_argument("--output_dir", default="output/bfpa_precision_tasks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    for task_name in args.tasks:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task {task_name}; choices={sorted(TASKS)}")
        dataset, task_type = TASKS[task_name]
        print(f"[Task] {task_name} dataset={dataset} type={task_type}")
        rows = evaluate_task(
            task_name=task_name,
            dataset=dataset,
            task_type=task_type,
            bits=args.bits,
            runs=args.runs,
            seed=args.seed,
            model_name=args.model_name,
            reference_tag=args.reference_tag,
            target_block=args.target_block,
            link_epochs=args.link_epochs,
        )
        all_rows.extend(rows)
        _write_tsv(output_dir / f"{task_name}_raw.tsv", rows)
        _write_tsv(output_dir / "raw.tsv", all_rows)
        summary, markdown = summarize(all_rows, args.bits)
        _write_tsv(output_dir / "summary.tsv", summary)
        (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
        print(markdown)


if __name__ == "__main__":
    main()
