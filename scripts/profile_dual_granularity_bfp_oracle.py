#!/usr/bin/env python3
"""Evaluate dual-granularity BFPA4->BFPA6 dynamic BFP pools.

For each task, this reports:
  - BFPA4 baseline drop
  - dynamic BFPA4->BFPA6 drop
  - BFPA6 upper-bound drop
  - refined block ratio from the dynamic-pool metadata

Node tasks use accuracy; link tasks use sampled link AUC.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _fmt_score(x: float) -> str:
    return f"{x:.4f}" if not math.isnan(float(x)) else "-"


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%" if not math.isnan(float(x)) else "-"


def _pool(dataset: str, model_name: str, tag: str, device: torch.device) -> torch.Tensor:
    path = OFA_ROOT / default_pool_path(dataset, model_name, tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing pool: {path}")
    return load_tensor_pool(str(path), device)


def _meta(dataset: str, model_name: str, tag: str) -> dict[str, Any]:
    path = (OFA_ROOT / default_pool_path(dataset, model_name, tag)).with_suffix(".json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _score_hidden(task_type: str, gnn_model, data, hidden, link_model, pos_test, neg_test) -> float:
    if task_type == "node":
        return float(evaluate_gnn_embeddings(gnn_model, data, hidden))
    auc, _ = eval_link(link_model, hidden, pos_test, neg_test)
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["CN", "PN", "AR", "WK"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="llama2_7b")
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--base_tag", default="W4BFPA4_B256")
    parser.add_argument("--refine_tag", default="W4BFPA6_B256")
    parser.add_argument("--dynamic_tag", required=True)
    parser.add_argument("--link_epochs", type=int, default=300)
    parser.add_argument("--output_dir", default="output/dual_granularity_bfp_oracle")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for task_name in args.tasks:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task {task_name}; choices={sorted(TASKS)}")
        dataset, task_type = TASKS[task_name]
        meta = _meta(dataset, args.model_name, args.dynamic_tag)
        refined_ratio = float(meta.get("refined_ratio", float("nan")))
        effective_bits = float(meta.get("effective_bits", float("nan")))
        total_blocks = int(meta.get("total_blocks", 0) or 0)
        refined_blocks = int(meta.get("refined_blocks", 0) or 0)
        print(f"[Task] {task_name} dataset={dataset} type={task_type} dynamic={args.dynamic_tag}")

        for run_idx in range(int(args.runs)):
            seed = int(args.seed) + run_idx
            run_args = make_run_args(make_profile_args(seed, dataset), seed)
            _conf, data, _verify_features, device = load_run_state(dataset, run_args, seed)

            ref = _pool(dataset, args.model_name, args.reference_tag, device)
            base = _pool(dataset, args.model_name, args.base_tag, device)
            refine = _pool(dataset, args.model_name, args.refine_tag, device)
            dyn = _pool(dataset, args.model_name, args.dynamic_tag, device)

            data.x = ref
            gnn_model, node_base_score, ref_hidden, _ = train_baseline_model(data, run_args, device)
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
                base_score = float(node_base_score)
                metric = "Acc"

            with torch.no_grad():
                base_hidden = gnn_model.encoder(base)
                dyn_hidden = gnn_model.encoder(dyn)
                refine_hidden = gnn_model.encoder(refine)
            scores = {
                "BFPA4": _score_hidden(task_type, gnn_model, data, base_hidden, link_model, pos_test, neg_test),
                "Dynamic": _score_hidden(task_type, gnn_model, data, dyn_hidden, link_model, pos_test, neg_test),
                "BFPA6": _score_hidden(task_type, gnn_model, data, refine_hidden, link_model, pos_test, neg_test),
            }
            for policy, score in scores.items():
                rows.append(
                    {
                        "task": task_name,
                        "dataset": dataset,
                        "metric": metric,
                        "run": run_idx,
                        "seed": seed,
                        "policy": policy,
                        "reference_score": float(base_score),
                        "score": float(score),
                        "drop": float(base_score - score),
                        "dynamic_tag": args.dynamic_tag,
                        "refined_blocks": refined_blocks,
                        "total_blocks": total_blocks,
                        "refined_ratio": refined_ratio,
                        "effective_bits": effective_bits,
                    }
                )

    out_dir = Path(args.output_dir)
    _write_tsv(out_dir / "raw.tsv", rows)

    summary: list[dict[str, str]] = []
    for task in args.tasks:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        base = _mean([float(r["reference_score"]) for r in task_rows if r["policy"] == "BFPA4"])
        item = {
            "Task": task,
            "Ref": _fmt_score(base),
            "Lifted Blocks": _fmt_pct(_mean([float(r["refined_ratio"]) for r in task_rows])),
            "Eff. Bits": _fmt_score(_mean([float(r["effective_bits"]) for r in task_rows])),
        }
        for policy in ["BFPA4", "Dynamic", "BFPA6"]:
            sub = [r for r in task_rows if r["policy"] == policy]
            item[f"{policy} Score"] = _fmt_score(_mean([float(r["score"]) for r in sub]))
            item[f"{policy} Drop"] = _fmt_pct(_mean([float(r["drop"]) for r in sub]))
        summary.append(item)

    _write_tsv(out_dir / "summary.tsv", summary)
    lines = [
        "# Dual-Granularity BFPA4->BFPA6 Oracle/Profile",
        "",
        f"Dynamic tag: `{args.dynamic_tag}`",
        "",
        "| Task | Ref. | BFPA4 Drop | Dynamic Drop | BFPA6 Drop | Lifted Blocks | Eff. Bits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['Task']} | {row['Ref']} | {row['BFPA4 Drop']} | {row['Dynamic Drop']} | "
            f"{row['BFPA6 Drop']} | {row['Lifted Blocks']} | {row['Eff. Bits']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
