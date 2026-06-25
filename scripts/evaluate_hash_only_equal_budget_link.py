#!/usr/bin/env python3
"""Evaluate hash-only equal-budget node substitution on link prediction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.real_quant import load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import train_baseline_model  # noqa: E402
from GraphhopSimhash.scripts.evaluate_cora_tser_t45_link_reuse import (  # noqa: E402
    eval_link,
    split_edges,
    train_link_predictor,
)
from GraphhopSimhash.scripts.replay_llama7b_tser_equal_budget import (  # noqa: E402
    equal_budget_mask,
    make_eval_args,
)
from GraphhopSimhash.scripts.replay_llama7b_tser_from_trace import (  # noqa: E402
    discover_traces,
    load_trace,
)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cora", "pubmed"], required=True)
    parser.add_argument("--trace_dir", required=True, type=Path)
    parser.add_argument("--trace_tag_contains", default="")
    parser.add_argument("--target_reuse", type=float, default=0.30)
    parser.add_argument("--soft_support", type=int, default=3)
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    trace_paths = discover_traces(args.trace_dir, [args.dataset], args.trace_tag_contains)
    if not trace_paths:
        raise SystemExit(f"No traces found in {args.trace_dir} for {args.dataset}")

    pool_path = ROOT / "cache_data" / f"{args.dataset}_llama2_7b_oracle_{args.reference_tag}.pt"
    reference = load_tensor_pool(str(pool_path), device="cpu").float()
    device = torch.device(args.device)

    rows: list[dict[str, float | int]] = []
    for path in trace_paths:
        trace = load_trace(path)
        run_args = make_eval_args(trace.seed)
        _conf, data, _verify_features, run_device = load_run_state(trace.dataset, run_args, trace.seed)
        if device.type != "cpu":
            run_device = device
        ref_dev = reference.to(run_device)
        data.x = ref_dev
        model, _node_acc, baseline_hidden, _logits = train_baseline_model(data, run_args, run_device)
        baseline_hidden = baseline_hidden.detach()

        mask_cpu = equal_budget_mask(trace, None, args.target_reuse, args.soft_support)
        mask = mask_cpu.to(run_device)
        source_id = trace.source_id.to(run_device)
        reuse_features = ref_dev.clone()
        accepted_idx = torch.nonzero(mask, as_tuple=False).flatten()
        if accepted_idx.numel() > 0:
            reuse_features[accepted_idx] = ref_dev[source_id[accepted_idx]]
        model.eval()
        with torch.no_grad():
            reuse_hidden = model.encoder(reuse_features).detach()

        train_edges, val_edges, test_edges = split_edges(data.edge_index, data.num_nodes, trace.seed)
        link_model = train_link_predictor(baseline_hidden, train_edges, val_edges, trace.seed, args.epochs)
        pos_test, neg_test = test_edges
        base_auc, base_ap = eval_link(link_model, baseline_hidden, pos_test, neg_test)
        reuse_auc, reuse_ap = eval_link(link_model, reuse_hidden, pos_test, neg_test)
        rows.append(
            {
                "run": int(trace.run),
                "seed": int(trace.seed),
                "reuse": float(mask.float().mean().item()),
                "base_auc": float(base_auc),
                "reuse_auc": float(reuse_auc),
                "auc_drop": float(base_auc - reuse_auc),
                "base_ap": float(base_ap),
                "reuse_ap": float(reuse_ap),
                "ap_drop": float(base_ap - reuse_ap),
            }
        )
        print(
            f"[{trace.dataset}] run={trace.run} reuse={pct(rows[-1]['reuse'])} "
            f"AUC={base_auc:.4f}->{reuse_auc:.4f} drop={pct(rows[-1]['auc_drop'])}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = args.output_dir / f"{args.dataset}_hash_only_equal_budget_link.tsv"
    with out_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "seed",
                "reuse",
                "base_auc",
                "reuse_auc",
                "auc_drop",
                "base_ap",
                "reuse_ap",
                "ap_drop",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    mean = {key: float(np.mean([float(r[key]) for r in rows])) for key in rows[0] if key not in {"run", "seed"}}
    out_md = args.output_dir / f"{args.dataset}_hash_only_equal_budget_link.md"
    lines = [
        f"# {args.dataset} Hash-Only Equal-Budget Link Substitution",
        "",
        f"Target reuse: {pct(args.target_reuse)}.",
        "",
        "| Reuse | Base AUC | Reuse AUC | AUC Drop | Base AP | Reuse AP | AP Drop |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {pct(mean['reuse'])} | {mean['base_auc']:.4f} | {mean['reuse_auc']:.4f} | "
            f"{pct(mean['auc_drop'])} | {mean['base_ap']:.4f} | {mean['reuse_ap']:.4f} | {pct(mean['ap_drop'])} |"
        ),
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Saved] {out_tsv}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
