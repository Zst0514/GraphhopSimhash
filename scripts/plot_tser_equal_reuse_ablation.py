#!/usr/bin/env python3
"""Plot equal-reuse TSER component ablation.

The plot uses the closest operating point for each policy at a fixed target
reuse budget. It is intentionally driven by the already-generated TSV files so
rerunning it after new datasets finish will automatically update the figure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("cora", "CR"),
    ("pubmed", "PB"),
    ("wikics", "WK"),
    ("tape_products", "PR"),
    ("tape_arxiv23", "TA23"),
    ("arxiv", "AR"),
]

POLICIES = [
    ("P only", "P"),
    ("P+C", "P+C"),
    ("P+U", "P+U"),
    ("Full TSER", "TSER"),
]

COLORS = {
    "P only": "#9ecae1",
    "P+C": "#a1d99b",
    "P+U": "#fdae6b",
    "Full TSER": "#756bb1",
}

HATCHES = {
    "P only": "",
    "P+C": "//",
    "P+U": "\\\\",
    "Full TSER": "..",
}


def pct_to_float(value: str) -> float:
    return float(value.strip().replace("%", ""))


def load_rows(output_root: Path, target: str, kind: str):
    rows = []
    for ds_name, ds_label in DATASETS:
        path = (
            output_root
            / f"llama7b_tser_equal_reuse_sweep_{ds_name}"
            / "llama7b_tser_equal_reuse_closest.tsv"
        )
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("target_reuse") != target or row.get("kind") != kind:
                    continue
                if row.get("policy") not in dict(POLICIES):
                    continue
                rows.append(
                    {
                        "dataset": ds_label,
                        "policy": row["policy"],
                        "reuse": pct_to_float(row["reuse"]),
                        "drop": pct_to_float(row["drop"]),
                        "T": row["T"],
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=Path, default=Path("output"))
    ap.add_argument("--target_reuse", default="40.00%")
    ap.add_argument("--kind", default="ResidualReuse")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("GraphhopSimhash/HPCA_2027_GFMAcc/Figure/tser_equal_reuse_ablation.pdf"),
    )
    ap.add_argument(
        "--tsv",
        type=Path,
        default=Path("output/tser_equal_reuse_ablation_plot_data.tsv"),
    )
    args = ap.parse_args()

    rows = load_rows(args.output_root, args.target_reuse, args.kind)
    if not rows:
        raise SystemExit("No rows found. Check --output_root/--target_reuse/--kind.")

    datasets = [label for _, label in DATASETS if any(r["dataset"] == label for r in rows)]
    policy_names = [name for name, _ in POLICIES]
    data = {(r["dataset"], r["policy"]): r for r in rows}

    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "policy", "reuse", "drop", "T"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    width = max(4.0, 0.9 * len(datasets) + 2.2)
    fig, ax = plt.subplots(figsize=(width, 2.45))

    x = np.arange(len(datasets))
    bar_w = 0.18
    offsets = (np.arange(len(policy_names)) - (len(policy_names) - 1) / 2) * bar_w

    for i, policy in enumerate(policy_names):
        drops = [data.get((ds, policy), {}).get("drop", np.nan) for ds in datasets]
        reuses = [data.get((ds, policy), {}).get("reuse", np.nan) for ds in datasets]
        bars = ax.bar(
            x + offsets[i],
            drops,
            width=bar_w,
            label=dict(POLICIES)[policy],
            color=COLORS[policy],
            edgecolor="black",
            linewidth=0.6,
            hatch=HATCHES[policy],
            zorder=3,
        )
        for bar, reuse in zip(bars, reuses):
            if np.isnan(reuse):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.055,
                f"{reuse:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                rotation=90,
                color="#333333",
            )

    ax.set_ylabel("Accuracy drop (%)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, max(2.25, max(r["drop"] for r in rows) + 0.65))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncol=len(policy_names),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    ax.text(
        0.99,
        0.98,
        f"target reuse = {args.target_reuse.replace('.00', '')}; numbers above bars are actual reuse (%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)
    print(args.tsv)


if __name__ == "__main__":
    main()
