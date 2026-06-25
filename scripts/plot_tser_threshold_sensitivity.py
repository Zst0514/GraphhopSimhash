#!/usr/bin/env python3
"""Plot Full-TSER reuse-threshold sensitivity from existing frontier TSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = [
    ("cora", "CN", "#1f77b4", "o"),
    ("pubmed", "PN", "#d62728", "s"),
    ("wikics", "WK", "#2ca02c", "^"),
]


def pct(value: str) -> float:
    return float(value.strip().replace("%", ""))


def load_full_tser_rows(output_root: Path, kind: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, label, _color, _marker in DATASETS:
        path = output_root / f"llama7b_tser_equal_reuse_sweep_{dataset}" / "llama7b_tser_equal_reuse_frontier.tsv"
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("kind") != kind or row.get("policy") != "Full TSER":
                    continue
                rows.append(
                    {
                        "dataset": label,
                        "T": int(row["T"]),
                        "reuse": pct(row["reuse"]),
                        "drop": pct(row["drop"]),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--kind", default="ResidualReuse")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GraphhopSimhash/HPCA_2027_GFMAcc/Figure/Experiment/tser_threshold_sensitivity.pdf"),
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path("output/tser_threshold_sensitivity_plot_data.tsv"),
    )
    args = parser.parse_args()

    rows = load_full_tser_rows(args.output_root, args.kind)
    if not rows:
        raise SystemExit("No Full TSER frontier rows found.")

    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "T", "reuse", "drop"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.25))

    for dataset, label, color, marker in DATASETS:
        ds_rows = sorted([r for r in rows if r["dataset"] == label], key=lambda r: (r["reuse"], r["T"]))
        if not ds_rows:
            continue
        reuse = [float(r["reuse"]) for r in ds_rows]
        drop = [float(r["drop"]) for r in ds_rows]
        ax.plot(
            reuse,
            drop,
            marker=marker,
            markersize=4.5,
            linewidth=1.6,
            color=color,
            label=label,
        )

    ax.set_xlabel("Reuse rate (%)")
    ax.set_ylabel("Accuracy drop (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="both", color="#dddddd", linewidth=0.65, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=3, frameon=True, borderpad=0.25, columnspacing=0.8, handletextpad=0.35)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)
    print(args.tsv)


if __name__ == "__main__":
    main()
