#!/usr/bin/env python3
"""Summarize matched-replacement Motivation profiling across tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASK_SOURCES = [
    ("CN", "Node Acc.", Path("output/matched_replacement_cora_runs5/topology_risk_sensitivity_raw.tsv")),
    ("CL", "Link AUC", Path("output/matched_replacement_cora_link_runs5/topology_risk_sensitivity_raw.tsv")),
    ("PN", "Node Acc.", Path("output/matched_replacement_pubmed_runs5/topology_risk_sensitivity_raw.tsv")),
    ("PL", "Link AUC", Path("output/matched_replacement_pubmed_link_runs5/topology_risk_sensitivity_raw.tsv")),
    ("AR", "Node Acc.", Path("output/matched_replacement_arxiv_runs5/topology_risk_sensitivity_raw.tsv")),
    ("WK", "Node Acc.", Path("output/matched_replacement_wikics_runs5/topology_risk_sensitivity_raw.tsv")),
]

GROUP_ORDER = ["High-P", "Low-P", "High-C", "Low-C", "High-U", "Low-U", "Random"]


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def load_summary(task: str, metric: str, path: Path) -> dict[str, str]:
    if not path.exists():
        row = {"Task": task, "Metric": metric}
        for group in GROUP_ORDER:
            row[group] = "-"
        row["Replaced"] = "-"
        row["Support"] = "-"
        row["Ham."] = "-"
        return row
    df = pd.read_csv(path, sep="\t")
    row = {"Task": task, "Metric": metric}
    grouped = df.groupby("group")
    for group in GROUP_ORDER:
        if group in grouped.groups:
            row[group] = pct(grouped.get_group(group)["drop"].mean())
        else:
            row[group] = "-"
    row["Replaced"] = pct(df["replaced_rate"].mean())
    row["Support"] = f"{df['mean_support'].mean():.2f}"
    row["Ham."] = f"{df['mean_hamming'].mean():.2f}"
    return row


def write_markdown(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Matched Replacement Motivation Table",
        "",
        "Each task replaces the same node budget with real SimHash-CAM anchors. "
        "High/low groups are matched by support and Hamming-distance buckets, so "
        "drop differences reflect graph-position sensitivity rather than a larger "
        "replacement count or looser candidate-distance distribution.",
        "",
        "| Task | Metric | Replaced | Support | Ham. | High-P | Low-P | High-C | Low-C | High-U | Low-U | Random |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Task']} | {row['Metric']} | {row['Replaced']} | {row['Support']} | {row['Ham.']} | "
            f"{row['High-P']} | {row['Low-P']} | {row['High-C']} | {row['Low-C']} | "
            f"{row['High-U']} | {row['Low-U']} | {row['Random']} |"
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "",
            "- `P`: propagation / fanout-related position.",
            "- `C`: graph-context boundary / neighborhood mismatch.",
            "- `U`: rare-tail / low-redundancy position.",
            "- `Drop` is accuracy drop for node tasks and AUC drop for link tasks; smaller is better.",
            "",
            "This table is intended for Motivation. It should be used to state that semantic candidate quality is not sufficient by itself; downstream damage changes with graph position, and degree alone is not a complete explanation.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/MOTIVATION_MATCHED_REPLACEMENT_FIVE_TASKS.md"),
    )
    args = parser.parse_args()
    rows = [load_summary(task, metric, path) for task, metric, path in TASK_SOURCES]
    write_markdown(rows, args.output)
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()
