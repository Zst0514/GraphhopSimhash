#!/usr/bin/env python3
"""Aggregate equal-reuse TSER ablation results across datasets."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


POLICY_ORDER = {
    "Hash only": 0,
    "P only": 1,
    "P+C": 2,
    "P+U": 3,
    "Full TSER": 4,
}


@dataclass(frozen=True)
class Row:
    target: str
    dataset: str
    kind: str
    policy: str
    threshold: str
    reuse: float
    drop: float
    acc: float
    avg_err: float
    log: str


def parse_pct(text: str) -> float:
    text = text.strip()
    if not text or text == "-":
        return math.nan
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def pct(value: float) -> str:
    if math.isnan(value):
        return "-"
    return f"{value * 100:.2f}%"


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            rows.append(
                Row(
                    target=rec["target_reuse"],
                    dataset=rec["dataset"],
                    kind=rec["kind"],
                    policy=rec["policy"],
                    threshold=rec["T"],
                    reuse=parse_pct(rec["reuse"]),
                    drop=parse_pct(rec["drop"]),
                    acc=float(rec["acc"]),
                    avg_err=float(rec["avg_err"]),
                    log=rec["log"],
                )
            )
    return rows


def collect(output_root: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(output_root.glob("llama7b_tser_equal_reuse_sweep_*/llama7b_tser_equal_reuse_closest.tsv")):
        rows.extend(read_rows(path))
    return rows


def group_rows(rows: list[Row]) -> dict[tuple[str, str, str], list[Row]]:
    grouped: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row.target, row.kind, row.dataset)].append(row)
    return grouped


def best_policy(rows: list[Row]) -> Row:
    return min(rows, key=lambda r: (r.drop, abs(r.reuse - parse_pct(r.target))))


def mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def write_markdown(rows: list[Row], output: Path, output_root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped = group_rows(rows)
    datasets = sorted({r.dataset for r in rows})
    targets = sorted({r.target for r in rows}, key=parse_pct)

    lines: list[str] = [
        "# TSER Equal-Reuse Ablation",
        "",
        "This file aggregates the equal-reuse TSER component ablation. Candidate",
        "discovery is fixed; each policy is compared at the closest available reuse",
        "point to the target budget.",
        "",
        "Policies:",
        "",
        "| Policy | Risk Terms | Meaning |",
        "| --- | --- | --- |",
        "| Hash only | none | SimHash support/distance without graph-risk scoring. |",
        "| P only | P | Propagation risk only. |",
        "| P+C | P+C | Adds graph-context/boundary risk. |",
        "| P+U | P+U | Adds low-degree uniqueness risk. |",
        "| Full TSER | P+C+U | Uses all three TSER terms. |",
        "",
        f"Current datasets found: {', '.join(datasets) if datasets else 'none'}.",
        "",
    ]

    preferred_target = "40.00%" if "40.00%" in targets else (targets[-1] if targets else "")
    if preferred_target:
        lines.extend(
            [
                f"## Main Comparison Around {preferred_target} Reuse",
                "",
                "The table uses the `ResidualReuse` row, which includes the full TSER",
                "filtering plus residual repair path.",
                "",
                "| Dataset | Policy | T | Actual reuse | Drop | AvgErr |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for ds in datasets:
            rs = sorted(
                grouped.get((preferred_target, "ResidualReuse", ds), []),
                key=lambda r: POLICY_ORDER.get(r.policy, 99),
            )
            for r in rs:
                lines.append(
                    f"| {ds} | {r.policy} | {r.threshold} | {pct(r.reuse)} | "
                    f"{pct(r.drop)} | {r.avg_err:.5f} |"
                )
        lines.append("")

        lines.extend(
            [
                "## Best Policy Per Dataset",
                "",
                "| Dataset | Best policy | Actual reuse | Drop | Hash-only drop | Gain vs Hash |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        gains: list[float] = []
        for ds in datasets:
            rs = grouped.get((preferred_target, "ResidualReuse", ds), [])
            if not rs:
                continue
            best = best_policy(rs)
            hash_rows = [r for r in rs if r.policy == "Hash only"]
            hash_drop = hash_rows[0].drop if hash_rows else math.nan
            gain = hash_drop - best.drop if not math.isnan(hash_drop) else math.nan
            if not math.isnan(gain):
                gains.append(gain)
            lines.append(
                f"| {ds} | {best.policy} | {pct(best.reuse)} | {pct(best.drop)} | "
                f"{pct(hash_drop)} | {pct(gain)} |"
            )
        lines.append("")
        if gains:
            lines.append(
                f"Average drop reduction of the best policy over Hash-only at "
                f"{preferred_target} reuse: **{pct(mean(gains))}**."
            )
            lines.append("")

        lines.extend(
            [
                "## Average Drop By Policy",
                "",
                "| Policy | Avg. reuse | Avg. drop |",
                "| --- | ---: | ---: |",
            ]
        )
        for policy in sorted(POLICY_ORDER, key=POLICY_ORDER.get):
            prs = [
                r
                for ds in datasets
                for r in grouped.get((preferred_target, "ResidualReuse", ds), [])
                if r.policy == policy
            ]
            if prs:
                lines.append(
                    f"| {policy} | {pct(mean([r.reuse for r in prs]))} | "
                    f"{pct(mean([r.drop for r in prs]))} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Source Files",
            "",
        ]
    )
    for path in sorted(output_root.glob("llama7b_tser_equal_reuse_sweep_*/llama7b_tser_equal_reuse_closest.tsv")):
        lines.append(f"- `{path}`")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("/home/zhangshangtong/Transformer/OFA/output"))
    parser.add_argument("--output", type=Path, default=Path("docs/results/TSER_EQUAL_REUSE_ABLATION.md"))
    args = parser.parse_args()

    rows = collect(args.output_root)
    if not rows:
        raise SystemExit(f"No equal-reuse closest TSV files found under {args.output_root}")
    write_markdown(rows, args.output, args.output_root)
    print(args.output)


if __name__ == "__main__":
    main()
