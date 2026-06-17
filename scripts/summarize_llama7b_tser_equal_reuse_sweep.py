#!/usr/bin/env python3
"""Summarize TSER sweeps by aligning policies at comparable reuse rates."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


DATASET_LABELS = {
    "cora": "CR",
    "pubmed": "PB",
    "arxiv": "AR",
    "wikics": "WK",
    "tape_products": "PR",
    "tape_arxiv23": "TA23",
}

POLICY_LABELS = {
    "no_graph_risk": "Hash only",
    "degree_only": "P only",
    "degree_context": "P+C",
    "degree_unique": "P+U",
    "tser": "Full TSER",
}

POLICY_ORDER = {
    "no_graph_risk": 0,
    "degree_only": 1,
    "degree_context": 2,
    "degree_unique": 3,
    "tser": 4,
}

ROW_KINDS = ("SoftDirectReuse", "ResidualReuse")


@dataclass(frozen=True)
class Row:
    dataset: str
    policy: str
    threshold: float
    kind: str
    reuse: float
    drop: float
    acc: float
    avg_err: float
    log: str


def parse_percent(text: str) -> float:
    text = text.strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def pct(value: float) -> str:
    if math.isnan(value):
        return "-"
    return f"{value * 100:.2f}%"


def parse_name(path: Path) -> tuple[str, str, float]:
    stem = path.name.removesuffix(".log")
    m = re.match(
        r"(?P<dataset>.+)_(?P<policy>no_graph_risk|degree_only|degree_context|degree_unique|tser)_T(?P<t>[-0-9.]+)_runs\d+$",
        stem,
    )
    if not m:
        raise ValueError(path.name)
    return m.group("dataset"), m.group("policy"), float(m.group("t"))


def parse_summary_rows(path: Path) -> list[Row]:
    dataset, policy, threshold = parse_name(path)
    rows: list[Row] = []
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 9 or parts[0] not in ROW_KINDS:
            continue
        rows.append(
            Row(
                dataset=dataset,
                policy=policy,
                threshold=threshold,
                kind=parts[0],
                reuse=parse_percent(parts[1]),
                acc=float(parts[3]),
                drop=parse_percent(parts[4]),
                avg_err=float(parts[5]),
                log=str(path),
            )
        )
    return rows


def pick_closest(rows: list[Row], target: float) -> Row:
    return min(rows, key=lambda r: (abs(r.reuse - target), r.drop))


def write_outputs(rows: list[Row], output_dir: Path, targets: list[float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (
            DATASET_LABELS.get(r.dataset, r.dataset),
            r.kind,
            POLICY_ORDER.get(r.policy, 99),
            r.threshold,
        ),
    )

    frontier_tsv = output_dir / "llama7b_tser_equal_reuse_frontier.tsv"
    with frontier_tsv.open("w") as f:
        f.write("dataset\tkind\tpolicy\tT\treuse\tdrop\tacc\tavg_err\tlog\n")
        for r in rows:
            f.write(
                f"{DATASET_LABELS.get(r.dataset, r.dataset)}\t{r.kind}\t"
                f"{POLICY_LABELS.get(r.policy, r.policy)}\t{r.threshold:g}\t"
                f"{pct(r.reuse)}\t{pct(r.drop)}\t{r.acc:.4f}\t{r.avg_err:.5f}\t{r.log}\n"
            )

    equal_rows: list[tuple[float, Row]] = []
    groups: dict[tuple[str, str, str], list[Row]] = {}
    for r in rows:
        groups.setdefault((r.dataset, r.kind, r.policy), []).append(r)
    for target in targets:
        for key_rows in groups.values():
            equal_rows.append((target, pick_closest(key_rows, target)))

    equal_tsv = output_dir / "llama7b_tser_equal_reuse_closest.tsv"
    with equal_tsv.open("w") as f:
        f.write("target_reuse\tdataset\tkind\tpolicy\tT\treuse\tdrop\tacc\tavg_err\tlog\n")
        for target, r in sorted(
            equal_rows,
            key=lambda x: (
                x[0],
                DATASET_LABELS.get(x[1].dataset, x[1].dataset),
                x[1].kind,
                POLICY_ORDER.get(x[1].policy, 99),
            ),
        ):
            f.write(
                f"{pct(target)}\t{DATASET_LABELS.get(r.dataset, r.dataset)}\t{r.kind}\t"
                f"{POLICY_LABELS.get(r.policy, r.policy)}\t{r.threshold:g}\t"
                f"{pct(r.reuse)}\t{pct(r.drop)}\t{r.acc:.4f}\t{r.avg_err:.5f}\t{r.log}\n"
            )

    md = output_dir / "llama7b_tser_equal_reuse_sweep.md"
    lines = [
        "# LLaMA2-7B TSER Equal-Reuse Sweep",
        "",
        "This sweep compares TSER components at comparable reuse rates. The",
        "SimHash-CAM candidate discovery path is fixed; only the graph-risk",
        "signals and the score threshold vary. `SoftDirectReuse` isolates the",
        "TSER filter before residual repair, while `ResidualReuse` reports the",
        "full lightweight repair path.",
        "",
    ]
    for kind in ROW_KINDS:
        lines.extend(
            [
                f"## Closest Points: {kind}",
                "",
                "| Target reuse | Dataset | Policy | T | Actual reuse | Drop | AvgErr |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for target, r in sorted(
            [x for x in equal_rows if x[1].kind == kind],
            key=lambda x: (
                x[0],
                DATASET_LABELS.get(x[1].dataset, x[1].dataset),
                POLICY_ORDER.get(x[1].policy, 99),
            ),
        ):
            lines.append(
                f"| {pct(target)} | {DATASET_LABELS.get(r.dataset, r.dataset)} | "
                f"{POLICY_LABELS.get(r.policy, r.policy)} | {r.threshold:g} | "
                f"{pct(r.reuse)} | {pct(r.drop)} | {r.avg_err:.5f} |"
            )
        lines.append("")
    lines.extend(
        [
            f"Frontier TSV: `{frontier_tsv}`",
            f"Closest-point TSV: `{equal_tsv}`",
        ]
    )
    md.write_text("\n".join(lines) + "\n")
    print(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", type=float, default=[0.35, 0.40, 0.45, 0.50])
    args = parser.parse_args()

    rows: list[Row] = []
    for path in sorted(args.log_dir.glob("*.log")):
        try:
            rows.extend(parse_summary_rows(path))
        except ValueError:
            continue
    if not rows:
        raise SystemExit(f"No rows parsed from {args.log_dir}")
    write_outputs(rows, args.output_dir, args.targets)


if __name__ == "__main__":
    main()
