#!/usr/bin/env python3
"""Summarize LLaMA2-7B TSER score-gate ablation logs.

The experiment isolates the score design before residual repair.  For every
policy log, we report the SoftDirectReuse row because it is the candidate set
after the configured score gate but before residual correction.
"""

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

POLICY_TERMS = {
    "no_graph_risk": "none",
    "degree_only": "P",
    "degree_context": "P+C",
    "degree_unique": "P+U",
    "tser": "P+C+U",
}

POLICY_ORDER = {
    "no_graph_risk": 0,
    "degree_only": 1,
    "degree_context": 2,
    "degree_unique": 3,
    "tser": 4,
}


@dataclass
class Row:
    dataset: str
    policy: str
    threshold: str
    runs: int | None
    reuse: float
    acc: float
    drop: float
    avg_err: float
    log: str


def parse_percent(text: str) -> float:
    text = text.strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def parse_name(path: Path) -> tuple[str, str, str]:
    stem = path.name.removesuffix(".log")
    m = re.match(r"(?P<dataset>.+)_(?P<policy>no_graph_risk|degree_only|degree_context|degree_unique|tser)_T(?P<t>[-0-9.]+)_runs\d+$", stem)
    if not m:
        raise ValueError(f"Cannot parse log name: {path.name}")
    return m.group("dataset"), m.group("policy"), m.group("t")


def parse_log(path: Path) -> Row | None:
    dataset, policy, threshold = parse_name(path)
    text = path.read_text(errors="replace")
    runs_match = re.search(r"FINAL RESIDUAL REUSE SUMMARY \((\d+) Runs\)", text)
    runs = int(runs_match.group(1)) if runs_match else None

    # Parse the final summary table row:
    # SoftDirectReuse | 46.5% | 0.7023 | 0.7107 | ...
    parsed: Row | None = None
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 9 or parts[0] != "SoftDirectReuse":
            continue
        parsed = Row(
            dataset=dataset,
            policy=policy,
            threshold=threshold,
            runs=runs,
            reuse=parse_percent(parts[1]),
            acc=float(parts[3]),
            drop=parse_percent(parts[4]),
            avg_err=float(parts[5]),
            log=str(path),
        )
    return parsed


def pct(value: float) -> str:
    if math.isnan(value):
        return "-"
    return f"{value * 100:.2f}%"


def write_outputs(rows: list[Row], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (DATASET_LABELS.get(r.dataset, r.dataset), POLICY_ORDER.get(r.policy, 99)))

    tsv = output_dir / "llama7b_tser_score_ablation.tsv"
    with tsv.open("w") as f:
        f.write("dataset\tpolicy\trisk_terms\tthreshold\truns\treuse\tdrop\tacc\tavg_err\tlog\n")
        for r in rows:
            f.write(
                f"{DATASET_LABELS.get(r.dataset, r.dataset)}\t"
                f"{POLICY_LABELS.get(r.policy, r.policy)}\t"
                f"{POLICY_TERMS.get(r.policy, '')}\t"
                f"{r.threshold}\t{r.runs or ''}\t{pct(r.reuse)}\t{pct(r.drop)}\t"
                f"{r.acc:.4f}\t{r.avg_err:.5f}\t{r.log}\n"
            )

    md = output_dir / "llama7b_tser_score_ablation.md"
    lines = [
        "# LLaMA2-7B TSER Score Ablation",
        "",
        "This is a component ablation, not a parameter sweep. The SimHash-CAM",
        "candidate discovery frontend is fixed, and each row reports the",
        "`SoftDirectReuse` result before residual repair. Only the graph-risk",
        "signal used by TSER changes.",
        "",
        "| Dataset | Policy | Risk terms | T | Reuse | Drop | AvgErr |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {DATASET_LABELS.get(r.dataset, r.dataset)} | {POLICY_LABELS.get(r.policy, r.policy)} | "
            f"{POLICY_TERMS.get(r.policy, '')} | {r.threshold} | "
            f"{pct(r.reuse)} | {pct(r.drop)} | {r.avg_err:.5f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `Hash only` keeps the same SimHash/CAM candidates but removes graph-risk filtering.",
            "- `P only` keeps only propagation risk from node degree.",
            "- `P+C` adds graph-context/boundary risk.",
            "- `P+U` adds rare low-degree uniqueness risk.",
            "- `Full TSER` uses all three terms, `P+C+U`.",
        ]
    )
    lines.extend(
        [
            "",
            f"Source logs: `{output_dir / 'logs'}`",
            f"Decision traces: `{output_dir / 'traces'}`",
            f"TSV: `{tsv}`",
        ]
    )
    md.write_text("\n".join(lines) + "\n")
    print(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    rows: list[Row] = []
    for path in sorted(args.log_dir.glob("*.log")):
        try:
            row = parse_log(path)
        except ValueError:
            continue
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No rows parsed from {args.log_dir}")
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
