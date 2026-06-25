#!/usr/bin/env python3
"""Build the Progressive BFP NPU evaluation table.

The input `policy_summary.tsv` is produced by
`summarize_progressive_bfp_policy_eval.py` and contains one row per
dynamic-pool tag and task.  This script converts it into the table needed by
the paper:

  BFPA4, BFPA6, Rand20, Stress20, Graph-only20, GraphxStress20

For each policy it reports drop, lifted block ratio, effective mantissa bits,
and cycle proxies relative to BFPA4/BFPA6.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


TASKS = ["CN", "CL", "PN", "PL", "AR", "WK"]
POLICIES = [
    ("BFPA4", "", ""),
    ("BFPA6", "", ""),
    ("Rand20", "Random", "20%"),
    ("Stress20", "Stress", "20%"),
    ("Graph-only20", "GraphRisk", "20%"),
    ("GraphxStress20", "GraphxStress", "20%"),
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def pct_to_float(value: str) -> float:
    value = str(value or "").strip()
    if not value or value == "-":
        return float("nan")
    return float(value.replace("%", ""))


def fmt_pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.2f}%"


def fmt_num(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.3f}"


def make_detail_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_policy = {(r["task"], r["selector"], r["ratio"]): r for r in rows}
    detail: list[dict[str, str]] = []
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        rep = task_rows[0] if task_rows else None
        for label, selector, ratio in POLICIES:
            if label == "BFPA4":
                if rep is None:
                    continue
                drop = pct_to_float(rep["bfpa4_drop"])
                lifted = 0.0
                eff_bits = 4.0
            elif label == "BFPA6":
                if rep is None:
                    continue
                drop = pct_to_float(rep["bfpa6_drop"])
                lifted = 100.0
                eff_bits = 6.0
            else:
                row = by_policy.get((task, selector, ratio))
                if row is None:
                    continue
                drop = pct_to_float(row["dynamic_drop"])
                lifted = pct_to_float(row["lifted_blocks"])
                eff_bits = float(row["effective_bits"])
            detail.append(
                {
                    "Task": task,
                    "Policy": label,
                    "Drop": fmt_pct(drop),
                    "Lifted Blocks": fmt_pct(lifted),
                    "Effective Bits": fmt_num(eff_bits),
                    "Cycle/BFPA4": fmt_num(eff_bits / 4.0),
                    "Cycle/BFPA6": fmt_num(eff_bits / 6.0),
                }
            )
    return detail


def read_threshold_rows(threshold_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in sorted(threshold_root.glob("*W4GraphBFPA4to6_B256_tser_top25_t0.2/summary.tsv")):
        for row in read_tsv(summary):
            task = row.get("Task", "")
            if not task:
                continue
            eff_bits = float(row.get("Eff. Bits", "nan"))
            rows.append(
                {
                    "Task": task,
                    "Policy": "GraphxStress-threshold",
                    "Drop": row.get("Dynamic Drop", "-"),
                    "Lifted Blocks": row.get("Lifted Blocks", "-"),
                    "Effective Bits": fmt_num(eff_bits),
                    "Cycle/BFPA4": fmt_num(eff_bits / 4.0),
                    "Cycle/BFPA6": fmt_num(eff_bits / 6.0),
                }
            )
    return rows


def markdown(detail: list[dict[str, str]], threshold: list[dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# Progressive BFP NPU Policy Evaluation")
    lines.append("")
    lines.append("Cycle proxies use effective mantissa bits: Cycle/BFPA4 = EffBits/4 and Cycle/BFPA6 = EffBits/6.")
    lines.append("")
    lines.append("## Fixed 20% Lift Selector Comparison")
    lines.append("")
    lines.append("| Task | BFPA4 | BFPA6 | Rand20 | Stress20 | Graph-only20 | GraphxStress20 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for task in TASKS:
        cells = {r["Policy"]: r["Drop"] for r in detail if r["Task"] == task}
        if not cells:
            continue
        lines.append(
            f"| {task} | {cells.get('BFPA4', '-')} | {cells.get('BFPA6', '-')} | "
            f"{cells.get('Rand20', '-')} | {cells.get('Stress20', '-')} | "
            f"{cells.get('Graph-only20', '-')} | {cells.get('GraphxStress20', '-')} |"
        )
    lines.append("")
    lines.append("## Detailed Cost Rows")
    lines.append("")
    lines.append("| Task | Policy | Drop | Lifted Blocks | Eff. Bits | Cycle/BFPA4 | Cycle/BFPA6 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in detail:
        lines.append(
            f"| {row['Task']} | {row['Policy']} | {row['Drop']} | {row['Lifted Blocks']} | "
            f"{row['Effective Bits']} | {row['Cycle/BFPA4']} | {row['Cycle/BFPA6']} |"
        )
    if threshold:
        lines.append("")
        lines.append("## Threshold-Style GraphxStress Gate")
        lines.append("")
        lines.append("| Task | Policy | Drop | Lifted Blocks | Eff. Bits | Cycle/BFPA4 | Cycle/BFPA6 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in threshold:
            lines.append(
                f"| {row['Task']} | {row['Policy']} | {row['Drop']} | {row['Lifted Blocks']} | "
                f"{row['Effective Bits']} | {row['Cycle/BFPA4']} | {row['Cycle/BFPA6']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy_summary",
        default="/home/zhangshangtong/Transformer/OFA/output/progressive_bfp_policy_eval/policy_summary.tsv",
    )
    parser.add_argument(
        "--threshold_root",
        default="/home/zhangshangtong/Transformer/OFA/output/dual_granularity_bfp_oracle",
    )
    parser.add_argument(
        "--detail_out",
        default="/home/zhangshangtong/Transformer/OFA/output/progressive_bfp_policy_eval/progressive_bfp_main_table.tsv",
    )
    parser.add_argument(
        "--threshold_out",
        default="/home/zhangshangtong/Transformer/OFA/output/progressive_bfp_policy_eval/progressive_bfp_threshold_gate.tsv",
    )
    parser.add_argument(
        "--md_out",
        default="/home/zhangshangtong/Transformer/OFA/GraphhopSimhash/docs/results/PROGRESSIVE_BFP_NPU_EVAL.md",
    )
    args = parser.parse_args()

    rows = read_tsv(Path(args.policy_summary))
    detail = make_detail_rows(rows)
    threshold = read_threshold_rows(Path(args.threshold_root))
    write_tsv(Path(args.detail_out), detail)
    write_tsv(Path(args.threshold_out), threshold)
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text(markdown(detail, threshold), encoding="utf-8")
    print(f"[Saved] {args.detail_out} rows={len(detail)}")
    print(f"[Saved] {args.threshold_out} rows={len(threshold)}")
    print(f"[Saved] {args.md_out}")


if __name__ == "__main__":
    main()
