#!/usr/bin/env python3
"""Summarize progressive BFPA4->BFPA6 selector evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


def parse_tag(tag: str) -> dict[str, Any]:
    if "random" in tag:
        selector = "Random"
    elif "oracle" in tag:
        selector = "OracleError"
    elif "graphstress" in tag:
        selector = "GraphxStress"
    elif "_stress" in tag:
        selector = "Stress"
    elif "_graph" in tag:
        selector = "GraphRisk"
    elif "_t" in tag:
        selector = "GraphxStressThreshold"
    else:
        selector = "Unknown"

    ratio = ""
    match = re.search(r"(?:random|oracle|stress|graphstress|graph)(\d+)$", tag)
    if match:
        ratio = f"{int(match.group(1))}%"
    top_match = re.search(r"_top(\d+)_t([0-9.]+)", tag)
    if top_match:
        ratio = f"top{top_match.group(1)}@t{top_match.group(2)}"
    return {"selector": selector, "ratio": ratio}


def read_tsv(path: Path) -> list[dict[str, str]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", default="output/progressive_bfp_policy_eval/eval")
    parser.add_argument("--output", default="output/progressive_bfp_policy_eval/policy_summary.tsv")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    rows: list[dict[str, Any]] = []
    for summary in sorted(eval_dir.glob("*/summary.tsv")):
        tag = summary.parent.name
        meta = parse_tag(tag)
        for item in read_tsv(summary):
            rows.append(
                {
                    "tag": tag,
                    "selector": meta["selector"],
                    "ratio": meta["ratio"],
                    "task": item.get("Task", ""),
                    "ref": item.get("Ref", ""),
                    "bfpa4_drop": item.get("BFPA4 Drop", ""),
                    "dynamic_drop": item.get("Dynamic Drop", ""),
                    "bfpa6_drop": item.get("BFPA6 Drop", ""),
                    "lifted_blocks": item.get("Lifted Blocks", ""),
                    "effective_bits": item.get("Eff. Bits", ""),
                }
            )
    write_tsv(Path(args.output), rows)
    print(f"[Saved] {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
