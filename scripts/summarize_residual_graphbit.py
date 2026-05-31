#!/usr/bin/env python3
"""Summarize residual-reuse + Graph-Bit logs into compact tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CONFIG_ALIASES = {
    "RandomDepthBudget": "Rand",
    "DegreeDepthBudget": "Deg",
    "TSERDepthBudget": "TSER",
    "ContextDepthBudget": "Ctx",
    "LowUniqueDepthBudget": "Uniq",
}


RUN_RE = re.compile(r"T(?P<T>\d+)_(?P<budget>.+)_runs(?P<runs>\d+)\.log$")
SUMMARY_RE = re.compile(r"FINAL RESIDUAL \+ GRAPH-BIT SUMMARY \((?P<runs>\d+) Runs\) \| (?P<dataset>\w+)")


def is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_log_meta(path: Path) -> dict[str, str]:
    match = RUN_RE.search(path.name)
    meta = {
        "dataset": "",
        "heads": "",
        "T": "",
        "budget": "",
        "runs": "",
    }
    parts = path.parts
    if len(parts) >= 3:
        meta["heads"] = parts[-2]
        meta["dataset"] = parts[-3]
    if match:
        meta.update(match.groupdict())
    return meta


def parse_table_row(line: str) -> dict[str, str] | None:
    if "|" not in line:
        return None
    fields = [part.strip() for part in line.split("|")]
    if len(fields) < 12:
        return None
    cfg = fields[0]
    if cfg in {"Config", "Cfg", ""}:
        return None
    if set(cfg) <= {"-"}:
        return None
    if not fields[1].endswith("%"):
        return None
    has_p7 = len(fields) >= 15 and is_number(fields[9])
    has_p5 = len(fields) >= 14 and is_number(fields[8])
    has_no_p5 = len(fields) >= 13 and is_number(fields[7])
    if has_p7:
        p7 = fields[5]
        p6 = fields[6]
        p5 = fields[7]
        p4 = fields[8]
        cost = fields[9]
        acc = fields[10]
        drop = fields[11]
        finalerr = fields[12]
        train = fields[13]
        alpha = fields[14] if len(fields) > 14 else ""
    elif has_p5:
        p7 = "0.0%"
        p6 = fields[5]
        p5 = fields[6]
        p4 = fields[7]
        cost = fields[8]
        acc = fields[9]
        drop = fields[10]
        finalerr = fields[11]
        train = fields[12]
        alpha = fields[13] if len(fields) > 13 else ""
    elif has_no_p5:
        p7 = "0.0%"
        p6 = fields[5]
        p5 = "0.0%"
        p4 = fields[6]
        cost = fields[7]
        acc = fields[8]
        drop = fields[9]
        finalerr = fields[10]
        train = fields[11]
        alpha = fields[12] if len(fields) > 12 else ""
    else:
        return None
    return {
        "config": CONFIG_ALIASES.get(cfg, cfg),
        "reuse": fields[1],
        "direct": fields[2],
        "residual": fields[3],
        "P8": fields[4],
        "P7": p7,
        "P6": p6,
        "P5": p5,
        "P4": p4,
        "cost": cost,
        "acc": acc,
        "drop": drop,
        "finalerr": finalerr,
        "train": train,
        "alpha": alpha,
    }


def parse_log(path: Path) -> list[dict[str, str]]:
    meta = infer_log_meta(path)
    rows: list[dict[str, str]] = []
    in_summary = False
    for line in path.read_text(errors="replace").splitlines():
        summary_match = SUMMARY_RE.search(line)
        if summary_match:
            in_summary = True
            meta["dataset"] = summary_match.group("dataset").lower()
            meta["runs"] = summary_match.group("runs")
            continue
        if not in_summary:
            continue
        row = parse_table_row(line)
        if row:
            rows.append({**meta, **row, "log": str(path)})
        elif line.startswith("=") and rows:
            break
    return rows


def sort_key(row: dict[str, str]) -> tuple:
    config_order = {
        "FullP8": 0,
        "AllP6": 1,
        "AllP5": 2,
        "AllP4": 3,
        "Rand": 4,
        "Deg": 5,
        "TSER": 6,
        "Ctx": 7,
        "Uniq": 8,
    }
    return (
        row["dataset"],
        row["heads"],
        int(row["T"] or 0),
        row["budget"],
        config_order.get(row["config"], 99),
    )


def write_tsv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "heads",
        "T",
        "budget",
        "runs",
        "config",
        "reuse",
        "direct",
        "residual",
        "P8",
        "P7",
        "P6",
        "P5",
        "P4",
        "cost",
        "acc",
        "drop",
        "finalerr",
        "train",
        "alpha",
        "log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_compact(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["heads"], row["T"], row["budget"])
        grouped.setdefault(key, {})[row["config"]] = row

    lines = [
        "Residual + Graph-Bit compact summary",
        "FullP8 = accepted reuse/residual hits plus all misses at P8.",
        "Rand/Deg/TSER = same reuse set; only miss nodes are routed to P8/P6/P4 by that policy.",
        "",
        "dataset heads T  budget        reuse  FullDrop FullCost RandDrop DegDrop TSERDrop DegCost",
        "-----------------------------------------------------------------------------------------",
    ]
    for key in sorted(grouped, key=lambda item: (item[0], item[1], int(item[2] or 0), item[3])):
        dataset, heads, threshold, budget = key
        group = grouped[key]
        full = group.get("FullP8", {})
        rand = group.get("Rand", {})
        deg = group.get("Deg", {})
        tser = group.get("TSER", {})
        reuse = full.get("reuse") or deg.get("reuse") or ""
        lines.append(
            f"{dataset:<7} {heads:<5} {threshold:<2} {budget:<13} "
            f"{reuse:>6} {full.get('drop', ''):>8} {full.get('cost', ''):>8} "
            f"{rand.get('drop', ''):>8} {deg.get('drop', ''):>7} "
            f"{tser.get('drop', ''):>8} {deg.get('cost', ''):>7}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=root / "output" / "residual_graphbit_three_depth_probe" / "logs")
    parser.add_argument("--output-tsv", type=Path, default=root / "output" / "residual_graphbit_three_depth_probe" / "three_depth_summary.tsv")
    parser.add_argument("--output-txt", type=Path, default=root / "output" / "residual_graphbit_three_depth_probe" / "three_depth_summary.txt")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for log_path in sorted(args.log_dir.rglob("*.log")):
        rows.extend(parse_log(log_path))
    rows = sorted(rows, key=sort_key)
    if not rows:
        raise SystemExit(f"No summary rows found under {args.log_dir}")

    write_tsv(rows, args.output_tsv)
    write_compact(rows, args.output_txt)
    print(f"[ResidualGraphBitSummary] wrote {args.output_tsv} | rows={len(rows)}")
    print(f"[ResidualGraphBitSummary] wrote {args.output_txt}")


if __name__ == "__main__":
    main()
