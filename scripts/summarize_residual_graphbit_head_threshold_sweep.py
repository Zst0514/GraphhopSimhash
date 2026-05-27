#!/usr/bin/env python3
import csv
import math
import re
import sys
from pathlib import Path


CONFIGS = {
    "FullP8",
    "AllP6",
    "AllP5",
    "AllP4",
    "RandomDepthBudget",
    "DegreeDepthBudget",
    "TSERDepthBudget",
    "ContextDepthBudget",
    "LowUniqueDepthBudget",
}


def parse_percent(value):
    value = value.strip().rstrip("%")
    return float(value) if value else math.nan


def parse_float(value):
    value = value.strip()
    return float(value) if value else math.nan


def parse_log_path(log_path):
    # .../logs/cora/h4/T20_runs3.log
    dataset = log_path.parent.parent.name
    heads_match = re.match(r"h(\d+)", log_path.parent.name)
    threshold_match = re.match(r"T(\d+)_runs(\d+)", log_path.stem)
    if not heads_match or not threshold_match:
        return None
    return {
        "dataset": dataset,
        "heads": int(heads_match.group(1)),
        "threshold": int(threshold_match.group(1)),
        "runs": int(threshold_match.group(2)),
    }


def parse_log(log_path):
    meta = parse_log_path(log_path)
    if meta is None:
        return []

    text = log_path.read_text(errors="replace")
    baseline = ""
    baseline_matches = re.findall(r"Baseline Acc:\s+([-0-9.]+)", text)
    if baseline_matches:
        baseline = baseline_matches[-1]

    rows = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 14 or parts[0] not in CONFIGS:
            continue
        rows.append(
            {
                **meta,
                "config": parts[0],
                "reuse": parse_percent(parts[1]),
                "direct": parse_percent(parts[2]),
                "residual": parse_percent(parts[3]),
                "P8": parse_percent(parts[4]),
                "P6": parse_percent(parts[5]),
                "P5": parse_percent(parts[6]),
                "P4": parse_percent(parts[7]),
                "cost": parse_float(parts[8]),
                "acc": parse_float(parts[9]),
                "drop": parse_percent(parts[10]),
                "finalerr": parse_float(parts[11]),
                "train_pairs": parse_float(parts[12]),
                "alpha": parse_float(parts[13]),
                "baseline": baseline,
                "log": str(log_path),
            }
        )
    return rows


def fmt(value, width=7, suffix=""):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA".rjust(width)
    if suffix == "%":
        return f"{value:>{width}.1f}%"
    return f"{value:>{width}.3f}"


def write_pivot(rows, out_dir):
    by_key = {}
    for row in rows:
        key = (row["dataset"], row["threshold"], row["heads"], row["config"])
        by_key[key] = row

    dataset_thresholds = sorted({(r["dataset"], r["threshold"]) for r in rows})
    lines = []
    lines.append("Residual + Graph-Bit head/threshold sweep")
    lines.append("Rows compare 4x16 and 8x16 at the same dataset and reuse threshold.")
    lines.append("")
    header = (
        f"{'dataset':<8} {'T':>4} | "
        f"{'h4 reuse':>9} {'h4 Full':>8} {'h4 Deg':>8} {'h4 cost':>8} | "
        f"{'h8 reuse':>9} {'h8 Full':>8} {'h8 Deg':>8} {'h8 cost':>8} | "
        f"{'dReuse':>8} {'dFull':>8} {'dDeg':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for dataset, threshold in dataset_thresholds:
        h4_full = by_key.get((dataset, threshold, 4, "FullP8"))
        h4_deg = by_key.get((dataset, threshold, 4, "DegreeDepthBudget"))
        h8_full = by_key.get((dataset, threshold, 8, "FullP8"))
        h8_deg = by_key.get((dataset, threshold, 8, "DegreeDepthBudget"))

        h4_reuse = h4_full["reuse"] if h4_full else math.nan
        h8_reuse = h8_full["reuse"] if h8_full else math.nan
        h4_full_drop = h4_full["drop"] if h4_full else math.nan
        h8_full_drop = h8_full["drop"] if h8_full else math.nan
        h4_deg_drop = h4_deg["drop"] if h4_deg else math.nan
        h8_deg_drop = h8_deg["drop"] if h8_deg else math.nan
        h4_deg_cost = h4_deg["cost"] if h4_deg else math.nan
        h8_deg_cost = h8_deg["cost"] if h8_deg else math.nan

        delta_reuse = h8_reuse - h4_reuse
        delta_full = h8_full_drop - h4_full_drop
        delta_deg = h8_deg_drop - h4_deg_drop

        lines.append(
            f"{dataset:<8} {threshold:>4} | "
            f"{fmt(h4_reuse, 8, '%')} {fmt(h4_full_drop, 7, '%')} {fmt(h4_deg_drop, 7, '%')} {fmt(h4_deg_cost, 8)} | "
            f"{fmt(h8_reuse, 8, '%')} {fmt(h8_full_drop, 7, '%')} {fmt(h8_deg_drop, 7, '%')} {fmt(h8_deg_cost, 8)} | "
            f"{fmt(delta_reuse, 7, '%')} {fmt(delta_full, 7, '%')} {fmt(delta_deg, 7, '%')}"
        )

    pivot_path = out_dir / "head_threshold_pivot.txt"
    pivot_path.write_text("\n".join(lines) + "\n")


def write_best(rows, out_dir):
    lines = []
    lines.append("Best deployable DegreeDepthBudget points by dataset/head")
    lines.append("Sorted by FullP8 drop first, then Degree drop, then cost.")
    lines.append("")
    header = (
        f"{'dataset':<8} {'heads':>5} {'T':>4} {'reuse':>8} "
        f"{'full_drop':>10} {'degree_drop':>11} {'cost':>7} {'acc':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    keys = sorted({(r["dataset"], r["heads"]) for r in rows})
    for dataset, heads in keys:
        subset = [r for r in rows if r["dataset"] == dataset and r["heads"] == heads]
        full_by_t = {r["threshold"]: r for r in subset if r["config"] == "FullP8"}
        degree = [r for r in subset if r["config"] == "DegreeDepthBudget"]
        candidates = []
        for row in degree:
            full = full_by_t.get(row["threshold"])
            if not full:
                continue
            candidates.append((full["drop"], row["drop"], row["cost"], row, full))
        for _full_drop, _degree_drop, _cost, row, full in sorted(candidates)[:5]:
            lines.append(
                f"{dataset:<8} {heads:>5} {row['threshold']:>4} "
                f"{fmt(full['reuse'], 7, '%')} {fmt(full['drop'], 9, '%')} "
                f"{fmt(row['drop'], 10, '%')} {fmt(row['cost'], 7)} {row['acc']:>8.4f}"
            )
        lines.append("")

    best_path = out_dir / "best_degree_points.txt"
    best_path.write_text("\n".join(lines).rstrip() + "\n")


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_residual_graphbit_head_threshold_sweep.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1])
    log_dir = out_dir / "logs"
    rows = []
    for log_path in sorted(log_dir.glob("*/*/*.log")):
        rows.extend(parse_log(log_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.tsv"
    fieldnames = [
        "dataset",
        "heads",
        "threshold",
        "runs",
        "config",
        "reuse",
        "direct",
        "residual",
        "P8",
        "P6",
        "P5",
        "P4",
        "cost",
        "acc",
        "drop",
        "finalerr",
        "train_pairs",
        "alpha",
        "baseline",
        "log",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_pivot(rows, out_dir)
    write_best(rows, out_dir)
    print(f"[Summary] wrote {summary_path}")
    print(f"[Summary] wrote {out_dir / 'head_threshold_pivot.txt'}")
    print(f"[Summary] wrote {out_dir / 'best_degree_points.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
