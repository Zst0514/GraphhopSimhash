#!/usr/bin/env python3
"""Summarize T31 fixed-front-end Graph-Bit bound policy sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PERCENT_COLS = ("reuse", "direct", "residual", "P8", "P7", "P6", "P5", "P4", "drop")
DEPTH_BITS = {"P8": 8.0, "P7": 7.0, "P6": 6.0, "P5": 5.0, "P4": 4.0}
PREFERRED_CONFIGS = (
    "DegBoundNode",
    "DegBound",
    "Deg",
    "TSERBoundNode",
    "TSERBound",
    "TSER",
    "CtxBoundNode",
    "CtxBound",
    "Ctx",
)


def parse_pct(value: str) -> float:
    text = str(value or "").strip().rstrip("%")
    return float(text) if text else 0.0


def parse_float(value: str) -> float:
    text = str(value or "").strip().rstrip("%")
    return float(text) if text else 0.0


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_num(value: float) -> str:
    return f"{value:.4f}"


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["policy"]: row for row in csv.DictReader(handle, delimiter="\t")}


def policy_from_summary(path: Path) -> str:
    # Expected: root / dataset_frontend / policy / summary.tsv
    return path.parent.name


def dataset_from_summary(path: Path) -> str:
    # Prefer table content later; this is only fallback.
    name = path.parent.parent.name
    return name.split("_", 1)[0]


def read_summary(path: Path, manifest: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    policy = policy_from_summary(path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            row = dict(row)
            row["policy"] = policy
            row["dataset"] = row.get("dataset") or dataset_from_summary(path)
            for key, value in manifest.get(policy, {}).items():
                if key != "policy":
                    row[key] = value
            rows.append(row)
    return rows


def avg_depth(row: dict[str, str]) -> float:
    total = 0.0
    weighted = 0.0
    for key, bit in DEPTH_BITS.items():
        pct = parse_pct(row.get(key, "0"))
        total += pct
        weighted += pct * bit
    return weighted / total if total > 0.0 else 0.0


def choose_graphbit_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    by_config = {row.get("config", ""): row for row in rows}
    for config in PREFERRED_CONFIGS:
        if config in by_config:
            return by_config[config]
    for row in rows:
        config = row.get("config", "")
        if config not in {"FullP8", "AllP6", "AllP5", "AllP4", "Rand"}:
            return row
    return None


def build_dataset_rows(all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in all_rows:
        grouped.setdefault((row["dataset"], row["policy"]), []).append(row)

    out: list[dict[str, str]] = []
    for (dataset, policy), rows in sorted(grouped.items()):
        full = next((row for row in rows if row.get("config") == "FullP8"), None)
        graph = choose_graphbit_row(rows)
        if not full or not graph:
            continue

        full_drop = parse_pct(full.get("drop", "0"))
        graph_drop = parse_pct(graph.get("drop", "0"))
        full_cost = parse_float(full.get("cost", "0"))
        graph_cost = parse_float(graph.get("cost", "0"))
        cost_save = 0.0 if full_cost <= 0 else 100.0 * (full_cost - graph_cost) / full_cost

        entry = {
            "dataset": dataset,
            "policy": policy,
            "config": graph.get("config", ""),
            "reuse": graph.get("reuse", ""),
            "direct": graph.get("direct", ""),
            "residual": graph.get("residual", ""),
            "P8": graph.get("P8", ""),
            "P7": graph.get("P7", ""),
            "P6": graph.get("P6", ""),
            "P5": graph.get("P5", ""),
            "P4": graph.get("P4", ""),
            "avg_depth": fmt_num(avg_depth(graph)),
            "full_drop": fmt_pct(full_drop),
            "drop": fmt_pct(graph_drop),
            "extra_drop": fmt_pct(graph_drop - full_drop),
            "full_cost": fmt_num(full_cost),
            "cost": fmt_num(graph_cost),
            "cost_save_vs_full": fmt_pct(cost_save),
            "acc": graph.get("acc", ""),
            "log": graph.get("log", ""),
        }
        for key in (
            "nodewise_min_depth",
            "nodewise_min_tol",
            "nodewise_max_tol",
            "nodewise_gamma",
            "nodewise_risk_max",
            "high_min",
            "high_tol",
            "mid_min",
            "mid_tol",
            "low_min",
            "low_tol",
            "scale",
            "high_ratio",
            "mid_ratio",
            "low_ratio",
        ):
            entry[key] = graph.get(key, "")
        out.append(entry)
    return out


def build_pareto_rows(dataset_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in dataset_rows:
        grouped.setdefault(row["policy"], []).append(row)

    out: list[dict[str, str]] = []
    for policy, rows in sorted(grouped.items()):
        drops = [parse_pct(row["drop"]) for row in rows]
        extra = [parse_pct(row["extra_drop"]) for row in rows]
        avg_depths = [parse_float(row["avg_depth"]) for row in rows]
        cost_saves = [parse_pct(row["cost_save_vs_full"]) for row in rows]
        reuses = [parse_pct(row["reuse"]) for row in rows]
        entry = {
            "policy": policy,
            "datasets": ",".join(row["dataset"] for row in rows),
            "min_reuse": fmt_pct(min(reuses) if reuses else 0.0),
            "max_drop": fmt_pct(max(drops) if drops else 0.0),
            "max_extra_drop": fmt_pct(max(extra) if extra else 0.0),
            "mean_avg_depth": fmt_num(sum(avg_depths) / len(avg_depths) if avg_depths else 0.0),
            "mean_cost_save_vs_full": fmt_pct(sum(cost_saves) / len(cost_saves) if cost_saves else 0.0),
        }
        if rows:
            for key in (
                "nodewise_min_depth",
                "nodewise_min_tol",
                "nodewise_max_tol",
                "nodewise_gamma",
                "nodewise_risk_max",
                "high_min",
                "high_tol",
                "mid_min",
                "mid_tol",
                "low_min",
                "low_tol",
                "scale",
                "high_ratio",
                "mid_ratio",
                "low_ratio",
            ):
                entry[key] = rows[0].get(key, "")
        out.append(entry)
    return out


def write_tsv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_aligned(rows: list[dict[str, str]], path: Path, title: str, columns: list[str]) -> None:
    lines = [title]
    if not rows:
        lines.append("(empty)")
        path.write_text("\n".join(lines) + "\n")
        return
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
    lines.append(" ".join(col.ljust(widths[col]) for col in columns))
    lines.append(" ".join("-" * widths[col] for col in columns))
    for row in rows:
        lines.append(" ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    parser.add_argument("--pareto-tsv", type=Path, required=True)
    parser.add_argument("--pareto-txt", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    all_rows: list[dict[str, str]] = []
    for summary in sorted(args.root.glob("*/*/summary.tsv")):
        all_rows.extend(read_summary(summary, manifest))

    dataset_rows = build_dataset_rows(all_rows)
    pareto_rows = build_pareto_rows(dataset_rows)
    write_tsv(dataset_rows, args.output_tsv)
    write_tsv(pareto_rows, args.pareto_tsv)

    write_aligned(
        dataset_rows,
        args.output_txt,
        "T31 fixed-front-end Graph-Bit bound policy sweep",
        [
            "dataset",
            "policy",
            "config",
            "nodewise_min_depth",
            "nodewise_min_tol",
            "nodewise_max_tol",
            "nodewise_gamma",
            "reuse",
            "P8",
            "P7",
            "P6",
            "P5",
            "P4",
            "avg_depth",
            "full_drop",
            "drop",
            "extra_drop",
            "cost_save_vs_full",
        ],
    )
    write_aligned(
        pareto_rows,
        args.pareto_txt,
        "Cross-dataset Pareto summary",
        [
            "policy",
            "datasets",
            "nodewise_min_depth",
            "nodewise_min_tol",
            "nodewise_max_tol",
            "nodewise_gamma",
            "min_reuse",
            "max_drop",
            "max_extra_drop",
            "mean_avg_depth",
            "mean_cost_save_vs_full",
            "high_min",
            "high_tol",
            "mid_min",
            "mid_tol",
            "low_min",
            "low_tol",
        ],
    )
    print(f"[T31BoundSweepSummary] wrote {args.output_tsv}")
    print(f"[T31BoundSweepSummary] wrote {args.output_txt}")
    print(f"[T31BoundSweepSummary] wrote {args.pareto_tsv}")
    print(f"[T31BoundSweepSummary] wrote {args.pareto_txt}")


if __name__ == "__main__":
    main()
