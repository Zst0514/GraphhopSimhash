#!/usr/bin/env python3
"""Summarize LLaMA2-7B frontend reuse logs.

The runner writes one log per dataset/threshold. This parser extracts the final
residual-reuse summary table and emits compact TSV/Markdown files for paper
table construction.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DATASET_LABELS = {
    "cora": "CR",
    "pubmed": "PB",
    "arxiv": "AR",
    "wikics": "WK",
    "tape_products": "PR",
    "tape_arxiv23": "AR23",
}


def parse_percent(text: str) -> float:
    text = text.strip()
    if not text or text == "-":
        return float("nan")
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def parse_float(text: str) -> float:
    text = text.strip()
    if not text or text == "-":
        return float("nan")
    return float(text)


def infer_dataset_t(log_path: Path) -> tuple[str, str]:
    stem = log_path.stem
    m = re.match(r"(?P<dataset>.+)_T(?P<t>[-0-9.]+)_runs[0-9]+$", stem)
    if m:
        return m.group("dataset"), m.group("t")
    return stem, ""


def parse_log(log_path: Path) -> list[dict[str, object]]:
    dataset, threshold = infer_dataset_t(log_path)
    text = log_path.read_text(errors="replace")
    runs_match = re.search(r"FINAL RESIDUAL REUSE SUMMARY \((\d+) Runs\)", text)
    runs = int(runs_match.group(1)) if runs_match else None
    baseline_matches = re.findall(r"Baseline Acc:\s*([0-9.]+)", text)
    baseline = float(baseline_matches[-1]) if baseline_matches else float("nan")

    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not any(name in line for name in ("DirectReuse", "SoftDirectReuse", "ResidualReuse")):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 9:
            continue
        config = parts[0]
        if config not in {"DirectReuse", "SoftDirectReuse", "ResidualReuse"}:
            continue
        rows.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "threshold": threshold,
                "runs": runs,
                "baseline_acc": baseline,
                "config": config,
                "reuse": parse_percent(parts[1]),
                "train_pairs": parts[2],
                "acc": parse_float(parts[3]),
                "drop": parse_percent(parts[4]),
                "avg_err": parse_float(parts[5]),
                "hit_err": parse_float(parts[6]),
                "alpha": parts[7],
                "reuse_n_d": parts[8],
                "log": str(log_path),
            }
        )
    return rows


def fmt_pct(value: float) -> str:
    if value != value:
        return "-"
    return f"{value * 100:.2f}%"


def fmt_float(value: float) -> str:
    if value != value:
        return "-"
    return f"{value:.4f}"


def write_tsv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "dataset",
        "label",
        "threshold",
        "runs",
        "baseline_acc",
        "config",
        "reuse",
        "acc",
        "drop",
        "avg_err",
        "hit_err",
        "alpha",
        "reuse_n_d",
        "log",
    ]
    with path.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            out = []
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    out.append(f"{value:.8g}")
                else:
                    out.append("" if value is None else str(value))
            f.write("\t".join(out) + "\n")


def choose_operating_points(rows: list[dict[str, object]], max_drop: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["config"] != "ResidualReuse":
            continue
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset, candidates in sorted(by_dataset.items()):
        feasible = [r for r in candidates if float(r["drop"]) <= max_drop]
        pool = feasible if feasible else candidates
        if not pool:
            continue
        # Highest reuse first; if tied, lowest drop.
        best = sorted(pool, key=lambda r: (-float(r["reuse"]), float(r["drop"])))[0]
        selected.append(best)
    return selected


def write_markdown(rows: list[dict[str, object]], selected: list[dict[str, object]], path: Path, max_drop: float) -> None:
    lines: list[str] = []
    lines.append("# LLaMA2-7B Frontend Reuse Summary")
    lines.append("")
    lines.append(
        "Target pool: `W4BFPA8_B128`. The table below selects the highest-reuse "
        f"`ResidualReuse` point whose drop is no more than `{max_drop * 100:.1f}%` when available."
    )
    lines.append("")
    lines.append("## Selected Operating Points")
    lines.append("")
    lines.append("| Dataset | T | Runs | Baseline Acc | Reuse | Acc | Drop | Log |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["threshold"]),
                    str(row["runs"]),
                    fmt_float(float(row["baseline_acc"])),
                    fmt_pct(float(row["reuse"])),
                    fmt_float(float(row["acc"])),
                    fmt_pct(float(row["drop"])),
                    f"`{Path(str(row['log'])).name}`",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## All Parsed Points")
    lines.append("")
    lines.append("| Dataset | T | Config | Reuse | Acc | Drop |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["threshold"]),
                    str(row["config"]),
                    fmt_pct(float(row["reuse"])),
                    fmt_float(float(row["acc"])),
                    fmt_pct(float(row["drop"])),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_drop", type=float, default=0.02)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for log_path in sorted(log_dir.glob("*.log")):
        rows.extend(parse_log(log_path))
    rows.sort(key=lambda r: (str(r["dataset"]), float(r["threshold"] or 0), str(r["config"])))

    selected = choose_operating_points(rows, args.max_drop)
    write_tsv(rows, output_dir / "summary.tsv")
    write_tsv(selected, output_dir / "selected_operating_points.tsv")
    write_markdown(rows, selected, output_dir / "summary.md", args.max_drop)

    print(f"[Summary] parsed_rows={len(rows)}")
    print(f"[Summary] selected={len(selected)}")
    print(f"[Summary] wrote {output_dir / 'summary.tsv'}")
    print(f"[Summary] wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
