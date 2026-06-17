#!/usr/bin/env python3
"""Summarize reuse-safety ablation logs.

The residual-reuse runner reports three evaluated feature sets per run:

* DirectReuse: only high-support anchors are used.
* SoftDirectReuse: all support-accepted anchors are used without residual repair.
* ResidualReuse: fuzzy anchors are repaired and may be rejected by the gate.

When the log is produced with ``--enable_score_gate``, SoftDirectReuse is the
TSER-filtered no-repair path.  When the log is produced with
``--disable_score_gate``, SoftDirectReuse is the support-only no-TSER baseline.
This parser keeps those two cases separate so the paper table does not blur
candidate discovery with graph-risk filtering.
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
    "tape_arxiv23": "AR23",
}


CONFIGS = {"DirectReuse", "SoftDirectReuse", "ResidualReuse"}


@dataclass
class ParsedConfig:
    dataset: str
    threshold: str
    runs: int | None
    gate: str
    baseline_acc: float
    config: str
    reuse: float
    acc: float
    drop: float
    avg_err: float
    hit_err: float
    reuse_num: float
    reuse_den: float
    log: str
    corrected_mean: float | None = None


def infer_dataset_t(log_path: Path) -> tuple[str, str]:
    stem = log_path.name
    stem = stem.removesuffix(".log")
    m = re.match(r"(?P<dataset>.+)_T(?P<t>[-0-9.]+)_runs[0-9]+$", stem)
    if m:
        return m.group("dataset"), m.group("t")
    return stem, ""


def parse_percent(text: str) -> float:
    text = text.strip()
    if not text or text == "-":
        return math.nan
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def parse_float(text: str) -> float:
    text = text.strip()
    if not text or text == "-":
        return math.nan
    return float(text)


def parse_reuse_fraction(text: str) -> tuple[float, float]:
    text = text.strip()
    m = re.match(r"([0-9.]+)/([0-9.]+)", text)
    if not m:
        return math.nan, math.nan
    return float(m.group(1)), float(m.group(2))


def mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def parse_log(log_path: Path, default_gate: str = "unknown") -> list[ParsedConfig]:
    dataset, threshold = infer_dataset_t(log_path)
    text = log_path.read_text(errors="replace")
    runs_match = re.search(r"FINAL RESIDUAL REUSE SUMMARY \((\d+) Runs\)", text)
    runs = int(runs_match.group(1)) if runs_match else None

    gate_matches = re.findall(r"\[RiskScore\]\s+reuse_gate=(enabled|disabled)", text)
    gate = gate_matches[-1] if gate_matches else default_gate

    baseline_matches = re.findall(r"Baseline Acc:\s*([0-9.]+)", text)
    baseline = float(baseline_matches[-1]) if baseline_matches else math.nan

    corrected_values = [
        float(m.group(1))
        for m in re.finditer(r"\[ResidualReuse\].*?\|\s*Corrected=([0-9]+)\s*\|", text)
    ]
    corrected_mean = mean(corrected_values)

    rows: list[ParsedConfig] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 9 or parts[0] not in CONFIGS:
            continue
        reuse_num, reuse_den = parse_reuse_fraction(parts[8])
        rows.append(
            ParsedConfig(
                dataset=dataset,
                threshold=threshold,
                runs=runs,
                gate=gate,
                baseline_acc=baseline,
                config=parts[0],
                reuse=parse_percent(parts[1]),
                acc=parse_float(parts[3]),
                drop=parse_percent(parts[4]),
                avg_err=parse_float(parts[5]),
                hit_err=parse_float(parts[6]),
                reuse_num=reuse_num,
                reuse_den=reuse_den,
                log=str(log_path),
                corrected_mean=corrected_mean if parts[0] == "ResidualReuse" else None,
            )
        )
    return rows


def choose_threshold(rows: list[ParsedConfig], max_drop: float) -> dict[str, str]:
    """Pick the highest-reuse residual point within drop budget per dataset."""
    by_dataset: dict[str, list[ParsedConfig]] = {}
    for row in rows:
        if row.gate == "enabled" and row.config == "ResidualReuse":
            by_dataset.setdefault(row.dataset, []).append(row)
    selected: dict[str, str] = {}
    for dataset, candidates in by_dataset.items():
        feasible = [r for r in candidates if r.drop <= max_drop]
        pool = feasible if feasible else candidates
        if not pool:
            continue
        best = sorted(pool, key=lambda r: (-r.reuse, r.drop))[0]
        selected[dataset] = best.threshold
    return selected


def get_row(rows: list[ParsedConfig], dataset: str, threshold: str, gate: str, config: str) -> ParsedConfig | None:
    candidates = [
        r
        for r in rows
        if r.dataset == dataset and r.threshold == threshold and r.gate == gate and r.config == config
    ]
    return candidates[0] if candidates else None


def get_any_row(rows: list[ParsedConfig], dataset: str, gate: str, config: str) -> ParsedConfig | None:
    candidates = [r for r in rows if r.dataset == dataset and r.gate == gate and r.config == config]
    return sorted(candidates, key=lambda r: r.threshold)[0] if candidates else None


def pct(value: float) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:.2f}%"


def num(value: float) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.5f}"


def acc(value: float) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.4f}"


def split_for(row: ParsedConfig, action: str) -> tuple[float, float, float]:
    if action == "no_reuse":
        return 0.0, 0.0, 1.0
    if action in {"direct", "soft", "tser"}:
        return row.reuse, 0.0, max(0.0, 1.0 - row.reuse)
    if action == "residual":
        if row.corrected_mean is None or math.isnan(row.corrected_mean) or math.isnan(row.reuse_den):
            return math.nan, math.nan, math.nan
        residual = row.corrected_mean / max(1.0, row.reuse_den)
        direct = max(0.0, row.reuse - residual)
        return direct, residual, max(0.0, 1.0 - row.reuse)
    raise ValueError(action)


def source_name(row: ParsedConfig) -> str:
    prefix = "no_tser" if row.gate == "disabled" else "tser"
    return f"{prefix}/{Path(row.log).name}"


def table_rows(rows: list[ParsedConfig], max_drop: float) -> list[dict[str, object]]:
    thresholds = choose_threshold(rows, max_drop)
    out: list[dict[str, object]] = []
    for dataset in sorted(thresholds, key=lambda d: DATASET_LABELS.get(d, d)):
        threshold = thresholds[dataset]
        enabled_direct = get_row(rows, dataset, threshold, "enabled", "DirectReuse")
        enabled_soft = get_row(rows, dataset, threshold, "enabled", "SoftDirectReuse")
        enabled_res = get_row(rows, dataset, threshold, "enabled", "ResidualReuse")
        disabled_soft = get_row(rows, dataset, threshold, "disabled", "SoftDirectReuse")
        if disabled_soft is None:
            disabled_soft = get_any_row(rows, dataset, "disabled", "SoftDirectReuse")
        if enabled_res is None:
            continue
        base = enabled_res.baseline_acc
        out.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "threshold": threshold,
                "method": "No reuse",
                "definition": "Full encoder reference",
                "reuse": 0.0,
                "direct": 0.0,
                "residual": 0.0,
                "compute": 1.0,
                "acc": base,
                "drop": 0.0,
                "avg_err": 0.0,
                "hit_err": 0.0,
                "source": source_name(enabled_res),
            }
        )
        if enabled_direct is not None:
            direct, residual, compute = split_for(enabled_direct, "direct")
            out.append(
                {
                    "dataset": dataset,
                    "label": DATASET_LABELS.get(dataset, dataset),
                    "threshold": threshold,
                    "method": "Direct only",
                    "definition": "high-support anchors only",
                    "reuse": enabled_direct.reuse,
                    "direct": direct,
                    "residual": residual,
                    "compute": compute,
                    "acc": enabled_direct.acc,
                    "drop": enabled_direct.drop,
                    "avg_err": enabled_direct.avg_err,
                    "hit_err": enabled_direct.hit_err,
                    "source": source_name(enabled_direct),
                }
            )
        if disabled_soft is not None:
            direct, residual, compute = split_for(disabled_soft, "soft")
            out.append(
                {
                    "dataset": dataset,
                    "label": DATASET_LABELS.get(dataset, dataset),
                    "threshold": threshold,
                    "method": "Soft direct reuse",
                    "definition": "support-only fuzzy anchors, no TSER/residual",
                    "reuse": disabled_soft.reuse,
                    "direct": direct,
                    "residual": residual,
                    "compute": compute,
                    "acc": disabled_soft.acc,
                    "drop": disabled_soft.drop,
                    "avg_err": disabled_soft.avg_err,
                    "hit_err": disabled_soft.hit_err,
                    "source": source_name(disabled_soft),
                }
            )
        else:
            out.append(
                {
                    "dataset": dataset,
                    "label": DATASET_LABELS.get(dataset, dataset),
                    "threshold": threshold,
                    "method": "Soft direct reuse",
                    "definition": "support-only fuzzy anchors, no TSER/residual",
                    "reuse": math.nan,
                    "direct": math.nan,
                    "residual": math.nan,
                    "compute": math.nan,
                    "acc": math.nan,
                    "drop": math.nan,
                    "avg_err": math.nan,
                    "hit_err": math.nan,
                    "source": "pending no-TSER run",
                }
            )
        if enabled_soft is not None:
            direct, residual, compute = split_for(enabled_soft, "tser")
            out.append(
                {
                    "dataset": dataset,
                    "label": DATASET_LABELS.get(dataset, dataset),
                    "threshold": threshold,
                    "method": "TSER filtering",
                    "definition": "TSER-accepted anchors, no residual repair",
                    "reuse": enabled_soft.reuse,
                    "direct": direct,
                    "residual": residual,
                    "compute": compute,
                    "acc": enabled_soft.acc,
                    "drop": enabled_soft.drop,
                    "avg_err": enabled_soft.avg_err,
                    "hit_err": enabled_soft.hit_err,
                    "source": source_name(enabled_soft),
                }
            )
        direct, residual, compute = split_for(enabled_res, "residual")
        out.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "threshold": threshold,
                "method": "TSER + residual repair",
                "definition": "TSER fuzzy anchors repaired or rejected",
                "reuse": enabled_res.reuse,
                "direct": direct,
                "residual": residual,
                "compute": compute,
                "acc": enabled_res.acc,
                "drop": enabled_res.drop,
                "avg_err": enabled_res.avg_err,
                "hit_err": enabled_res.hit_err,
                "source": source_name(enabled_res),
            }
        )
    return out


def write_tsv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "dataset",
        "threshold",
        "method",
        "reuse",
        "direct",
        "residual",
        "compute",
        "acc",
        "drop",
        "avg_err",
        "hit_err",
        "source",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    str(row.get(field, ""))
                    if not isinstance(row.get(field, ""), float)
                    else f"{row[field]:.8g}"
                    for field in fields
                )
                + "\n"
            )


def write_markdown(rows: list[dict[str, object]], path: Path, max_drop: float) -> None:
    lines: list[str] = []
    lines.append("# Reuse Safety Ablation")
    lines.append("")
    lines.append(
        "This table isolates the frontend reuse-safety path. The selected threshold per dataset is "
        f"the highest-reuse `TSER + residual repair` point under a `{max_drop * 100:.1f}%` drop budget when available."
    )
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("- `No reuse`: all nodes use the reference encoder target pool.")
    lines.append("- `Direct only`: only high-support SimHash/CAM anchors are reused.")
    lines.append("- `Soft direct reuse`: support-only fuzzy reuse without TSER or residual repair; shown as pending if the no-TSER run has not been produced.")
    lines.append("- `TSER filtering`: TSER-accepted anchors are directly reused without residual repair.")
    lines.append("- `TSER + residual repair`: medium-support TSER candidates are corrected by the residual adapter or rejected to compute.")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append(
        "| Dataset | T | Method | Reuse | Direct | Residual | Compute | Acc | Drop | AvgErr | HitErr | Source |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["threshold"]),
                    str(row["method"]),
                    pct(float(row["reuse"])),
                    pct(float(row["direct"])),
                    pct(float(row["residual"])),
                    pct(float(row["compute"])),
                    acc(float(row["acc"])),
                    pct(float(row["drop"])),
                    num(float(row["avg_err"])),
                    num(float(row["hit_err"])),
                    f"`{row['source']}`",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "The expected trend is that candidate discovery alone can expose many anchors but may cause high error when fuzzy hits are accepted blindly. "
        "`TSER filtering` lowers unsafe reuse by using graph risk, and `TSER + residual repair` recovers part of the fuzzy bucket while keeping embedding error and accuracy drop controlled."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_log_dir", required=True, help="Logs produced with --enable_score_gate.")
    parser.add_argument("--no_score_log_dir", default=None, help="Optional logs produced with --disable_score_gate.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_drop", type=float, default=0.02)
    args = parser.parse_args()

    all_rows: list[ParsedConfig] = []
    log_dirs = [(Path(args.score_log_dir), "enabled")]
    if args.no_score_log_dir:
        log_dirs.append((Path(args.no_score_log_dir), "disabled"))
    for log_dir, default_gate in log_dirs:
        if not log_dir.exists():
            continue
        for log_path in sorted(log_dir.glob("*.log")):
            all_rows.extend(parse_log(log_path, default_gate=default_gate))

    rows = table_rows(all_rows, args.max_drop)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(rows, output_dir / "reuse_safety_ablation.tsv")
    write_markdown(rows, output_dir / "reuse_safety_ablation.md", args.max_drop)
    print(f"[Saved] {output_dir / 'reuse_safety_ablation.tsv'}")
    print(f"[Saved] {output_dir / 'reuse_safety_ablation.md'}")


if __name__ == "__main__":
    main()
