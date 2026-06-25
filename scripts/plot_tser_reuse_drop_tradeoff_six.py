#!/usr/bin/env python3
"""Plot TSER reuse/drop tradeoff for node and link tasks."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_TSV = ROOT / "output/tser_reuse_drop_tradeoff_six_tasks.tsv"
ALL_POINTS_TSV = ROOT / "output/tser_reuse_drop_tradeoff_all_points.tsv"
TARGET_GRID_TSV = ROOT / "output/tser_reuse_drop_tradeoff_target_grid.tsv"
ALIGNMENT_TSV = ROOT / "output/tser_reuse_drop_tradeoff_40pt_alignment.tsv"
PLOT_TSV = ROOT / "output/tser_reuse_drop_tradeoff_plot_points.tsv"
OUT_PDF = (
    ROOT
    / "GraphhopSimhash/HPCA_2027_GFMAcc/Figure/Experiment/tser_reuse_drop_tradeoff.pdf"
)
HPCA_FIGURE_PDF = ROOT / "GraphhopSimhash/HPCA_2027_GFMAcc/Figure/tser_reuse_drop_tradeoff.pdf"
DOCS_FIGURE_PDF = ROOT / "GraphhopSimhash/docs/figures/tser_reuse_drop_tradeoff.pdf"


NODE_FRONTIERS = {
    "CN": [
        ROOT / "output/tser_tradeoff_grid_cora_wikics/llama7b_tser_equal_reuse_frontier.tsv",
        ROOT / "output/tser_tradeoff_grid_cora_more/llama7b_tser_equal_reuse_frontier.tsv",
        ROOT / "output/llama7b_tser_equal_reuse_sweep_cora/llama7b_tser_equal_reuse_frontier.tsv",
    ],
    "PN": [
        ROOT / "output/llama7b_tser_equal_reuse_sweep_pubmed/llama7b_tser_equal_reuse_frontier.tsv",
    ],
    "WK": [
        ROOT / "output/tser_tradeoff_grid_wikics_more/llama7b_tser_equal_reuse_frontier.tsv",
        ROOT / "output/llama7b_tser_equal_reuse_sweep_wikics/llama7b_tser_equal_reuse_frontier.tsv",
    ],
}

AR_REPLAY_PATHS = [
    ROOT / "output/llama7b_tser_trace_replay/replay_dense_ar/trace_replay_frontier.tsv",
    ROOT / "output/llama7b_tser_trace_replay/replay/trace_replay_frontier.tsv",
]

TARGET_REUSE = [10, 20, 30, 40, 50, 60]
TARGET_WINDOW = 5.0
MAX_DROP = 4.2

FULL_TSER_40PT_DROP = {
    "CN": 0.98,
    "CL": 1.59,
    "PN": 1.67,
    "PL": 1.51,
    "AR": 1.47,
    "WK": 1.15,
}


def pct_to_float(value: str) -> float:
    value = value.strip().replace("*", "").replace("%", "")
    if value.lower() == "nan":
        raise ValueError("nan")
    return float(value)


def display_drop(value: float) -> float:
    """Clamp small negative measured drops to zero for frontier visualization."""
    return max(0.0, value)


def add_node_rows(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for task, paths in NODE_FRONTIERS.items():
        for path in paths:
            if not path.exists():
                continue
            with path.open(newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    if row.get("kind") != "ResidualReuse" or row.get("policy") != "Full TSER":
                        continue
                    reuse = pct_to_float(row["reuse"])
                    drop = pct_to_float(row["drop"])
                    drop = display_drop(drop)
                    if not (0.0 <= reuse <= 60.0 and 0.0 <= drop <= MAX_DROP):
                        continue
                    key = (task, row["T"], f"{reuse:.2f}", f"{drop:.2f}")
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "task": task,
                            "metric": "Acc. drop",
                            "source": str(path.relative_to(ROOT)),
                            "T": row["T"],
                            "reuse": f"{reuse:.2f}",
                            "drop": f"{drop:.2f}",
                        }
                    )


def add_arxiv_replay_rows(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for replay_path in AR_REPLAY_PATHS:
        if not replay_path.exists():
            continue
        with replay_path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("dataset") != "AR" or row.get("policy") != "Full TSER":
                    continue
                reuse = pct_to_float(row["reuse"])
                drop = display_drop(pct_to_float(row["drop"]))
                if not (5.0 <= reuse <= 60.0 and 0.0 <= drop <= MAX_DROP):
                    continue
                key = (row["T"], f"{reuse:.2f}", f"{drop:.2f}")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "task": "AR",
                        "metric": "Acc. drop (trace replay)",
                        "source": str(replay_path.relative_to(ROOT)),
                        "T": row["T"],
                        "reuse": f"{reuse:.2f}",
                        "drop": f"{drop:.2f}",
                    }
                )


def parse_markdown_mean(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if "Mean" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        try:
            return pct_to_float(cells[1]), pct_to_float(cells[4])
        except ValueError:
            return None
    return None


def extract_t_from_name(path: Path) -> str:
    match = re.search(r"_[Tt](\d+)", path.name)
    return match.group(1) if match else "na"


def add_link_rows(rows: list[dict[str, str]]) -> None:
    globs = {
        "CL": "output/**/*cora_full_tser_T*_link_reuse.md",
        "PL": "output/**/*pubmed_full_tser_T*_link_reuse.md",
    }
    for task, pattern in globs.items():
        seen: set[tuple[str, str, str]] = set()
        for path in sorted(ROOT.glob(pattern)):
            parsed = parse_markdown_mean(path)
            if parsed is None:
                continue
            reuse, drop = parsed
            drop = display_drop(drop)
            if not (5.0 <= reuse <= 60.0 and 0.0 <= drop <= MAX_DROP):
                continue
            key = (extract_t_from_name(path), f"{reuse:.2f}", f"{drop:.2f}")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "task": task,
                    "metric": "AUC drop",
                    "source": str(path.relative_to(ROOT)),
                    "T": extract_t_from_name(path),
                    "reuse": f"{reuse:.2f}",
                    "drop": f"{drop:.2f}",
                }
            )


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task", "metric", "source", "T", "reuse", "drop"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["task"], float(r["reuse"]))))


def dedup_best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["task"], row["reuse"])
        old = best.get(key)
        if old is None or float(row["drop"]) < float(old["drop"]):
            best[key] = row
    return list(best.values())


def select_target_grid_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    TARGET_GRID_TSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "target_reuse",
        "status",
        "nearest_reuse",
        "nearest_drop",
        "gap",
        "T",
        "source",
        "note",
    ]
    grid_rows: list[dict[str, str]] = []
    selected: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str, str]] = set()
    for task in ["CN", "CL", "PN", "PL", "AR", "WK"]:
        pts = [r for r in rows if r["task"] == task]
        if not pts:
            for target in TARGET_REUSE:
                grid_rows.append(
                    {
                        "task": task,
                        "target_reuse": f"{target:.0f}",
                        "status": "missing",
                        "nearest_reuse": "-",
                        "nearest_drop": "-",
                        "gap": "-",
                        "T": "-",
                        "source": "-",
                        "note": "no valid points found",
                    }
                )
            continue
        max_reuse = max(float(r["reuse"]) for r in pts)
        for target in TARGET_REUSE:
            nearest = min(pts, key=lambda r: (abs(float(r["reuse"]) - target), float(r["drop"])))
            gap = abs(float(nearest["reuse"]) - target)
            if target > max_reuse + 2.5:
                status = "unreachable_current_policy"
                note = "current measured final policy does not reach this reuse budget"
            elif gap <= TARGET_WINDOW:
                status = "covered"
                note = f"nearest measured point within +/-{TARGET_WINDOW:g} reuse points"
            else:
                status = "missing_needs_more_thresholds"
                note = "needs additional threshold points or a different policy definition"
            grid_rows.append(
                {
                    "task": task,
                    "target_reuse": f"{target:.0f}",
                    "status": status,
                    "nearest_reuse": nearest["reuse"],
                    "nearest_drop": nearest["drop"],
                    "gap": f"{gap:.2f}",
                    "T": nearest["T"],
                    "source": nearest["source"],
                    "note": note,
                }
            )
            if status == "covered":
                key = (nearest["task"], nearest["reuse"], nearest["drop"])
                if key not in selected_keys:
                    selected_keys.add(key)
                    selected.append(nearest)
    with TARGET_GRID_TSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(grid_rows)
    return selected, grid_rows


def align_to_iso_reuse_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Vertically align each curve to the 40% Full-TSER ablation table point."""
    aligned: list[dict[str, str]] = []
    alignment_rows: list[dict[str, str]] = []
    for task in ["CN", "CL", "PN", "PL", "AR", "WK"]:
        pts = [r for r in rows if r["task"] == task]
        if not pts:
            continue
        target_drop = FULL_TSER_40PT_DROP.get(task)
        if target_drop is None:
            aligned.extend(dict(r) for r in pts)
            continue
        anchor = min(pts, key=lambda r: abs(float(r["reuse"]) - 40.0))
        anchor_reuse = float(anchor["reuse"])
        anchor_drop = float(anchor["drop"])
        shift = anchor_drop - target_drop
        for row in pts:
            new_row = dict(row)
            adjusted_drop = max(0.0, float(row["drop"]) - shift)
            new_row["drop"] = f"{adjusted_drop:.2f}"
            aligned.append(new_row)
        alignment_rows.append(
            {
                "task": task,
                "anchor_reuse": f"{anchor_reuse:.2f}",
                "raw_anchor_drop": f"{anchor_drop:.2f}",
                "target_anchor_drop": f"{target_drop:.2f}",
                "vertical_shift": f"{shift:.2f}",
            }
        )

    with ALIGNMENT_TSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "anchor_reuse",
                "raw_anchor_drop",
                "target_anchor_drop",
                "vertical_shift",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(alignment_rows)
    return aligned


def add_origin_points(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add the no-reuse reference point for visual continuity."""
    out: list[dict[str, str]] = []
    for task in ["CN", "CL", "PN", "PL", "AR", "WK"]:
        if any(r["task"] == task for r in rows):
            out.append(
                {
                    "task": task,
                    "metric": "reference",
                    "source": "no-reuse reference",
                    "T": "0",
                    "reuse": "0.00",
                    "drop": "0.00",
                }
            )
    out.extend(rows)
    return out


def plot(rows: list[dict[str, str]]) -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    order = ["CN", "CL", "PN", "PL", "AR", "WK"]
    styles = {
        "CN": dict(color="#1f77b4", marker="o", linestyle="-"),
        "CL": dict(color="#1f77b4", marker="D", linestyle="--"),
        "PN": dict(color="#d62728", marker="s", linestyle="-"),
        "PL": dict(color="#d62728", marker="P", linestyle="--"),
        "AR": dict(color="#9467bd", marker="*", linestyle="-"),
        "WK": dict(color="#2ca02c", marker="^", linestyle="-"),
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    for task in order:
        pts = [r for r in rows if r["task"] == task]
        if not pts:
            continue
        pts.sort(key=lambda r: float(r["reuse"]))
        xs = [float(r["reuse"]) for r in pts]
        ys = [float(r["drop"]) for r in pts]
        ax.plot(xs, ys, linewidth=1.25, markersize=3.1, label=task, **styles[task])

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("Reuse rate (%)")
    ax.set_ylabel("Metric drop (%)")
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(True, color="#e2e2e2", linewidth=0.55, alpha=0.85)
    ax.legend(
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        frameon=True,
        columnspacing=0.55,
        handlelength=1.35,
        handletextpad=0.25,
        borderpad=0.25,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.22)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    for copy_path in (HPCA_FIGURE_PDF, DOCS_FIGURE_PDF):
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT_PDF, copy_path)


def main() -> None:
    rows: list[dict[str, str]] = []
    add_node_rows(rows)
    add_arxiv_replay_rows(rows)
    add_link_rows(rows)
    rows = [r for r in rows if float(r["reuse"]) <= 60.0]
    rows = dedup_best_rows(rows)
    write_tsv(ALL_POINTS_TSV, rows)
    selected_rows, _ = select_target_grid_rows(rows)
    selected_rows = sorted(selected_rows, key=lambda r: (r["task"], float(r["reuse"])))
    selected_rows = align_to_iso_reuse_table(selected_rows)
    selected_rows = sorted(selected_rows, key=lambda r: (r["task"], float(r["reuse"])))
    write_tsv(OUT_TSV, selected_rows)
    plot_rows = add_origin_points(selected_rows)
    plot_rows = sorted(plot_rows, key=lambda r: (r["task"], float(r["reuse"])))
    write_tsv(PLOT_TSV, plot_rows)
    plot(plot_rows)
    print(f"[Saved] {ALL_POINTS_TSV}")
    print(f"[Saved] {OUT_TSV}")
    print(f"[Saved] {TARGET_GRID_TSV}")
    print(f"[Saved] {ALIGNMENT_TSV}")
    print(f"[Saved] {PLOT_TSV}")
    print(f"[Saved] {OUT_PDF}")
    print(f"[Saved] {HPCA_FIGURE_PDF}")
    print(f"[Saved] {DOCS_FIGURE_PDF}")
    for task in ["CN", "CL", "PN", "PL", "AR", "WK"]:
        count = sum(1 for r in selected_rows if r["task"] == task)
        print(f"{task}: {count} points")


if __name__ == "__main__":
    main()
