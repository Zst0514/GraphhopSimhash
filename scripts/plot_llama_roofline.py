#!/usr/bin/env python3
"""Plot LLaMA ONNXim time breakdown and roofline figure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing TSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def m_label(source: str, m: float | int) -> str:
    return f"M{int(m)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("output/onnxim_graphbit/llama_roofline_p8_m16_m128"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/llama_roofline_profile.png"),
    )
    parser.add_argument("--peak-tflops", type=float, default=131.072)
    parser.add_argument("--peak-gbps", type=float, default=614.4)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    components = read_tsv(args.profile_dir / "llama_roofline_components.tsv")
    layers = read_tsv(args.profile_dir / "llama_roofline_layers.tsv")

    sources = [row["source"] for row in layers]
    labels = [m_label(row["source"], f(row, "m")) for row in layers]
    parts = ["proj", "ffn_up", "ffn_down"]
    part_labels = {"proj": "QKV/O Proj", "ffn_up": "FFN Gate/Up", "ffn_down": "FFN Down"}
    part_colors = {"proj": "#4C78A8", "ffn_up": "#9B59B6", "ffn_down": "#43A2CA"}
    markers = {"proj": "o", "ffn_up": "^", "ffn_down": "s", "layer_total": "D"}

    comp_by_source = {source: [] for source in sources}
    for row in components:
        comp_by_source.setdefault(row["source"], []).append(row)
    total_cycles = {row["source"]: f(row, "cycles") for row in layers}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=180)
    ax0, ax1 = axes

    x = np.arange(len(sources))
    bottom = np.zeros(len(sources))
    for part in parts:
        values = []
        for source in sources:
            rows = [row for row in comp_by_source[source] if row["name"] == part]
            if rows:
                values.append(100.0 * f(rows[0], "weighted_cycles") / total_cycles[source])
            else:
                values.append(0.0)
        ax0.bar(
            x,
            values,
            bottom=bottom,
            width=0.66,
            label=part_labels[part],
            color=part_colors[part],
            edgecolor="black",
            linewidth=0.5,
            hatch="////" if part == "ffn_up" else None,
            alpha=0.88,
        )
        bottom += np.array(values)

    ax0.set_ylim(0, 100)
    ax0.set_xticks(x, labels)
    ax0.set_ylabel("Layer Time Percentage")
    ax0.set_xlabel("LLaMA GEMM M dimension")
    ax0.set_title("(a) GEMM Time Breakdown")
    ax0.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=True, fontsize=8)

    oi_min = min(f(row, "oi_flop_per_byte") for row in components + layers) * 0.65
    oi_max = max(args.peak_tflops * 1000.0 / args.peak_gbps * 1.8, max(f(row, "oi_flop_per_byte") for row in components + layers) * 1.6)
    xs = np.logspace(math.log10(max(1.0, oi_min)), math.log10(oi_max), 300)
    mem_roof = xs * args.peak_gbps / 1000.0
    roof = np.minimum(mem_roof, args.peak_tflops)
    ridge = args.peak_tflops * 1000.0 / args.peak_gbps

    ax1.plot(xs, roof, color="#555555", linestyle="--", linewidth=1.5, label="Roofline")
    ax1.axvline(ridge, color="#D62728", linestyle="--", linewidth=1.2)
    ax1.axhline(args.peak_tflops, color="#D62728", linestyle="--", linewidth=1.0)
    ax1.text(ridge * 0.72, args.peak_tflops * 0.72, "Memory Bound", color="#B00000", ha="right", va="top", fontsize=9, weight="bold")
    ax1.text(ridge * 1.08, args.peak_tflops * 0.72, "Compute Bound", color="#B00000", ha="left", va="top", fontsize=9, weight="bold")

    for part in parts:
        rows = [row for row in components if row["name"] == part]
        ax1.scatter(
            [f(row, "oi_flop_per_byte") for row in rows],
            [f(row, "achieved_gflops") / 1000.0 for row in rows],
            marker=markers[part],
            s=54,
            color=part_colors[part],
            edgecolor="black",
            linewidth=0.5,
            label=part_labels[part],
            alpha=0.92,
        )

    ax1.scatter(
        [f(row, "oi_flop_per_byte") for row in layers],
        [f(row, "achieved_gflops") / 1000.0 for row in layers],
        marker=markers["layer_total"],
        s=52,
        color="#222222",
        edgecolor="white",
        linewidth=0.5,
        label="Layer Total",
        alpha=0.9,
    )
    for row in layers:
        ax1.annotate(
            m_label(row["source"], f(row, "m")),
            (f(row, "oi_flop_per_byte"), f(row, "achieved_gflops") / 1000.0),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color="#222222",
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax1.set_ylabel("Performance (TFLOP/s)")
    ax1.set_title("(b) LLaMA GEMM Roofline")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.45, alpha=0.35)
    ax1.set_xlim(max(1.0, oi_min), oi_max)
    ax1.set_ylim(1.0, args.peak_tflops * 1.5)
    ax1.legend(loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    print(f"[LlamaRooflinePlot] wrote {args.output}")
    print(f"[LlamaRooflinePlot] wrote {args.output.with_suffix('.pdf')}")
    print(f"[LlamaRooflinePlot] wrote {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
