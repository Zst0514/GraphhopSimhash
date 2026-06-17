#!/usr/bin/env python3
"""Profile reuse-filter frontend vs encoder-backend workload character.

This script builds a compact analytical profile for Motivation.  It does not
claim cycle accuracy.  The goal is to separate the two execution characters:

* graph retrieval frontend: metadata-access dominated, low arithmetic intensity;
* LLaMA encoder backend: dense GEMM dominated, high arithmetic intensity.

The output includes a TSV table, a short Markdown summary, and a vector PDF.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class WorkloadPoint:
    name: str
    kind: str
    ops: float
    bytes_moved: float
    intensity: float
    roofline_perf: float
    note: str


def ratio(num: float, den: float) -> float:
    return num / den if den else math.inf


def reuse_frontend_point(
    heads: int,
    bits_per_head: int,
    candidate_result_bytes: int,
    cache_metadata_bytes: int,
    graph_metadata_bytes: int,
    reuse_queue_bytes: int,
    score_ops: int,
    ridge_intensity: float,
) -> WorkloadPoint:
    signature_bits = heads * bits_per_head
    signature_bytes = math.ceil(signature_bits / 8)
    # These are logical operations.  In a CAM implementation, the bit matching
    # happens inside the array; the count is used only to characterize intensity.
    bit_match_ops = signature_bits
    metadata_bytes = (
        signature_bytes
        + candidate_result_bytes
        + cache_metadata_bytes
        + graph_metadata_bytes
        + reuse_queue_bytes
    )
    ops = bit_match_ops + score_ops
    return WorkloadPoint(
        name="Reuse filter",
        kind="retrieval",
        ops=float(ops),
        bytes_moved=float(metadata_bytes),
        intensity=ratio(float(ops), float(metadata_bytes)),
        roofline_perf=min(1.0, ratio(ratio(float(ops), float(metadata_bytes)), ridge_intensity)),
        note=(
            f"{heads}x{bits_per_head}-bit signature, metadata includes "
            "candidate/cache/graph/queue fields"
        ),
    )


def gemm_point(
    name: str,
    m: int,
    k: int,
    n: int,
    activation_bytes: float,
    weight_bytes: float,
    output_bytes: float,
    ridge_intensity: float,
) -> WorkloadPoint:
    ops = 2.0 * m * k * n
    bytes_moved = m * k * activation_bytes + k * n * weight_bytes + m * n * output_bytes
    intensity = ratio(ops, bytes_moved)
    return WorkloadPoint(
        name=name,
        kind="dense_gemm",
        ops=ops,
        bytes_moved=bytes_moved,
        intensity=intensity,
        roofline_perf=min(1.0, ratio(intensity, ridge_intensity)),
        note=f"M={m}, K={k}, N={n}",
    )


def write_tsv(path: Path, rows: list[WorkloadPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "kind", "ops", "bytes_moved", "intensity", "roofline_perf", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[WorkloadPoint], args: argparse.Namespace) -> None:
    reuse = next(r for r in rows if r.name == "Reuse filter")
    gemms = [r for r in rows if r.kind == "dense_gemm"]
    min_gemm = min(gemms, key=lambda r: r.intensity)
    max_gemm = max(gemms, key=lambda r: r.intensity)
    avg_reuse = sum(args.reuse_rates) / len(args.reuse_rates)
    avg_miss = 1.0 - avg_reuse

    lines = [
        "# Reuse/Encoder Workload Character Profile",
        "",
        "This profile separates the graph retrieval frontend from the dense LLaMA encoder backend.",
        "It is an analytical workload-character estimate, not a cycle-accurate simulation.",
        "",
        "## Assumptions",
        "",
        f"- SimHash heads: `{args.heads}`",
        f"- Bits per head: `{args.bits_per_head}`",
        f"- Token-row batch M: `{args.node_batch} x {args.seq_len} = {args.node_batch * args.seq_len}`",
        f"- LLaMA hidden/intermediate/layers: `{args.hidden}` / `{args.intermediate}` / `{args.layers}`",
        f"- Activation / weight / output bytes: `{args.activation_bytes}` / `{args.weight_bytes}` / `{args.output_bytes}`",
        f"- Normalized roofline ridge intensity: `{args.ridge_intensity}` ops/byte",
        f"- Average bypass rate used for split context: `{avg_reuse * 100:.1f}%`",
        f"- Average miss-node rate after compaction: `{avg_miss * 100:.1f}%`",
        "",
        "## Key Result",
        "",
        (
            f"- Reuse filter intensity: `{reuse.intensity:.2f}` logical ops/byte. "
            f"Under the normalized roofline, its attainable performance is `{reuse.roofline_perf:.3f}` "
            "of peak, so it is metadata-access dominated."
        ),
        (
            f"- Dense GEMM intensity range: `{min_gemm.intensity:.0f}`--`{max_gemm.intensity:.0f}` "
            "MAC-equivalent ops/byte. These points sit on the compute roof."
        ),
        (
            f"- The lowest GEMM intensity is about `{min_gemm.intensity / reuse.intensity:.0f}x` "
            "higher than reuse filtering."
        ),
        "",
        "## TSV",
        "",
        f"- `{path.parent / 'workload_character.tsv'}`",
        "",
        "## Figure",
        "",
        f"- `{args.figure}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(path: Path, rows: list[WorkloadPoint], ridge_intensity: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.15, 1.78))
    xs = np.logspace(-1, 4.0, 300)
    ys = np.minimum(1.0, xs / ridge_intensity)
    ax.plot(xs, ys, color="black", linewidth=1.0)
    bound_color = "#8c2d2d"
    ax.axvline(ridge_intensity, color=bound_color, linestyle="--", linewidth=0.75)
    ax.annotate("", xy=(0.13, 1.07), xytext=(ridge_intensity * 0.9, 1.07),
                arrowprops={"arrowstyle": "<->", "color": bound_color, "lw": 0.65, "ls": "--"})
    ax.annotate("", xy=(ridge_intensity * 1.1, 1.07), xytext=(8500, 1.07),
                arrowprops={"arrowstyle": "<->", "color": bound_color, "lw": 0.65, "ls": "--"})
    mem_label_x = math.sqrt(0.13 * ridge_intensity * 0.9)
    comp_label_x = math.sqrt(ridge_intensity * 1.1 * 8500)
    ax.text(mem_label_x, 0.62, "Memory bound", fontsize=6.8, color=bound_color,
            weight="bold", ha="center")
    ax.text(comp_label_x, 0.62, "Compute bound", fontsize=6.8, color=bound_color,
            weight="bold", ha="center")

    reuse_row = next(row for row in rows if row.kind == "retrieval")
    self_attn_row = next(row for row in rows if "Self-Attn" in row.name)
    ffn_rows = [row for row in rows if row.kind == "dense_gemm" and "FFN" in row.name]
    ffn_intensity = math.exp(sum(math.log(row.intensity) for row in ffn_rows) / len(ffn_rows))
    ffn_perf = min(1.0, ffn_intensity / ridge_intensity)

    ax.scatter(reuse_row.intensity, reuse_row.roofline_perf, s=28, marker="D",
               color="#4c5f8f", edgecolor="black", linewidth=0.35, zorder=3,
               label="Reuse filter")

    ax.scatter(self_attn_row.intensity, self_attn_row.roofline_perf, s=28, marker="o",
               color="#17becf", edgecolor="black", linewidth=0.35, zorder=3,
               label="Self-Attn")
    ax.scatter(ffn_intensity, ffn_perf, s=34, marker="^",
               color="#9467bd", edgecolor="black", linewidth=0.35, zorder=3,
               label="FFN")

    ax.legend(loc="lower right", fontsize=5.8,
              frameon=True, framealpha=0.9, borderpad=0.25, handlelength=1.0,
              handletextpad=0.35, labelspacing=0.25)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (ops/byte)", fontsize=7)
    ax.set_ylabel("Normalized performance", fontsize=7)
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.01, 1.35)
    ax.tick_params(axis="both", which="major", labelsize=7, width=0.8, length=3)
    ax.tick_params(axis="both", which="minor", width=0.6, length=2)
    ax.grid(which="both", alpha=0.22, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/reuse_encoder_workload_profile"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("HPCA_2027_GFMAcc/Figure/route_encoder_workload_character.pdf"),
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bits-per-head", type=int, default=16)
    parser.add_argument("--candidate-result-bytes", type=int, default=32)
    parser.add_argument("--cache-metadata-bytes", type=int, default=64)
    parser.add_argument("--graph-metadata-bytes", type=int, default=32)
    parser.add_argument("--reuse-queue-bytes", type=int, default=16)
    parser.add_argument("--score-ops", type=int, default=64)
    parser.add_argument("--node-batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--activation-bytes", type=float, default=0.5)
    parser.add_argument("--weight-bytes", type=float, default=0.5)
    parser.add_argument("--output-bytes", type=float, default=2.0)
    parser.add_argument(
        "--ridge-intensity",
        type=float,
        default=64.0,
        help="Ops/byte at which the normalized roofline reaches peak compute.",
    )
    parser.add_argument(
        "--reuse-rates",
        type=float,
        nargs="+",
        default=[0.391, 0.299, 0.300, 0.369],
        help="Bypass/reuse rates used only to summarize reuse split context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    m = args.node_batch * args.seq_len
    h = args.hidden
    i = args.intermediate

    rows = [
        reuse_frontend_point(
            heads=args.heads,
            bits_per_head=args.bits_per_head,
            candidate_result_bytes=args.candidate_result_bytes,
            cache_metadata_bytes=args.cache_metadata_bytes,
            graph_metadata_bytes=args.graph_metadata_bytes,
            reuse_queue_bytes=args.reuse_queue_bytes,
            score_ops=args.score_ops,
            ridge_intensity=args.ridge_intensity,
        ),
        gemm_point("Self-Attn Proj", m, h, h, args.activation_bytes, args.weight_bytes, args.output_bytes, args.ridge_intensity),
        gemm_point("FFN up/gate GEMM", m, h, i, args.activation_bytes, args.weight_bytes, args.output_bytes, args.ridge_intensity),
        gemm_point("FFN down GEMM", m, i, h, args.activation_bytes, args.weight_bytes, args.output_bytes, args.ridge_intensity),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "workload_character.tsv", rows)
    write_plot(args.figure, rows, args.ridge_intensity)
    write_summary(args.output_dir / "README.md", rows, args)

    print(f"[Profile] wrote {args.output_dir / 'workload_character.tsv'}")
    print(f"[Profile] wrote {args.output_dir / 'README.md'}")
    print(f"[Profile] wrote {args.figure}")
    for row in rows:
        print(
            f"{row.name}\t{row.kind}\tintensity={row.intensity:.2f}"
            f"\troofline={row.roofline_perf:.3f}\tops={row.ops:.3e}"
            f"\tbytes={row.bytes_moved:.3e}"
        )


if __name__ == "__main__":
    main()
