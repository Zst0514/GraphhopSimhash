#!/usr/bin/env python3
"""Draw the graph-aware progressive BFP array datapath figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "HPCA_2027_GFMAcc" / "Figure"
DOC_FIG_DIR = ROOT / "docs" / "figures"


COLORS = {
    "ink": "#111827",
    "muted": "#475467",
    "line": "#334155",
    "blue": "#DBEAFE",
    "blue_edge": "#2563EB",
    "green": "#DCFCE7",
    "green_edge": "#16A34A",
    "orange": "#FFEDD5",
    "orange_edge": "#EA580C",
    "purple": "#EDE9FE",
    "purple_edge": "#7C3AED",
    "gray": "#F1F5F9",
    "gray_edge": "#94A3B8",
    "yellow": "#FEF3C7",
    "yellow_edge": "#D97706",
    "red": "#FEE2E2",
    "red_edge": "#DC2626",
}


def box(ax, x, y, w, h, text="", fc="gray", ec=None, lw=1.5, fs=10,
        weight="semibold", ha="center", va="center", radius=0.08, z=1):
    if ec is None:
        ec = f"{fc}_edge"
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw,
        edgecolor=COLORS.get(ec, ec),
        facecolor=COLORS.get(fc, fc),
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va, fontsize=fs,
                color=COLORS["ink"], weight=weight, zorder=z + 1)
    return patch


def arrow(ax, x1, y1, x2, y2, color="line", lw=1.8, style="-", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=COLORS.get(color, color),
        linestyle=style,
        connectionstyle=f"arc3,rad={rad}",
        zorder=3,
    )
    ax.add_patch(arr)
    return arr


def bus_arrow(ax, pts, color="line", lw=1.8, style="-"):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=COLORS.get(color, color),
                linewidth=lw, linestyle=style, zorder=3)
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    return arrow(ax, x1, y1, x2, y2, color=color, lw=lw, style=style)


def label(ax, x, y, s, fs=10, weight="regular", color="ink",
          ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, weight=weight, color=COLORS.get(color, color),
            ha=ha, va=va, style=style)


def bit_cells(ax, x, y, bits, colors, cell_w=0.25, cell_h=0.28, fs=9):
    for i, (bit, c) in enumerate(zip(bits, colors)):
        ax.add_patch(Rectangle((x + i * cell_w, y), cell_w, cell_h,
                               linewidth=1.0, edgecolor=COLORS["ink"],
                               facecolor=c, zorder=2))
        ax.text(x + i * cell_w + cell_w / 2, y + cell_h / 2, bit,
                ha="center", va="center", fontsize=fs, weight="semibold",
                color=COLORS["ink"], zorder=3)


def pe_array(ax, x, y, w, h, rows=3, cols=4, title=True):
    box(ax, x, y, w, h, fc="#F8FAFC", ec="ink", lw=1.6, radius=0.06)
    pad_x, pad_y = 0.18, 0.18
    cw = (w - 2 * pad_x) / cols
    ch = (h - 2 * pad_y) / rows
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (x + pad_x + c * cw + 0.04, y + pad_y + r * ch + 0.04),
                cw - 0.08, ch - 0.08,
                linewidth=1.0,
                edgecolor=COLORS["ink"],
                facecolor="#FFFFFF",
                zorder=2,
            ))
    if title:
        label(ax, x + w / 2, y + h + 0.18, "W4 x mantissa INT MAC array", fs=10,
              weight="bold")


def draw():
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    label(ax, 7, 7.16, "Graph-Aware Progressive BFP Array", fs=17, weight="bold")
    label(ax, 7, 6.88,
          "Shared-exponent BFP prealignment feeds a BFPA4 base array and an optional low-2-bit lift lane.",
          fs=10.5, color="muted")

    # Top-level chip boundary.
    box(ax, 0.25, 0.55, 13.45, 6.05, fc="#FFFFFF", ec="ink", lw=1.8, radius=0.06)
    label(ax, 0.55, 6.40, "Progressive BFP compute tile", fs=10.5, weight="bold", ha="left")

    # Input and metadata buffers.
    box(ax, 0.55, 4.65, 1.70, 1.25, "Activation\nSRAM", fc="blue", ec="blue_edge", fs=10.5)
    box(ax, 0.55, 3.10, 1.70, 1.15, "BlockMeta\nE, graph risk", fc="yellow", ec="yellow_edge", fs=9.8)
    box(ax, 0.55, 1.75, 1.70, 0.95, "Weight RF\nW4 tile", fc="gray", ec="gray_edge", fs=10.2)

    # Prealignment and stress.
    box(ax, 2.75, 4.35, 1.70, 1.65, "BFP\nPrealigner\nshared exp E", fc="green", ec="green_edge", fs=9.5)
    box(ax, 2.75, 2.95, 1.70, 1.00, "Stress Unit\nmax / median", fc="orange", ec="orange_edge", fs=9.2)
    box(ax, 2.75, 1.55, 1.70, 0.95, "Priority\ncompare", fc="orange", ec="orange_edge", fs=9.2)
    label(ax, 3.60, 1.26, "graph risk x stress", fs=8.3, color="orange_edge", weight="bold")

    # Mantissa split.
    box(ax, 4.95, 4.45, 1.85, 1.45, fc="gray", ec="gray_edge", fs=10.2)
    label(ax, 5.875, 5.66, "Mantissa Slicer", fs=10.2, weight="bold")
    label(ax, 5.16, 5.32, "q6", fs=8.8, ha="left")
    bit_cells(ax, 5.54, 5.18, ["b5", "b4", "b3", "b2", "b1", "b0"],
              ["#BFDBFE", "#BFDBFE", "#BFDBFE", "#BFDBFE", "#FED7AA", "#FED7AA"],
              cell_w=0.18, cell_h=0.22, fs=6.5)
    label(ax, 5.18, 4.91, "q4 = high 4", fs=8.2, ha="left")
    label(ax, 5.18, 4.67, "q2 = low 2", fs=8.2, ha="left")

    # Base array lane.
    box(ax, 7.25, 4.05, 3.25, 1.95, fc="green", ec="green_edge", fs=11, weight="bold")
    label(ax, 8.875, 5.78, "BFPA4 Base Lane", fs=11, weight="bold")
    box(ax, 7.55, 5.05, 0.90, 0.48, "q4", fc="blue", ec="blue_edge", fs=9.5)
    box(ax, 7.55, 4.45, 0.90, 0.42, "E", fc="yellow", ec="yellow_edge", fs=9.5)
    pe_array(ax, 8.85, 4.45, 1.25, 0.82, rows=2, cols=3, title=False)
    label(ax, 9.48, 5.46, "INT MAC array", fs=8.8, weight="bold")

    # Refine scheduler and lane.
    box(ax, 4.95, 1.25, 1.85, 1.20, "Refine\nScheduler", fc="orange", ec="orange_edge", fs=10.0)
    box(ax, 7.25, 1.02, 3.25, 1.78, fc="#FFF7ED", ec="orange_edge", fs=11, weight="bold")
    label(ax, 8.45, 2.62, "BFPA6 Lift Lane", fs=10.5, weight="bold")
    box(ax, 7.55, 1.92, 0.90, 0.48, "q2", fc="orange", ec="orange_edge", fs=9.5)
    box(ax, 7.55, 1.30, 0.90, 0.42, "E", fc="yellow", ec="yellow_edge", fs=9.5)
    pe_array(ax, 8.85, 1.45, 1.25, 0.82, rows=2, cols=3, title=False)
    label(ax, 9.48, 2.35, "extra INT MAC", fs=8.4, weight="bold")

    # Shared W fanout.
    box(ax, 6.95, 3.42, 0.75, 0.52, "W4", fc="gray", ec="gray_edge", fs=9.0)
    bus_arrow(ax, [(2.25, 2.22), (2.42, 2.22), (2.42, 4.02), (6.95, 4.02), (6.95, 3.68)])
    arrow(ax, 7.70, 3.74, 8.85, 4.72, color="line", rad=0.02)
    arrow(ax, 7.70, 3.50, 8.85, 1.72, color="line", rad=-0.02)
    label(ax, 7.32, 3.25, "broadcast", fs=7.8, color="muted")

    # Accumulation and output.
    box(ax, 11.00, 3.25, 1.45, 1.28, "Psum SRAM\nread / update", fc="purple", ec="purple_edge", fs=9.3)
    box(ax, 11.00, 1.58, 1.45, 0.92, "Accumulate\nDeltaY", fc="red", ec="red_edge", fs=9.2)
    box(ax, 12.75, 2.62, 0.70, 0.52, "+", fc="#FFFFFF", ec="ink", fs=14)
    box(ax, 12.55, 3.55, 0.90, 0.62, "Output\ntile", fc="purple", ec="purple_edge", fs=8.6)

    # Dataflow arrows.
    arrow(ax, 2.25, 5.25, 2.75, 5.25)
    arrow(ax, 2.25, 3.65, 2.75, 3.45)
    arrow(ax, 4.45, 5.25, 4.95, 5.25)
    arrow(ax, 6.80, 5.18, 7.55, 5.28)
    arrow(ax, 8.45, 5.28, 8.85, 4.95)
    arrow(ax, 10.10, 4.86, 11.00, 3.98)
    arrow(ax, 12.45, 3.90, 12.75, 3.00)
    arrow(ax, 13.10, 3.14, 13.10, 3.55)

    arrow(ax, 4.45, 3.45, 4.95, 2.05, color="orange_edge", style="--", rad=-0.1)
    arrow(ax, 4.45, 2.02, 4.95, 1.85, color="orange_edge", style="--")
    arrow(ax, 6.80, 1.85, 7.55, 2.16, color="orange_edge", style="--")
    arrow(ax, 8.45, 2.16, 8.85, 1.95, color="orange_edge", style="--")
    arrow(ax, 10.10, 1.86, 11.00, 2.04, color="red_edge", style="--")
    arrow(ax, 11.72, 2.50, 11.72, 3.25, color="red_edge", style="--")
    arrow(ax, 12.45, 3.90, 12.75, 3.00)

    # Control and phase annotations.
    box(ax, 4.95, 3.00, 1.85, 0.70, "Refine flag\n+ psum addr", fc="yellow", ec="yellow_edge", fs=8.6)
    arrow(ax, 3.60, 2.50, 5.15, 3.00, color="orange_edge", style="--", rad=0.08)
    arrow(ax, 5.88, 3.00, 5.88, 2.45, color="orange_edge", style="--")
    label(ax, 8.90, 6.20, "all blocks", fs=9.0, color="green_edge", weight="bold")
    # Small equations inside the datapath, not explanatory notes.
    box(ax, 0.55, 0.70, 3.90, 0.48, "q6 = (q4 << 2) + q2", fc="#FFFFFF", ec="gray_edge", fs=9.2)
    box(ax, 4.95, 0.70, 3.15, 0.48, "Y4 = (q4 x W4) << 2", fc="#FFFFFF", ec="gray_edge", fs=8.9)
    box(ax, 8.35, 0.70, 3.40, 0.48, "Ydyn = Y4 + flag x (q2 x W4)", fc="#FFFFFF", ec="gray_edge", fs=8.7)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (FIG_DIR, DOC_FIG_DIR):
        fig.savefig(out_dir / "progressive_bfp_array_datapath.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "progressive_bfp_array_datapath.svg", bbox_inches="tight")
        fig.savefig(out_dir / "progressive_bfp_array_datapath.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
