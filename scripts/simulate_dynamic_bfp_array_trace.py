#!/usr/bin/env python3
"""Summarize dynamic BFPA4/BFPA6 array activity from a generated pool trace.

The input is the metadata JSON emitted by
``generate_graph_aware_bfp_dynamic_pool.py``.  That metadata records, for each
wrapped Linear module, how many activation BFP blocks were executed at the
BFPA4 base precision and how many were refined with two extra mantissa bits.

This script turns that trace into a lightweight array-side estimate:

    dynamic cycles = BFPA4 base bit-MACs + selected extra 2-bit bit-MACs

It is not a full RTL/cycle-accurate simulator.  It is an activity/cycle proxy
for the progressive BFP PE array, grounded in the real per-module block trace
from the LLaMA forward pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _module_kind(name: str) -> str:
    if "q_proj" in name:
        return "q_proj"
    if "k_proj" in name:
        return "k_proj"
    if "v_proj" in name:
        return "v_proj"
    if "o_proj" in name:
        return "o_proj"
    if "gate_proj" in name:
        return "gate_proj"
    if "up_proj" in name:
        return "up_proj"
    if "down_proj" in name:
        return "down_proj"
    return "other"


def _cycles(bit_macs: int, bit_macs_per_cycle: float, utilization: float) -> float:
    denom = max(1e-9, float(bit_macs_per_cycle) * float(utilization))
    return float(bit_macs) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Dynamic BFP metadata JSON.")
    parser.add_argument("--output_dir", default=None, help="Directory for TSV/MD output. Defaults next to metadata.")
    parser.add_argument(
        "--bit_macs_per_cycle",
        type=float,
        default=128 * 128,
        help="Effective bit-MAC issue throughput of the modeled array.",
    )
    parser.add_argument(
        "--base_utilization",
        type=float,
        default=0.90,
        help="Utilization for the always-on BFPA4 base phase.",
    )
    parser.add_argument(
        "--refine_utilization",
        type=float,
        default=0.80,
        help="Utilization for the optional extra 2-bit refinement phase.",
    )
    parser.add_argument(
        "--weight_bits",
        type=int,
        default=4,
        help="Weight bit-width used for rough traffic accounting.",
    )
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trace = meta.get("array_trace") or {}
    modules: list[dict[str, Any]] = list(trace.get("modules") or [])
    if not modules:
        raise SystemExit(
            f"No array_trace.modules found in {meta_path}. Regenerate the dynamic pool with the updated script."
        )

    out_dir = Path(args.output_dir) if args.output_dir else meta_path.parent / f"{meta_path.stem}_array_trace"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    totals = {
        "total_blocks": 0,
        "refined_blocks": 0,
        "base_bit_macs": 0,
        "extra_bit_macs": 0,
        "dynamic_bit_macs": 0,
        "full_bfpa4_bit_macs": 0,
        "full_bfpa6_bit_macs": 0,
        "full_bfpa8_bit_macs": 0,
        "dynamic_cycles": 0.0,
        "full_bfpa4_cycles": 0.0,
        "full_bfpa6_cycles": 0.0,
        "full_bfpa8_cycles": 0.0,
    }
    by_kind: dict[str, dict[str, Any]] = {}

    for module in modules:
        total_blocks = int(module["total_blocks"])
        refined_blocks = int(module["refined_blocks"])
        base_bit_macs = int(module["base_bit_macs"])
        extra_bit_macs = int(module["extra_bit_macs"])
        dynamic_bit_macs = base_bit_macs + extra_bit_macs
        full_bfpa6_bit_macs = int(module["full_refine_bit_macs"])
        full_bfpa8_bit_macs = int(module["full_p8_bit_macs"])
        # BFPA4 base is exactly the base-bit path for every block.
        full_bfpa4_bit_macs = base_bit_macs

        base_cycles = _cycles(base_bit_macs, args.bit_macs_per_cycle, args.base_utilization)
        extra_cycles = _cycles(extra_bit_macs, args.bit_macs_per_cycle, args.refine_utilization)
        dynamic_cycles = base_cycles + extra_cycles
        full_bfpa4_cycles = _cycles(full_bfpa4_bit_macs, args.bit_macs_per_cycle, args.base_utilization)
        full_bfpa6_cycles = _cycles(full_bfpa6_bit_macs, args.bit_macs_per_cycle, args.base_utilization)
        full_bfpa8_cycles = _cycles(full_bfpa8_bit_macs, args.bit_macs_per_cycle, args.base_utilization)

        row = {
            "module": module["module"],
            "kind": _module_kind(module["module"]),
            "calls": int(module["calls"]),
            "token_rows": int(module["token_rows"]),
            "in_features": int(module["in_features"]),
            "out_features": int(module["out_features"]),
            "total_blocks": total_blocks,
            "refined_blocks": refined_blocks,
            "refined_ratio": refined_blocks / max(1, total_blocks),
            "effective_bits": float(module["effective_bits"]),
            "dynamic_bit_macs": dynamic_bit_macs,
            "dynamic_cycles": dynamic_cycles,
            "full_bfpa4_cycles": full_bfpa4_cycles,
            "full_bfpa6_cycles": full_bfpa6_cycles,
            "full_bfpa8_cycles": full_bfpa8_cycles,
            "dynamic_vs_bfpa4_cycles": dynamic_cycles / max(1e-9, full_bfpa4_cycles),
            "dynamic_vs_bfpa6_cycles": dynamic_cycles / max(1e-9, full_bfpa6_cycles),
            "dynamic_vs_bfpa8_cycles": dynamic_cycles / max(1e-9, full_bfpa8_cycles),
        }
        rows.append(row)

        for key in (
            "total_blocks",
            "refined_blocks",
            "base_bit_macs",
            "extra_bit_macs",
            "dynamic_bit_macs",
            "full_bfpa4_bit_macs",
            "full_bfpa6_bit_macs",
            "full_bfpa8_bit_macs",
        ):
            totals[key] += locals()[key]
        for key, value in (
            ("dynamic_cycles", dynamic_cycles),
            ("full_bfpa4_cycles", full_bfpa4_cycles),
            ("full_bfpa6_cycles", full_bfpa6_cycles),
            ("full_bfpa8_cycles", full_bfpa8_cycles),
        ):
            totals[key] += value

        kind = row["kind"]
        bucket = by_kind.setdefault(
            kind,
            {
                "kind": kind,
                "modules": 0,
                "total_blocks": 0,
                "refined_blocks": 0,
                "dynamic_cycles": 0.0,
                "full_bfpa4_cycles": 0.0,
                "full_bfpa6_cycles": 0.0,
                "full_bfpa8_cycles": 0.0,
            },
        )
        bucket["modules"] += 1
        bucket["total_blocks"] += total_blocks
        bucket["refined_blocks"] += refined_blocks
        bucket["dynamic_cycles"] += dynamic_cycles
        bucket["full_bfpa4_cycles"] += full_bfpa4_cycles
        bucket["full_bfpa6_cycles"] += full_bfpa6_cycles
        bucket["full_bfpa8_cycles"] += full_bfpa8_cycles

    rows.sort(key=lambda row: row["dynamic_cycles"], reverse=True)

    tsv_path = out_dir / "module_array_trace.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    kind_rows = []
    for bucket in sorted(by_kind.values(), key=lambda item: item["dynamic_cycles"], reverse=True):
        refined_ratio = bucket["refined_blocks"] / max(1, bucket["total_blocks"])
        kind_rows.append(
            {
                "kind": bucket["kind"],
                "modules": bucket["modules"],
                "total_blocks": bucket["total_blocks"],
                "refined_blocks": bucket["refined_blocks"],
                "refined_ratio": refined_ratio,
                "dynamic_cycles": bucket["dynamic_cycles"],
                "dynamic_vs_bfpa4": bucket["dynamic_cycles"] / max(1e-9, bucket["full_bfpa4_cycles"]),
                "dynamic_vs_bfpa6": bucket["dynamic_cycles"] / max(1e-9, bucket["full_bfpa6_cycles"]),
                "dynamic_vs_bfpa8": bucket["dynamic_cycles"] / max(1e-9, bucket["full_bfpa8_cycles"]),
            }
        )

    kind_tsv_path = out_dir / "kind_array_trace.tsv"
    with kind_tsv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(kind_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(kind_rows)

    total_refined_ratio = totals["refined_blocks"] / max(1, totals["total_blocks"])
    total_effective_bits = int(meta.get("base_mantissa", 4)) + (
        int(meta.get("refine_mantissa", 6)) - int(meta.get("base_mantissa", 4))
    ) * total_refined_ratio
    summary = {
        "metadata": str(meta_path),
        "dataset": meta.get("dataset"),
        "tag": meta.get("tag"),
        "threshold": meta.get("threshold"),
        "stress_scale": meta.get("stress_scale"),
        "block_size": meta.get("block_size"),
        "base_mantissa": meta.get("base_mantissa"),
        "refine_mantissa": meta.get("refine_mantissa"),
        "total_blocks": totals["total_blocks"],
        "refined_blocks": totals["refined_blocks"],
        "refined_ratio": total_refined_ratio,
        "effective_bits": total_effective_bits,
        "dynamic_cycles": totals["dynamic_cycles"],
        "full_bfpa4_cycles": totals["full_bfpa4_cycles"],
        "full_bfpa6_cycles": totals["full_bfpa6_cycles"],
        "full_bfpa8_cycles": totals["full_bfpa8_cycles"],
        "dynamic_vs_bfpa4_cycles": totals["dynamic_cycles"] / max(1e-9, totals["full_bfpa4_cycles"]),
        "dynamic_vs_bfpa6_cycles": totals["dynamic_cycles"] / max(1e-9, totals["full_bfpa6_cycles"]),
        "dynamic_vs_bfpa8_cycles": totals["dynamic_cycles"] / max(1e-9, totals["full_bfpa8_cycles"]),
        "bit_macs_per_cycle": args.bit_macs_per_cycle,
        "base_utilization": args.base_utilization,
        "refine_utilization": args.refine_utilization,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Dynamic BFP Array Trace Summary",
        "",
        f"metadata: `{meta_path}`",
        "",
        "## Overall",
        "",
        f"- refined blocks: `{totals['refined_blocks']}/{totals['total_blocks']}` ({_fmt_pct(total_refined_ratio)})",
        f"- effective mantissa bits: `{total_effective_bits:.3f}`",
        f"- dynamic cycles: `{totals['dynamic_cycles']:.0f}`",
        f"- dynamic / BFPA4 cycles: `{summary['dynamic_vs_bfpa4_cycles']:.3f}x`",
        f"- dynamic / BFPA6 cycles: `{summary['dynamic_vs_bfpa6_cycles']:.3f}x`",
        f"- dynamic / BFPA8 cycles: `{summary['dynamic_vs_bfpa8_cycles']:.3f}x`",
        "",
        "## By Module Kind",
        "",
        "| Kind | Refined | Dynamic/BFPA4 | Dynamic/BFPA6 | Dynamic/BFPA8 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in kind_rows:
        md_lines.append(
            f"| {row['kind']} | {_fmt_pct(row['refined_ratio'])} | "
            f"{row['dynamic_vs_bfpa4']:.3f}x | {row['dynamic_vs_bfpa6']:.3f}x | {row['dynamic_vs_bfpa8']:.3f}x |"
        )
    md_lines += [
        "",
        "## Output Files",
        "",
        f"- `{tsv_path}`",
        f"- `{kind_tsv_path}`",
        f"- `{out_dir / 'summary.json'}`",
        "",
    ]
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
