#!/usr/bin/env python3
"""Summarize ONNXim FFN block-gating microbenchmarks.

This is a hardware-only probe for the next Graph-Bit+ direction: reducing FFN
intermediate blocks so weight traffic can fall, not only activation bit-plane
traffic.  It compares LLaMA-7B encoder GEMM microbenchmarks with different FFN
intermediate dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing aggregate: {path}")
    return json.loads(path.read_text())


def enc(agg: dict[str, Any]) -> dict[str, Any]:
    return agg["encoder"]


def reqs(agg: dict[str, Any]) -> float:
    e = enc(agg)
    return float(e.get("dram_read_requests", 0.0)) + float(e.get("dram_write_requests", 0.0))


def metric(agg: dict[str, Any], key: str) -> float:
    return float(enc(agg).get(key, 0.0) or 0.0)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=root / "output" / "onnxim_graphbit" / "microbench_s64_internal_p8",
    )
    parser.add_argument("--workspaces", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "onnxim_graphbit" / "ffn_block_gating",
    )
    args = parser.parse_args()

    base = load_json(args.baseline / "aggregate.json")
    base_summary = read_summary_rows(args.baseline / "summary.tsv")
    base_intermediate = max(int(row["n"]) for row in base_summary if row["name"] == "ffn_up")

    rows = []
    for workspace in args.workspaces:
        agg = load_json(workspace / "aggregate.json")
        summary_rows = read_summary_rows(workspace / "summary.tsv")
        intermediate = max(int(row["n"]) for row in summary_rows if row["name"] == "ffn_up")
        keep = intermediate / base_intermediate
        rows.append(
            {
                "workspace": str(workspace),
                "keep_ratio": keep,
                "intermediate": intermediate,
                "cycles_norm": safe_div(metric(agg, "cycles"), metric(base, "cycles")),
                "matmul_norm": safe_div(metric(agg, "matmul_active_cycles"), metric(base, "matmul_active_cycles")),
                "traffic_norm": safe_div(reqs(agg), reqs(base)),
                "input_read_norm": safe_div(metric(agg, "mem_read_input_actual"), metric(base, "mem_read_input_actual")),
                "weight_read_norm": safe_div(metric(agg, "mem_read_weight"), metric(base, "mem_read_weight")),
                "output_write_norm": safe_div(metric(agg, "mem_write_output"), metric(base, "mem_write_output")),
                "gflops_norm": safe_div(metric(agg, "gflops"), metric(base, "gflops")),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "ffn_block_gating.tsv"
    txt_path = args.output_dir / "ffn_block_gating.txt"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "ONNXim LLaMA-7B FFN block-gating hardware probe",
        f"Baseline intermediate={base_intermediate}",
        "",
        "Keep  Interm  Cycles MatMul Traffic InRead WeightRd OutWr GFLOPs",
        "---------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{100.0 * row['keep_ratio']:>4.0f}% {row['intermediate']:>7} "
            f"{row['cycles_norm']:>7.3f} {row['matmul_norm']:>6.3f} "
            f"{row['traffic_norm']:>7.3f} {row['input_read_norm']:>6.3f} "
            f"{row['weight_read_norm']:>8.3f} {row['output_write_norm']:>5.3f} "
            f"{row['gflops_norm']:>6.3f}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This probe changes FFN intermediate blocks, so FFN weight traffic and FFN output shape shrink.",
            "- Unlike activation bit-plane early-stop, this is the path that can reduce weight-read pressure.",
            "- It is hardware-only here; accuracy must be validated separately before becoming a main policy.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[FFNBlockGating] wrote {tsv_path}")
    print(f"[FFNBlockGating] wrote {txt_path}")
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
