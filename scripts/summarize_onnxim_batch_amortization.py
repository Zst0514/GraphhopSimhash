#!/usr/bin/env python3
"""Summarize ONNXim batch-size amortization for LLaMA encoder GEMMs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_agg(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing aggregate: {path}")
    return json.loads(path.read_text())


def enc(agg: dict[str, Any]) -> dict[str, Any]:
    return agg["encoder"]


def metric(agg: dict[str, Any], key: str) -> float:
    return float(enc(agg).get(key, 0.0) or 0.0)


def reqs(agg: dict[str, Any]) -> float:
    return metric(agg, "dram_read_requests") + metric(agg, "dram_write_requests")


def seq_from_workspace(path: Path) -> int:
    match = re.search(r"s(\d+)", path.name)
    if not match:
        raise SystemExit(f"Cannot infer seq-len from workspace name: {path}")
    return int(match.group(1))


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspaces", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "onnxim_graphbit" / "batch_amortization",
    )
    args = parser.parse_args()

    rows = []
    for workspace in sorted(args.workspaces, key=seq_from_workspace):
        seq = seq_from_workspace(workspace)
        agg = load_agg(workspace / "aggregate.json")
        rows.append(
            {
                "seq_len": seq,
                "workspace": str(workspace),
                "cycles": metric(agg, "cycles"),
                "traffic": reqs(agg),
                "input_read": metric(agg, "mem_read_input_actual"),
                "weight_read": metric(agg, "mem_read_weight"),
                "output_write": metric(agg, "mem_write_output"),
                "cycles_per_node": metric(agg, "cycles") / seq,
                "traffic_per_node": reqs(agg) / seq,
                "input_read_per_node": metric(agg, "mem_read_input_actual") / seq,
                "weight_read_per_node": metric(agg, "mem_read_weight") / seq,
                "output_write_per_node": metric(agg, "mem_write_output") / seq,
            }
        )

    base = rows[0]
    for row in rows:
        row["cycles_per_node_norm"] = safe_div(row["cycles_per_node"], base["cycles_per_node"])
        row["traffic_per_node_norm"] = safe_div(row["traffic_per_node"], base["traffic_per_node"])
        row["weight_per_node_norm"] = safe_div(row["weight_read_per_node"], base["weight_read_per_node"])
        row["input_per_node_norm"] = safe_div(row["input_read_per_node"], base["input_read_per_node"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / "batch_amortization.tsv"
    txt_path = args.output_dir / "batch_amortization.txt"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "ONNXim LLaMA-7B FullP8 batch-size amortization",
        "Per-node values are normalized to the smallest tested batch.",
        "",
        "Seq  Cyc/Node  Traf/Node Weight/Node Input/Node  Out/Node  WNorm TNorm CNorm",
        "----------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['seq_len']:>3} {row['cycles_per_node']:>9.1f} "
            f"{row['traffic_per_node']:>10.1f} {row['weight_read_per_node']:>11.1f} "
            f"{row['input_read_per_node']:>9.1f} {row['output_write_per_node']:>8.1f} "
            f"{row['weight_per_node_norm']:>6.3f} {row['traffic_per_node_norm']:>5.3f} "
            f"{row['cycles_per_node_norm']:>5.3f}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Larger risk-bucket micro-batches amortize weight reads across more node tokens.",
            "- This is the memory-side reason to batch miss nodes by risk instead of issuing many tiny mixed batches.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[BatchAmortization] wrote {tsv_path}")
    print(f"[BatchAmortization] wrote {txt_path}")
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
