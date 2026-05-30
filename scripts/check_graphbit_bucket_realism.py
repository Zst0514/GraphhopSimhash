#!/usr/bin/env python3
"""Check whether Graph-Bit risk buckets are large enough for batching.

This script deliberately does *not* claim extra HBM weight savings.  Its job is
to separate two questions:

1. Are there enough nodes in each depth bucket to form normal micro-batches?
2. Is any additional weight-stationary HBM amortization being assumed?

If buckets are large, Graph-Bit can keep PE utilization and ordinary
weight-stationary reuse healthy.  That is different from claiming a new 4x
weight HBM reduction.  Extra W HBM reduction must be treated as a sensitivity
parameter or justified by a concrete scheduler/capacity model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEPTH_KEYS = ("p8", "p6", "p5", "p4")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing workload: {path}")
    return json.loads(path.read_text())


def choose_profile(workload: dict[str, Any], contains: str) -> dict[str, Any]:
    needle = contains.lower()
    for profile in workload["profiles"]:
        text = " ".join(
            [
                str(profile.get("id", "")),
                str(profile.get("route", {}).get("method", "")),
                str(profile.get("route", {}).get("config", "")),
            ]
        ).lower()
        if needle in text:
            return profile
    raise SystemExit(f"No profile matching '{contains}'")


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--profile-match", default="degree static")
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-weight-tile-batch", type=int, default=16)
    parser.add_argument("--assumed-weight-stationary-tile-batch", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    workload = load_json(args.workload)
    profile = choose_profile(workload, args.profile_match)
    ratios = {key: float(profile["ratios"].get(key, 0.0) or 0.0) for key in DEPTH_KEYS}
    miss_ratio = sum(ratios.values())
    miss_nodes = int(round(args.node_count * miss_ratio))

    rows = []
    total_padded = 0
    total_nonempty = 0
    for key in DEPTH_KEYS:
        count = int(round(args.node_count * ratios[key]))
        batches = math.ceil(count / args.batch_size) if count else 0
        padded = batches * args.batch_size
        util = (count / padded) if padded else 0.0
        total_padded += padded
        total_nonempty += count
        rows.append(
            {
                "bucket": key.upper(),
                "nodes": count,
                "ratio_all": ratios[key],
                "batches": batches,
                "tail_util": util,
            }
        )

    bucket_padding = (total_padded / total_nonempty) if total_nonempty else 1.0
    assumed_gain = (
        args.assumed_weight_stationary_tile_batch / args.baseline_weight_tile_batch
        if args.baseline_weight_tile_batch
        else 1.0
    )
    assumed_w_hbm_scale = min(1.0, 1.0 / assumed_gain) if assumed_gain > 0 else 1.0

    output_dir = args.output_dir or (args.workload.parent / "bucket_realism")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "graphbit_bucket_realism.v1",
        "workload": str(args.workload),
        "profile": profile.get("id"),
        "node_count": args.node_count,
        "batch_size": args.batch_size,
        "miss_nodes": miss_nodes,
        "miss_ratio": miss_ratio,
        "bucket_padding_overhead": bucket_padding,
        "baseline_weight_tile_batch": args.baseline_weight_tile_batch,
        "assumed_weight_stationary_tile_batch": args.assumed_weight_stationary_tile_batch,
        "assumed_weight_hbm_scale": assumed_w_hbm_scale,
        "important_note": (
            "The bucket table only validates batching feasibility. It does not "
            "prove extra weight HBM reduction. If assumed_weight_hbm_scale < 1, "
            "that is an explicit sensitivity assumption."
        ),
        "rows": rows,
    }
    (output_dir / "bucket_realism.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "Graph-Bit bucket realism check",
        f"workload: {args.workload}",
        f"profile: {profile.get('id')}",
        f"nodes: {args.node_count} | miss: {miss_nodes} ({fmt_pct(miss_ratio)}) | batch={args.batch_size}",
        "",
        f"{'bucket':>8s} {'nodes':>8s} {'all%':>8s} {'batches':>8s} {'tail_util':>10s}",
        "-" * 48,
    ]
    for row in rows:
        lines.append(
            f"{row['bucket']:>8s} {row['nodes']:8d} {fmt_pct(row['ratio_all']):>8s} "
            f"{row['batches']:8d} {fmt_pct(row['tail_util']):>10s}"
        )
    lines.extend(
        [
            "",
            f"bucket padding overhead: {bucket_padding:.3f}x",
            "",
            "Weight HBM reuse:",
            f"  baseline tile batch: {args.baseline_weight_tile_batch}",
            f"  assumed WS tile batch: {args.assumed_weight_stationary_tile_batch}",
            f"  assumed W HBM scale: {assumed_w_hbm_scale:.3f}",
            "",
            "Interpretation:",
            "  - If assumed W HBM scale is 1.000, this is the conservative mainline.",
            "  - If it is below 1.000, that is an explicit sensitivity assumption, not a measured fact.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "bucket_realism.txt").write_text(text)
    print(text)
    print(f"[BucketRealism] wrote {output_dir / 'bucket_realism.txt'}")


if __name__ == "__main__":
    main()
