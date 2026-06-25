#!/usr/bin/env python3
"""Compare GRACE TSER40 frontend timing against a HEAT-style W8/W4 frontend.

This script is intentionally analytical.  It fixes the local baseline to the
project's W4BFPA4 encoder path and separates compute, weight-load,
activation-load, and output/cache movement terms.  The goal is to avoid
comparing HEAT's W8A10/W4A2 bit-serial policy against the wrong W4BFPA8
baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent
DEFAULT_REUSE_TSV = OFA_ROOT / "output" / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"

TASK_NODES = {
    "CN": 2708,
    "CL": 2708,
    "PN": 19717,
    "PL": 19717,
    "AR": 169343,
    "WK": 11701,
}

HEAT_OVERLAP = {"CN", "CL", "PN", "PL", "AR"}


def read_reuse_points(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            task = str(row.get("task", "")).strip()
            if not task:
                continue
            reuse = float(row.get("anchor_reuse", row.get("reuse", 40.0))) / 100.0
            drop = float(row.get("target_anchor_drop", row.get("drop", 0.0))) / 100.0
            out[task] = {"reuse": reuse, "drop": drop}
    return out


def ceil_batches(nodes: float, batch_size: int) -> int:
    return max(1, int(math.ceil(float(nodes) / float(batch_size))))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def num(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def weighted_total(row: dict[str, Any], args: argparse.Namespace, *, cache_read_scale: float) -> float:
    output_norm = float(row["encoder_output_write_norm"]) + cache_read_scale * float(row["reuse_cache_read_norm"])
    return (
        float(args.compute_share) * float(row["compute_norm"])
        + float(args.weight_share) * float(row["weight_stream_norm"])
        + float(args.activation_share) * float(row["activation_load_norm"])
        + float(args.output_share) * output_norm
        + float(args.control_share) * float(row["control_norm"])
    )


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    reuse_points = read_reuse_points(Path(args.reuse_tsv))
    rows: list[dict[str, Any]] = []

    base_w = int(args.local_weight_bits)
    base_a = int(args.local_activation_bits)
    base_planes = float(base_w * base_a)
    baseline_batches_by_task = {
        task: ceil_batches(nodes, int(args.batch_size)) for task, nodes in TASK_NODES.items()
    }

    for task, nodes in TASK_NODES.items():
        reuse = float(reuse_points.get(task, {}).get("reuse", float(args.default_reuse)))
        drop = float(reuse_points.get(task, {}).get("drop", float("nan")))
        miss = 1.0 - reuse
        baseline_batches = baseline_batches_by_task[task]

        # Local fixed W4BFPA4 baseline.
        rows.append(
            {
                "task": task,
                "policy": "NoReuse+W4BFPA4",
                "nodes": nodes,
                "reuse": 0.0,
                "drop": 0.0,
                "compute_norm": 1.0,
                "weight_stream_norm": 1.0,
                "weight_lower_bound_norm": 1.0,
                "activation_load_norm": 1.0,
                "encoder_output_write_norm": 1.0,
                "reuse_cache_read_norm": 0.0,
                "control_norm": 1.0,
                "batches": baseline_batches,
                "note": "local fixed W4BFPA4 baseline",
            }
        )

        # HEAT-style: key vertices use W8A10; non-key vertices use W4A2.
        key_nodes = float(nodes) * float(args.heat_alpha)
        low_nodes = float(nodes) - key_nodes
        key_batches = ceil_batches(key_nodes, int(args.batch_size)) if key_nodes > 0 else 0
        low_batches = ceil_batches(low_nodes, int(args.batch_size)) if low_nodes > 0 else 0
        heat_avg_planes = (
            float(args.heat_alpha) * float(args.heat_hi_weight_bits * args.heat_hi_activation_bits)
            + (1.0 - float(args.heat_alpha))
            * float(args.heat_lo_weight_bits * args.heat_lo_activation_bits)
        )
        heat_avg_w = (
            float(args.heat_alpha) * float(args.heat_hi_weight_bits)
            + (1.0 - float(args.heat_alpha)) * float(args.heat_lo_weight_bits)
        )
        heat_avg_a = (
            float(args.heat_alpha) * float(args.heat_hi_activation_bits)
            + (1.0 - float(args.heat_alpha)) * float(args.heat_lo_activation_bits)
        )
        heat_stream_weight = (
            key_batches * float(args.heat_hi_weight_bits)
            + low_batches * float(args.heat_lo_weight_bits)
        ) / (baseline_batches * float(base_w))
        heat_w8_only_weight = float(args.heat_hi_weight_bits) / float(base_w)
        rows.append(
            {
                "task": task,
                "policy": "HEAT-style W8A10/W4A2",
                "nodes": nodes,
                "reuse": 0.0,
                "drop": float("nan"),
                "compute_norm": heat_avg_planes / base_planes,
                "weight_stream_norm": heat_stream_weight,
                "weight_lower_bound_norm": heat_avg_w / float(base_w),
                "weight_w8_only_norm": heat_w8_only_weight,
                "activation_load_norm": heat_avg_a / float(base_a),
                "encoder_output_write_norm": 1.0,
                "reuse_cache_read_norm": 0.0,
                "control_norm": heat_avg_planes / base_planes,
                "batches": key_batches + low_batches,
                "key_batches": key_batches,
                "low_batches": low_batches,
                "note": "HEAT Sec.5.2.1 bit-serial precision split; no TSER reuse",
            }
        )

        # GRACE TSER40 with the fixed local W4BFPA4 miss path.
        miss_batches = ceil_batches(float(nodes) * miss, int(args.batch_size)) if miss > 0 else 0
        rows.append(
            {
                "task": task,
                "policy": "TSER40+W4BFPA4",
                "nodes": nodes,
                "reuse": reuse,
                "drop": drop,
                "compute_norm": miss,
                "weight_stream_norm": float(miss_batches) / float(baseline_batches),
                "weight_lower_bound_norm": miss,
                "activation_load_norm": miss,
                "encoder_output_write_norm": miss,
                "reuse_cache_read_norm": reuse,
                "control_norm": miss,
                "batches": miss_batches,
                "note": "semantic reuse; miss nodes execute the same fixed W4BFPA4 path",
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope, tasks in (("AVG_HEAT5", HEAT_OVERLAP), ("AVG6", set(TASK_NODES))):
        for policy in ("HEAT-style W8A10/W4A2", "TSER40+W4BFPA4"):
            vals = [row for row in rows if row["task"] in tasks and row["policy"] == policy]
            if not vals:
                continue
            avg = {
                "scope": scope,
                "policy": policy,
                "reuse": sum(float(v["reuse"]) for v in vals) / len(vals),
                "drop": sum(float(v["drop"]) for v in vals if not math.isnan(float(v["drop"])))
                / max(1, sum(1 for v in vals if not math.isnan(float(v["drop"])))),
                "compute_norm": sum(float(v["compute_norm"]) for v in vals) / len(vals),
                "weight_stream_norm": sum(float(v["weight_stream_norm"]) for v in vals) / len(vals),
                "weight_lower_bound_norm": sum(float(v["weight_lower_bound_norm"]) for v in vals) / len(vals),
                "activation_load_norm": sum(float(v["activation_load_norm"]) for v in vals) / len(vals),
                "encoder_output_write_norm": sum(float(v["encoder_output_write_norm"]) for v in vals) / len(vals),
                "reuse_cache_read_norm": sum(float(v["reuse_cache_read_norm"]) for v in vals) / len(vals),
                "control_norm": sum(float(v["control_norm"]) for v in vals) / len(vals),
            }
            avg["weighted_total_offchip_cache"] = weighted_total(avg, args, cache_read_scale=1.0)
            avg["weighted_total_onchip_cache"] = weighted_total(avg, args, cache_read_scale=float(args.cache_read_scale))
            out.append(avg)
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]], aggregate: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Frontend Timing: TSER40 W4BFPA4 vs HEAT-Style Bit-Serial",
        "",
        "## Correct Baseline",
        "",
        "The local backend/frontend encoder path is fixed `W4BFPA4`, so HEAT-style bit-serial compute must be normalized to `W4 x A4 = 16` bit-plane GEMMs, not to W4A8 or INT8xINT8.",
        "",
        "HEAT-style Fig. 6 uses top-degree key vertices at `W8A10` and non-key vertices at `W4A2`.",
        "With `alpha=0.1`, its average compute is `15.2` bit-plane GEMMs per MAC, i.e. `15.2 / 16 = 0.95x` of the local W4BFPA4 compute path.",
        "",
        "## Timing Components",
        "",
        f"- Batch size for streamed weight-load rounding: `{args.batch_size}` nodes.",
        f"- Weighted total shares: compute `{args.compute_share}`, weight `{args.weight_share}`, activation `{args.activation_share}`, output `{args.output_share}`, control `{args.control_share}`.",
        f"- On-chip cache-read scale for reused embeddings: `{args.cache_read_scale}` of an encoder output write.",
        "",
        "Weight-load is reported in two forms:",
        "",
        "- `Weight stream`: split high/low precision passes with batch rounding.",
        "- `Weight lower`: per-node lower bound using average weight bits.",
        "",
        "## Aggregate",
        "",
        "| Scope | Policy | Reuse | Drop | Compute | Weight Stream | Weight Lower | Activation | Output Write | Cache Read | Weighted Total, Off-Chip Cache | Weighted Total, On-Chip Cache |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['scope']} | {row['policy']} | {pct(row['reuse'])} | "
            f"{pct(row['drop']) if not math.isnan(float(row['drop'])) else '-'} | "
            f"{num(row['compute_norm'])}x | {num(row['weight_stream_norm'])}x | "
            f"{num(row['weight_lower_bound_norm'])}x | {num(row['activation_load_norm'])}x | "
            f"{num(row['encoder_output_write_norm'])}x | {num(row['reuse_cache_read_norm'])}x | "
            f"{num(row['weighted_total_offchip_cache'])}x | {num(row['weighted_total_onchip_cache'])}x |"
        )

    lines.extend(
        [
            "",
            "## Per Task",
            "",
            "| Task | Policy | Reuse | Compute | Weight Stream | Activation | Output Write | Cache Read | Batches | Note |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if row["policy"] == "NoReuse+W4BFPA4":
            continue
        lines.append(
            f"| {row['task']} | {row['policy']} | {pct(row['reuse'])} | "
            f"{num(row['compute_norm'])}x | {num(row['weight_stream_norm'])}x | "
            f"{num(row['activation_load_norm'])}x | {num(row['encoder_output_write_norm'])}x | "
            f"{num(row['reuse_cache_read_norm'])}x | {row['batches']} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Against fixed `W4BFPA4`, HEAT-style compute is only about `0.95x`, because its high-precision `W8A10` key branch offsets the low `W4A2` branch.",
            "- HEAT-style weight loading is not free: even the per-node lower bound is `1.10x` of W4 weight traffic, and split precision streams are slightly higher after batch rounding.",
            "- TSER40 keeps the same fixed W4BFPA4 miss path, but executes it for only the miss nodes. Its compute, activation load, and lower-bound weight load scale with the miss rate.",
            "- The decisive comparison is therefore not HEAT bit-plane count alone; it is component timing under the same fixed W4BFPA4 baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse_tsv", default=str(DEFAULT_REUSE_TSV))
    parser.add_argument("--default_reuse", type=float, default=0.40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_weight_bits", type=int, default=4)
    parser.add_argument("--local_activation_bits", type=int, default=4)
    parser.add_argument("--heat_alpha", type=float, default=0.1)
    parser.add_argument("--heat_hi_weight_bits", type=int, default=8)
    parser.add_argument("--heat_hi_activation_bits", type=int, default=10)
    parser.add_argument("--heat_lo_weight_bits", type=int, default=4)
    parser.add_argument("--heat_lo_activation_bits", type=int, default=2)
    parser.add_argument("--compute_share", type=float, default=0.70)
    parser.add_argument("--weight_share", type=float, default=0.20)
    parser.add_argument("--activation_share", type=float, default=0.05)
    parser.add_argument("--output_share", type=float, default=0.04)
    parser.add_argument("--control_share", type=float, default=0.01)
    parser.add_argument("--cache_read_scale", type=float, default=0.10)
    parser.add_argument(
        "--output_dir",
        default=str(OFA_ROOT / "output" / "heat_frontend_timing_w4bfpa4"),
    )
    parser.add_argument(
        "--repo_report",
        default=str(REPO_ROOT / "HEAT" / "results" / "FRONTEND_TIMING_VS_HEAT.md"),
    )
    args = parser.parse_args()

    rows = build_rows(args)
    aggregate = aggregate_rows(rows, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "frontend_rows.tsv", rows)
    write_tsv(output_dir / "frontend_aggregate.tsv", aggregate)
    (output_dir / "frontend_timing.json").write_text(
        json.dumps({"config": vars(args), "rows": rows, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    report = render_markdown(rows, aggregate, args)
    (output_dir / "FRONTEND_TIMING_VS_HEAT.md").write_text(report, encoding="utf-8")
    repo_report = Path(args.repo_report)
    repo_report.parent.mkdir(parents=True, exist_ok=True)
    repo_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
