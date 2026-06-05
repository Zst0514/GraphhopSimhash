#!/usr/bin/env python3
"""Compare baseline and capacity-limited CAM simulator reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_decisions(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def resolve_existing_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    remapped = Path(str(candidate).replace("hardware_8h16b3vote", "CAM_sim"))
    if remapped.exists():
        return remapped
    return None


def compare_decisions(baseline: dict, limited: dict) -> tuple[int, int, int]:
    base_path = resolve_existing_path(baseline.get("decision_path"))
    limited_path = resolve_existing_path(limited.get("decision_path"))
    if base_path is None or limited_path is None:
        return -1, -1, -1

    base_rows = load_decisions(str(base_path))
    limited_rows = load_decisions(str(limited_path))
    total = min(len(base_rows), len(limited_rows))
    strict_mismatches = 0
    functional_mismatches = 0
    for idx in range(total):
        lhs = base_rows[idx]
        rhs = limited_rows[idx]
        if (
            lhs.get("node_id") != rhs.get("node_id")
            or lhs.get("hit") != rhs.get("hit")
            or lhs.get("source_id") != rhs.get("source_id")
            or lhs.get("support") != rhs.get("support")
            or lhs.get("min_dist") != rhs.get("min_dist")
            or lhs.get("kind") != rhs.get("kind")
        ):
            strict_mismatches += 1
        if (
            lhs.get("node_id") != rhs.get("node_id")
            or lhs.get("hit") != rhs.get("hit")
            or lhs.get("source_id") != rhs.get("source_id")
        ):
            functional_mismatches += 1

    size_delta = abs(len(base_rows) - len(limited_rows))
    strict_mismatches += size_delta
    functional_mismatches += size_delta
    return strict_mismatches, functional_mismatches, max(len(base_rows), len(limited_rows))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_pp_delta(baseline: float, limited: float) -> str:
    return f"{100.0 * (limited - baseline):+.2f} pp"


def fmt_num_delta(baseline: float, limited: float, digits: int = 0) -> str:
    if digits == 0:
        return f"{limited - baseline:+.0f}"
    return f"{limited - baseline:+.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and capacity-limited reports.")
    parser.add_argument("baseline_report")
    parser.add_argument("limited_report")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline_report)
    limited = load_json(args.limited_report)
    strict_mismatch_count, functional_mismatch_count, decision_count = compare_decisions(baseline, limited)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        out.write("# Capacity-Limited CAM Impact\n\n")
        out.write("| Metric | Baseline | Limited | Delta |\n")
        out.write("|---|---:|---:|---:|\n")
        out.write(
            f"| Replacement policy | `{baseline.get('replacement_policy', 'unknown')}` | "
            f"`{limited.get('replacement_policy', 'unknown')}` | - |\n"
        )
        out.write(
            f"| Capacity nodes | {baseline.get('capacity_limit_nodes', 0)} | "
            f"{limited.get('capacity_limit_nodes', 0)} | "
            f"{fmt_num_delta(float(baseline.get('capacity_limit_nodes', 0)), float(limited.get('capacity_limit_nodes', 0)))} |\n"
        )
        out.write(
            f"| Total CAM bytes | {baseline.get('total_cam_bytes', 0)} | {limited.get('total_cam_bytes', 0)} | "
            f"{fmt_num_delta(float(baseline.get('total_cam_bytes', 0)), float(limited.get('total_cam_bytes', 0)))} |\n"
        )
        out.write(
            f"| Node entry bytes | {baseline.get('node_entry_bytes', 0)} | {limited.get('node_entry_bytes', 0)} | "
            f"{fmt_num_delta(float(baseline.get('node_entry_bytes', 0)), float(limited.get('node_entry_bytes', 0)))} |\n"
        )
        out.write(
            f"| Reuse rate | {fmt_pct(float(baseline['reuse_rate']))} | {fmt_pct(float(limited['reuse_rate']))} | "
            f"{fmt_pp_delta(float(baseline['reuse_rate']), float(limited['reuse_rate']))} |\n"
        )
        out.write(
            f"| Reuse count | {baseline['reuse']} | {limited['reuse']} | "
            f"{fmt_num_delta(float(baseline['reuse']), float(limited['reuse']))} |\n"
        )
        out.write(
            f"| Computed count | {baseline['computed']} | {limited['computed']} | "
            f"{fmt_num_delta(float(baseline['computed']), float(limited['computed']))} |\n"
        )
        out.write(
            f"| Candidate overflows | {baseline.get('candidate_overflows', 0)} | {limited.get('candidate_overflows', 0)} | "
            f"{fmt_num_delta(float(baseline.get('candidate_overflows', 0)), float(limited.get('candidate_overflows', 0)))} |\n"
        )
        out.write(
            f"| CAM evictions | {baseline.get('cam_evictions', 0)} | {limited.get('cam_evictions', 0)} | "
            f"{fmt_num_delta(float(baseline.get('cam_evictions', 0)), float(limited.get('cam_evictions', 0)))} |\n"
        )
        out.write(
            f"| Max active nodes | {baseline.get('max_active_nodes', 0)} | {limited.get('max_active_nodes', 0)} | "
            f"{fmt_num_delta(float(baseline.get('max_active_nodes', 0)), float(limited.get('max_active_nodes', 0)))} |\n"
        )
        out.write(
            f"| Max active rows | {baseline.get('max_active_rows', 0)} | {limited.get('max_active_rows', 0)} | "
            f"{fmt_num_delta(float(baseline.get('max_active_rows', 0)), float(limited.get('max_active_rows', 0)))} |\n"
        )
        out.write(
            f"| Cycles/query | {float(baseline['cycles_per_query']):.3f} | {float(limited['cycles_per_query']):.3f} | "
            f"{fmt_num_delta(float(baseline['cycles_per_query']), float(limited['cycles_per_query']), 3)} |\n"
        )
        out.write(
            f"| Verified rows/query | {float(baseline.get('verified_rows_per_query', 0.0)):.3f} | "
            f"{float(limited.get('verified_rows_per_query', 0.0)):.3f} | "
            f"{fmt_num_delta(float(baseline.get('verified_rows_per_query', 0.0)), float(limited.get('verified_rows_per_query', 0.0)), 3)} |\n"
        )

        out.write("\n## Decision Drift\n\n")
        if strict_mismatch_count >= 0:
            out.write(f"- strict_mismatch: `{strict_mismatch_count}/{decision_count}`\n")
            out.write(f"- functional_mismatch: `{functional_mismatch_count}/{decision_count}`\n")
            out.write(
                f"- functional_mismatch_rate: "
                f"`{(100.0 * functional_mismatch_count / decision_count) if decision_count else 0.0:.4f}%`\n"
            )
        else:
            out.write("- decision_mismatch: `not checked; decision CSV missing`\n")

    print(f"[capacity-compare] wrote {out_path}")


if __name__ == "__main__":
    main()
