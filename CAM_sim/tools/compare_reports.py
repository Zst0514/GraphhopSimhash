#!/usr/bin/env python3
"""Compare digital and analog CAM hash-reuse simulator reports."""

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


def winner(left: dict, right: dict, key: str) -> str:
    lv = float(left.get(key, 0.0))
    rv = float(right.get(key, 0.0))
    if abs(lv - rv) <= max(abs(lv), abs(rv), 1.0) * 1e-9:
        return "tie"
    return left["implementation"] if lv < rv else right["implementation"]


def compare_decisions(left: dict, right: dict) -> tuple[int, int, int]:
    left_path = left.get("decision_path")
    right_path = right.get("decision_path")
    if not left_path or not right_path or not Path(left_path).exists() or not Path(right_path).exists():
        return -1, -1, -1
    left_rows = load_decisions(left_path)
    right_rows = load_decisions(right_path)
    total = min(len(left_rows), len(right_rows))
    strict_mismatches = 0
    functional_mismatches = 0
    for idx in range(total):
        lhs = left_rows[idx]
        rhs = right_rows[idx]
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
    size_delta = abs(len(left_rows) - len(right_rows))
    strict_mismatches += size_delta
    functional_mismatches += size_delta
    return strict_mismatches, functional_mismatches, max(len(left_rows), len(right_rows))


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two hardware simulator reports.")
    parser.add_argument("digital_report")
    parser.add_argument("analog_report")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    digital = load_json(args.digital_report)
    analog = load_json(args.analog_report)
    strict_mismatch_count, functional_mismatch_count, decision_count = compare_decisions(digital, analog)

    latency_winner = winner(digital, analog, "cycles_per_query")
    energy_winner = winner(digital, analog, "energy_per_query_pj")
    edp_winner = winner(digital, analog, "edp_pj_cycle_per_query")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        out.write("# 8h16b3vote Hardware Comparison\n\n")
        out.write("| Metric | Digital Logic | Analog CAM |\n")
        out.write("|---|---:|---:|\n")
        out.write(f"| Reuse | {format_pct(float(digital['reuse_rate']))} | {format_pct(float(analog['reuse_rate']))} |\n")
        out.write(f"| Reuse n/d | {digital['reuse']}/{digital['total_queries']} | {analog['reuse']}/{analog['total_queries']} |\n")
        out.write(f"| Exact reuse | {digital['exact_reuse']} | {analog['exact_reuse']} |\n")
        out.write(f"| Fuzzy reuse | {digital['fuzzy_reuse']} | {analog['fuzzy_reuse']} |\n")
        out.write(f"| Cycles/query | {float(digital['cycles_per_query']):.3f} | {float(analog['cycles_per_query']):.3f} |\n")
        out.write(f"| Search cycles/query | {float(digital.get('search_cycles_per_query', 0.0)):.3f} | {float(analog.get('search_cycles_per_query', 0.0)):.3f} |\n")
        out.write(f"| Verify cycles/query | {float(digital.get('verify_cycles_per_query', 0.0)):.3f} | {float(analog.get('verify_cycles_per_query', 0.0)):.3f} |\n")
        out.write(f"| Verified rows/query | {float(digital.get('verified_rows_per_query', 0.0)):.3f} | {float(analog.get('verified_rows_per_query', 0.0)):.3f} |\n")
        out.write(f"| Throughput qps | {float(digital['throughput_qps']):.2f} | {float(analog['throughput_qps']):.2f} |\n")
        out.write(f"| Energy/query pJ | {float(digital['energy_per_query_pj']):.6f} | {float(analog['energy_per_query_pj']):.6f} |\n")
        out.write(f"| EDP pJ*cycle/query | {float(digital['edp_pj_cycle_per_query']):.6f} | {float(analog['edp_pj_cycle_per_query']):.6f} |\n")
        out.write(f"| Area proxy um2 | {float(digital['area_proxy_um2']):.2f} | {float(analog['area_proxy_um2']):.2f} |\n")
        out.write(f"| Candidate inserts | {digital['candidate_inserts']} | {analog['candidate_inserts']} |\n")
        out.write(f"| Candidate overflows | {digital['candidate_overflows']} | {analog['candidate_overflows']} |\n\n")

        out.write("## Winners\n\n")
        out.write(f"- latency_winner: `{latency_winner}`\n")
        out.write(f"- energy_winner: `{energy_winner}`\n")
        out.write(f"- edp_winner: `{edp_winner}`\n")
        out.write(f"- analog_calibration: `{analog.get('calibration', 'unknown')}`\n")
        if strict_mismatch_count >= 0:
            out.write(f"- decision_mismatch_strict: `{strict_mismatch_count}/{decision_count}`\n")
            out.write(f"- decision_mismatch_functional: `{functional_mismatch_count}/{decision_count}`\n")
        else:
            out.write("- decision_mismatch: `not checked; decision CSV missing`\n")

        out.write("\n## Interpretation\n\n")
        out.write(
            "Digital Logic uses ordinary CAM chunk coarse filtering plus XOR/popcount verification. "
            "It is usually better when the coarse filter is selective enough that the survivor set stays small.\n\n"
        )
        out.write(
            "Analog CAM uses threshold search across active rows. It can become attractive when fuzzy enumeration "
            "or large candidate spaces dominate, but the current result is a proxy unless calibrated by CAMASim/EvaCAM.\n"
        )

    print(f"[compare] wrote {out_path}")


if __name__ == "__main__":
    main()
