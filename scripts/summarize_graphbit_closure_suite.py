#!/usr/bin/env python3
"""Summarize the Cora Graph-Bit closure evidence chain.

The goal is to put the key hardware ablations in one table:

1. FullP8-miss baseline.
2. Degree compute-mask only.
3. Degree random-mixed demand-fetch.
4. Degree risk-bucket demand-fetch.

This table is intentionally hardware-facing.  Accuracy/drop is inherited from
the corresponding embedding proxy profile, while cycles/traffic/energy come
from the demand-fetch model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEEP_METHODS = [
    "FullP8-miss",
    "Degree compute-mask only",
    "Degree random-mixed",
    "Degree demand-fetch",
]


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_rows(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    args = payload["args"]
    workload = json.loads(Path(args["workload"]).read_text())
    profile = workload["profiles"][0]
    frontend = str(profile["route"]["frontend"])
    budget = str(profile["route"]["budget"])
    return frontend, budget, rows


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def row_id(frontend: str, row: dict[str, Any]) -> str:
    return f"{frontend}:{row['method']}"


def write_outputs(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "closure_table.tsv"
    txt_path = out_dir / "closure_table.txt"
    json_path = out_dir / "closure_table.json"

    fieldnames = [
        "frontend",
        "budget",
        "method",
        "schedule",
        "dataflow",
        "reuse",
        "miss",
        "useful_depth",
        "executed_depth",
        "bit_util",
        "miss_bitcomp",
        "miss_act_read",
        "miss_weight_read",
        "miss_out_write",
        "full_cycles",
        "full_traffic",
        "energy_proxy",
        "drop",
        "cycles_save_vs_fullp8",
        "traffic_save_vs_fullp8",
        "extra_drop_vs_fullp8",
    ]
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "Cora Graph-Bit closure table",
        "FullP8-miss is the baseline within each frontend. Lower FullC/FullT is better; lower Drop is better.",
        "",
        "Frontend       Budget   Method                       Sched        Dataflow      Reuse  UsefulD ExecD Util   BitC  ActRd WgtRd FullC FullT Energy Drop  SaveC SaveT +Drop",
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------",
    ]
    for r in records:
        lines.append(
            f"{r['frontend']:<14} {r['budget']:<8} {r['method']:<28} "
            f"{r['schedule']:<12} {r['dataflow']:<13} "
            f"{pct(r['reuse']):>5} {r['useful_depth']:>7.2f} {r['executed_depth']:>5.2f} "
            f"{pct(r['bit_util']):>5} {r['miss_bitcomp']:>6.3f} {r['miss_act_read']:>6.3f} "
            f"{r['miss_weight_read']:>5.3f} {r['full_cycles']:>5.3f} {r['full_traffic']:>5.3f} "
            f"{r['energy_proxy']:>6.3f} {r['drop']:>4.2f}% "
            f"{pct(r['cycles_save_vs_fullp8']):>5} {pct(r['traffic_save_vs_fullp8']):>5} "
            f"{r['extra_drop_vs_fullp8']:>5.2f}%"
        )

    lines.extend(
        [
            "",
            "Reading guide:",
            "1. compute-mask only isolates arithmetic masking without activation demand-fetch.",
            "2. random-mixed isolates the batching problem: useful depth can be low while executed depth returns to P8.",
            "3. demand-fetch + risk-bucket is the actual Graph-Bit NPU target.",
            "4. p8heavy is accuracy-first; balanced exposes the hardware mechanism more clearly.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"[GraphBitClosure] wrote {tsv_path}")
    print(f"[GraphBitClosure] wrote {txt_path}")
    print(f"[GraphBitClosure] wrote {json_path}")
    print(txt_path.read_text())


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demand-json",
        type=Path,
        nargs="+",
        default=[
            root / "output" / "graphbit_predictor_free" / "cora_h8_53_T30" / "demand_fetch_model" / "demand_fetch_model.json",
            root / "output" / "graphbit_predictor_free" / "cora_h8_54_T40" / "demand_fetch_model" / "demand_fetch_model.json",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "graphbit_closure" / "cora")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.demand_json:
        frontend, budget, rows = load_rows(path)
        by_method = {row["method"]: row for row in rows}
        base = by_method["FullP8-miss"]
        for method in KEEP_METHODS:
            row = dict(by_method[method])
            row["frontend"] = frontend
            row["budget"] = budget
            row["cycles_save_vs_fullp8"] = 1.0 - row["full_cycles"] / base["full_cycles"]
            row["traffic_save_vs_fullp8"] = 1.0 - row["full_traffic"] / base["full_traffic"]
            row["extra_drop_vs_fullp8"] = row["drop"] - base["drop"]
            row["id"] = row_id(frontend, row)
            records.append(row)

    write_outputs(records, args.output_dir)


if __name__ == "__main__":
    main()
