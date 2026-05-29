#!/usr/bin/env python3
"""Export residual-reuse + Graph-Bit workload profiles.

The simulator side should not parse long experiment logs directly.  This script
normalizes the current TSV summaries into a small JSON profile:

    direct reuse ratio
    residual reuse ratio
    miss-node P8/P6/P5/P4 ratios

The ratios are fractions of all nodes, so they can be directly multiplied by an
encoder cost model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONFIG_ALIASES = {
    "RandomDepthBudget": "Rand",
    "DegreeDepthBudget": "Deg",
    "TSERDepthBudget": "TSER",
    "ContextDepthBudget": "Ctx",
    "LowUniqueDepthBudget": "Uniq",
    "PredictorDepthBudget": "Pred",
    "OracleDamageBudget": "Oracle",
}


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_percent(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    number = float(text)
    return number / 100.0 if abs(number) > 1.0 else number


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    if text.endswith("%"):
        return float(text[:-1])
    return float(text)


def normalize_config(name: str) -> str:
    return CONFIG_ALIASES.get(name, name)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def row_matches(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.dataset and row.get("dataset") != args.dataset:
        return False
    if args.config and normalize_config(row.get("config", "")) != normalize_config(args.config):
        return False
    if args.budget and row.get("budget") != args.budget:
        return False
    if args.heads and row.get("heads") != args.heads:
        return False
    if args.threshold is not None and str(row.get("T", row.get("threshold", ""))) != str(args.threshold):
        return False
    if args.runs is not None and str(row.get("runs", "")) != str(args.runs):
        return False
    return True


def profile_from_row(row: dict[str, str], source: Path) -> dict[str, Any]:
    config = normalize_config(row.get("config", ""))
    p8 = parse_percent(row.get("P8", row.get("P8%", 0.0)))
    p6 = parse_percent(row.get("P6", row.get("P6%", 0.0)))
    p5 = parse_percent(row.get("P5", row.get("P5%", 0.0)))
    p4 = parse_percent(row.get("P4", row.get("P4%", 0.0)))
    direct = parse_percent(row.get("direct", row.get("Dir", 0.0)))
    residual = parse_percent(row.get("residual", row.get("Res", 0.0)))
    reuse = parse_percent(row.get("reuse", row.get("Reuse", direct + residual)))
    miss = max(0.0, 1.0 - reuse)

    return {
        "id": "_".join(
            str(part)
            for part in [
                row.get("dataset", "dataset"),
                row.get("heads", "h?"),
                f"T{row.get('T', row.get('threshold', 'x'))}",
                row.get("budget", "budget"),
                config,
            ]
            if part not in ("", None)
        ),
        "dataset": row.get("dataset", ""),
        "model": "llama2_7b",
        "source": str(source),
        "route": {
            "heads": row.get("heads", ""),
            "threshold": row.get("T", row.get("threshold", "")),
            "budget": row.get("budget", ""),
            "config": config,
        },
        "ratios": {
            "reuse": reuse,
            "direct": direct,
            "residual": residual,
            "miss": miss,
            "p8": p8,
            "p6": p6,
            "p5": p5,
            "p4": p4,
        },
        "metrics": {
            "cost": parse_float(row.get("cost"), None),
            "acc": parse_float(row.get("acc"), None),
            "drop_percent": parse_float(row.get("drop"), None),
            "finalerr": parse_float(row.get("finalerr"), None),
        },
    }


def main() -> None:
    root = ofa_root()
    default_source = root / "output" / "residual_graphbit_three_depth_probe" / "three_depth_summary.tsv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dataset", choices=["cora", "pubmed", "arxiv"], default=None)
    parser.add_argument("--config", default=None, help="Example: Deg, TSER, FullP8")
    parser.add_argument("--budget", default=None, help="Example: balanced, conservative, aggressive")
    parser.add_argument("--heads", default=None, help="Example: h4 or h8")
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    rows = [row for row in read_tsv(args.source) if row_matches(row, args)]
    if not rows:
        raise SystemExit(f"No rows matched filters in {args.source}")

    profiles = [profile_from_row(row, args.source) for row in rows]
    payload = {
        "schema": "graphbit_workload_profile.v1",
        "description": "Ratios are fractions of all graph nodes.",
        "profiles": profiles,
    }

    if args.output is None:
        out_dir = root / "output" / "onnxim_graphbit" / "workloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "profiles"
        if len(profiles) == 1:
            stem = profiles[0]["id"]
        args.output = out_dir / f"{stem}.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2 if args.pretty else None)
        handle.write("\n")

    print(f"[WorkloadProfile] wrote {args.output} | profiles={len(profiles)}")
    for profile in profiles:
        ratios = profile["ratios"]
        print(
            "[WorkloadProfile] "
            f"{profile['id']} | reuse={ratios['reuse']:.3f} "
            f"direct={ratios['direct']:.3f} residual={ratios['residual']:.3f} "
            f"P8={ratios['p8']:.3f} P6={ratios['p6']:.3f} "
            f"P5={ratios['p5']:.3f} P4={ratios['p4']:.3f}"
        )


if __name__ == "__main__":
    main()
