#!/usr/bin/env python3
"""Build Graph-Bit nodewise bound policies from a LLaMA W-tile profile."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LLAMA7B_KIND_WEIGHTS = {
    # Approximate per-layer MAC share for LLaMA-2-7B linear layers.
    # q/k/v/o: 4096 x 4096
    # gate/up/down: 4096 x 11008 or 11008 x 4096
    "self_attn.q_proj": 4096 * 4096,
    "self_attn.k_proj": 4096 * 4096,
    "self_attn.v_proj": 4096 * 4096,
    "self_attn.o_proj": 4096 * 4096,
    "mlp.gate_proj": 4096 * 11008,
    "mlp.up_proj": 4096 * 11008,
    "mlp.down_proj": 11008 * 4096,
}


def read_global_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_row(rows: list[dict[str, str]], scope: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row.get("scope") == scope and row.get("metric") == metric:
            return row
    raise KeyError(f"Missing scope={scope} metric={metric}")


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def weighted_kind_strength(rows: list[dict[str, str]], quantile_key: str) -> float:
    total_weight = 0.0
    weighted = 0.0
    for kind, weight in LLAMA7B_KIND_WEIGHTS.items():
        row = find_row(rows, kind, "strength_p95")
        total_weight += float(weight)
        weighted += float(weight) * f(row, quantile_key)
    return weighted / max(total_weight, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile_dir", default="output/graphbit_w_tile_strength/llama2_7b_k128_n128")
    parser.add_argument("--min_depth", type=int, default=4)
    parser.add_argument("--min_tol", type=float, default=0.0)
    parser.add_argument("--max_tol", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--risk_max", type=float, default=15.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    rows = read_global_summary(profile_dir / "global_summary.tsv")
    global_row = find_row(rows, "all", "strength_p95")

    specs: list[tuple[str, float]] = [
        ("now_no_w", 1.0),
        ("global_p75", f(global_row, "p75")),
        ("global_p90", f(global_row, "p90")),
        ("global_p95", f(global_row, "p95")),
        ("module_p75", weighted_kind_strength(rows, "p75")),
        ("module_p90", weighted_kind_strength(rows, "p90")),
        ("module_p95", weighted_kind_strength(rows, "p95")),
    ]

    policies = [
        (
            f"{name}:{args.min_depth}:{args.min_tol}:{args.max_tol}:"
            f"{args.gamma}:{args.risk_max}:{args.scale}:{strength:.6f}"
        )
        for name, strength in specs
    ]
    text = "\n".join(policies)

    summary = {
        "profile_dir": str(profile_dir),
        "policy_format": "id:min_depth:min_tol:max_tol:gamma:risk_max:scale:w_strength",
        "settings": {
            "min_depth": args.min_depth,
            "min_tol": args.min_tol,
            "max_tol": args.max_tol,
            "gamma": args.gamma,
            "risk_max": args.risk_max,
            "scale": args.scale,
        },
        "policies": {name: strength for name, strength in specs},
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
