#!/usr/bin/env python3
"""Compose residual/reuse workload, ONNXim traces, and bucket feasibility.

The output table is normalized to a full graph where every node executes the
FullP8 encoder.  Reuse/residual hits are treated as cache/adapter-side paths
and do not enter the encoder.  Miss nodes are weighted by the real P8/P6/P5/P4
ratios from the workload profile and by ONNXim component traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEPTH_KEYS = ("p8", "p6", "p5", "p4")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing json: {path}")
    return json.loads(path.read_text())


def profile_text(profile: dict[str, Any]) -> str:
    route = profile.get("route", {}) or {}
    parts = [str(profile.get("id", ""))]
    for key in ("method", "config", "budget", "frontend", "heads", "threshold"):
        parts.append(str(route.get(key, "")))
    return " ".join(parts).lower()


def select_profile(workload: dict[str, Any], match: str) -> dict[str, Any]:
    needle = match.lower()
    for profile in workload.get("profiles", []):
        if needle in profile_text(profile):
            return profile
    raise SystemExit(f"No profile matching '{match}'")


def load_encoder(root: Path, case: str) -> dict[str, Any]:
    return load_json(root / case / "aggregate.json")["encoder"]


def norm(row: dict[str, Any], base: dict[str, Any], key: str) -> float:
    den = float(base.get(key, 0.0) or 0.0)
    if den == 0.0:
        return 0.0
    return float(row.get(key, 0.0) or 0.0) / den


def traffic(row: dict[str, Any]) -> float:
    return float(row.get("dram_read_requests", 0.0) or 0.0) + float(
        row.get("dram_write_requests", 0.0) or 0.0
    )


def traffic_norm(row: dict[str, Any], base: dict[str, Any]) -> float:
    den = traffic(base)
    return 0.0 if den == 0.0 else traffic(row) / den


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_hist(hist: dict[str, float]) -> str:
    if not hist:
        return "-"
    return ",".join(f"D{k}:{100.0 * v:.1f}%" for k, v in sorted(hist.items(), key=lambda x: int(x[0])))


def slug(text: str) -> str:
    text = text.strip() or "unknown"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def hist_share(row: dict[str, Any]) -> dict[str, float]:
    hist = row.get("graphbit_effective_depth_hist") or {}
    total = sum(float(v) for v in hist.values())
    if total <= 0.0:
        depth = row.get("graphbit_avg_depth")
        if depth:
            return {str(int(round(float(depth)))): 1.0}
        return {}
    return {str(k): float(v) / total for k, v in hist.items()}


def load_feasibility(path: Path, dataset: str, profile_match: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("dataset", "").lower() != dataset.lower():
                continue
            if profile_match.lower() not in row.get("profile", "").lower():
                continue
            rows[int(row["candidate_batch"])] = row
    return rows


def combine_components(
    ratios: dict[str, float],
    components: dict[str, dict[str, Any]],
    base: dict[str, Any],
    suffix: str,
) -> dict[str, Any]:
    cycles = 0.0
    traff = 0.0
    depth_sum = 0.0
    fetch_sum = 0.0
    issue_sum = 0.0
    hist: dict[str, float] = {}
    miss = sum(ratios.values())

    for depth_key in DEPTH_KEYS:
        ratio = ratios.get(depth_key, 0.0)
        if ratio <= 0.0:
            continue
        row = components[f"{depth_key}_{suffix}"]
        cycles += ratio * norm(row, base, "cycles")
        traff += ratio * traffic_norm(row, base)
        depth_sum += ratio * float(row.get("graphbit_avg_depth", 8.0) or 8.0)
        fetch_sum += ratio * float(row.get("graphbit_avg_fetch_depth", 8.0) or 8.0)
        issue_sum += ratio * float(row.get("graphbit_avg_issue_depth", 8.0) or 8.0)
        for depth, share in hist_share(row).items():
            hist[depth] = hist.get(depth, 0.0) + ratio * share

    if miss > 0.0:
        hist = {depth: value / miss for depth, value in hist.items()}
    energy = 0.5 * cycles + 0.5 * traff
    return {
        "cycles_norm": cycles,
        "traffic_norm": traff,
        "energy_norm": energy,
        "avg_depth": depth_sum / miss if miss else 0.0,
        "avg_fetch_depth": fetch_sum / miss if miss else 0.0,
        "avg_issue_depth": issue_sum / miss if miss else 0.0,
        "depth_hist": hist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-json", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-profile-match", default="fullp8-miss")
    parser.add_argument("--graphbit-profile-match", default="degree_runtime-bound")
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--feasibility-tsv", type=Path, required=True)
    parser.add_argument("--feasibility-profile-match", default="degree_runtime-bound")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional output file prefix. Defaults to dataset + workload frontend.",
    )
    args = parser.parse_args()

    workload = load_json(args.workload_json)
    full_profile = select_profile(workload, args.full_profile_match)
    graphbit_profile = select_profile(workload, args.graphbit_profile_match)
    full_ratios = full_profile.get("ratios", {}) or {}
    gb_ratios = graphbit_profile.get("ratios", {}) or {}
    reuse = float(gb_ratios.get("reuse", 0.0) or 0.0)
    direct = float(gb_ratios.get("direct", 0.0) or 0.0)
    residual = float(gb_ratios.get("residual", 0.0) or 0.0)
    depth_ratios = {key: float(gb_ratios.get(key, 0.0) or 0.0) for key in DEPTH_KEYS}
    miss = sum(depth_ratios.values())

    base = load_encoder(args.components_root, "full_p8")
    components: dict[str, dict[str, Any]] = {}
    for key in DEPTH_KEYS:
        components[f"{key}_now"] = load_encoder(args.components_root, f"{key}_now")
        for batch in (32, 64):
            components[f"{key}_ws_b{batch}"] = load_encoder(
                args.components_root, f"{key}_ws_b{batch}"
            )

    feasibility = load_feasibility(
        args.feasibility_tsv, args.dataset, args.feasibility_profile_match
    )

    rows: list[dict[str, Any]] = []
    fullp8_drop = float((full_profile.get("metrics", {}) or {}).get("drop_percent", 0.0) or 0.0)
    graphbit_drop = float(
        (graphbit_profile.get("metrics", {}) or {}).get("drop_percent", 0.0) or 0.0
    )

    rows.append(
        {
            "method": "FullP8-miss",
            "reuse": reuse,
            "direct": direct,
            "residual": residual,
            "miss": float(full_ratios.get("p8", miss) or miss),
            "cycles_norm": miss,
            "traffic_norm": miss,
            "energy_norm": miss,
            "drop_percent": fullp8_drop,
            "avg_depth": 8.0,
            "avg_fetch_depth": 8.0,
            "avg_issue_depth": 8.0,
            "depth_hist": {"8": 1.0},
            "wscale": 1.0,
            "sram_fit": "yes",
            "note": "all miss nodes execute FullP8",
        }
    )

    for method, suffix, batch in (
        ("GraphBit-now", "now", 16),
        ("GraphBit-bucket32", "ws_b32", 32),
        ("GraphBit-bucket64", "ws_b64", 64),
    ):
        combined = combine_components(depth_ratios, components, base, suffix)
        feasible = feasibility.get(batch, {})
        rows.append(
            {
                "method": method,
                "reuse": reuse,
                "direct": direct,
                "residual": residual,
                "miss": miss,
                "cycles_norm": combined["cycles_norm"],
                "traffic_norm": combined["traffic_norm"],
                "energy_norm": combined["energy_norm"],
                "drop_percent": graphbit_drop,
                "avg_depth": combined["avg_depth"],
                "avg_fetch_depth": combined["avg_fetch_depth"],
                "avg_issue_depth": combined["avg_issue_depth"],
                "depth_hist": combined["depth_hist"],
                "wscale": float(feasible.get("w_hbm_scale", 1.0) or 1.0),
                "sram_fit": "yes" if feasible.get("sram_fit", "1") in ("1", "yes", "true") else "no",
                "note": "runtime bound + same-risk W tile scheduling",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frontend = str((graphbit_profile.get("route", {}) or {}).get("frontend", "unknown"))
    prefix = slug(args.output_prefix) if args.output_prefix else f"{slug(args.dataset)}_{slug(frontend)}"
    tsv_path = args.output_dir / f"{prefix}_fullstack_hardware.tsv"
    txt_path = args.output_dir / f"{prefix}_fullstack_hardware.txt"
    json_path = args.output_dir / f"{prefix}_fullstack_hardware.json"

    serializable = []
    for row in rows:
        copied = dict(row)
        copied["depth_hist"] = dict(row["depth_hist"])
        serializable.append(copied)
    json_path.write_text(json.dumps({"rows": serializable}, indent=2) + "\n")

    fieldnames = [
        "method",
        "reuse",
        "direct",
        "residual",
        "miss",
        "cycles_norm",
        "traffic_norm",
        "energy_norm",
        "drop_percent",
        "avg_depth",
        "avg_fetch_depth",
        "avg_issue_depth",
        "depth_hist",
        "wscale",
        "sram_fit",
        "note",
    ]
    with tsv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["depth_hist"] = fmt_hist(row["depth_hist"])
            for key in (
                "reuse",
                "direct",
                "residual",
                "miss",
                "cycles_norm",
                "traffic_norm",
                "energy_norm",
                "avg_depth",
                "avg_fetch_depth",
                "avg_issue_depth",
                "wscale",
            ):
                out[key] = f"{float(out[key]):.6f}"
            out["drop_percent"] = f"{float(out['drop_percent']):.3f}"
            writer.writerow(out)

    lines = [
        f"Graph-Bit full-stack hardware table | {args.dataset} | {frontend}",
        f"workload: {args.workload_json}",
        f"components: {args.components_root}",
        "",
        "Norm columns are relative to all graph nodes executing FullP8 encoder.",
        "",
        (
            f"{'Method':<20s} {'Reuse':>7s} {'Miss':>7s} {'Cycles':>8s} "
            f"{'Traffic':>8s} {'Energy':>8s} {'Drop':>7s} {'AvgD':>6s} "
            f"{'Hist(miss)':<22s} {'Wscale':>7s} {'SRAM':>5s}"
        ),
        "-" * 120,
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<20s} {fmt_pct(row['reuse']):>7s} {fmt_pct(row['miss']):>7s} "
            f"{row['cycles_norm']:8.3f} {row['traffic_norm']:8.3f} "
            f"{row['energy_norm']:8.3f} {row['drop_percent']:6.2f}% "
            f"{row['avg_depth']:6.2f} {fmt_hist(row['depth_hist']):<22s} "
            f"{row['wscale']:7.3f} {row['sram_fit']:>5s}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- FullP8-miss keeps reuse/residual front-end fixed and sends every miss node to FullP8.",
            "- GraphBit-now uses predictor-free bit-plane stop and activation demand fetch, without extra W-HBM amortization.",
            "- GraphBit-bucket32/64 additionally use same-risk bucket scheduling when the feasibility model says SRAM and bucket size are sufficient.",
            "- Drop is inherited from the corresponding workload accuracy profile; W-tile scheduling does not change numerical output.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    print(txt_path.read_text())
    print(f"[GraphBitFullStack] wrote {tsv_path}")
    print(f"[GraphBitFullStack] wrote {json_path}")


if __name__ == "__main__":
    main()
