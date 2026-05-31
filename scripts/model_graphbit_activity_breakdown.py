#!/usr/bin/env python3
"""Break Graph-Bit replay rows into bit-depth-sensitive NPU activities.

`replay_graphbit_trace_scheduler.py` gives an ONNXim component-level cycles
table.  That table is useful for W-tile batching, but it currently under-exposes
the activity saved by mixed activation depth because P8/P6/P5 component cycles
are nearly identical.

This script keeps the same replay rows and adds a separate architectural
activity model:

* W_HBM: weight tile loads from HBM, scaled by replayed Wloads/Wscale;
* A_HBM: first activation tile read, unchanged for byte-major activations;
* A_RF: activation RF / issue read, scaled by actual stop depth;
* PE: bit-plane MAC issue, scaled by actual stop depth;
* W_RF: on-chip weight RF / broadcast cycles, scaled by actual stop depth;
* Psum: partial-sum read/update/write, scaled by actual stop depth;
* Out: output write, unchanged for each miss node;
* Sched: optional scheduler overhead already carried by replay rows.

The output is not RTL timing.  It is a bit-depth-sensitive activity breakdown
that answers whether mixed-depth helps beyond W-stationary bucket batching.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ACTIVITY_FIELDS = ("w_hbm", "a_hbm", "a_rf", "pe", "w_rf", "psum", "out", "sched")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing json: {path}")
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_weights(spec: str, defaults: dict[str, float]) -> dict[str, float]:
    weights = dict(defaults)
    if spec:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"Bad weight item '{item}', expected key=value")
            key, value = item.split("=", 1)
            key = key.strip()
            if key not in ACTIVITY_FIELDS:
                raise SystemExit(f"Unknown activity weight key '{key}'")
            weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0.0:
        raise SystemExit("Activity weights sum to zero")
    return {key: value / total for key, value in weights.items()}


def activity_for_row(row: dict[str, Any], *, plane_group_hbm: bool, group_bits: int) -> dict[str, float]:
    miss = as_float(row, "miss")
    avg_depth = as_float(row, "avg_depth", 8.0)
    depth_ratio = max(0.0, min(1.0, avg_depth / 8.0))
    wscale = as_float(row, "wscale", 1.0)
    sched = as_float(row, "scheduler_cycles", 0.0)

    if plane_group_hbm:
        fetch_depth = min(8.0, group_bits * ((avg_depth + group_bits - 1e-9) // group_bits))
        a_hbm = miss * max(0.0, min(1.0, fetch_depth / 8.0))
    else:
        a_hbm = miss

    return {
        "w_hbm": miss * wscale,
        "a_hbm": a_hbm,
        "a_rf": miss * depth_ratio,
        "pe": miss * depth_ratio,
        "w_rf": miss * depth_ratio,
        "psum": miss * depth_ratio,
        "out": miss,
        "sched": sched,
    }


def weighted_sum(activity: dict[str, float], weights: dict[str, float]) -> float:
    return sum(activity.get(key, 0.0) * weights.get(key, 0.0) for key in ACTIVITY_FIELDS)


def build_rows(
    replay: dict[str, Any],
    *,
    cycle_weights: dict[str, float],
    energy_weights: dict[str, float],
    plane_group_hbm: bool,
    group_bits: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in replay["rows"]:
        activity = activity_for_row(row, plane_group_hbm=plane_group_hbm, group_bits=group_bits)
        out.append(
            {
                "method": row["method"],
                "reuse": as_float(row, "reuse"),
                "miss": as_float(row, "miss"),
                "drop": as_float(row, "drop"),
                "avg_depth": as_float(row, "avg_depth"),
                "wloads": int(row.get("wloads", 0) or 0),
                "wscale": as_float(row, "wscale"),
                "onnx_cycles": as_float(row, "cycles"),
                "onnx_traffic": as_float(row, "traffic"),
                "activity_cycles": weighted_sum(activity, cycle_weights),
                "activity_energy": weighted_sum(activity, energy_weights),
                **activity,
            }
        )
    return out


def row_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method"]): row for row in rows}


def improvement(new: float, base: float) -> float:
    return 0.0 if abs(base) < 1e-12 else 1.0 - new / base


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    replay_path: Path,
    cycle_weights: dict[str, float],
    energy_weights: dict[str, float],
    plane_group_hbm: bool,
    group_bits: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "graphbit_activity_breakdown.tsv"
    txt_path = output_dir / "graphbit_activity_breakdown.txt"
    json_path = output_dir / "graphbit_activity_breakdown.json"

    fieldnames = [
        "method",
        "reuse",
        "miss",
        "drop",
        "avg_depth",
        "wloads",
        "wscale",
        "onnx_cycles",
        "onnx_traffic",
        "activity_cycles",
        "activity_energy",
        *ACTIVITY_FIELDS,
    ]
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    by_method = row_by_method(rows)
    lines = [
        "Graph-Bit bit-depth-sensitive activity breakdown",
        f"replay: {replay_path}",
        f"plane_group_hbm={plane_group_hbm} | group_bits={group_bits}",
        "cycle_weights=" + ", ".join(f"{k}={v:.3f}" for k, v in cycle_weights.items()),
        "energy_weights=" + ", ".join(f"{k}={v:.3f}" for k, v in energy_weights.items()),
        "",
        (
            f"{'Method':<20s} {'Drop':>7s} {'AvgD':>6s} {'Wscale':>7s} "
            f"{'ONNX-C':>7s} {'Act-C':>7s} {'Act-E':>7s} "
            f"{'W_HBM':>7s} {'A_HBM':>7s} {'A_RF':>7s} {'PE':>7s} "
            f"{'W_RF':>7s} {'Psum':>7s} {'Out':>7s}"
        ),
        "-" * 128,
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<20s} {row['drop']:6.2f}% {row['avg_depth']:6.2f} "
            f"{row['wscale']:7.3f} {row['onnx_cycles']:7.3f} "
            f"{row['activity_cycles']:7.3f} {row['activity_energy']:7.3f} "
            f"{row['w_hbm']:7.3f} {row['a_hbm']:7.3f} {row['a_rf']:7.3f} "
            f"{row['pe']:7.3f} {row['w_rf']:7.3f} {row['psum']:7.3f} {row['out']:7.3f}"
        )

    lines.extend(["", "Mixed-depth contribution over FullP8-bucket:", ""])
    for batch in (32, 64, 128):
        full_key = f"FullP8-bucket-b{batch}"
        risk_key = f"RiskBucket-b{batch}"
        if full_key not in by_method or risk_key not in by_method:
            continue
        full = by_method[full_key]
        risk = by_method[risk_key]
        lines.append(
            f"b{batch}: "
            f"ONNX-C {pct(improvement(risk['onnx_cycles'], full['onnx_cycles']))}, "
            f"Act-C {pct(improvement(risk['activity_cycles'], full['activity_cycles']))}, "
            f"Act-E {pct(improvement(risk['activity_energy'], full['activity_energy']))}, "
            f"PE {pct(improvement(risk['pe'], full['pe']))}, "
            f"W_RF {pct(improvement(risk['w_rf'], full['w_rf']))}, "
            f"Psum {pct(improvement(risk['psum'], full['psum']))}, "
            f"extra_drop={risk['drop'] - full['drop']:.2f}%"
        )

    lines.extend(
        [
            "",
            "Reading guide:",
            "- ONNX-C/ONNX traffic are the existing component replay totals.",
            "- Act-C/Act-E are activity-sensitive proxies with configurable weights.",
            "- W_HBM follows replayed Wloads/Wscale.",
            "- A_HBM is unchanged by default because external activations remain byte-major.",
            "- A_RF/PE/W_RF/Psum scale with AvgDepth/8 and expose mixed-depth savings.",
        ]
    )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "schema": "graphbit_activity_breakdown.v1",
                "replay": str(replay_path),
                "assumptions": {
                    "plane_group_hbm": plane_group_hbm,
                    "group_bits": group_bits,
                    "external_activation_byte_major_by_default": not plane_group_hbm,
                    "w_hbm_from_trace_replayed_wloads": True,
                    "rf_pe_psum_scale_with_avg_depth": True,
                },
                "cycle_weights": cycle_weights,
                "energy_weights": energy_weights,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(txt_path.read_text(encoding="utf-8"))
    print(f"[GraphBitActivity] wrote {tsv_path}")
    print(f"[GraphBitActivity] wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cycle-weights",
        default="w_hbm=0.50,a_hbm=0.05,a_rf=0.05,pe=0.15,w_rf=0.10,psum=0.10,out=0.05,sched=0.00",
        help="Comma-separated activity weights for cycle proxy.",
    )
    parser.add_argument(
        "--energy-weights",
        default="w_hbm=0.30,a_hbm=0.08,a_rf=0.12,pe=0.20,w_rf=0.15,psum=0.10,out=0.05,sched=0.00",
        help="Comma-separated activity weights for energy proxy.",
    )
    parser.add_argument(
        "--plane-group-hbm",
        action="store_true",
        help="Also scale first activation memory read by grouped stop depth. Default keeps A_HBM byte-major.",
    )
    parser.add_argument("--group-bits", type=int, default=2)
    args = parser.parse_args()

    cycle_defaults = {key: 0.0 for key in ACTIVITY_FIELDS}
    energy_defaults = {key: 0.0 for key in ACTIVITY_FIELDS}
    cycle_weights = parse_weights(args.cycle_weights, cycle_defaults)
    energy_weights = parse_weights(args.energy_weights, energy_defaults)
    replay = load_json(args.replay_json)
    rows = build_rows(
        replay,
        cycle_weights=cycle_weights,
        energy_weights=energy_weights,
        plane_group_hbm=args.plane_group_hbm,
        group_bits=args.group_bits,
    )
    write_outputs(
        rows,
        output_dir=args.output_dir,
        replay_path=args.replay_json,
        cycle_weights=cycle_weights,
        energy_weights=energy_weights,
        plane_group_hbm=args.plane_group_hbm,
        group_bits=args.group_bits,
    )


if __name__ == "__main__":
    main()
