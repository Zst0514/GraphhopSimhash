#!/usr/bin/env python3
"""Component-level Graph-Bit NPU dataflow model.

This model is intentionally one level more concrete than the embedding-level
precision-depth proxy.  It asks: after Graph-Bit decides that miss nodes need
less activation precision, which hardware events actually disappear?

The modeled knobs are:

* byte-major vs bit-plane/plane-group-major activation layout;
* issue gating for skipped bit-plane cycles;
* weight SRAM/RF broadcast gating for skipped bit-plane cycles;
* partial-sum update gating;
* risk-bucket batching vs random mixed batches;
* weight-stationary reuse that amortizes HBM weight tile reads.

It is a conservative architectural model, not RTL.  It should be used to
compare dataflow choices and expose the design requirements before writing
Verilog or cycle-accurate PE logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEVEL_ORDER = ("p8", "p6", "p5", "p4")
LEVEL_DEPTH = {"p8": 8.0, "p6": 6.0, "p5": 5.0, "p4": 4.0}


@dataclass(frozen=True)
class Dataflow:
    name: str
    profile_match: str
    schedule: str
    activation_layout: str
    issue_gate: bool
    weight_rf_gate: bool
    psum_gate: bool
    weight_stationary: bool
    include_pack_overhead: bool


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return json.loads(path.read_text())


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if abs(den) < 1e-12 else num / den


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def choose_profile(workload: dict[str, Any], contains: str) -> dict[str, Any]:
    contains_lower = contains.lower()
    for profile in workload["profiles"]:
        text = " ".join(
            [
                str(profile.get("id", "")),
                str(profile.get("route", {}).get("method", "")),
                str(profile.get("route", {}).get("config", "")),
            ]
        ).lower()
        if contains_lower in text:
            return profile
    raise SystemExit(f"No workload profile matching '{contains}'")


def ratios(profile: dict[str, Any]) -> dict[str, float]:
    raw = profile["ratios"]
    return {key: float(raw.get(key, 0.0) or 0.0) for key in ["reuse", "direct", "residual", *LEVEL_ORDER]}


def class_probs(r: dict[str, float]) -> dict[str, float]:
    miss = sum(r.get(level, 0.0) for level in LEVEL_ORDER)
    if miss <= 0:
        return {level: 0.0 for level in LEVEL_ORDER}
    return {level: r.get(level, 0.0) / miss for level in LEVEL_ORDER}


def random_mixed_exec_probs(probs: dict[str, float], batch_size: int) -> dict[str, float]:
    """Expected max-depth class when one batch randomly mixes risk classes."""

    out: dict[str, float] = {}
    cumulative_higher = 0.0
    remaining = 1.0
    for level in LEVEL_ORDER:
        p_level = probs.get(level, 0.0)
        p_no_higher = max(0.0, 1.0 - cumulative_higher) ** batch_size
        p_no_this_or_higher = max(0.0, 1.0 - cumulative_higher - p_level) ** batch_size
        out[level] = max(0.0, p_no_higher - p_no_this_or_higher)
        remaining -= out[level]
        cumulative_higher += p_level
    out[LEVEL_ORDER[-1]] += max(0.0, remaining)
    total = sum(out.values())
    return {level: safe_div(value, total, 0.0) for level, value in out.items()}


def weighted_depth(probs: dict[str, float]) -> float:
    return sum(probs.get(level, 0.0) * LEVEL_DEPTH[level] for level in LEVEL_ORDER)


def grouped_depth(depth: float, group_bits: int) -> float:
    return min(8.0, math.ceil(depth / group_bits) * group_bits)


def weighted_group_depth(probs: dict[str, float], group_bits: int) -> float:
    return sum(probs.get(level, 0.0) * grouped_depth(LEVEL_DEPTH[level], group_bits) for level in LEVEL_ORDER)


def bucket_padding(probs: dict[str, float], node_count: int, miss_ratio: float, batch_size: int) -> float:
    miss_nodes = max(1, int(round(node_count * miss_ratio)))
    padded = 0
    for level in LEVEL_ORDER:
        count = int(round(miss_nodes * probs.get(level, 0.0)))
        if count:
            padded += math.ceil(count / batch_size) * batch_size
    return safe_div(padded, miss_nodes, 1.0)


def component_model(
    *,
    profile: dict[str, Any],
    flow: Dataflow,
    batch_size: int,
    node_count: int,
    plane_group_bits: int,
    baseline_weight_tile_batch: int,
    weight_stationary_tile_batch: int,
    pack_overhead: float,
    cache_compute: float,
    residual_compute: float,
    cache_traffic: float,
    residual_traffic: float,
    cycle_compute_weight: float,
    cycle_act_weight: float,
    cycle_weight_weight: float,
    cycle_fixed_weight: float,
    traffic_act_weight: float,
    traffic_weight_weight: float,
    traffic_output_weight: float,
    energy_compute_weight: float,
    energy_act_weight: float,
    energy_weight_hbm_weight: float,
    energy_weight_rf_weight: float,
    energy_psum_weight: float,
    energy_output_weight: float,
) -> dict[str, Any]:
    r = ratios(profile)
    probs = class_probs(r)
    miss_ratio = sum(r.get(level, 0.0) for level in LEVEL_ORDER)
    exec_probs = random_mixed_exec_probs(probs, batch_size) if flow.schedule == "random_mixed" else probs

    useful_depth = weighted_depth(probs)
    executed_depth = weighted_depth(exec_probs)
    fetched_depth = 8.0
    if flow.activation_layout == "plane_group":
        fetched_depth = weighted_group_depth(exec_probs, plane_group_bits)

    # Byte-major layout cannot avoid reading full A8 even if compute stops.
    act_read = 1.0 if flow.activation_layout == "byte_major" else fetched_depth / 8.0
    pe_issue = executed_depth / 8.0 if flow.issue_gate else 1.0
    weight_rf = executed_depth / 8.0 if flow.weight_rf_gate else 1.0
    psum = executed_depth / 8.0 if flow.psum_gate else 1.0

    if flow.weight_stationary:
        reuse_gain = safe_div(weight_stationary_tile_batch, baseline_weight_tile_batch, 1.0)
        weight_hbm = max(0.0, min(1.0, 1.0 / reuse_gain))
    else:
        weight_hbm = 1.0

    pack = pack_overhead if flow.include_pack_overhead and flow.activation_layout == "plane_group" else 0.0
    output_write = 1.0

    miss_cycles = (
        cycle_compute_weight * pe_issue
        + cycle_act_weight * act_read
        + cycle_weight_weight * weight_hbm
        + cycle_fixed_weight
    )
    miss_traffic = traffic_act_weight * act_read + traffic_weight_weight * weight_hbm + traffic_output_weight * output_write
    miss_energy = (
        energy_compute_weight * pe_issue
        + energy_act_weight * act_read
        + energy_weight_hbm_weight * weight_hbm
        + energy_weight_rf_weight * weight_rf
        + energy_psum_weight * psum
        + energy_output_weight * output_write
        + pack
    )

    full_cycles = (
        miss_ratio * miss_cycles
        + r.get("direct", 0.0) * cache_compute
        + r.get("residual", 0.0) * residual_compute
    )
    full_traffic = (
        miss_ratio * miss_traffic
        + r.get("direct", 0.0) * cache_traffic
        + r.get("residual", 0.0) * (cache_traffic + residual_traffic)
    )
    full_energy = (
        miss_ratio * miss_energy
        + r.get("direct", 0.0) * cache_compute
        + r.get("residual", 0.0) * residual_compute
    )

    metrics = profile.get("metrics", {})
    return {
        "method": flow.name,
        "schedule": flow.schedule,
        "activation_layout": flow.activation_layout,
        "issue_gate": flow.issue_gate,
        "weight_rf_gate": flow.weight_rf_gate,
        "psum_gate": flow.psum_gate,
        "weight_stationary": flow.weight_stationary,
        "reuse": r.get("reuse", 0.0),
        "direct": r.get("direct", 0.0),
        "residual": r.get("residual", 0.0),
        "miss": miss_ratio,
        "p8": r.get("p8", 0.0),
        "p6": r.get("p6", 0.0),
        "p5": r.get("p5", 0.0),
        "p4": r.get("p4", 0.0),
        "useful_depth": useful_depth,
        "executed_depth": executed_depth,
        "fetched_depth": fetched_depth,
        "batch_util": safe_div(useful_depth, executed_depth, 1.0),
        "padding_overhead": bucket_padding(probs, node_count, miss_ratio, batch_size) if flow.schedule == "risk_bucket" else 1.0,
        "pe_issue": pe_issue,
        "act_read": act_read,
        "weight_hbm": weight_hbm,
        "weight_rf": weight_rf,
        "psum": psum,
        "out_write": output_write,
        "pack_overhead": pack,
        "miss_cycles": miss_cycles,
        "miss_traffic": miss_traffic,
        "miss_energy": miss_energy,
        "full_cycles": full_cycles,
        "full_traffic": full_traffic,
        "full_energy": full_energy,
        "drop": float(metrics.get("drop_percent", 0.0) or 0.0),
        "acc": float(metrics.get("acc", 0.0) or 0.0),
    }


def default_flows() -> list[Dataflow]:
    return [
        Dataflow(
            name="FullP8 byte baseline",
            profile_match="fullp8",
            schedule="risk_bucket",
            activation_layout="byte_major",
            issue_gate=False,
            weight_rf_gate=False,
            psum_gate=False,
            weight_stationary=False,
            include_pack_overhead=False,
        ),
        Dataflow(
            name="ByteMajor mask only",
            profile_match="degree static",
            schedule="risk_bucket",
            activation_layout="byte_major",
            issue_gate=False,
            weight_rf_gate=False,
            psum_gate=False,
            weight_stationary=False,
            include_pack_overhead=False,
        ),
        Dataflow(
            name="ByteMajor issue gate",
            profile_match="degree static",
            schedule="risk_bucket",
            activation_layout="byte_major",
            issue_gate=True,
            weight_rf_gate=True,
            psum_gate=True,
            weight_stationary=False,
            include_pack_overhead=False,
        ),
        Dataflow(
            name="PlaneGroup random mixed",
            profile_match="degree static",
            schedule="random_mixed",
            activation_layout="plane_group",
            issue_gate=True,
            weight_rf_gate=True,
            psum_gate=True,
            weight_stationary=False,
            include_pack_overhead=True,
        ),
        Dataflow(
            name="PlaneGroup risk bucket",
            profile_match="degree static",
            schedule="risk_bucket",
            activation_layout="plane_group",
            issue_gate=True,
            weight_rf_gate=True,
            psum_gate=True,
            weight_stationary=False,
            include_pack_overhead=True,
        ),
        Dataflow(
            name="RiskBucket + WS",
            profile_match="degree static",
            schedule="risk_bucket",
            activation_layout="plane_group",
            issue_gate=True,
            weight_rf_gate=True,
            psum_gate=True,
            weight_stationary=True,
            include_pack_overhead=True,
        ),
        Dataflow(
            name="Random risk full NPU",
            profile_match="random static",
            schedule="risk_bucket",
            activation_layout="plane_group",
            issue_gate=True,
            weight_rf_gate=True,
            psum_gate=True,
            weight_stationary=True,
            include_pack_overhead=True,
        ),
    ]


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, workload: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "npu_dataflow_model.tsv"
    txt_path = output_dir / "npu_dataflow_model.txt"
    json_path = output_dir / "npu_dataflow_model.json"

    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    base = rows[0]
    lines = [
        "Graph-Bit NPU dataflow model",
        f"Workload: {args.workload}",
        f"Dataset={workload['profiles'][0]['dataset']} | frontend={workload['profiles'][0]['route']['frontend']} | budget={workload['profiles'][0]['route']['budget']}",
        f"plane_group_bits={args.plane_group_bits} | batch_size={args.batch_size} | baseline_weight_tile_batch={args.baseline_weight_tile_batch} | weight_stationary_tile_batch={args.weight_stationary_tile_batch}",
        "",
        "Method                    Layout      Sched        Reuse Miss  UseD ExecD FetchD Util  PE    Aread WHBM  WRF   Psum  MissC MissT MissE FullC FullT FullE Drop",
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------",
    ]
    for r in rows:
        lines.append(
            f"{r['method']:<25} {r['activation_layout']:<11} {r['schedule']:<12} "
            f"{pct(r['reuse']):>5} {pct(r['miss']):>5} "
            f"{r['useful_depth']:>4.1f} {r['executed_depth']:>5.1f} {r['fetched_depth']:>6.1f} "
            f"{pct(r['batch_util']):>5} "
            f"{r['pe_issue']:>5.3f} {r['act_read']:>5.3f} {r['weight_hbm']:>5.3f} "
            f"{r['weight_rf']:>5.3f} {r['psum']:>5.3f} "
            f"{r['miss_cycles']:>5.3f} {r['miss_traffic']:>5.3f} {r['miss_energy']:>5.3f} "
            f"{r['full_cycles']:>5.3f} {r['full_traffic']:>5.3f} {r['full_energy']:>5.3f} "
            f"{r['drop']:>4.2f}%"
        )

    lines.extend(
        [
            "",
            "Delta versus FullP8 byte baseline:",
            "Method                    FullC-save FullT-save FullE-save ExtraDrop",
            "-------------------------------------------------------------------",
        ]
    )
    for r in rows[1:]:
        lines.append(
            f"{r['method']:<25} "
            f"{pct(1.0 - safe_div(r['full_cycles'], base['full_cycles'], 1.0)):>10} "
            f"{pct(1.0 - safe_div(r['full_traffic'], base['full_traffic'], 1.0)):>10} "
            f"{pct(1.0 - safe_div(r['full_energy'], base['full_energy'], 1.0)):>10} "
            f"{r['drop'] - base['drop']:>8.2f}%"
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "- ByteMajor mask only keeps ordinary A8 byte reads; it should not show traffic benefit.",
            "- ByteMajor issue gate removes PE/WRF/psum low-bit cycles but still reads full activations.",
            "- PlaneGroup random mixed has plane-group layout but loses depth savings when high-risk nodes share a batch.",
            "- PlaneGroup risk bucket is the minimum viable Graph-Bit datapath: plane-group layout plus risk-bucket scheduling.",
            "- RiskBucket + WS adds weight-stationary tile reuse; this is how skipped bit-planes stop being dominated by HBM weight reads.",
            "- Random risk full NPU uses the same hardware as RiskBucket + WS but random assignment, isolating why graph risk matters.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    payload = {
        "schema": "graphbit_npu_dataflow_model.v1",
        "assumptions": {
            "activation_byte_major_cannot_reduce_activation_reads": True,
            "plane_group_fetch_granularity_bits": args.plane_group_bits,
            "weight_hbm_saving_requires_weight_stationary_tile_reuse": True,
            "p5_fetches_one_2bit_group_more_than_p4_when_plane_group_bits_is_2": args.plane_group_bits == 2,
        },
        "args": vars(args) | {"workload": str(args.workload), "output_dir": str(args.output_dir)},
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[GraphBitNPUDataflow] wrote {tsv_path}")
    print(f"[GraphBitNPUDataflow] wrote {txt_path}")
    print(f"[GraphBitNPUDataflow] wrote {json_path}")
    print(txt_path.read_text())


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        type=Path,
        default=root / "output" / "graphbit_predictor_free" / "cora_h8_54_T40_dynp5" / "predictor_free_workload.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--node-count", type=int, default=2708)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plane-group-bits", type=int, default=2)
    parser.add_argument("--baseline-weight-tile-batch", type=int, default=16)
    parser.add_argument("--weight-stationary-tile-batch", type=int, default=64)
    parser.add_argument("--pack-overhead", type=float, default=0.015)
    parser.add_argument("--cache-compute-cost", type=float, default=0.001)
    parser.add_argument("--residual-compute-cost", type=float, default=0.005)
    parser.add_argument("--cache-traffic-cost", type=float, default=0.003)
    parser.add_argument("--residual-traffic-cost", type=float, default=0.005)
    parser.add_argument("--cycle-compute-weight", type=float, default=0.55)
    parser.add_argument("--cycle-act-weight", type=float, default=0.15)
    parser.add_argument("--cycle-weight-weight", type=float, default=0.20)
    parser.add_argument("--cycle-fixed-weight", type=float, default=0.10)
    parser.add_argument("--traffic-act-weight", type=float, default=0.35)
    parser.add_argument("--traffic-weight-weight", type=float, default=0.50)
    parser.add_argument("--traffic-output-weight", type=float, default=0.15)
    parser.add_argument("--energy-compute-weight", type=float, default=0.30)
    parser.add_argument("--energy-act-weight", type=float, default=0.20)
    parser.add_argument("--energy-weight-hbm-weight", type=float, default=0.20)
    parser.add_argument("--energy-weight-rf-weight", type=float, default=0.15)
    parser.add_argument("--energy-psum-weight", type=float, default=0.10)
    parser.add_argument("--energy-output-weight", type=float, default=0.05)
    args = parser.parse_args()

    workload = load_json(args.workload)
    output_dir = args.output_dir or (args.workload.parent / "npu_dataflow_model")
    args.output_dir = output_dir

    rows: list[dict[str, Any]] = []
    for flow in default_flows():
        profile = choose_profile(workload, flow.profile_match)
        rows.append(
            component_model(
                profile=profile,
                flow=flow,
                batch_size=args.batch_size,
                node_count=args.node_count,
                plane_group_bits=args.plane_group_bits,
                baseline_weight_tile_batch=args.baseline_weight_tile_batch,
                weight_stationary_tile_batch=args.weight_stationary_tile_batch,
                pack_overhead=args.pack_overhead,
                cache_compute=args.cache_compute_cost,
                residual_compute=args.residual_compute_cost,
                cache_traffic=args.cache_traffic_cost,
                residual_traffic=args.residual_traffic_cost,
                cycle_compute_weight=args.cycle_compute_weight,
                cycle_act_weight=args.cycle_act_weight,
                cycle_weight_weight=args.cycle_weight_weight,
                cycle_fixed_weight=args.cycle_fixed_weight,
                traffic_act_weight=args.traffic_act_weight,
                traffic_weight_weight=args.traffic_weight_weight,
                traffic_output_weight=args.traffic_output_weight,
                energy_compute_weight=args.energy_compute_weight,
                energy_act_weight=args.energy_act_weight,
                energy_weight_hbm_weight=args.energy_weight_hbm_weight,
                energy_weight_rf_weight=args.energy_weight_rf_weight,
                energy_psum_weight=args.energy_psum_weight,
                energy_output_weight=args.energy_output_weight,
            )
        )

    write_outputs(rows, output_dir, workload, args)


if __name__ == "__main__":
    main()
