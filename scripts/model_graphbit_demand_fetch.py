#!/usr/bin/env python3
"""Model Graph-Bit bit-plane demand-fetch dataflow.

This script turns the current Graph-Bit workload profile into a hardware-facing
breakdown.  It separates four effects:

1. Graph/reuse front-end: direct/residual hits skip the encoder.
2. Precision-depth assignment: miss nodes are assigned to P8/P6/P4.
3. Bit-plane demand fetch: lower activation bit-planes are not requested after
   early-stop.
4. Micro-batch utilization: random mixed risk classes execute to the maximum
   depth in the batch, while risk-bucketed scheduling avoids this waste.

The model is intentionally conservative: weight reads and output writes do not
fall with activation depth.  This makes the fixed costs explicit instead of
claiming unrealistic end-to-end speedups.
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
class Agg:
    name: str
    depth: float
    cycles: float
    traffic: float
    act_read: float
    weight_read: float
    out_write: float
    bitcomp: float


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else num / den


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return json.loads(path.read_text())


def encoder(agg: dict[str, Any]) -> dict[str, Any]:
    return agg["encoder"]


def reqs(enc: dict[str, Any]) -> float:
    return float(enc.get("dram_read_requests", 0.0) or 0.0) + float(enc.get("dram_write_requests", 0.0) or 0.0)


def bitcomp(enc: dict[str, Any], fallback_depth: float) -> float:
    raw = float(enc.get("graphbit_raw_compute_cycles", 0.0) or 0.0)
    eff = float(enc.get("graphbit_effective_compute_cycles", 0.0) or 0.0)
    if raw > 0:
        return eff / raw
    return fallback_depth / 8.0


def load_agg(path: Path, name: str, base: dict[str, Any] | None = None, fallback_depth: float = 8.0) -> Agg:
    payload = load_json(path)
    enc = encoder(payload)
    depth = enc.get("graphbit_avg_depth")
    depth_f = fallback_depth if depth is None else float(depth)
    if base is None:
        base_enc = enc
    else:
        base_enc = encoder(base)
    return Agg(
        name=name,
        depth=depth_f,
        cycles=safe_div(float(enc.get("cycles", 0.0) or 0.0), float(base_enc.get("cycles", 0.0) or 0.0), 1.0),
        traffic=safe_div(reqs(enc), reqs(base_enc), 1.0),
        act_read=safe_div(
            float(enc.get("mem_read_input_actual", 0.0) or 0.0),
            float(base_enc.get("mem_read_input_actual", 0.0) or 0.0),
            1.0,
        ),
        weight_read=safe_div(
            float(enc.get("mem_read_weight", 0.0) or 0.0),
            float(base_enc.get("mem_read_weight", 0.0) or 0.0),
            1.0,
        ),
        out_write=safe_div(
            float(enc.get("mem_write_output", 0.0) or 0.0),
            float(base_enc.get("mem_write_output", 0.0) or 0.0),
            1.0,
        ),
        bitcomp=bitcomp(enc, depth_f),
    )


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
    """Expected max-depth class when a micro-batch randomly mixes risk classes."""

    remaining = 1.0
    out: dict[str, float] = {}
    cumulative_higher = 0.0
    for level in LEVEL_ORDER:
        p_level = probs.get(level, 0.0)
        p_no_higher = max(0.0, 1.0 - cumulative_higher) ** batch_size
        p_no_this_or_higher = max(0.0, 1.0 - cumulative_higher - p_level) ** batch_size
        out[level] = max(0.0, p_no_higher - p_no_this_or_higher)
        remaining -= out[level]
        cumulative_higher += p_level
    # Numerical residue means an all-empty impossible batch; assign to lowest.
    out[LEVEL_ORDER[-1]] += max(0.0, remaining)
    total = sum(out.values())
    return {level: safe_div(value, total, 0.0) for level, value in out.items()}


def weighted(probs: dict[str, float], aggs: dict[str, Agg], field: str) -> float:
    return sum(probs.get(level, 0.0) * float(getattr(aggs[level], field)) for level in LEVEL_ORDER)


def useful_depth(probs: dict[str, float], aggs: dict[str, Agg]) -> float:
    return weighted(probs, aggs, "depth")


def padding_overhead(probs: dict[str, float], batch_size: int, miss_nodes: int) -> float:
    useful = max(1, int(miss_nodes))
    padded = 0
    for level in LEVEL_ORDER:
        count = int(round(probs.get(level, 0.0) * useful))
        if count == 0:
            continue
        padded += math.ceil(count / batch_size) * batch_size
    return safe_div(padded, useful, 1.0)


def make_row(
    *,
    method: str,
    profile: dict[str, Any],
    aggs: dict[str, Agg],
    schedule: str,
    dataflow: str,
    batch_size: int,
    node_count: int,
    cache_compute: float,
    residual_compute: float,
    cache_traffic: float,
    residual_traffic: float,
    energy_compute_weight: float,
    energy_traffic_weight: float,
) -> dict[str, Any]:
    r = ratios(profile)
    probs = class_probs(r)
    miss_ratio = sum(r.get(level, 0.0) for level in LEVEL_ORDER)
    exec_probs = random_mixed_exec_probs(probs, batch_size) if schedule == "random_mixed" else probs

    if dataflow == "compute_only":
        # Only PE bit work is reduced.  Memory and issued tile cycles remain at P8.
        miss_cycles = 1.0
        miss_traffic = 1.0
        act_read = 1.0
        weight_read = 1.0
        out_write = 1.0
        bitcomp_val = weighted(exec_probs, aggs, "bitcomp")
    else:
        miss_cycles = weighted(exec_probs, aggs, "cycles")
        miss_traffic = weighted(exec_probs, aggs, "traffic")
        act_read = weighted(exec_probs, aggs, "act_read")
        weight_read = weighted(exec_probs, aggs, "weight_read")
        out_write = weighted(exec_probs, aggs, "out_write")
        bitcomp_val = weighted(exec_probs, aggs, "bitcomp")

    useful_d = useful_depth(probs, aggs)
    exec_d = useful_depth(exec_probs, aggs)
    bit_util = safe_div(useful_d, exec_d, 1.0)
    padding = padding_overhead(probs, batch_size, int(round(node_count * miss_ratio))) if schedule == "risk_bucket" else 1.0

    direct = r.get("direct", 0.0)
    residual = r.get("residual", 0.0)
    full_cycles = (
        miss_ratio * miss_cycles
        + direct * cache_compute
        + residual * residual_compute
    )
    full_traffic = (
        miss_ratio * miss_traffic
        + direct * cache_traffic
        + residual * (cache_traffic + residual_traffic)
    )
    energy = energy_compute_weight * full_cycles + energy_traffic_weight * full_traffic

    metrics = profile.get("metrics", {})
    return {
        "method": method,
        "schedule": schedule,
        "dataflow": dataflow,
        "reuse": r.get("reuse", 0.0),
        "direct": direct,
        "residual": residual,
        "miss": miss_ratio,
        "p8": r.get("p8", 0.0),
        "p6": r.get("p6", 0.0),
        "p5": r.get("p5", 0.0),
        "p4": r.get("p4", 0.0),
        "useful_depth": useful_d,
        "executed_depth": exec_d,
        "bit_util": bit_util,
        "wasted_bitplanes": 1.0 - bit_util,
        "padding_overhead": padding,
        "miss_bitcomp": bitcomp_val,
        "miss_act_read": act_read,
        "miss_weight_read": weight_read,
        "miss_out_write": out_write,
        "miss_cycles": miss_cycles,
        "miss_traffic": miss_traffic,
        "full_cycles": full_cycles,
        "full_traffic": full_traffic,
        "energy_proxy": energy,
        "drop": float(metrics.get("drop_percent", 0.0) or 0.0),
        "acc": float(metrics.get("acc", 0.0) or 0.0),
        "finalerr": float(metrics.get("finalerr", 0.0) or 0.0),
    }


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, workload: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "demand_fetch_model.tsv"
    txt_path = output_dir / "demand_fetch_model.txt"
    json_path = output_dir / "demand_fetch_model.json"

    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    base = rows[0]
    lines = [
        "Graph-Bit bit-plane demand-fetch model",
        f"Workload: {args.workload}",
        f"Dataset={workload['profiles'][0]['dataset']} | frontend={workload['profiles'][0]['route']['frontend']} | budget={workload['profiles'][0]['route']['budget']}",
        f"Batch size={args.batch_size} | node_count={args.node_count}",
        "",
        "Method                         Sched        Dataflow       Reuse Miss  UsefulD ExecD Util  BitC  ActRd WgtRd OutWr MissC MissT FullC FullT Energy Drop",
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<30} {row['schedule']:<12} {row['dataflow']:<13} "
            f"{fmt_pct(row['reuse']):>5} {fmt_pct(row['miss']):>5} "
            f"{row['useful_depth']:>7.2f} {row['executed_depth']:>5.2f} "
            f"{fmt_pct(row['bit_util']):>5} "
            f"{row['miss_bitcomp']:>5.3f} "
            f"{row['miss_act_read']:>5.3f} "
            f"{row['miss_weight_read']:>5.3f} "
            f"{row['miss_out_write']:>5.3f} "
            f"{row['miss_cycles']:>5.3f} {row['miss_traffic']:>5.3f} "
            f"{row['full_cycles']:>5.3f} {row['full_traffic']:>5.3f} "
            f"{row['energy_proxy']:>6.3f} {row['drop']:>4.2f}%"
        )
    lines.extend(
        [
            "",
            "Delta versus FullP8-miss:",
            "Method                         FullC-save FullT-save Energy-save ExtraDrop",
            "---------------------------------------------------------------------",
        ]
    )
    for row in rows[1:]:
        lines.append(
            f"{row['method']:<30} "
            f"{fmt_pct(1.0 - safe_div(row['full_cycles'], base['full_cycles'], 1.0)):>10} "
            f"{fmt_pct(1.0 - safe_div(row['full_traffic'], base['full_traffic'], 1.0)):>10} "
            f"{fmt_pct(1.0 - safe_div(row['energy_proxy'], base['energy_proxy'], 1.0)):>11} "
            f"{row['drop'] - base['drop']:>8.2f}%"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- compute_only: lower bit MACs are masked, but A8 activation bytes are still fetched.",
            "- random_mixed demand_fetch: bit-plane layout exists, but a mixed batch runs to the deepest node.",
            "- risk_bucket demand_fetch: graph-risk buckets are batched separately, so useful and executed depths match.",
            "- weight/output costs remain explicit and unchanged; this is why full-stack gains are smaller than bitcomp gains.",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n")
    payload = {
        "schema": "graphbit_demand_fetch_model.v1",
        "assumptions": {
            "weight_reads_do_not_scale_with_activation_depth": True,
            "output_writes_do_not_scale_with_activation_depth": True,
            "risk_bucket_padding_is_reported_not_applied_to_cycles": True,
            "drop_source": "embedding proxy from workload profile",
        },
        "args": vars(args) | {"workload": str(args.workload), "output_dir": str(args.output_dir), "microbench_dir": str(args.microbench_dir)},
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[GraphBitDemandFetch] wrote {tsv_path}")
    print(f"[GraphBitDemandFetch] wrote {txt_path}")
    print(f"[GraphBitDemandFetch] wrote {json_path}")
    print(txt_path.read_text())


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        type=Path,
        default=root / "output" / "graphbit_predictor_free" / "cora_h8_53_T30" / "predictor_free_workload.json",
    )
    parser.add_argument(
        "--microbench-dir",
        type=Path,
        default=root / "output" / "onnxim_graphbit",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--node-count", type=int, default=2708)
    parser.add_argument("--degree-profile-match", default="degree static")
    parser.add_argument("--random-profile-match", default="random static")
    parser.add_argument("--full-profile-match", default="fullp8")
    parser.add_argument("--cache-compute-cost", type=float, default=0.001)
    parser.add_argument("--residual-compute-cost", type=float, default=0.005)
    parser.add_argument("--cache-traffic-cost", type=float, default=0.003)
    parser.add_argument("--residual-traffic-cost", type=float, default=0.005)
    parser.add_argument("--energy-compute-weight", type=float, default=0.55)
    parser.add_argument("--energy-traffic-weight", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    workload = load_json(args.workload)
    out_dir = args.output_dir or (args.workload.parent / "demand_fetch_model")
    args.output_dir = out_dir

    base_payload = load_json(args.microbench_dir / f"microbench_s{args.seq_len}_internal_p8" / "aggregate.json")
    full = load_agg(
        args.microbench_dir / f"microbench_s{args.seq_len}_internal_p8" / "aggregate.json",
        "p8",
        base=base_payload,
        fallback_depth=8.0,
    )
    p6 = load_agg(
        args.microbench_dir / f"microbench_s{args.seq_len}_internal_p6" / "aggregate.json",
        "p6",
        base=base_payload,
        fallback_depth=6.0,
    )
    p4 = load_agg(
        args.microbench_dir / f"microbench_s{args.seq_len}_internal_p4" / "aggregate.json",
        "p4",
        base=base_payload,
        fallback_depth=4.0,
    )
    mid_early = load_agg(
        args.microbench_dir / f"microbench_s{args.seq_len}_internal_bound_mid_min6_t0p02" / "aggregate.json",
        "p6_early",
        base=base_payload,
        fallback_depth=6.0,
    )
    low_early = load_agg(
        args.microbench_dir / f"microbench_s{args.seq_len}_internal_bound_low_min4_t0p04" / "aggregate.json",
        "p4_early",
        base=base_payload,
        fallback_depth=5.0,
    )

    static_aggs = {"p8": full, "p6": p6, "p5": p6, "p4": p4}
    early_aggs = {"p8": full, "p6": mid_early, "p5": mid_early, "p4": low_early}

    full_profile = choose_profile(workload, args.full_profile_match)
    degree_profile = choose_profile(workload, args.degree_profile_match)
    random_profile = choose_profile(workload, args.random_profile_match)

    rows = [
        make_row(
            method="FullP8-miss",
            profile=full_profile,
            aggs=static_aggs,
            schedule="risk_bucket",
            dataflow="demand_fetch",
            batch_size=args.batch_size,
            node_count=args.node_count,
            cache_compute=args.cache_compute_cost,
            residual_compute=args.residual_compute_cost,
            cache_traffic=args.cache_traffic_cost,
            residual_traffic=args.residual_traffic_cost,
            energy_compute_weight=args.energy_compute_weight,
            energy_traffic_weight=args.energy_traffic_weight,
        ),
        make_row(
            method="Degree compute-mask only",
            profile=degree_profile,
            aggs=static_aggs,
            schedule="risk_bucket",
            dataflow="compute_only",
            batch_size=args.batch_size,
            node_count=args.node_count,
            cache_compute=args.cache_compute_cost,
            residual_compute=args.residual_compute_cost,
            cache_traffic=args.cache_traffic_cost,
            residual_traffic=args.residual_traffic_cost,
            energy_compute_weight=args.energy_compute_weight,
            energy_traffic_weight=args.energy_traffic_weight,
        ),
        make_row(
            method="Random demand-fetch",
            profile=random_profile,
            aggs=static_aggs,
            schedule="risk_bucket",
            dataflow="demand_fetch",
            batch_size=args.batch_size,
            node_count=args.node_count,
            cache_compute=args.cache_compute_cost,
            residual_compute=args.residual_compute_cost,
            cache_traffic=args.cache_traffic_cost,
            residual_traffic=args.residual_traffic_cost,
            energy_compute_weight=args.energy_compute_weight,
            energy_traffic_weight=args.energy_traffic_weight,
        ),
        make_row(
            method="Degree random-mixed",
            profile=degree_profile,
            aggs=early_aggs,
            schedule="random_mixed",
            dataflow="demand_fetch",
            batch_size=args.batch_size,
            node_count=args.node_count,
            cache_compute=args.cache_compute_cost,
            residual_compute=args.residual_compute_cost,
            cache_traffic=args.cache_traffic_cost,
            residual_traffic=args.residual_traffic_cost,
            energy_compute_weight=args.energy_compute_weight,
            energy_traffic_weight=args.energy_traffic_weight,
        ),
        make_row(
            method="Degree demand-fetch",
            profile=degree_profile,
            aggs=early_aggs,
            schedule="risk_bucket",
            dataflow="demand_fetch",
            batch_size=args.batch_size,
            node_count=args.node_count,
            cache_compute=args.cache_compute_cost,
            residual_compute=args.residual_compute_cost,
            cache_traffic=args.cache_traffic_cost,
            residual_traffic=args.residual_traffic_cost,
            energy_compute_weight=args.energy_compute_weight,
            energy_traffic_weight=args.energy_traffic_weight,
        ),
    ]
    write_outputs(rows, out_dir, workload, args)


if __name__ == "__main__":
    main()
