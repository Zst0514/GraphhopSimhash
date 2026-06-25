#!/usr/bin/env python3
"""Cora BFP activation-lift timing breakdown in the ONNXim environment.

This runner does not regenerate LLaMA embeddings.  It composes the existing
Cora progressive-BFP array trace with ONNXim's vector/scalar latency knobs, so
the runtime BFP loader/control path is visible separately from the BFP MAC
array cycles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OFA_ROOT = REPO_ROOT.parent

DEFAULT_CONFIG = REPO_ROOT / "ONNXim" / "configs" / "systolic_ws_128x128_c4_simple_noc_tpuv4.json"
DEFAULT_TRACE_DIR = OFA_ROOT / "output" / "e2e_time_breakdown_40reuse" / "array_cora_graphstress20"
DEFAULT_REUSE = OFA_ROOT / "output" / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"
DEFAULT_OUTPUT_DIR = OFA_ROOT / "output" / "onnxim_cora_bfp_lift_breakdown"
DEFAULT_REPO_REPORT = REPO_ROOT / "docs" / "results" / "CORA_ONNXIM_BFP_LIFT_BREAKDOWN.md"


@dataclass(frozen=True)
class OnnximLatencies:
    add_tree_latency: int
    exp_latency: int
    scalar_add_latency: int
    scalar_mul_latency: int


@dataclass(frozen=True)
class ScenarioRow:
    scenario: str
    reuse_pct: float
    miss_pct: float
    total_blocks: float
    refined_blocks: float
    exponent_select_cycles: float
    mantissa_pack_cycles: float
    stress_priority_cycles: float
    refine_queue_push_cycles: float
    loader_control_raw_cycles: float
    bfpa4_base_mac_cycles: float
    bfplift_extra_mac_cycles: float
    dynamic_mac_cycles: float
    serial_total_cycles: float
    overlap_exposed_total_cycles: float
    exponent_select_s: float
    mantissa_pack_s: float
    stress_priority_s: float
    refine_queue_push_s: float
    loader_control_raw_s: float
    bfpa4_base_mac_s: float
    bfplift_extra_mac_s: float
    dynamic_mac_s: float
    serial_total_s: float
    overlap_exposed_total_s: float
    loader_raw_pct_of_dynamic_mac: float


def read_config(path: Path) -> tuple[OnnximLatencies, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    core = payload["core_config"]["core_0"]
    return (
        OnnximLatencies(
            add_tree_latency=int(core.get("add_tree_latency", 1)),
            exp_latency=int(core.get("exp_latency", 1)),
            scalar_add_latency=int(core.get("scalar_add_latency", 1)),
            scalar_mul_latency=int(core.get("scalar_mul_latency", 1)),
        ),
        int(payload.get("core_freq", 0)),
    )


def read_reuse(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["task"] in {"CN", "CL"}:
                out[row["task"]] = float(row["anchor_reuse"])
    missing = {"CN", "CL"} - set(out)
    if missing:
        raise ValueError(f"missing reuse rows: {sorted(missing)}")
    return out


def seconds(cycles: float, clock_mhz: float) -> float:
    return float(cycles) / (float(clock_mhz) * 1.0e6)


def fmt_s(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}s"
    if value >= 1.0e-3:
        return f"{value * 1.0e3:.3f}ms"
    if value >= 1.0e-6:
        return f"{value * 1.0e6:.3f}us"
    return f"{value * 1.0e9:.3f}ns"


def fmt_cycles(value: float) -> str:
    if abs(value) >= 1.0e12:
        return f"{value / 1.0e12:.3f}T"
    if abs(value) >= 1.0e9:
        return f"{value / 1.0e9:.3f}B"
    if abs(value) >= 1.0e6:
        return f"{value / 1.0e6:.3f}M"
    return f"{value:.0f}"


def build_row(
    *,
    scenario: str,
    reuse_pct: float,
    miss_scale: float,
    summary: dict[str, Any],
    exp_cycles_per_block: float,
    mantissa_pack_cycles_per_block: float,
    stress_priority_cycles_per_block: float,
    queue_push_cycles_per_refined_block: float,
    clock_mhz: float,
) -> ScenarioRow:
    total_blocks = float(summary["total_blocks"]) * miss_scale
    refined_blocks = float(summary["refined_blocks"]) * miss_scale
    exp_cycles = total_blocks * exp_cycles_per_block
    pack_cycles = total_blocks * mantissa_pack_cycles_per_block
    stress_cycles = total_blocks * stress_priority_cycles_per_block
    queue_cycles = refined_blocks * queue_push_cycles_per_refined_block
    loader_cycles = exp_cycles + pack_cycles + stress_cycles + queue_cycles
    bfpa4_cycles = float(summary["full_bfpa4_cycles"]) * miss_scale
    dynamic_cycles = float(summary["dynamic_cycles"]) * miss_scale
    extra_cycles = dynamic_cycles - bfpa4_cycles
    serial_total = loader_cycles + dynamic_cycles
    overlap_total = dynamic_cycles + max(0.0, loader_cycles - dynamic_cycles)
    loader_pct = 100.0 * loader_cycles / max(1.0, dynamic_cycles)
    return ScenarioRow(
        scenario=scenario,
        reuse_pct=reuse_pct,
        miss_pct=100.0 * miss_scale,
        total_blocks=total_blocks,
        refined_blocks=refined_blocks,
        exponent_select_cycles=exp_cycles,
        mantissa_pack_cycles=pack_cycles,
        stress_priority_cycles=stress_cycles,
        refine_queue_push_cycles=queue_cycles,
        loader_control_raw_cycles=loader_cycles,
        bfpa4_base_mac_cycles=bfpa4_cycles,
        bfplift_extra_mac_cycles=extra_cycles,
        dynamic_mac_cycles=dynamic_cycles,
        serial_total_cycles=serial_total,
        overlap_exposed_total_cycles=overlap_total,
        exponent_select_s=seconds(exp_cycles, clock_mhz),
        mantissa_pack_s=seconds(pack_cycles, clock_mhz),
        stress_priority_s=seconds(stress_cycles, clock_mhz),
        refine_queue_push_s=seconds(queue_cycles, clock_mhz),
        loader_control_raw_s=seconds(loader_cycles, clock_mhz),
        bfpa4_base_mac_s=seconds(bfpa4_cycles, clock_mhz),
        bfplift_extra_mac_s=seconds(extra_cycles, clock_mhz),
        dynamic_mac_s=seconds(dynamic_cycles, clock_mhz),
        serial_total_s=seconds(serial_total, clock_mhz),
        overlap_exposed_total_s=seconds(overlap_total, clock_mhz),
        loader_raw_pct_of_dynamic_mac=loader_pct,
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    *,
    rows: list[ScenarioRow],
    summary: dict[str, Any],
    config_path: Path,
    trace_dir: Path,
    lat: OnnximLatencies,
    config_core_freq_mhz: int,
    clock_mhz: float,
    reduction_levels: int,
    exp_cycles_per_block: float,
    mantissa_pack_cycles_per_block: float,
    stress_priority_cycles_per_block: float,
    queue_push_cycles_per_refined_block: float,
) -> str:
    lines = [
        "# Cora ONNXim BFP Lift Runtime Breakdown",
        "",
        "## Inputs",
        "",
        f"- ONNXim config: `{config_path}`.",
        f"- Cora BFP array trace: `{trace_dir}`.",
        f"- Trace tag: `{summary.get('tag', '')}`.",
        f"- Trace block size: `{int(summary['block_size'])}` activation values/block.",
        f"- Full Cora trace blocks: `{int(summary['total_blocks'])}` total, `{int(summary['refined_blocks'])}` refined ({100.0 * float(summary['refined_ratio']):.2f}%).",
        f"- ONNXim config core frequency field: `{config_core_freq_mhz} MHz`; report clock override: `{clock_mhz} MHz`.",
        "",
        "## Runtime BFP Loader Model",
        "",
        f"- Exponent select per block: `ceil(log2(block_size)) * add_tree_latency + exp_latency = {reduction_levels} * {lat.add_tree_latency} + {lat.exp_latency} = {exp_cycles_per_block:.0f}` cycles.",
        f"- Mantissa pack/slice per block: `scalar_mul_latency + scalar_add_latency = {mantissa_pack_cycles_per_block:.0f}` cycles.",
        f"- Stress/priority/refine-flag per block: `scalar_mul_latency + scalar_add_latency = {stress_priority_cycles_per_block:.0f}` cycles.",
        f"- RefineQueue push per selected block: `{queue_push_cycles_per_refined_block:.0f}` cycle.",
        "- `dynamic_mac` is the existing ONNXim-style BFP array trace: BFPA4 base MAC plus selected BFPA6 low-2-bit correction MAC.",
        "- `serial_total` is conservative no-overlap time. `overlap_exposed_total` assumes the BFP loader is double-buffered with the MAC array; only loader work exceeding MAC time is exposed.",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Reuse | Miss | Exp Select | Pack/Slice | Stress+Flag | Queue Push | Loader Raw | BFPA4 MAC | BFPLift Extra MAC | Dynamic MAC | Serial Total | Overlap Exposed | Loader Raw / Dynamic MAC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.scenario} | {row.reuse_pct:.2f}% | {row.miss_pct:.2f}% | "
            f"{fmt_s(row.exponent_select_s)} | {fmt_s(row.mantissa_pack_s)} | "
            f"{fmt_s(row.stress_priority_s)} | {fmt_s(row.refine_queue_push_s)} | "
            f"{fmt_s(row.loader_control_raw_s)} | {fmt_s(row.bfpa4_base_mac_s)} | "
            f"{fmt_s(row.bfplift_extra_mac_s)} | {fmt_s(row.dynamic_mac_s)} | "
            f"{fmt_s(row.serial_total_s)} | {fmt_s(row.overlap_exposed_total_s)} | "
            f"{row.loader_raw_pct_of_dynamic_mac:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Cycle Details",
            "",
            "| Scenario | Blocks | Refined Blocks | Exp Select | Pack/Slice | Stress+Flag | Queue Push | Loader Raw | BFPA4 MAC | BFPLift Extra MAC | Dynamic MAC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.scenario} | {fmt_cycles(row.total_blocks)} | {fmt_cycles(row.refined_blocks)} | "
            f"{fmt_cycles(row.exponent_select_cycles)} | {fmt_cycles(row.mantissa_pack_cycles)} | "
            f"{fmt_cycles(row.stress_priority_cycles)} | {fmt_cycles(row.refine_queue_push_cycles)} | "
            f"{fmt_cycles(row.loader_control_raw_cycles)} | {fmt_cycles(row.bfpa4_base_mac_cycles)} | "
            f"{fmt_cycles(row.bfplift_extra_mac_cycles)} | {fmt_cycles(row.dynamic_mac_cycles)} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- The exponent-selection and lift-selection work is online runtime work, not Table VI offline preprocessing.",
            "- For Cora CN at the 39.90% reuse point, the raw loader/control work is tens of seconds if serialized, but it is only a few percent of the dynamic MAC time and is hidden under a double-buffered loader/MAC pipeline in this model.",
            "- If a reviewer asks for the unhidden worst case, use `Serial Total`; if discussing the actual pipelined NPU critical path, use `Overlap Exposed` plus separately report `Loader Raw` as work performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--reuse-input", type=Path, default=DEFAULT_REUSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-report", type=Path, default=DEFAULT_REPO_REPORT)
    parser.add_argument("--clock-mhz", type=float, default=500.0)
    parser.add_argument("--queue-push-cycles-per-refined-block", type=float, default=1.0)
    args = parser.parse_args()

    lat, config_core_freq_mhz = read_config(args.config)
    summary = json.loads((args.trace_dir / "summary.json").read_text(encoding="utf-8"))
    reuse = read_reuse(args.reuse_input)
    block_size = int(summary["block_size"])
    reduction_levels = int(math.ceil(math.log2(block_size)))
    exp_cycles_per_block = reduction_levels * lat.add_tree_latency + lat.exp_latency
    mantissa_pack_cycles_per_block = lat.scalar_mul_latency + lat.scalar_add_latency
    stress_priority_cycles_per_block = lat.scalar_mul_latency + lat.scalar_add_latency

    scenario_specs = [
        ("CoraFull+BFPLift", 0.0, 1.0),
        ("CN_TSER40_Miss+BFPLift", reuse["CN"], 1.0 - reuse["CN"] / 100.0),
        ("CL_TSER40_Miss+BFPLift", reuse["CL"], 1.0 - reuse["CL"] / 100.0),
    ]
    rows = [
        build_row(
            scenario=name,
            reuse_pct=reuse_pct,
            miss_scale=miss_scale,
            summary=summary,
            exp_cycles_per_block=exp_cycles_per_block,
            mantissa_pack_cycles_per_block=mantissa_pack_cycles_per_block,
            stress_priority_cycles_per_block=stress_priority_cycles_per_block,
            queue_push_cycles_per_refined_block=args.queue_push_cycles_per_refined_block,
            clock_mhz=args.clock_mhz,
        )
        for name, reuse_pct, miss_scale in scenario_specs
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    write_tsv(args.output_dir / "cora_bfp_lift_breakdown.tsv", row_dicts)
    payload = {
        "config": {
            "onnxim_config": str(args.config),
            "trace_dir": str(args.trace_dir),
            "clock_mhz": args.clock_mhz,
            "latencies": asdict(lat),
            "config_core_freq_mhz": config_core_freq_mhz,
            "reduction_levels": reduction_levels,
            "exponent_select_cycles_per_block": exp_cycles_per_block,
            "mantissa_pack_cycles_per_block": mantissa_pack_cycles_per_block,
            "stress_priority_cycles_per_block": stress_priority_cycles_per_block,
            "queue_push_cycles_per_refined_block": args.queue_push_cycles_per_refined_block,
        },
        "trace_summary": summary,
        "rows": row_dicts,
    }
    (args.output_dir / "cora_bfp_lift_breakdown.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = render_report(
        rows=rows,
        summary=summary,
        config_path=args.config,
        trace_dir=args.trace_dir,
        lat=lat,
        config_core_freq_mhz=config_core_freq_mhz,
        clock_mhz=args.clock_mhz,
        reduction_levels=reduction_levels,
        exp_cycles_per_block=exp_cycles_per_block,
        mantissa_pack_cycles_per_block=mantissa_pack_cycles_per_block,
        stress_priority_cycles_per_block=stress_priority_cycles_per_block,
        queue_push_cycles_per_refined_block=args.queue_push_cycles_per_refined_block,
    )
    (args.output_dir / "CORA_ONNXIM_BFP_LIFT_BREAKDOWN.md").write_text(report, encoding="utf-8")
    args.repo_report.parent.mkdir(parents=True, exist_ok=True)
    args.repo_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
