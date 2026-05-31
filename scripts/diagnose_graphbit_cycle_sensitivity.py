#!/usr/bin/env python3
"""Diagnose why Graph-Bit depth changes do or do not reduce cycles.

ONNXim already records Graph-Bit effective compute cycles, but the end-to-end
component `cycles` can stay nearly unchanged when memory and pipeline overlap
hide the shorter bit-plane execution.  This script separates:

* measured ONNXim wall cycles;
* summed Graph-Bit per-instruction compute cycles;
* a depth-scaled PE active critical-path proxy;
* DRAM request ratios;
* a simple roofline sensitivity sweep over memory exposure.

The goal is not to replace ONNXim.  The goal is to make the bottleneck visible
and to state when mixed activation depth is latency-visible versus only an
activity/energy optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEPTHS = ("p8", "p6", "p5", "p4")
MODES = ("now", "ws_b32", "ws_b64")
DEPTH_VALUE = {"p8": 8, "p6": 6, "p5": 5, "p4": 4}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing json: {path}")
    return json.loads(path.read_text())


def encoder(root: Path, case: str) -> dict[str, Any]:
    return load_json(root / case / "aggregate.json")["encoder"]


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def traffic(row: dict[str, Any]) -> float:
    return f(row, "dram_read_requests") + f(row, "dram_write_requests")


def ratio(num: float, den: float) -> float:
    return 0.0 if abs(den) < 1e-12 else num / den


def case_name(depth: str, mode: str) -> str:
    if mode == "now":
        return f"{depth}_now"
    return f"{depth}_{mode}"


def measured_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        base = encoder(root, case_name("p8", mode))
        base_cycles = f(base, "cycles")
        base_traffic = traffic(base)
        base_eff_compute = f(base, "graphbit_effective_compute_cycles")
        base_matmul_active = f(base, "matmul_active_cycles")
        base_weight = f(base, "mem_read_weight")
        base_input = f(base, "mem_read_input_actual")
        for depth in DEPTHS:
            row = encoder(root, case_name(depth, mode))
            depth_bits = DEPTH_VALUE[depth]
            avg_issue = f(row, "graphbit_avg_issue_depth", depth_bits)
            pe_active_depth_scaled = base_matmul_active * avg_issue / 8.0
            rows.append(
                {
                    "mode": mode,
                    "depth": depth.upper(),
                    "depth_bits": depth_bits,
                    "onnx_cycles": f(row, "cycles"),
                    "onnx_cycles_norm_to_p8": ratio(f(row, "cycles"), base_cycles),
                    "onnx_cycle_save_vs_p8": 1.0 - ratio(f(row, "cycles"), base_cycles),
                    "traffic": traffic(row),
                    "traffic_norm_to_p8": ratio(traffic(row), base_traffic),
                    "traffic_save_vs_p8": 1.0 - ratio(traffic(row), base_traffic),
                    "eff_compute": f(row, "graphbit_effective_compute_cycles"),
                    "raw_compute": f(row, "graphbit_raw_compute_cycles"),
                    "eff_compute_norm_to_p8": ratio(
                        f(row, "graphbit_effective_compute_cycles"), base_eff_compute
                    ),
                    "matmul_active": f(row, "matmul_active_cycles"),
                    "matmul_active_depth_scaled": pe_active_depth_scaled,
                    "matmul_active_depth_scaled_norm_to_p8": ratio(
                        pe_active_depth_scaled, base_matmul_active
                    ),
                    "weight_read": f(row, "mem_read_weight"),
                    "weight_read_norm_to_p8": ratio(f(row, "mem_read_weight"), base_weight),
                    "input_read": f(row, "mem_read_input_actual"),
                    "input_read_norm_to_p8": ratio(f(row, "mem_read_input_actual"), base_input),
                    "avg_fetch_depth": f(row, "graphbit_avg_fetch_depth", depth_bits),
                    "avg_issue_depth": avg_issue,
                    "avg_weight_rf_depth": f(row, "graphbit_avg_weight_rf_depth", depth_bits),
                    "avg_psum_depth": f(row, "graphbit_avg_psum_depth", depth_bits),
                }
            )
    return rows


def roofline_rows(root: Path, memory_scales: list[float]) -> list[dict[str, Any]]:
    """Build a what-if critical path model.

    For each mode, use P8 measured ONNX cycles as the current memory/fixed path
    upper envelope, and use ONNXim matmul_active_cycles scaled by issue depth as
    the bit-serial PE critical path proxy.  A memory_scale < 1 models improved
    W reuse / bandwidth / overlap.  Latency-visible mixed-depth savings appear
    only when the memory path falls below the depth-scaled compute path.
    """

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        p8 = encoder(root, case_name("p8", mode))
        p8_cycles = f(p8, "cycles")
        p8_active = f(p8, "matmul_active_cycles")
        for mem_scale in memory_scales:
            memory_path = p8_cycles * mem_scale
            p8_total = max(memory_path, p8_active)
            for depth in DEPTHS:
                row = encoder(root, case_name(depth, mode))
                issue_depth = f(row, "graphbit_avg_issue_depth", DEPTH_VALUE[depth])
                compute_path = p8_active * issue_depth / 8.0
                total = max(memory_path, compute_path)
                rows.append(
                    {
                        "mode": mode,
                        "memory_scale": mem_scale,
                        "depth": depth.upper(),
                        "memory_path": memory_path,
                        "compute_path": compute_path,
                        "roofline_cycles": total,
                        "roofline_norm_to_p8": ratio(total, p8_total),
                        "roofline_save_vs_p8": 1.0 - ratio(total, p8_total),
                        "bottleneck": "memory" if memory_path >= compute_path else "compute",
                    }
                )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_summary(
    path: Path,
    *,
    component_root: Path,
    measured: list[dict[str, Any]],
    roofline: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "Graph-Bit cycle sensitivity diagnosis",
        f"components: {component_root}",
        "",
        "Measured ONNXim component behavior",
        (
            f"{'mode':<7s} {'depth':>5s} {'cycles':>10s} {'C/P8':>7s} "
            f"{'Csave':>7s} {'traffic/P8':>10s} {'Eff/P8':>8s} "
            f"{'PEcrit/P8':>10s} {'W/P8':>7s} {'A/P8':>7s}"
        ),
        "-" * 92,
    ]
    for row in measured:
        lines.append(
            f"{row['mode']:<7s} {row['depth']:>5s} {row['onnx_cycles']:10.0f} "
            f"{fmt(row['onnx_cycles_norm_to_p8']):>7s} "
            f"{pct(row['onnx_cycle_save_vs_p8']):>7s} "
            f"{fmt(row['traffic_norm_to_p8']):>10s} "
            f"{fmt(row['eff_compute_norm_to_p8']):>8s} "
            f"{fmt(row['matmul_active_depth_scaled_norm_to_p8']):>10s} "
            f"{fmt(row['weight_read_norm_to_p8']):>7s} "
            f"{fmt(row['input_read_norm_to_p8']):>7s}"
        )

    lines.extend(
        [
            "",
            "Key diagnosis:",
            "- Eff/P8 shows summed bit-plane compute is scaling correctly.",
            "- PEcrit/P8 shows the intended bit-serial PE critical path if compute becomes exposed.",
            "- C/P8 shows ONNXim wall cycles.  When C/P8 stays near 1.0 while Eff/P8 drops, the saved bit-plane work is hidden by memory/pipeline overlap.",
            "",
            "Roofline what-if sweep",
            "memory_scale shrinks the P8 measured memory/fixed path.  Lower values model stronger W reuse, higher bandwidth, or less memory bottleneck.",
            (
                f"{'mode':<7s} {'mem':>5s} {'P6save':>8s} {'P5save':>8s} "
                f"{'P4save':>8s} {'P6bn':>8s} {'P5bn':>8s} {'P4bn':>8s}"
            ),
            "-" * 68,
        ]
    )

    by_key = {(r["mode"], r["memory_scale"], r["depth"]): r for r in roofline}
    modes = sorted({r["mode"] for r in roofline})
    mems = sorted({r["memory_scale"] for r in roofline}, reverse=True)
    for mode in modes:
        for mem in mems:
            p6 = by_key[(mode, mem, "P6")]
            p5 = by_key[(mode, mem, "P5")]
            p4 = by_key[(mode, mem, "P4")]
            lines.append(
                f"{mode:<7s} {mem:5.2f} {pct(p6['roofline_save_vs_p8']):>8s} "
                f"{pct(p5['roofline_save_vs_p8']):>8s} {pct(p4['roofline_save_vs_p8']):>8s} "
                f"{p6['bottleneck']:>8s} {p5['bottleneck']:>8s} {p4['bottleneck']:>8s}"
            )
        lines.append("")

    lines.extend(
        [
            "Conclusion:",
            "- The current ONNXim components already reduce GraphBit effective compute cycles with depth.",
            "- The reason p8/p6/p5 wall cycles are close is that the exposed critical path is still memory/fixed-path dominated.",
            "- Mixed activation depth should be reported as activity/energy unless the design also exposes PE issue cycles by reducing W/memory pressure or by using a compute-bound bit-serial array.",
            "- For latency claims, use this sweep to state the hardware condition under which A-depth becomes visible.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-root",
        type=Path,
        default=Path("output/onnxim_graphbit/risk_bucket_components_s8"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/onnxim_graphbit/cycle_sensitivity"),
    )
    parser.add_argument(
        "--memory-scales",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.2, 0.15, 0.1],
    )
    args = parser.parse_args()

    measured = measured_rows(args.component_root)
    roofline = roofline_rows(args.component_root, args.memory_scales)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measured_fields = list(measured[0].keys()) if measured else []
    roofline_fields = list(roofline[0].keys()) if roofline else []
    write_tsv(args.output_dir / "measured_components.tsv", measured, measured_fields)
    write_tsv(args.output_dir / "roofline_sensitivity.tsv", roofline, roofline_fields)
    payload = {
        "component_root": str(args.component_root),
        "measured": measured,
        "roofline": roofline,
    }
    (args.output_dir / "cycle_sensitivity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    summary_path = args.output_dir / "cycle_sensitivity.txt"
    write_summary(
        summary_path,
        component_root=args.component_root,
        measured=measured,
        roofline=roofline,
    )
    print(summary_path.read_text(encoding="utf-8"))
    print(f"[GraphBitCycleSensitivity] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
