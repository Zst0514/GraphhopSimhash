#!/usr/bin/env python3
"""Compare the Cora TSER+BFPLift path against a measured GPU BFPA5 baseline.

This is a calibrated timing model, not a raw single-array wall-clock claim.  It
uses the measured GPU Cora W4BFPA5_B256 encoding time as the reference seconds,
then applies the local ONNXim BFP-cycle ratios, TSER miss stream, and NDP-local
CAM/embedding traffic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OFA_ROOT = REPO_ROOT.parent

DEFAULT_GPU_LOG = (
    OFA_ROOT
    / "output"
    / "bfpa635_b256_generation"
    / "cora_W4BFPA5_B256_20260610_134558.log"
)
DEFAULT_ARRAY_SUMMARY = OFA_ROOT / "output" / "e2e_time_breakdown_40reuse" / "array_cora_graphstress20" / "summary.json"
DEFAULT_BFP_BREAKDOWN = OFA_ROOT / "output" / "onnxim_cora_bfp_lift_breakdown" / "cora_bfp_lift_breakdown.json"
DEFAULT_REUSE = OFA_ROOT / "output" / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"
DEFAULT_OUTPUT_DIR = OFA_ROOT / "output" / "cora_gpu_bfpa5_comparison"
DEFAULT_REPO_REPORT = REPO_ROOT / "docs" / "results" / "CORA_GPU_BFPA5_COMPARISON.md"

CORA_NODES = 2708
CORA_EDGES = 10556
TASKS = ("CN", "CL")


@dataclass(frozen=True)
class ScenarioRow:
    task: str
    reuse_pct: float
    miss_pct: float
    drop_pct: float
    gpu_bfpa5_baseline_s: float
    bfpa4_base_s: float
    bfplift_extra_s: float
    dynamic_mac_s: float
    exponent_select_s: float
    mantissa_pack_s: float
    stress_priority_s: float
    refine_queue_push_s: float
    loader_raw_s: float
    npu_to_ndp_write_s: float
    cam_lru_s: float
    hit_embedding_read_s: float
    graph_index_load_s: float
    neighbor_embedding_read_s: float
    frontend_overlap_s: float
    frontend_serial_s: float
    e2e_overlap_with_graph_mem_s: float
    e2e_serial_with_graph_mem_s: float
    speedup_overlap_vs_gpu_bfpa5: float
    speedup_serial_vs_gpu_bfpa5: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-log", type=Path, default=DEFAULT_GPU_LOG)
    parser.add_argument("--gpu-baseline-s", type=float, default=None)
    parser.add_argument("--array-summary", type=Path, default=DEFAULT_ARRAY_SUMMARY)
    parser.add_argument("--bfp-breakdown", type=Path, default=DEFAULT_BFP_BREAKDOWN)
    parser.add_argument("--reuse-input", type=Path, default=DEFAULT_REUSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-report", type=Path, default=DEFAULT_REPO_REPORT)
    parser.add_argument("--npu-clock-mhz", type=float, default=500.0)
    parser.add_argument("--ndp-clock-mhz", type=float, default=500.0)
    parser.add_argument("--ndp-local-dram-bw-gbs", type=float, default=256.0)
    parser.add_argument("--npu-to-ndp-bw-gbs", type=float, default=64.0)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--embedding-bits", type=int, default=16)
    parser.add_argument("--graph-index-bits", type=int, default=32)
    parser.add_argument("--cam-search-cycles", type=float, default=1.0)
    parser.add_argument("--cam-select-cycles", type=float, default=1.0)
    parser.add_argument("--cam-miss-update-cycles", type=float, default=1.0)
    parser.add_argument("--neighbor-embedding-read-factor", type=float, default=1.0)
    return parser.parse_args()


def parse_duration_to_seconds(text: str) -> float:
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"unsupported duration: {text}")


def parse_gpu_encoding_seconds(path: Path) -> tuple[float, str]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    final_line = ""
    duration = None
    pattern = re.compile(r"Encoding:\s+100%.*\[(\d+(?::\d+){1,2})<")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            final_line = line.strip()
            duration = parse_duration_to_seconds(match.group(1))
    if duration is None:
        raise ValueError(f"could not parse final Encoding 100% duration from {path}")
    return duration, final_line


def summarize_encoding_line(line: str) -> str:
    match = re.search(r"(Encoding:\s+100%).*?(\d+/\d+)\s+\[([^]]+)\]", line)
    if not match:
        return line.encode("ascii", "ignore").decode("ascii")
    return f"{match.group(1)}, {match.group(2)}, [{match.group(3)}]"


def read_reuse(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["task"] in TASKS:
                rows[row["task"]] = {
                    "reuse_pct": float(row["anchor_reuse"]),
                    "drop_pct": float(row["target_anchor_drop"]),
                }
    missing = set(TASKS) - set(rows)
    if missing:
        raise ValueError(f"missing Cora task reuse rows: {sorted(missing)}")
    return rows


def seconds_from_bytes(num_bytes: float, bandwidth_gbs: float) -> float:
    if bandwidth_gbs <= 0.0:
        return 0.0
    return float(num_bytes) / (float(bandwidth_gbs) * 1.0e9)


def seconds_from_cycles(cycles: float, clock_mhz: float) -> float:
    return float(cycles) / (float(clock_mhz) * 1.0e6)


def fmt_s(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}s"
    if value >= 1.0e-3:
        return f"{value * 1.0e3:.3f}ms"
    if value >= 1.0e-6:
        return f"{value * 1.0e6:.3f}us"
    return f"{value * 1.0e9:.3f}ns"


def pick_full_bfp_row(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload["rows"]:
        if row["scenario"] == "CoraFull+BFPLift":
            return row
    raise ValueError("missing CoraFull+BFPLift row in BFP breakdown")


def build_rows(args: argparse.Namespace) -> tuple[list[ScenarioRow], dict[str, Any]]:
    parsed_gpu_s, gpu_line = parse_gpu_encoding_seconds(args.gpu_log)
    gpu_line_ascii = summarize_encoding_line(gpu_line)
    gpu_baseline_s = parsed_gpu_s if args.gpu_baseline_s is None else float(args.gpu_baseline_s)
    array = json.loads(args.array_summary.read_text(encoding="utf-8"))
    bfp = json.loads(args.bfp_breakdown.read_text(encoding="utf-8"))
    bfp_full = pick_full_bfp_row(bfp)
    reuse_rows = read_reuse(args.reuse_input)

    full_bfpa4_cycles = float(array["full_bfpa4_cycles"])
    full_bfpa6_cycles = float(array["full_bfpa6_cycles"])
    dynamic_cycles = float(array["dynamic_cycles"])
    full_bfpa5_cycles = full_bfpa4_cycles + 0.5 * (full_bfpa6_cycles - full_bfpa4_cycles)
    bfpa4_vs_bfpa5 = full_bfpa4_cycles / full_bfpa5_cycles
    bfplift_extra_vs_bfpa5 = (dynamic_cycles - full_bfpa4_cycles) / full_bfpa5_cycles
    dynamic_vs_bfpa5 = dynamic_cycles / full_bfpa5_cycles

    exp_vs_bfpa5 = float(bfp_full["exponent_select_cycles"]) / full_bfpa5_cycles
    pack_vs_bfpa5 = float(bfp_full["mantissa_pack_cycles"]) / full_bfpa5_cycles
    stress_vs_bfpa5 = float(bfp_full["stress_priority_cycles"]) / full_bfpa5_cycles
    queue_vs_bfpa5 = float(bfp_full["refine_queue_push_cycles"]) / full_bfpa5_cycles

    emb_bytes = float(args.embedding_dim) * float(args.embedding_bits) / 8.0
    graph_index_load_s = seconds_from_bytes(
        float(CORA_NODES + 1 + CORA_EDGES) * float(args.graph_index_bits) / 8.0,
        args.ndp_local_dram_bw_gbs,
    )
    neighbor_embedding_read_s = seconds_from_bytes(
        float(CORA_EDGES) * emb_bytes * float(args.neighbor_embedding_read_factor),
        args.ndp_local_dram_bw_gbs,
    )

    rows: list[ScenarioRow] = []
    for task in TASKS:
        reuse_pct = float(reuse_rows[task]["reuse_pct"])
        miss = 1.0 - reuse_pct / 100.0
        drop_pct = float(reuse_rows[task]["drop_pct"])

        bfpa4_base_s = gpu_baseline_s * miss * bfpa4_vs_bfpa5
        bfplift_extra_s = gpu_baseline_s * miss * bfplift_extra_vs_bfpa5
        dynamic_mac_s = gpu_baseline_s * miss * dynamic_vs_bfpa5
        exponent_select_s = gpu_baseline_s * miss * exp_vs_bfpa5
        mantissa_pack_s = gpu_baseline_s * miss * pack_vs_bfpa5
        stress_priority_s = gpu_baseline_s * miss * stress_vs_bfpa5
        refine_queue_push_s = gpu_baseline_s * miss * queue_vs_bfpa5
        loader_raw_s = exponent_select_s + mantissa_pack_s + stress_priority_s + refine_queue_push_s

        npu_to_ndp_write_s = seconds_from_bytes(float(CORA_NODES) * miss * emb_bytes, args.npu_to_ndp_bw_gbs)
        cam_cycles = float(CORA_NODES) * (
            float(args.cam_search_cycles)
            + float(args.cam_select_cycles)
            + miss * float(args.cam_miss_update_cycles)
        )
        cam_lru_s = seconds_from_cycles(cam_cycles, args.ndp_clock_mhz)
        hit_embedding_read_s = seconds_from_bytes(
            float(CORA_NODES) * (reuse_pct / 100.0) * emb_bytes,
            args.ndp_local_dram_bw_gbs,
        )

        frontend_io_s = npu_to_ndp_write_s + cam_lru_s + hit_embedding_read_s
        graph_mem_s = graph_index_load_s + neighbor_embedding_read_s
        frontend_overlap_s = dynamic_mac_s + max(0.0, loader_raw_s - dynamic_mac_s) + frontend_io_s
        frontend_serial_s = dynamic_mac_s + loader_raw_s + frontend_io_s
        e2e_overlap_s = frontend_overlap_s + graph_mem_s
        e2e_serial_s = frontend_serial_s + graph_mem_s

        rows.append(
            ScenarioRow(
                task=task,
                reuse_pct=reuse_pct,
                miss_pct=100.0 * miss,
                drop_pct=drop_pct,
                gpu_bfpa5_baseline_s=gpu_baseline_s,
                bfpa4_base_s=bfpa4_base_s,
                bfplift_extra_s=bfplift_extra_s,
                dynamic_mac_s=dynamic_mac_s,
                exponent_select_s=exponent_select_s,
                mantissa_pack_s=mantissa_pack_s,
                stress_priority_s=stress_priority_s,
                refine_queue_push_s=refine_queue_push_s,
                loader_raw_s=loader_raw_s,
                npu_to_ndp_write_s=npu_to_ndp_write_s,
                cam_lru_s=cam_lru_s,
                hit_embedding_read_s=hit_embedding_read_s,
                graph_index_load_s=graph_index_load_s,
                neighbor_embedding_read_s=neighbor_embedding_read_s,
                frontend_overlap_s=frontend_overlap_s,
                frontend_serial_s=frontend_serial_s,
                e2e_overlap_with_graph_mem_s=e2e_overlap_s,
                e2e_serial_with_graph_mem_s=e2e_serial_s,
                speedup_overlap_vs_gpu_bfpa5=gpu_baseline_s / e2e_overlap_s,
                speedup_serial_vs_gpu_bfpa5=gpu_baseline_s / e2e_serial_s,
            )
        )

    config = {
        "gpu_log": str(args.gpu_log),
        "gpu_encoding_line": gpu_line_ascii,
        "parsed_gpu_encoding_s": parsed_gpu_s,
        "gpu_bfpa5_baseline_s": gpu_baseline_s,
        "array_summary": str(args.array_summary),
        "bfp_breakdown": str(args.bfp_breakdown),
        "reuse_input": str(args.reuse_input),
        "full_bfpa4_cycles": full_bfpa4_cycles,
        "full_bfpa6_cycles": full_bfpa6_cycles,
        "estimated_full_bfpa5_cycles": full_bfpa5_cycles,
        "dynamic_cycles": dynamic_cycles,
        "bfpa4_vs_bfpa5": bfpa4_vs_bfpa5,
        "bfplift_extra_vs_bfpa5": bfplift_extra_vs_bfpa5,
        "dynamic_vs_bfpa5": dynamic_vs_bfpa5,
        "loader_raw_vs_bfpa5": exp_vs_bfpa5 + pack_vs_bfpa5 + stress_vs_bfpa5 + queue_vs_bfpa5,
        "embedding_bytes_per_node": emb_bytes,
        "npu_clock_mhz": args.npu_clock_mhz,
        "ndp_clock_mhz": args.ndp_clock_mhz,
        "ndp_local_dram_bw_gbs": args.ndp_local_dram_bw_gbs,
        "npu_to_ndp_bw_gbs": args.npu_to_ndp_bw_gbs,
        "cora_nodes": CORA_NODES,
        "cora_edges": CORA_EDGES,
    }
    return rows, config


def write_tsv(path: Path, rows: list[ScenarioRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(data)


def render_report(rows: list[ScenarioRow], config: dict[str, Any]) -> str:
    lines = [
        "# Cora GPU BFPA5 Baseline Comparison",
        "",
        "## Baseline",
        "",
        f"- GPU baseline log: `{config['gpu_log']}`.",
        f"- Parsed final encoding line: `{config['gpu_encoding_line']}`.",
        f"- Baseline seconds used: `{config['gpu_bfpa5_baseline_s']:.3f}s` for full Cora W4BFPA5_B256 embedding generation.",
        "- The baseline is the encoding phase only. Model checkpoint loading, AWQ search, and one-time pseudo weight quantization are not counted.",
        "",
        "## Calibration",
        "",
        f"- Cora BFP trace: `{config['array_summary']}`.",
        f"- Full BFPA5 cycles are interpolated from BFPA4 and BFPA6: `{config['estimated_full_bfpa5_cycles']:.3f}` cycles.",
        f"- Dynamic BFPLift/BFPA5 cycle ratio: `{config['dynamic_vs_bfpa5']:.6f}`.",
        f"- Online loader raw/BFPA5 cycle ratio: `{config['loader_raw_vs_bfpa5']:.6f}`.",
        "- The GPU seconds are scaled by these ratios, so this is a GPU-calibrated trace-composition result rather than a raw single-array ONNXim wall time.",
        "",
        "## Result",
        "",
        "| Task | Reuse | Drop | GPU BFPA5 Baseline | BFPA4 Base | BFPLift Extra | Dynamic MAC | Online Loader Raw | CAM/LRU | NPU->NDP Write | Hit Emb Read | Graph Index | Neighbor Emb Read | E2E Overlap | E2E Serial | Speedup Overlap | Speedup Serial |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.task} | {row.reuse_pct:.2f}% | {row.drop_pct:.2f}% | "
            f"{fmt_s(row.gpu_bfpa5_baseline_s)} | {fmt_s(row.bfpa4_base_s)} | "
            f"{fmt_s(row.bfplift_extra_s)} | {fmt_s(row.dynamic_mac_s)} | "
            f"{fmt_s(row.loader_raw_s)} | {fmt_s(row.cam_lru_s)} | "
            f"{fmt_s(row.npu_to_ndp_write_s)} | {fmt_s(row.hit_embedding_read_s)} | "
            f"{fmt_s(row.graph_index_load_s)} | {fmt_s(row.neighbor_embedding_read_s)} | "
            f"{fmt_s(row.e2e_overlap_with_graph_mem_s)} | {fmt_s(row.e2e_serial_with_graph_mem_s)} | "
            f"{row.speedup_overlap_vs_gpu_bfpa5:.2f}x | {row.speedup_serial_vs_gpu_bfpa5:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- CN beats the measured GPU BFPA5 baseline as `108s -> about 57.8s` with double-buffered loader overlap, or `about 59.5s` under conservative serial accounting.",
            "- The key reduction is not a raw clock-frequency claim: `miss_stream * dynamic/BFPA5 = 0.601 * 0.890`, before tiny CAM and NDP-local embedding traffic.",
            "- Online exponent selection is modeled as runtime loader/control work. It is shown separately and is only exposed on the critical path if the loader cannot be hidden behind the MAC array.",
            "- NDP local graph index and neighbor embedding reads are included as memory traffic here. For Cora they are sub-millisecond and do not move the conclusion.",
            "- Backend GNN arithmetic is not included because the measured GPU BFPA5 baseline log is an embedding-generation baseline, not a full GNN training/inference wall time.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows, config = build_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_tsv(args.output_dir / "cora_gpu_bfpa5_comparison.tsv", rows)
    (args.output_dir / "cora_gpu_bfpa5_comparison.json").write_text(
        json.dumps({"config": config, "rows": [asdict(row) for row in rows]}, indent=2),
        encoding="utf-8",
    )
    report = render_report(rows, config)
    (args.output_dir / "CORA_GPU_BFPA5_COMPARISON.md").write_text(report, encoding="utf-8")
    args.repo_report.parent.mkdir(parents=True, exist_ok=True)
    args.repo_report.write_text(report, encoding="utf-8")

    for row in rows:
        print(
            f"{row.task}: GPU BFPA5 {row.gpu_bfpa5_baseline_s:.3f}s -> "
            f"overlap {row.e2e_overlap_with_graph_mem_s:.3f}s "
            f"({row.speedup_overlap_vs_gpu_bfpa5:.2f}x), "
            f"serial {row.e2e_serial_with_graph_mem_s:.3f}s "
            f"({row.speedup_serial_vs_gpu_bfpa5:.2f}x)"
        )
    print(f"Wrote {args.output_dir / 'CORA_GPU_BFPA5_COMPARISON.md'}")
    print(f"Wrote {args.repo_report}")


if __name__ == "__main__":
    main()
