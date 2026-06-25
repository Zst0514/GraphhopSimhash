#!/usr/bin/env python3
"""Build the missing Section 6 data package from existing traces.

This script is a summarizer.  It does not run the encoder, train a model, or
regenerate embedding pools.  It composes existing BFPA5 GPU logs, ONNXim BFP
cycle traces, HEAT/GFMEngine path-level reconstructions, and CACTI CAM numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent
OUT_ROOT = OFA_ROOT / "output"

DEFAULT_OUTPUT_DIR = OUT_ROOT / "sec6_missing_data_package"
DEFAULT_REPORT = REPO_ROOT / "docs" / "results" / "SEC6_MISSING_DATA_PACKAGE.md"

TASKS = ("CN", "CL", "PN", "PL", "AR", "WK")
TASK_TO_DATASET = {
    "CN": "cora",
    "CL": "cora",
    "PN": "pubmed",
    "PL": "pubmed",
    "AR": "arxiv",
    "WK": "wikics",
}

TASK_NODES = {
    "CN": 2708,
    "CL": 2708,
    "PN": 19717,
    "PL": 19717,
    "AR": 169343,
    "WK": 11701,
}

# PyG-style directed edge_index counts.  These affect only the NDP graph-memory
# tail, which is subdominant in the current LLaMA encoder workload.
TASK_EDGES = {
    "CN": 10556,
    "CL": 10556,
    "PN": 88648,
    "PL": 88648,
    "AR": 1166243,
    "WK": 216123,
}

BFPA5_LOGS = {
    "cora": OUT_ROOT / "bfpa635_b256_generation/cora_W4BFPA5_B256_20260610_134558.log",
    "pubmed": OUT_ROOT / "bfpa635_b256_generation/pubmed_W4BFPA5_B256_20260610_141456.log",
    "arxiv": OUT_ROOT / "bfpa635_b256_generation/arxiv_W4BFPA5_B256_20260610_175543.log",
    "wikics": OUT_ROOT / "bfpa635_b256_generation/wikics_W4BFPA5_B256_20260610_151626.log",
}

ARRAY_SUMMARY = {
    "cora": OUT_ROOT / "e2e_time_breakdown_40reuse/array_cora_graphstress20/summary.json",
    "pubmed": OUT_ROOT / "e2e_time_breakdown_40reuse/array_pubmed_graphstress20/summary.json",
    "arxiv": OUT_ROOT / "e2e_time_breakdown_40reuse/array_arxiv_graphstress10/summary.json",
    "wikics": OUT_ROOT / "e2e_time_breakdown_40reuse/array_wikics_graphstress20/summary.json",
}

REUSE_TSV = OUT_ROOT / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"
BFPA_BOUNDARY_TSV = OUT_ROOT / "bfpa_precision_tasks_cnclpnplarwk/summary.tsv"
BFPA_BLOCK_TSV = OUT_ROOT / "bfpa_boundary_table_runs10/all_blocks/summary.tsv"
TSER_THRESHOLD_TSV = OUT_ROOT / "tser_threshold_sensitivity_plot_data.tsv"

HEAT_JSON = OUT_ROOT / "e2e_run_heat_systolic_tile/frontend_path_timing.json"
GFM_JSONS = {
    "GFMEngine-PQ-M16-BW256": OUT_ROOT / "e2e_run_gfmengine_m16_bw256/pq_frontend_timing.json",
    "GFMEngine-PQ-M64-BW64": OUT_ROOT / "e2e_run_gfmengine_m64_bw64/pq_frontend_timing.json",
    "GFMEngine-PQ-M128-BW64": OUT_ROOT / "e2e_run_gfmengine_m128_bw64/pq_frontend_timing.json",
}

# Cora measured online loader ratio from
# output/cora_gpu_bfpa5_comparison/cora_gpu_bfpa5_comparison.json.
DEFAULT_LOADER_RAW_VS_BFPA5 = 0.026738


@dataclass(frozen=True)
class E2ERow:
    task: str
    dataset: str
    platform: str
    time_s: float
    normalized_time: float
    speedup_vs_gpu: float
    reuse_pct: float
    raw_drop_pct: float
    target_drop_pct: float
    note: str


def parse_duration_to_seconds(text: str) -> float:
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 2:
        return float(60 * parts[0] + parts[1])
    if len(parts) == 3:
        return float(3600 * parts[0] + 60 * parts[1] + parts[2])
    raise ValueError(f"unsupported duration: {text}")


def parse_gpu_log(path: Path) -> tuple[float, str]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    last_line = ""
    seconds = None
    pattern = re.compile(r"Encoding:\s+100%.*\[(\d+(?::\d+){1,2})<")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            last_line = line.strip()
            seconds = parse_duration_to_seconds(match.group(1))
    if seconds is None:
        raise ValueError(f"cannot parse Encoding 100% duration from {path}")
    return seconds, last_line


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def pct(value: float) -> str:
    return f"{value:.2f}%"


def sec(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}s"
    if value >= 1.0e-3:
        return f"{value * 1.0e3:.3f}ms"
    if value >= 1.0e-6:
        return f"{value * 1.0e6:.3f}us"
    return f"{value * 1.0e9:.3f}ns"


def load_reuse_rows() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_tsv(REUSE_TSV):
        task = row["task"]
        out[task] = {
            "reuse_pct": float(row["anchor_reuse"]),
            "raw_drop_pct": float(row["raw_anchor_drop"]),
            "target_drop_pct": float(row["target_anchor_drop"]),
        }
    return out


def load_array_rows(target_refine_ratio: float) -> dict[str, dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for dataset, path in ARRAY_SUMMARY.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        full4 = float(data["full_bfpa4_cycles"])
        full6 = float(data["full_bfpa6_cycles"])
        full5 = full4 + 0.5 * (full6 - full4)
        observed_ratio = float(data["refined_ratio"])
        observed_dynamic = float(data["dynamic_cycles"])
        if observed_ratio > 0.0:
            added_per_ratio = (observed_dynamic - full4) / observed_ratio
            dynamic20 = full4 + target_refine_ratio * added_per_ratio
        else:
            dynamic20 = observed_dynamic
        out[dataset] = {
            "full_bfpa4_cycles": full4,
            "full_bfpa6_cycles": full6,
            "full_bfpa5_cycles": full5,
            "observed_refined_ratio": observed_ratio,
            "target_refined_ratio": target_refine_ratio,
            "dynamic_cycles_target": dynamic20,
            "dynamic_vs_bfpa5_target": dynamic20 / full5,
            "cycle_vs_bfpa4_target": dynamic20 / full4,
            "effective_bits_target": 4.0 + 2.0 * target_refine_ratio,
            "tag": str(data["tag"]),
        }
    return out


def load_platform_norms() -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    heat = json.loads(HEAT_JSON.read_text(encoding="utf-8"))
    heat_norm: dict[str, float] = {}
    heat_drop: dict[str, float] = {}
    for row in heat["rows"]:
        if row["policy"] == "HEAT-style W8A10/W4A2":
            heat_norm[row["task"]] = float(row["total_overlap_norm"])
            heat_drop[row["task"]] = float(row.get("heat_proxy_drop") or math.nan) * 100.0

    gfm_norms: dict[str, dict[str, float]] = {}
    gfm_drops: dict[str, dict[str, float]] = {}
    for label, path in GFM_JSONS.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        gfm_norms[label] = {}
        gfm_drops[label] = {}
        for row in data["rows"]:
            if row["policy"] == "GFMEngine-PQ":
                gfm_norms[label][row["task"]] = float(row["total_overlap_norm"])
                gfm_drops[label][row["task"]] = float(row.get("gfmengine_accuracy_loss") or 0.0) * 100.0
    return heat_norm, gfm_norms, {"HEAT-style": heat_drop, **gfm_drops}


def seconds_from_bytes(num_bytes: float, bandwidth_gbs: float) -> float:
    return num_bytes / (bandwidth_gbs * 1.0e9)


def seconds_from_cycles(cycles: float, clock_mhz: float) -> float:
    return cycles / (clock_mhz * 1.0e6)


def build_e2e_tables(args: argparse.Namespace) -> tuple[list[E2ERow], list[dict], dict[str, float]]:
    gpu_seconds = {dataset: parse_gpu_log(path)[0] for dataset, path in BFPA5_LOGS.items()}
    reuse = load_reuse_rows()
    arrays = load_array_rows(args.target_refine_ratio)
    heat_norm, gfm_norms, _drops = load_platform_norms()

    rows: list[E2ERow] = []
    breakdown: list[dict] = []
    for task in TASKS:
        dataset = TASK_TO_DATASET[task]
        base_s = gpu_seconds[dataset]
        r = reuse[task]
        reuse_frac = r["reuse_pct"] / 100.0
        miss_frac = 1.0 - reuse_frac
        array = arrays[dataset]
        dynamic_vs_bfpa5 = float(array["dynamic_vs_bfpa5_target"])
        encoder_s = base_s * miss_frac * dynamic_vs_bfpa5
        loader_raw_s = base_s * miss_frac * args.loader_raw_vs_bfpa5

        emb_bytes = float(args.embedding_dim) * float(args.embedding_bits) / 8.0
        nodes = TASK_NODES[task]
        edges = TASK_EDGES[task]
        cam_cycles = nodes * (args.cam_search_cycles + args.cam_select_cycles + miss_frac * args.cam_update_cycles)
        cam_s = seconds_from_cycles(cam_cycles, args.ndp_clock_mhz)
        npu_to_ndp_s = seconds_from_bytes(nodes * miss_frac * emb_bytes, args.npu_to_ndp_bw_gbs)
        hit_read_s = seconds_from_bytes(nodes * reuse_frac * emb_bytes, args.ndp_local_dram_bw_gbs)
        graph_index_s = seconds_from_bytes((nodes + 1 + edges) * args.graph_index_bits / 8.0, args.ndp_local_dram_bw_gbs)
        neighbor_read_s = seconds_from_bytes(edges * emb_bytes, args.ndp_local_dram_bw_gbs)
        io_s = cam_s + npu_to_ndp_s + hit_read_s
        graph_mem_s = graph_index_s + neighbor_read_s
        total_overlap_s = encoder_s + io_s + graph_mem_s
        total_serial_s = encoder_s + loader_raw_s + io_s + graph_mem_s

        rows.append(E2ERow(task, dataset, "A100-GPU-BFPA5-proxy", base_s, 1.0, 1.0, 0.0, 0.0, 0.0, "normalized GPU BFPA5 baseline; current seconds parsed from local BFPA5 logs"))
        rows.append(E2ERow(task, dataset, "HEAT-style-bitserial", base_s * heat_norm[task], heat_norm[task], 1.0 / heat_norm[task], 0.0, float("nan"), float("nan"), "path-level W8A10/W4A2 bit-serial reconstruction"))
        gfm_label = args.main_gfm_label
        gfm_norm = gfm_norms[gfm_label][task]
        rows.append(E2ERow(task, dataset, "GFMEngine-PQ-M128-BW64", base_s * gfm_norm, gfm_norm, 1.0 / gfm_norm, 0.0, float("nan"), float("nan"), "quality/lookup-heavy PQ sensitivity point"))
        rows.append(E2ERow(task, dataset, "GRACE-TSER40-BFPLift20", total_overlap_s, total_overlap_s / base_s, base_s / total_overlap_s, r["reuse_pct"], r["raw_drop_pct"], r["target_drop_pct"], "GPU-calibrated trace composition; loader double-buffered"))

        breakdown.append(
            {
                "task": task,
                "dataset": dataset,
                "gpu_bfpa5_baseline_s": base_s,
                "reuse_pct": r["reuse_pct"],
                "miss_pct": 100.0 * miss_frac,
                "raw_drop_pct": r["raw_drop_pct"],
                "target_drop_pct": r["target_drop_pct"],
                "dynamic_vs_bfpa5": dynamic_vs_bfpa5,
                "npu_bfplift_encoder_s": encoder_s,
                "bfp_loader_raw_s": loader_raw_s,
                "cam_lru_s": cam_s,
                "npu_to_ndp_write_s": npu_to_ndp_s,
                "hit_embedding_read_s": hit_read_s,
                "graph_index_load_s": graph_index_s,
                "neighbor_embedding_read_s": neighbor_read_s,
                "total_overlap_s": total_overlap_s,
                "total_serial_s": total_serial_s,
                "speedup_overlap": base_s / total_overlap_s,
                "speedup_serial": base_s / total_serial_s,
            }
        )
    return rows, breakdown, gpu_seconds


def build_energy_rows(e2e_rows: list[E2ERow], args: argparse.Namespace) -> list[dict]:
    power_w = {
        "A100-GPU-BFPA5-proxy": args.gpu_power_w,
        "HEAT-style-bitserial": args.heat_power_w,
        "GFMEngine-PQ-M128-BW64": args.gfm_power_w,
        "GRACE-TSER40-BFPLift20": args.grace_power_w,
    }
    by_task_gpu_energy = {
        row.task: row.time_s * power_w[row.platform]
        for row in e2e_rows
        if row.platform == "A100-GPU-BFPA5-proxy"
    }
    out = []
    for row in e2e_rows:
        e_j = row.time_s * power_w[row.platform]
        out.append(
            {
                "task": row.task,
                "platform": row.platform,
                "time_s": row.time_s,
                "power_w": power_w[row.platform],
                "energy_j": e_j,
                "energy_efficiency_vs_gpu": by_task_gpu_energy[row.task] / e_j,
                "note": "power-model input; replace with RTL power after synthesis",
            }
        )
    return out


def build_cam_rows(breakdown: list[dict]) -> tuple[list[dict], list[dict]]:
    cacti = [
        {
            "entries": 1024,
            "directory_area_mm2": 0.0306,
            "search_energy_nj": 5.88,
            "directory_leakage_mw": 17.69,
            "hot_entries": 64,
            "hot_buffer_area_mm2": 0.5387,
            "hot_embedding_read_nj": 8.07,
            "total_area_with_hot_mm2": 0.5693,
        },
        {
            "entries": 4096,
            "directory_area_mm2": 0.1339,
            "search_energy_nj": 61.18,
            "directory_leakage_mw": 61.75,
            "hot_entries": 64,
            "hot_buffer_area_mm2": 0.5387,
            "hot_embedding_read_nj": 8.07,
            "total_area_with_hot_mm2": 0.6726,
        },
        {
            "entries": 32768,
            "directory_area_mm2": 0.9209,
            "search_energy_nj": 1777.72,
            "directory_leakage_mw": 449.87,
            "hot_entries": 64,
            "hot_buffer_area_mm2": 0.5387,
            "hot_embedding_read_nj": 8.07,
            "total_area_with_hot_mm2": 1.4596,
        },
    ]
    per_task = []
    search_energy_nj = 61.18
    hot_read_nj = 8.07
    for row in breakdown:
        nodes = TASK_NODES[row["task"]]
        reuse_frac = row["reuse_pct"] / 100.0
        cam_energy_j = nodes * search_energy_nj * 1.0e-9 + nodes * reuse_frac * hot_read_nj * 1.0e-9
        per_task.append(
            {
                "task": row["task"],
                "nodes": nodes,
                "reuse_pct": row["reuse_pct"],
                "cam_lru_s": row["cam_lru_s"],
                "cam_lru_us": row["cam_lru_s"] * 1.0e6,
                "cam_energy_j_4k_dir_plus_hot_hits": cam_energy_j,
                "cam_time_pct_of_total": 100.0 * row["cam_lru_s"] / row["total_overlap_s"],
            }
        )
    return cacti, per_task


def build_bfpa_boundary_rows() -> list[dict]:
    rows = []
    for row in read_tsv(BFPA_BOUNDARY_TSV):
        rows.append(
            {
                "task": row["Task"],
                "fp32_metric": row["Base"],
                "bfpa6_drop_pct": row["BFPA6"],
                "bfpa5_drop_pct": row["BFPA5"],
                "bfpa4_drop_pct": row["BFPA4"],
                "bfpa3_drop_pct": row["BFPA3"],
            }
        )
    return rows


def build_sensitivity_rows() -> tuple[list[dict], list[dict], list[dict]]:
    dataset_to_tasks = {
        "cora": ("CN", "CL"),
        "pubmed": ("PN", "PL"),
        "arxiv": ("AR",),
        "wikics": ("WK",),
    }
    block_rows: list[dict] = []
    for row in read_tsv(BFPA_BLOCK_TSV):
        dataset = row["dataset"]
        if dataset not in dataset_to_tasks:
            continue
        for task in dataset_to_tasks[dataset]:
            block_rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "block_size": row["block"],
                    "bfpa4_drop_pct": row["bfpa4_drop"],
                    "bfpa5_drop_pct": row["bfpa5_drop"],
                    "bfpa6_drop_pct": row["bfpa6_drop"],
                    "complete": row["complete"],
                    "source": str(BFPA_BLOCK_TSV),
                }
            )

    threshold_rows = []
    if TSER_THRESHOLD_TSV.exists():
        for row in read_tsv(TSER_THRESHOLD_TSV):
            threshold_rows.append(
                {
                    "task": row["dataset"],
                    "threshold_T": row["T"],
                    "reuse_pct": row["reuse"],
                    "drop_pct": row["drop"],
                    "source": str(TSER_THRESHOLD_TSV),
                }
            )

    hamming_model = []
    for radius in (0, 1, 2, 3, 4):
        # Direct hardware CAM compares all stored signatures in parallel and
        # thresholds the popcount, so lookup latency is fixed in this model.
        hamming_model.append(
            {
                "hamming_radius": radius,
                "cam_search_cycles": 1,
                "select_cycles": 1,
                "update_cycles_on_miss": 1,
                "candidate_discovery_status": "not_measured_in_current_package",
                "note": "existing candidate-discovery table fixes radius=2; run profile_candidate_discovery_ablation.py --radius R for real accuracy/yield sweep",
            }
        )
    return block_rows, threshold_rows, hamming_model


def build_gfm_sensitivity_rows() -> list[dict]:
    rows = []
    for label, path in GFM_JSONS.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for agg in data["aggregate"]:
            if agg["policy"] == "GFMEngine-PQ":
                rows.append(
                    {
                        "scenario": label,
                        "avg_total_norm": agg["total_overlap_seconds_norm"],
                        "avg_speedup": agg["speedup_overlap"],
                        "avg_memory_norm": agg["memory_seconds_norm"],
                        "avg_compute_norm": agg["compute_seconds_norm"],
                        "source": str(path),
                    }
                )
    return rows


def build_hardware_rows(args: argparse.Namespace) -> list[dict]:
    total_macs = args.npu_arrays * 128 * 128
    return [
        {
            "component": "SH-CAM active directory + 64-entry hot buffer",
            "config": "4096 entries, 8x16b signature, 64b metadata, 4096-d FP16 hot embeddings",
            "frequency_mhz": args.ndp_clock_mhz,
            "area_mm2": 0.6726,
            "power_w": 0.06175 + 0.6118,
            "latency_cycles": "2 hit / 3 miss",
            "source": "docs/results/CAM_LRU_CACTI_ESTIMATE.md; dynamic power at 10 Mlookup/s",
        },
        {
            "component": "BFP NPU systolic arrays",
            "config": f"{args.npu_arrays} x 128x128 arrays, aggregate {total_macs} MAC/cycle",
            "frequency_mhz": args.npu_clock_mhz,
            "area_mm2": args.npu_area_mm2,
            "power_w": args.npu_power_w,
            "latency_cycles": "trace driven",
            "source": "energy-model input; replace after RTL synthesis",
        },
        {
            "component": "NDP graph/embedding controller",
            "config": f"local DRAM {args.ndp_local_dram_bw_gbs} GB/s, NPU->NDP {args.npu_to_ndp_bw_gbs} GB/s",
            "frequency_mhz": args.ndp_clock_mhz,
            "area_mm2": args.ndp_area_mm2,
            "power_w": args.ndp_power_w,
            "latency_cycles": "streaming DMA + random local reads",
            "source": "energy-model input; embedding store in NDP-local DRAM",
        },
        {
            "component": "GRACE modeled total",
            "config": "NPU arrays + SH-CAM/NDP control, off-chip DRAM not counted as die area",
            "frequency_mhz": f"NPU {args.npu_clock_mhz}, NDP {args.ndp_clock_mhz}",
            "area_mm2": args.npu_area_mm2 + args.ndp_area_mm2 + 0.6726,
            "power_w": args.grace_power_w,
            "latency_cycles": "see latency_breakdown.tsv",
            "source": "sum of current model inputs",
        },
    ]


def maybe_plot(output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    made: list[str] = []
    e2e = read_tsv(output_dir / "e2e_speedup.tsv")
    platforms = ["A100-GPU-BFPA5-proxy", "HEAT-style-bitserial", "GFMEngine-PQ-M128-BW64", "GRACE-TSER40-BFPLift20"]
    x = range(len(TASKS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    for i, platform in enumerate(platforms):
        vals = [float(next(r for r in e2e if r["task"] == t and r["platform"] == platform)["speedup_vs_gpu"]) for t in TASKS]
        ax.bar([v + (i - 1.5) * width for v in x], vals, width=width, label=platform.replace("-BFPA5-proxy", ""))
    ax.set_xticks(list(x))
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Speedup vs GPU BFPA5")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = output_dir / "sec6_e2e_speedup.pdf"
    fig.savefig(path)
    plt.close(fig)
    made.append(str(path))

    energy = read_tsv(output_dir / "energy_efficiency.tsv")
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    for i, platform in enumerate(platforms):
        vals = [float(next(r for r in energy if r["task"] == t and r["platform"] == platform)["energy_efficiency_vs_gpu"]) for t in TASKS]
        ax.bar([v + (i - 1.5) * width for v in x], vals, width=width, label=platform.replace("-BFPA5-proxy", ""))
    ax.set_xticks(list(x))
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Energy efficiency vs GPU")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = output_dir / "sec6_energy_efficiency.pdf"
    fig.savefig(path)
    plt.close(fig)
    made.append(str(path))

    br = read_tsv(output_dir / "latency_breakdown.tsv")
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    bottoms = [0.0] * len(TASKS)
    fields = [
        ("cam_lru_s", "CAM/LRU"),
        ("npu_to_ndp_write_s", "NPU->NDP write"),
        ("hit_embedding_read_s", "Hit emb read"),
        ("graph_index_load_s", "Graph index"),
        ("neighbor_embedding_read_s", "Neighbor emb read"),
        ("npu_bfplift_encoder_s", "NPU encoder"),
    ]
    for field, label in fields:
        vals = [float(next(r for r in br if r["task"] == t)[field]) for t in TASKS]
        ax.bar(TASKS, vals, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path = output_dir / "sec6_latency_breakdown.pdf"
    fig.savefig(path)
    plt.close(fig)
    made.append(str(path))
    return made


def write_latex_snippets(
    output_dir: Path,
    bfpa_rows: list[dict],
    cam_task: list[dict],
    hw_rows: list[dict],
) -> None:
    latex_dir = output_dir / "latex_snippets"
    latex_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{BFPA precision boundary. Each cell reports metric drop versus FP32.}",
        r"\label{tab:bfpa-boundary-sec6}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & FP32 & BFPA6 & BFPA5 & BFPA4 & BFPA3 \\",
        r"\midrule",
    ]
    for row in bfpa_rows:
        lines.append(
            f"{row['task']} & {row['fp32_metric']} & {row['bfpa6_drop_pct']} & "
            f"{row['bfpa5_drop_pct']} & {row['bfpa4_drop_pct']} & {row['bfpa3_drop_pct']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (latex_dir / "table_bfpa_boundary.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Runtime CAM/LRU overhead at the TSER40 operating point.}",
        r"\label{tab:cam-overhead}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task & Nodes & Reuse & CAM latency ($\mu$s) & CAM energy ($\mu$J) \\",
        r"\midrule",
    ]
    for row in cam_task:
        lines.append(
            f"{row['task']} & {int(row['nodes'])} & {float(row['reuse_pct']):.2f}\\% & "
            f"{float(row['cam_lru_us']):.3f} & {float(row['cam_energy_j_4k_dir_plus_hot_hits']) * 1.0e6:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (latex_dir / "table_cam_overhead.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Current hardware area and power model. NPU and NDP power are model inputs pending RTL synthesis.}",
        r"\label{tab:hardware-area}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Component & Frequency & Area (mm$^2$) & Power (W) \\",
        r"\midrule",
    ]
    for row in hw_rows:
        component = row["component"].replace("_", r"\_")
        lines.append(
            f"{component} & {row['frequency_mhz']} & {float(row['area_mm2']):.3f} & {float(row['power_w']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (latex_dir / "table_hardware_area.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"% Copy the generated PDFs into the paper figure directory or update the paths below.",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{sec6_e2e_speedup.pdf}",
        r"\caption{End-to-end speedup normalized to the GPU BFPA5 encoder baseline.}",
        r"\label{fig:sec6-e2e-speedup}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{sec6_energy_efficiency.pdf}",
        r"\caption{Energy efficiency normalized to the GPU BFPA5 encoder baseline.}",
        r"\label{fig:sec6-energy}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{sec6_latency_breakdown.pdf}",
        r"\caption{GRACE latency breakdown at the TSER40 and BFPLift20 operating point.}",
        r"\label{fig:sec6-latency-breakdown}",
        r"\end{figure}",
        "",
    ]
    (latex_dir / "figures_sec6.tex").write_text("\n".join(lines), encoding="utf-8")


def render_report(
    e2e_rows: list[E2ERow],
    breakdown: list[dict],
    energy_rows: list[dict],
    gpu_seconds: dict[str, float],
    gfm_rows: list[dict],
    figures: list[str],
    args: argparse.Namespace,
) -> str:
    grace = [r for r in e2e_rows if r.platform == "GRACE-TSER40-BFPLift20"]
    heat = [r for r in e2e_rows if r.platform == "HEAT-style-bitserial"]
    gfm = [r for r in e2e_rows if r.platform == "GFMEngine-PQ-M128-BW64"]
    avg = lambda vals: sum(vals) / len(vals)
    energy_grace = [r for r in energy_rows if r["platform"] == "GRACE-TSER40-BFPLift20"]
    energy_heat = [r for r in energy_rows if r["platform"] == "HEAT-style-bitserial"]
    energy_gfm = [r for r in energy_rows if r["platform"] == "GFMEngine-PQ-M128-BW64"]

    lines = [
        "# Section 6 Missing Data Package",
        "",
        "## Scope",
        "",
        "This package fills the current Section 6 placeholders from existing local traces.",
        "It does not launch new encoder generation or training.",
        "",
        "Important provenance:",
        "",
        "- GPU BFPA5 seconds are parsed from existing `output/bfpa635_b256_generation/*W4BFPA5_B256*.log` files. They are local BFPA5 logs, so the A100 label in the paper should be replaced by a real A100 run before final submission.",
        "- GRACE timing is GPU-calibrated trace composition: same node batch/sequence baseline, TSER miss stream, ONNXim BFP cycle ratios, and NDP-local embedding traffic.",
        "- HEAT and GFMEngine are path-level reconstructions from the local `HEAT/` and `GFMEngine/` folders, not official private simulators.",
        "- Energy numbers use explicit power-model inputs. CAM area/energy is CACTI-backed; NPU/GFM/HEAT power should be replaced after RTL synthesis.",
        "",
        "## Main Numbers",
        "",
        f"- Average GRACE speedup vs GPU BFPA5 proxy: `{avg([r.speedup_vs_gpu for r in grace]):.2f}x`.",
        f"- Average HEAT-style bit-serial speedup vs GPU BFPA5 proxy: `{avg([r.speedup_vs_gpu for r in heat]):.2f}x`.",
        f"- Average GFMEngine-PQ M128/BW64 speedup vs GPU BFPA5 proxy: `{avg([r.speedup_vs_gpu for r in gfm]):.2f}x`.",
        f"- Average GRACE energy efficiency vs GPU under the current power model: `{avg([r['energy_efficiency_vs_gpu'] for r in energy_grace]):.2f}x`.",
        f"- GRACE energy-efficiency improvement over HEAT-style: `{avg([g['energy_efficiency_vs_gpu'] / h['energy_efficiency_vs_gpu'] for g, h in zip(energy_grace, energy_heat)]):.2f}x`.",
        f"- GRACE energy-efficiency improvement over GFMEngine-PQ M128/BW64: `{avg([g['energy_efficiency_vs_gpu'] / q['energy_efficiency_vs_gpu'] for g, q in zip(energy_grace, energy_gfm)]):.2f}x`.",
        "",
        "## GPU Timing Inputs",
        "",
        "| Encoder dataset | BFPA5 encoding time | Shared tasks |",
        "| --- | ---: | --- |",
        f"| Cora | {gpu_seconds['cora']:.1f}s | CN, CL |",
        f"| PubMed | {gpu_seconds['pubmed']:.1f}s | PN, PL |",
        f"| OGBN-Arxiv | {gpu_seconds['arxiv']:.1f}s | AR |",
        f"| Wiki-CS | {gpu_seconds['wikics']:.1f}s | WK |",
        "",
        "## E2E Speedup",
        "",
        "| Task | A100/GPU | HEAT-style | GFMEngine-PQ M128 | GRACE | GRACE time | Raw TSER drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        gpu = next(r for r in e2e_rows if r.task == task and r.platform == "A100-GPU-BFPA5-proxy")
        h = next(r for r in e2e_rows if r.task == task and r.platform == "HEAT-style-bitserial")
        q = next(r for r in e2e_rows if r.task == task and r.platform == "GFMEngine-PQ-M128-BW64")
        g = next(r for r in e2e_rows if r.task == task and r.platform == "GRACE-TSER40-BFPLift20")
        lines.append(
            f"| {task} | {gpu.speedup_vs_gpu:.2f}x | {h.speedup_vs_gpu:.2f}x | "
            f"{q.speedup_vs_gpu:.2f}x | {g.speedup_vs_gpu:.2f}x | {g.time_s:.1f}s | {g.raw_drop_pct:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## GRACE Latency Breakdown",
            "",
            "| Task | Total overlap | Encoder | Loader raw | CAM/LRU | NPU->NDP write | Hit emb read | Graph mem |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in breakdown:
        graph_mem = row["graph_index_load_s"] + row["neighbor_embedding_read_s"]
        lines.append(
            f"| {row['task']} | {sec(row['total_overlap_s'])} | {sec(row['npu_bfplift_encoder_s'])} | "
            f"{sec(row['bfp_loader_raw_s'])} | {sec(row['cam_lru_s'])} | {sec(row['npu_to_ndp_write_s'])} | "
            f"{sec(row['hit_embedding_read_s'])} | {sec(graph_mem)} |"
        )

    lines.extend(
        [
            "",
            "## GFMEngine Sensitivity",
            "",
            "| Scenario | Avg norm | Avg speedup | Memory norm | Compute norm |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gfm_rows:
        lines.append(
            f"| {row['scenario']} | {float(row['avg_total_norm']):.4f}x | {float(row['avg_speedup']):.2f}x | "
            f"{float(row['avg_memory_norm']):.4f}x | {float(row['avg_compute_norm']):.4f}x |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `e2e_speedup.tsv`: platform time/speedup table.",
            "- `energy_efficiency.tsv`: energy table under explicit power assumptions.",
            "- `latency_breakdown.tsv`: GRACE component timing in seconds.",
            "- `cam_cacti_table.tsv` and `cam_overhead_by_task.tsv`: CAM area/latency/energy.",
            "- `bfpa_boundary.tsv`: BFPA3/4/5/6 precision boundary.",
            "- `block_size_sensitivity.tsv`: B256/B512 BFPA sensitivity from existing runs.",
            "- `tser_threshold_sensitivity.tsv`: existing T sweep data.",
            "- `hamming_radius_model.tsv`: hardware-latency model only; accuracy/yield radius sweep is not measured in the current package.",
            "- `hardware_area_power.tsv`: current hardware model table.",
            "- `latex_snippets/`: copyable LaTeX tables and figure stubs.",
            "",
        ]
    )
    if figures:
        lines.append("Figures:")
        for fig in figures:
            lines.append(f"- `{fig}`")
        lines.append("")

    lines.extend(
        [
            "## Caveats For Paper Text",
            "",
            "- Do not claim the current absolute seconds are A100-measured until the BFPA5 encoder log is rerun on A100.",
            "- Do not call the Hamming-radius table an accuracy sensitivity result; it is only a CAM latency/candidate-sweep placeholder.",
            "- The current reproducible GRACE speedup is about 1.8x vs the BFPA5 GPU encoder proxy, not the 11.58x sentence currently present in `main.tex`.",
            "- The raw TSER 40% drop average from `tser_reuse_drop_tradeoff_40pt_alignment.tsv` is closer to the 1.7%-1.8% range; the lower target-drop column is a shifted plotting/alignment view.",
        ]
    )
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-refine-ratio", type=float, default=0.20)
    parser.add_argument("--loader-raw-vs-bfpa5", type=float, default=DEFAULT_LOADER_RAW_VS_BFPA5)
    parser.add_argument("--main-gfm-label", default="GFMEngine-PQ-M128-BW64", choices=sorted(GFM_JSONS))
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--embedding-bits", type=int, default=16)
    parser.add_argument("--graph-index-bits", type=int, default=32)
    parser.add_argument("--ndp-clock-mhz", type=float, default=500.0)
    parser.add_argument("--npu-clock-mhz", type=float, default=1000.0)
    parser.add_argument("--npu-arrays", type=int, default=16)
    parser.add_argument("--cam-search-cycles", type=float, default=1.0)
    parser.add_argument("--cam-select-cycles", type=float, default=1.0)
    parser.add_argument("--cam-update-cycles", type=float, default=1.0)
    parser.add_argument("--ndp-local-dram-bw-gbs", type=float, default=256.0)
    parser.add_argument("--npu-to-ndp-bw-gbs", type=float, default=64.0)
    parser.add_argument("--gpu-power-w", type=float, default=400.0)
    parser.add_argument("--heat-power-w", type=float, default=10.5)
    parser.add_argument("--gfm-power-w", type=float, default=98.0)
    parser.add_argument("--grace-power-w", type=float, default=36.0)
    parser.add_argument("--npu-power-w", type=float, default=34.0)
    parser.add_argument("--ndp-power-w", type=float, default=1.32645)
    parser.add_argument("--npu-area-mm2", type=float, default=48.0)
    parser.add_argument("--ndp-area-mm2", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    e2e_rows, breakdown, gpu_seconds = build_e2e_tables(args)
    energy_rows = build_energy_rows(e2e_rows, args)
    cam_cacti, cam_task = build_cam_rows(breakdown)
    bfpa_rows = build_bfpa_boundary_rows()
    block_rows, threshold_rows, hamming_rows = build_sensitivity_rows()
    gfm_rows = build_gfm_sensitivity_rows()
    hw_rows = build_hardware_rows(args)

    write_tsv(
        args.output_dir / "e2e_speedup.tsv",
        [row.__dict__ for row in e2e_rows],
        ["task", "dataset", "platform", "time_s", "normalized_time", "speedup_vs_gpu", "reuse_pct", "raw_drop_pct", "target_drop_pct", "note"],
    )
    write_tsv(args.output_dir / "latency_breakdown.tsv", breakdown, list(breakdown[0].keys()))
    write_tsv(args.output_dir / "energy_efficiency.tsv", energy_rows, list(energy_rows[0].keys()))
    write_tsv(args.output_dir / "cam_cacti_table.tsv", cam_cacti, list(cam_cacti[0].keys()))
    write_tsv(args.output_dir / "cam_overhead_by_task.tsv", cam_task, list(cam_task[0].keys()))
    write_tsv(args.output_dir / "bfpa_boundary.tsv", bfpa_rows, list(bfpa_rows[0].keys()))
    write_tsv(args.output_dir / "block_size_sensitivity.tsv", block_rows, list(block_rows[0].keys()))
    write_tsv(args.output_dir / "tser_threshold_sensitivity.tsv", threshold_rows, list(threshold_rows[0].keys()))
    write_tsv(args.output_dir / "hamming_radius_model.tsv", hamming_rows, list(hamming_rows[0].keys()))
    write_tsv(args.output_dir / "gfmengine_pq_sensitivity.tsv", gfm_rows, list(gfm_rows[0].keys()))
    write_tsv(args.output_dir / "hardware_area_power.tsv", hw_rows, list(hw_rows[0].keys()))
    write_latex_snippets(args.output_dir, bfpa_rows, cam_task, hw_rows)

    figures = maybe_plot(args.output_dir)
    report = render_report(e2e_rows, breakdown, energy_rows, gpu_seconds, gfm_rows, figures, args)
    (args.output_dir / "SEC6_MISSING_DATA_PACKAGE.md").write_text(report + "\n", encoding="utf-8")
    args.report.write_text(report + "\n", encoding="utf-8")
    print(f"[sec6] wrote {args.output_dir}")
    print(f"[sec6] wrote {args.report}")


if __name__ == "__main__":
    main()
