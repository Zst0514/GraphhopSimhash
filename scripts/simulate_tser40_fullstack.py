#!/usr/bin/env python3
"""Compose TSER-40 full-stack timing from frontend, BFP, and graph-head terms.

The script fixes the main policy point to the current paper setting:

    TSER-selected reuse near 40%

Inputs are intentionally small and auditable:

* a TSV containing the selected 40%-reuse TSER points;
* per-encoder-dataset progressive-BFP array summaries;
* configurable frontend/embedding/GNN timing knobs.

The output is a hardware-style timing table in cycles and seconds at the
configured clock.  It is not a monolithic cycle-accurate system simulator; it is
a trace/composition layer that keeps each component's contribution visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent

DEFAULT_REUSE_INPUT = OFA_ROOT / "output" / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"
DEFAULT_ARRAY_ROOT = OFA_ROOT / "output" / "e2e_time_breakdown_40reuse"
DEFAULT_OUTPUT_DIR = OFA_ROOT / "output" / "tser40_fullstack_sim"

TASKS = ("CN", "CL", "PN", "PL", "AR", "WK")

TASK_TO_ENCODER_DATASET = {
    "CN": "cora",
    "CL": "cora",
    "PN": "pubmed",
    "PL": "pubmed",
    "AR": "arxiv",
    "WK": "wikics",
}

# Node counts used by the current TAG workloads.  Keep this local and explicit
# so the timing table can be regenerated without loading PyG datasets.
TASK_NODE_COUNTS = {
    "CN": 2708,
    "CL": 2708,
    "PN": 19717,
    "PL": 19717,
    "AR": 169343,
    "WK": 11701,
}

ARRAY_SUMMARY_BY_DATASET = {
    "cora": DEFAULT_ARRAY_ROOT / "array_cora_graphstress20" / "summary.json",
    "pubmed": DEFAULT_ARRAY_ROOT / "array_pubmed_graphstress20" / "summary.json",
    "arxiv": DEFAULT_ARRAY_ROOT / "array_arxiv_graphstress10" / "summary.json",
    "wikics": DEFAULT_ARRAY_ROOT / "array_wikics_graphstress20" / "summary.json",
}


@dataclass
class ReusePoint:
    task: str
    reuse_pct: float
    drop_pct: float


@dataclass
class ArraySummary:
    dataset: str
    tag: str
    full_bfpa8_cycles: float
    dynamic_cycles: float
    dynamic_vs_bfpa8_cycles: float
    refined_ratio: float
    effective_bits: float


@dataclass
class TimingRow:
    task: str
    encoder_dataset: str
    policy: str
    nodes: int
    reuse_pct: float
    miss_pct: float
    drop_pct: float
    frontend_cycles: float
    embedding_cycles: float
    scheduler_cycles: float
    encoder_cycles: float
    gnn_cycles: float
    total_cycles: float
    time_ms: float
    time_s: float
    norm_vs_noreuse_bfpa8: float
    speedup_vs_noreuse_bfpa8: float
    bfp_tag: str
    refined_blocks_pct: float
    effective_bits: float


def read_reuse_points(path: Path, *, target: float, tolerance: float) -> dict[str, ReusePoint]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, ReusePoint] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            task = row["task"]
            reuse = float(row["anchor_reuse"])
            if abs(reuse - target) > tolerance:
                raise ValueError(
                    f"{task} reuse point {reuse:.2f}% is outside "
                    f"target {target:.2f}% +/- {tolerance:.2f}%"
                )
            out[task] = ReusePoint(
                task=task,
                reuse_pct=reuse,
                drop_pct=float(row["target_anchor_drop"]),
            )
    missing = sorted(set(TASKS) - set(out))
    if missing:
        raise ValueError(f"Missing reuse rows for tasks: {', '.join(missing)}")
    return out


def read_array_summaries(paths: dict[str, Path]) -> dict[str, ArraySummary]:
    out: dict[str, ArraySummary] = {}
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[dataset] = ArraySummary(
            dataset=dataset,
            tag=str(payload.get("tag", "unknown")),
            full_bfpa8_cycles=float(payload["full_bfpa8_cycles"]),
            dynamic_cycles=float(payload["dynamic_cycles"]),
            dynamic_vs_bfpa8_cycles=float(payload["dynamic_vs_bfpa8_cycles"]),
            refined_ratio=float(payload["refined_ratio"]),
            effective_bits=float(payload["effective_bits"]),
        )
    return out


def frontend_cycles_for_queries(
    *,
    nodes: int,
    miss_rate: float,
    search_cycles: float,
    select_cycles: float,
    miss_update_cycles: float,
    tser_cycles: float,
) -> float:
    return float(nodes) * (search_cycles + select_cycles + tser_cycles + miss_rate * miss_update_cycles)


def embedding_cycles(
    *,
    nodes: int,
    reuse_rate: float,
    miss_rate: float,
    embedding_dim: int,
    embedding_bits: int,
    bandwidth_gbps: float,
    clock_mhz: float,
) -> float:
    bytes_per_embedding = embedding_dim * embedding_bits / 8.0
    bytes_per_cycle = bandwidth_gbps * 1.0e9 / (clock_mhz * 1.0e6)
    if bytes_per_cycle <= 0.0:
        raise ValueError("embedding bandwidth must be positive")
    # Direct reuse reads cached embeddings; miss nodes write newly computed ones.
    transfer_bytes = float(nodes) * (reuse_rate + miss_rate) * bytes_per_embedding
    return transfer_bytes / bytes_per_cycle


def build_rows(args: argparse.Namespace) -> list[TimingRow]:
    reuse_points = read_reuse_points(args.reuse_input, target=args.target_reuse, tolerance=args.reuse_tolerance)
    array_rows = read_array_summaries(ARRAY_SUMMARY_BY_DATASET)

    rows: list[TimingRow] = []
    baseline_cycles_by_task: dict[str, float] = {}

    for task in TASKS:
        point = reuse_points[task]
        dataset = TASK_TO_ENCODER_DATASET[task]
        array = array_rows[dataset]
        nodes = TASK_NODE_COUNTS[task]
        reuse_rate = point.reuse_pct / 100.0
        miss_rate = 1.0 - reuse_rate

        gnn_cycles = args.gnn_norm_of_full_encoder * array.full_bfpa8_cycles
        scheduler_cycles = args.scheduler_norm_of_full_encoder * array.full_bfpa8_cycles

        baseline_embedding = embedding_cycles(
            nodes=nodes,
            reuse_rate=0.0,
            miss_rate=1.0,
            embedding_dim=args.embedding_dim,
            embedding_bits=args.embedding_bits,
            bandwidth_gbps=args.embedding_bandwidth_gbps,
            clock_mhz=args.clock_mhz,
        )
        baseline_total = array.full_bfpa8_cycles + baseline_embedding + gnn_cycles
        baseline_cycles_by_task[task] = baseline_total
        rows.append(
            TimingRow(
                task=task,
                encoder_dataset=dataset,
                policy="NoReuse+BFPA8",
                nodes=nodes,
                reuse_pct=0.0,
                miss_pct=100.0,
                drop_pct=0.0,
                frontend_cycles=0.0,
                embedding_cycles=baseline_embedding,
                scheduler_cycles=0.0,
                encoder_cycles=array.full_bfpa8_cycles,
                gnn_cycles=gnn_cycles,
                total_cycles=baseline_total,
                time_ms=baseline_total / (args.clock_mhz * 1.0e3),
                time_s=baseline_total / (args.clock_mhz * 1.0e6),
                norm_vs_noreuse_bfpa8=1.0,
                speedup_vs_noreuse_bfpa8=1.0,
                bfp_tag="W4BFPA8_B128",
                refined_blocks_pct=0.0,
                effective_bits=8.0,
            )
        )

        front = frontend_cycles_for_queries(
            nodes=nodes,
            miss_rate=miss_rate,
            search_cycles=args.frontend_search_cycles,
            select_cycles=args.frontend_select_cycles,
            miss_update_cycles=args.frontend_miss_update_cycles,
            tser_cycles=args.tser_cycles,
        )
        embed = embedding_cycles(
            nodes=nodes,
            reuse_rate=reuse_rate,
            miss_rate=miss_rate,
            embedding_dim=args.embedding_dim,
            embedding_bits=args.embedding_bits,
            bandwidth_gbps=args.embedding_bandwidth_gbps,
            clock_mhz=args.clock_mhz,
        )

        policy_specs = (
            ("TSER40+BFPA8", miss_rate * array.full_bfpa8_cycles, "W4BFPA8_B128", 0.0, 8.0),
            (
                "TSER40+DynBFP",
                miss_rate * array.dynamic_cycles,
                array.tag,
                array.refined_ratio * 100.0,
                array.effective_bits,
            ),
        )
        for policy, encoder_cycles, tag, refined_pct, eff_bits in policy_specs:
            total = front + embed + scheduler_cycles + encoder_cycles + gnn_cycles
            rows.append(
                TimingRow(
                    task=task,
                    encoder_dataset=dataset,
                    policy=policy,
                    nodes=nodes,
                    reuse_pct=point.reuse_pct,
                    miss_pct=miss_rate * 100.0,
                    drop_pct=point.drop_pct,
                    frontend_cycles=front,
                    embedding_cycles=embed,
                    scheduler_cycles=scheduler_cycles,
                    encoder_cycles=encoder_cycles,
                    gnn_cycles=gnn_cycles,
                    total_cycles=total,
                    time_ms=total / (args.clock_mhz * 1.0e3),
                    time_s=total / (args.clock_mhz * 1.0e6),
                    norm_vs_noreuse_bfpa8=total / baseline_cycles_by_task[task],
                    speedup_vs_noreuse_bfpa8=baseline_cycles_by_task[task] / total,
                    bfp_tag=tag,
                    refined_blocks_pct=refined_pct,
                    effective_bits=eff_bits,
                )
            )
    return rows


def write_tsv(path: Path, rows: Iterable[TimingRow]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_cycles(value: float) -> str:
    if abs(value) >= 1.0e12:
        return f"{value / 1.0e12:.3f}T"
    if abs(value) >= 1.0e9:
        return f"{value / 1.0e9:.3f}B"
    if abs(value) >= 1.0e6:
        return f"{value / 1.0e6:.3f}M"
    return f"{value:.0f}"


def write_markdown(path: Path, rows: list[TimingRow], args: argparse.Namespace) -> None:
    dyn_rows = [row for row in rows if row.policy == "TSER40+DynBFP"]
    avg_norm = sum(row.norm_vs_noreuse_bfpa8 for row in dyn_rows) / len(dyn_rows)
    avg_speedup = sum(row.speedup_vs_noreuse_bfpa8 for row in dyn_rows) / len(dyn_rows)

    lines = [
        "# TSER40 Full-Stack Timing Simulation",
        "",
        "## Scope",
        "",
        "This report fixes the frontend operating point to TSER-selected reuse near 40%.",
        "Timing is composed from CAM/LRU frontend cycles, embedding movement, progressive-BFP encoder cycles, and a configurable GNN/task-head term.",
        "",
        "## Timing Model",
        "",
        f"- Clock: `{args.clock_mhz:.0f} MHz`.",
        f"- Reuse input: `{args.reuse_input}`.",
        f"- Target reuse: `{args.target_reuse:.1f}% +/- {args.reuse_tolerance:.1f}%`.",
        f"- Frontend/query cycles: search `{args.frontend_search_cycles:g}` + select `{args.frontend_select_cycles:g}` + TSER `{args.tser_cycles:g}` + miss_update `{args.frontend_miss_update_cycles:g} * miss_rate`.",
        f"- Embedding movement: `{args.embedding_dim}` x `{args.embedding_bits}`b per node at `{args.embedding_bandwidth_gbps:g} GB/s`.",
        f"- Scheduler overhead: `{args.scheduler_norm_of_full_encoder:g}` x full BFPA8 encoder cycles.",
        f"- GNN/task-head overhead: `{args.gnn_norm_of_full_encoder:g}` x full BFPA8 encoder cycles.",
        "",
        "## Main TSER40 Result",
        "",
        "| Task | Nodes | Reuse | Drop | Dyn Tag | Eff. Bits | Frontend | Embed | Encoder | GNN | Total Time | Norm. | Speedup |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dyn_rows:
        lines.append(
            f"| {row.task} | {row.nodes:,} | {fmt_pct(row.reuse_pct)} | {fmt_pct(row.drop_pct)} | "
            f"`{row.bfp_tag}` | {row.effective_bits:.3f} | {fmt_cycles(row.frontend_cycles)} | "
            f"{fmt_cycles(row.embedding_cycles)} | {fmt_cycles(row.encoder_cycles)} | "
            f"{fmt_cycles(row.gnn_cycles)} | {row.time_s:.3f}s | "
            f"{row.norm_vs_noreuse_bfpa8:.3f}x | {row.speedup_vs_noreuse_bfpa8:.2f}x |"
        )
    lines.extend(
        [
            "",
            f"Average normalized time: `{avg_norm:.3f}x` of NoReuse+BFPA8.",
            f"Average speedup: `{avg_speedup:.2f}x` over NoReuse+BFPA8.",
            "",
            "## Policy Breakdown",
            "",
            "| Task | Policy | Reuse | Miss | Frontend | Embed | Scheduler | Encoder | GNN | Total Cycles | Time | Norm. | Speedup |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.task} | {row.policy} | {fmt_pct(row.reuse_pct)} | {fmt_pct(row.miss_pct)} | "
            f"{fmt_cycles(row.frontend_cycles)} | {fmt_cycles(row.embedding_cycles)} | "
            f"{fmt_cycles(row.scheduler_cycles)} | {fmt_cycles(row.encoder_cycles)} | "
            f"{fmt_cycles(row.gnn_cycles)} | {fmt_cycles(row.total_cycles)} | "
            f"{row.time_s:.3f}s | {row.norm_vs_noreuse_bfpa8:.3f}x | "
            f"{row.speedup_vs_noreuse_bfpa8:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The CAM/LRU frontend is modeled with actual 500MHz query cycles rather than a fixed normalized filter penalty.",
            "- Embedding traffic includes one 4096-d FP16 cached-embedding read for reuse nodes and one write for miss nodes.",
            "- Encoder time dominates under the configured BFP backend; frontend and embedding movement remain visible but small.",
            "- The absolute seconds are hardware-model seconds at the configured clock, not RTX4090 wall-clock seconds.",
            "",
            f"Raw TSV: `{path.with_suffix('.tsv')}`",
            f"Raw JSON: `{path.with_suffix('.json')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, rows: Iterable[TimingRow], args: argparse.Namespace) -> None:
    payload = {
        "model": {
            "clock_mhz": args.clock_mhz,
            "target_reuse": args.target_reuse,
            "reuse_tolerance": args.reuse_tolerance,
            "frontend_search_cycles": args.frontend_search_cycles,
            "frontend_select_cycles": args.frontend_select_cycles,
            "frontend_miss_update_cycles": args.frontend_miss_update_cycles,
            "tser_cycles": args.tser_cycles,
            "embedding_dim": args.embedding_dim,
            "embedding_bits": args.embedding_bits,
            "embedding_bandwidth_gbps": args.embedding_bandwidth_gbps,
            "scheduler_norm_of_full_encoder": args.scheduler_norm_of_full_encoder,
            "gnn_norm_of_full_encoder": args.gnn_norm_of_full_encoder,
        },
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-input", type=Path, default=DEFAULT_REUSE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clock-mhz", type=float, default=500.0)
    parser.add_argument("--target-reuse", type=float, default=40.0)
    parser.add_argument("--reuse-tolerance", type=float, default=3.0)
    parser.add_argument("--frontend-search-cycles", type=float, default=1.0)
    parser.add_argument("--frontend-select-cycles", type=float, default=1.0)
    parser.add_argument("--frontend-miss-update-cycles", type=float, default=1.0)
    parser.add_argument("--tser-cycles", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--embedding-bits", type=int, default=16)
    parser.add_argument("--embedding-bandwidth-gbps", type=float, default=25.6)
    parser.add_argument("--scheduler-norm-of-full-encoder", type=float, default=0.005)
    parser.add_argument("--gnn-norm-of-full-encoder", type=float, default=0.010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "tser40_fullstack_timing.md"
    tsv_path = md_path.with_suffix(".tsv")
    json_path = md_path.with_suffix(".json")
    write_tsv(tsv_path, rows)
    write_json(json_path, rows, args)
    write_markdown(md_path, rows, args)
    print(md_path)


if __name__ == "__main__":
    main()
