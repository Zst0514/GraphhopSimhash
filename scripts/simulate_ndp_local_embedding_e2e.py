#!/usr/bin/env python3
"""End-to-end timing with physically separated NPU and NDP memory domains.

Architecture contract modeled here:

* NPU memory is dedicated to streaming LLM weights/activations.
* NDP local DRAM stores graph CSR indices and generated/reused node embeddings.
* CAM/LRU is colocated with the NDP embedding store.
* On a CAM hit, the embedding is read from NDP local DRAM.
* On a CAM miss, the NPU computes the embedding and writes it one-way into the
  NDP embedding store.
* Backend GNN aggregation reads graph indices and neighbor embeddings locally
  from the NDP memory domain.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent

DEFAULT_REUSE_INPUT = OFA_ROOT / "output" / "tser_reuse_drop_tradeoff_40pt_alignment.tsv"
DEFAULT_ARRAY_ROOT = OFA_ROOT / "output" / "e2e_time_breakdown_40reuse"
DEFAULT_OUTPUT_DIR = OFA_ROOT / "output" / "ndp_local_embedding_e2e"

TASKS = ("CN", "CL", "PN", "PL", "AR", "WK")
TASK_TO_DATASET = {
    "CN": "cora",
    "CL": "cora",
    "PN": "pubmed",
    "PL": "pubmed",
    "AR": "arxiv",
    "WK": "wikics",
}
DATASET_STATS = {
    # Edges are from the GFMEngine dataset table / current TAG workload metadata.
    "cora": {"nodes": 2708, "edges": 10556},
    "pubmed": {"nodes": 19717, "edges": 44338},
    "arxiv": {"nodes": 169343, "edges": 1166243},
    "wikics": {"nodes": 11701, "edges": 216123},
}
ARRAY_SUMMARY_CANDIDATES_BY_DATASET = {
    "cora": (DEFAULT_ARRAY_ROOT / "array_cora_graphstress20" / "summary.json",),
    "pubmed": (DEFAULT_ARRAY_ROOT / "array_pubmed_graphstress20" / "summary.json",),
    "arxiv": (
        DEFAULT_ARRAY_ROOT / "array_arxiv_graphstress20" / "summary.json",
        DEFAULT_ARRAY_ROOT / "array_arxiv_graphstress10" / "summary.json",
    ),
    "wikics": (DEFAULT_ARRAY_ROOT / "array_wikics_graphstress20" / "summary.json",),
}
POLICIES = ("NoReuse+W4BFPA4", "TSER40+W4BFPA4", "TSER40+BFPLift")


@dataclass
class TimingRow:
    task: str
    dataset: str
    policy: str
    nodes: int
    edges: int
    reuse_pct: float
    drop_pct: float
    npu_encoder_s: float
    npu_to_ndp_write_s: float
    ndp_cam_lru_s: float
    ndp_hit_embedding_read_s: float
    ndp_graph_index_load_s: float
    ndp_neighbor_embedding_read_s: float
    ndp_gnn_compute_proxy_s: float
    total_s: float
    norm_vs_noreuse: float
    speedup_vs_noreuse: float
    full_bfpa4_cycles: float
    encoder_cycles: float
    bfp_tag: str
    lift_ratio_pct: float
    effective_bits: float


def read_reuse(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows[row["task"]] = {
                "reuse": float(row["anchor_reuse"]) / 100.0,
                "drop": float(row["target_anchor_drop"]),
            }
    return rows


def read_array_summaries() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset, candidates in ARRAY_SUMMARY_CANDIDATES_BY_DATASET.items():
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            candidate_list = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"missing array summary for {dataset}: {candidate_list}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[dataset] = {
            "full_bfpa4_cycles": float(payload["full_bfpa4_cycles"]),
            "dynamic_cycles": float(payload["dynamic_cycles"]),
            "dynamic_vs_bfpa4_cycles": float(payload["dynamic_vs_bfpa4_cycles"]),
            "refined_ratio": float(payload["refined_ratio"]),
            "effective_bits": float(payload["effective_bits"]),
            "tag": str(payload.get("tag", "")),
        }
    return out


def seconds_from_cycles(cycles: float, clock_mhz: float) -> float:
    return float(cycles) / (float(clock_mhz) * 1.0e6)


def seconds_from_bytes(num_bytes: float, bandwidth_gbs: float) -> float:
    if bandwidth_gbs <= 0:
        return 0.0
    return float(num_bytes) / (float(bandwidth_gbs) * 1.0e9)


def embedding_bytes(args: argparse.Namespace) -> float:
    return float(args.embedding_dim) * float(args.embedding_bits) / 8.0


def graph_index_bytes(nodes: int, edges: int, args: argparse.Namespace) -> float:
    return float(nodes + 1 + edges) * float(args.graph_index_bits) / 8.0


def neighbor_embedding_bytes(edges: int, args: argparse.Namespace) -> float:
    return (
        float(edges)
        * embedding_bytes(args)
        * float(args.neighbor_embedding_read_factor)
    )


def build_rows(args: argparse.Namespace) -> list[TimingRow]:
    reuse_points = read_reuse(args.reuse_input)
    arrays = read_array_summaries()
    rows: list[TimingRow] = []
    baseline_by_task: dict[str, float] = {}

    for task in TASKS:
        dataset = TASK_TO_DATASET[task]
        stats = DATASET_STATS[dataset]
        nodes = int(stats["nodes"])
        edges = int(stats["edges"])
        full_cycles = float(arrays[dataset]["full_bfpa4_cycles"])
        dynamic_cycles = float(arrays[dataset]["dynamic_cycles"])

        common_graph_index_s = seconds_from_bytes(
            graph_index_bytes(nodes, edges, args),
            args.ndp_local_dram_bw_gbs,
        )
        common_neighbor_read_s = seconds_from_bytes(
            neighbor_embedding_bytes(edges, args),
            args.ndp_local_dram_bw_gbs,
        )
        common_gnn_proxy_s = seconds_from_cycles(
            full_cycles * float(args.gnn_compute_norm_of_full_encoder),
            args.ndp_clock_mhz,
        )

        def make_row(
            policy: str,
            reuse: float,
            drop: float,
            use_cam: bool,
            encoder_cycles_full: float,
            bfp_tag: str,
            lift_ratio_pct: float,
            effective_bits: float,
        ) -> TimingRow:
            miss = 1.0 - reuse
            encoder_cycles = encoder_cycles_full * miss
            npu_encoder_s = seconds_from_cycles(encoder_cycles, args.npu_clock_mhz)
            npu_to_ndp_write_s = seconds_from_bytes(
                float(nodes) * miss * embedding_bytes(args),
                args.npu_to_ndp_bw_gbs,
            )
            ndp_cam_s = 0.0
            if use_cam:
                cam_cycles = float(nodes) * (
                    float(args.cam_search_cycles)
                    + float(args.cam_select_cycles)
                    + miss * float(args.cam_miss_update_cycles)
                )
                ndp_cam_s = seconds_from_cycles(cam_cycles, args.ndp_clock_mhz)
            ndp_hit_read_s = seconds_from_bytes(
                float(nodes) * reuse * embedding_bytes(args),
                args.ndp_local_dram_bw_gbs,
            )
            total_s = (
                npu_encoder_s
                + npu_to_ndp_write_s
                + ndp_cam_s
                + ndp_hit_read_s
                + common_graph_index_s
                + common_neighbor_read_s
                + common_gnn_proxy_s
            )
            return TimingRow(
                task=task,
                dataset=dataset,
                policy=policy,
                nodes=nodes,
                edges=edges,
                reuse_pct=100.0 * reuse,
                drop_pct=drop,
                npu_encoder_s=npu_encoder_s,
                npu_to_ndp_write_s=npu_to_ndp_write_s,
                ndp_cam_lru_s=ndp_cam_s,
                ndp_hit_embedding_read_s=ndp_hit_read_s,
                ndp_graph_index_load_s=common_graph_index_s,
                ndp_neighbor_embedding_read_s=common_neighbor_read_s,
                ndp_gnn_compute_proxy_s=common_gnn_proxy_s,
                total_s=total_s,
                norm_vs_noreuse=0.0,
                speedup_vs_noreuse=0.0,
                full_bfpa4_cycles=full_cycles,
                encoder_cycles=encoder_cycles,
                bfp_tag=bfp_tag,
                lift_ratio_pct=lift_ratio_pct,
                effective_bits=effective_bits,
            )

        base = make_row(
            "NoReuse+W4BFPA4",
            reuse=0.0,
            drop=0.0,
            use_cam=False,
            encoder_cycles_full=full_cycles,
            bfp_tag="W4BFPA4",
            lift_ratio_pct=0.0,
            effective_bits=4.0,
        )
        baseline_by_task[task] = base.total_s
        base.norm_vs_noreuse = 1.0
        base.speedup_vs_noreuse = 1.0
        rows.append(base)

        point = reuse_points[task]
        tser = make_row(
            "TSER40+W4BFPA4",
            reuse=float(point["reuse"]),
            drop=float(point["drop"]),
            use_cam=True,
            encoder_cycles_full=full_cycles,
            bfp_tag="W4BFPA4",
            lift_ratio_pct=0.0,
            effective_bits=4.0,
        )
        tser.norm_vs_noreuse = tser.total_s / baseline_by_task[task]
        tser.speedup_vs_noreuse = baseline_by_task[task] / tser.total_s
        rows.append(tser)

        lift = make_row(
            "TSER40+BFPLift",
            reuse=float(point["reuse"]),
            drop=float(point["drop"]),
            use_cam=True,
            encoder_cycles_full=dynamic_cycles,
            bfp_tag=str(arrays[dataset]["tag"]),
            lift_ratio_pct=100.0 * float(arrays[dataset]["refined_ratio"]),
            effective_bits=float(arrays[dataset]["effective_bits"]),
        )
        lift.norm_vs_noreuse = lift.total_s / baseline_by_task[task]
        lift.speedup_vs_noreuse = baseline_by_task[task] / lift.total_s
        rows.append(lift)

    return rows


def aggregate(rows: list[TimingRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for policy in POLICIES:
        vals = [r for r in rows if r.policy == policy]
        item: dict[str, Any] = {"policy": policy}
        for key in (
            "npu_encoder_s",
            "npu_to_ndp_write_s",
            "ndp_cam_lru_s",
            "ndp_hit_embedding_read_s",
            "ndp_graph_index_load_s",
            "ndp_neighbor_embedding_read_s",
            "ndp_gnn_compute_proxy_s",
            "total_s",
        ):
            item[key] = sum(float(getattr(r, key)) for r in vals)
        if policy == "NoReuse+W4BFPA4":
            item["norm_vs_noreuse"] = 1.0
            item["speedup_vs_noreuse"] = 1.0
            item["reuse_pct"] = 0.0
            item["drop_pct"] = 0.0
        else:
            base_total = sum(float(r.total_s) for r in rows if r.policy == "NoReuse+W4BFPA4")
            item["norm_vs_noreuse"] = item["total_s"] / base_total
            item["speedup_vs_noreuse"] = base_total / item["total_s"]
            item["reuse_pct"] = sum(float(r.reuse_pct) for r in vals) / len(vals)
            item["drop_pct"] = sum(float(r.drop_pct) for r in vals) / len(vals)
        item["lift_ratio_pct"] = sum(float(r.lift_ratio_pct) for r in vals) / len(vals)
        item["effective_bits"] = sum(float(r.effective_bits) for r in vals) / len(vals)
        out.append(item)
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def select_columns(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in keys} for row in rows]


def fmt_s(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}s"
    if value >= 1.0e-3:
        return f"{value * 1.0e3:.3f}ms"
    if value >= 1.0e-6:
        return f"{value * 1.0e6:.3f}us"
    return f"{value * 1.0e9:.3f}ns"


def render_report(rows: list[TimingRow], agg: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lift_rows = [r for r in rows if r.policy == "TSER40+BFPLift"]
    strict_lift20 = all(abs(r.lift_ratio_pct - 20.0) <= 0.05 for r in lift_rows)
    agg_by_policy = {str(row["policy"]): row for row in agg}
    tser_w4_encoder_s = float(agg_by_policy["TSER40+W4BFPA4"]["npu_encoder_s"])
    tser_w4_task_encoder_s = {
        row.task: row.npu_encoder_s for row in rows if row.policy == "TSER40+W4BFPA4"
    }
    lift_note = (
        "All BFPLift traces are at the 20% block-lift point."
        if strict_lift20
        else "Available BFPLift traces are used as-is; Arxiv currently uses graphstress10, while Cora/PubMed/WikiCS use graphstress20."
    )
    lines = [
        "# NDP-Local Embedding Store End-to-End Timing",
        "",
        "## Architecture Contract",
        "",
        "- NPU memory is reserved for streaming LLM weights and activations.",
        "- The NDP local DRAM stores graph CSR indices and generated/reused node embeddings.",
        "- CAM/LRU entries store compact SimHash signatures and pointers into the NDP embedding store.",
        "- CAM hits read full embeddings from NDP local DRAM and bypass the NPU.",
        "- CAM misses invoke the NPU encoder, then write the new embedding one-way into NDP local DRAM.",
        "- Backend GNN aggregation reads graph indices and neighbor embeddings in the NDP memory domain.",
        "",
        "## Configuration",
        "",
        f"- NPU clock: `{args.npu_clock_mhz} MHz`; NDP clock: `{args.ndp_clock_mhz} MHz`.",
        f"- NDP local DRAM bandwidth: `{args.ndp_local_dram_bw_gbs} GB/s`.",
        f"- NPU-to-NDP embedding write bandwidth: `{args.npu_to_ndp_bw_gbs} GB/s`.",
        f"- Embedding: `{args.embedding_dim}` x `{args.embedding_bits}`b = `{embedding_bytes(args) / 1024.0:.1f} KiB/node`.",
        f"- Graph index: `{args.graph_index_bits}`b CSR indices.",
        f"- Neighbor embedding read factor: `{args.neighbor_embedding_read_factor}` per edge.",
        f"- CAM cycles: search `{args.cam_search_cycles}` + select `{args.cam_select_cycles}` + miss update `{args.cam_miss_update_cycles}`.",
        f"- GNN compute proxy: `{args.gnn_compute_norm_of_full_encoder}` x full W4BFPA4 encoder cycles.",
        f"- BFPLift source: {lift_note}",
        "",
        "## Aggregate Timing",
        "",
        "| Policy | Reuse | Drop | Lift | Eff Bits | NPU Encoder | BFPLift Enc Extra | NPU->NDP Write | CAM/LRU | Hit Emb Read | Graph Index | Neighbor Emb Read | GNN Proxy | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in agg:
        bfp_extra_s = 0.0
        if row["policy"] == "TSER40+BFPLift":
            bfp_extra_s = float(row["npu_encoder_s"]) - tser_w4_encoder_s
        lines.append(
            f"| {row['policy']} | {row['reuse_pct']:.2f}% | {row['drop_pct']:.2f}% | "
            f"{row['lift_ratio_pct']:.2f}% | {row['effective_bits']:.3f} | "
            f"{fmt_s(row['npu_encoder_s'])} | {fmt_s(bfp_extra_s)} | "
            f"{fmt_s(row['npu_to_ndp_write_s'])} | "
            f"{fmt_s(row['ndp_cam_lru_s'])} | {fmt_s(row['ndp_hit_embedding_read_s'])} | "
            f"{fmt_s(row['ndp_graph_index_load_s'])} | {fmt_s(row['ndp_neighbor_embedding_read_s'])} | "
            f"{fmt_s(row['ndp_gnn_compute_proxy_s'])} | {fmt_s(row['total_s'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Task TSER40 Breakdown",
            "",
            "| Task | Policy | Reuse | Drop | BFP Tag | Lift | Eff Bits | NPU Encoder | BFPLift Enc Extra | NPU->NDP Write | CAM/LRU | Hit Emb Read | Graph Index | Neighbor Emb Read | GNN Proxy | Total |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row.policy not in ("TSER40+W4BFPA4", "TSER40+BFPLift"):
            continue
        bfp_extra_s = 0.0
        if row.policy == "TSER40+BFPLift":
            bfp_extra_s = row.npu_encoder_s - tser_w4_task_encoder_s[row.task]
        lines.append(
            f"| {row.task} | {row.policy} | {row.reuse_pct:.2f}% | {row.drop_pct:.2f}% | "
            f"`{row.bfp_tag}` | {row.lift_ratio_pct:.2f}% | {row.effective_bits:.3f} | "
            f"{fmt_s(row.npu_encoder_s)} | {fmt_s(bfp_extra_s)} | "
            f"{fmt_s(row.npu_to_ndp_write_s)} | "
            f"{fmt_s(row.ndp_cam_lru_s)} | {fmt_s(row.ndp_hit_embedding_read_s)} | "
            f"{fmt_s(row.ndp_graph_index_load_s)} | {fmt_s(row.ndp_neighbor_embedding_read_s)} | "
            f"{fmt_s(row.ndp_gnn_compute_proxy_s)} | {fmt_s(row.total_s)} |"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- The graph index load is tiny; neighbor embedding reads dominate the NDP-side graph memory traffic.",
            "- CAM/LRU lookup is colocated with the NDP embedding store and remains a microsecond-scale component.",
            "- BFPLift changes only the miss-side encoder cycles in this model; CAM, hit embedding reads, graph index loads, neighbor reads, and NPU-to-NDP miss writes are unchanged for the same reuse point.",
            "- The NPU is shielded from irregular graph/embedding reads; it only streams encoder data and writes miss embeddings to the NDP store.",
            "- This is a trace-composition model, not a bank-conflict or NoC arbitration simulator.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-input", type=Path, default=DEFAULT_REUSE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--npu-clock-mhz", type=float, default=500.0)
    parser.add_argument("--ndp-clock-mhz", type=float, default=500.0)
    parser.add_argument("--ndp-local-dram-bw-gbs", type=float, default=256.0)
    parser.add_argument("--npu-to-ndp-bw-gbs", type=float, default=64.0)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--embedding-bits", type=int, default=16)
    parser.add_argument("--graph-index-bits", type=int, default=32)
    parser.add_argument("--neighbor-embedding-read-factor", type=float, default=1.0)
    parser.add_argument("--cam-search-cycles", type=float, default=1.0)
    parser.add_argument("--cam-select-cycles", type=float, default=1.0)
    parser.add_argument("--cam-miss-update-cycles", type=float, default=1.0)
    parser.add_argument("--gnn-compute-norm-of-full-encoder", type=float, default=0.01)
    parser.add_argument(
        "--repo-report",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "NDP_LOCAL_EMBEDDING_E2E.md",
    )
    args = parser.parse_args()

    rows = build_rows(args)
    agg = aggregate(rows)
    row_dicts = [asdict(r) for r in rows]
    absolute_row_keys = (
        "task",
        "dataset",
        "policy",
        "nodes",
        "edges",
        "reuse_pct",
        "drop_pct",
        "bfp_tag",
        "lift_ratio_pct",
        "effective_bits",
        "npu_encoder_s",
        "npu_to_ndp_write_s",
        "ndp_cam_lru_s",
        "ndp_hit_embedding_read_s",
        "ndp_graph_index_load_s",
        "ndp_neighbor_embedding_read_s",
        "ndp_gnn_compute_proxy_s",
        "total_s",
    )
    absolute_aggregate_keys = (
        "policy",
        "reuse_pct",
        "drop_pct",
        "lift_ratio_pct",
        "effective_bits",
        "npu_encoder_s",
        "npu_to_ndp_write_s",
        "ndp_cam_lru_s",
        "ndp_hit_embedding_read_s",
        "ndp_graph_index_load_s",
        "ndp_neighbor_embedding_read_s",
        "ndp_gnn_compute_proxy_s",
        "total_s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "ndp_local_embedding_rows.tsv", row_dicts)
    write_tsv(args.output_dir / "ndp_local_embedding_aggregate.tsv", agg)
    write_tsv(
        args.output_dir / "absolute_time_breakdown_seconds.tsv",
        select_columns(row_dicts, absolute_row_keys),
    )
    write_tsv(
        args.output_dir / "absolute_time_aggregate_seconds.tsv",
        select_columns(agg, absolute_aggregate_keys),
    )
    (args.output_dir / "ndp_local_embedding_e2e.json").write_text(
        json.dumps({"config": vars(args), "rows": row_dicts, "aggregate": agg}, indent=2, default=str),
        encoding="utf-8",
    )
    report = render_report(rows, agg, args)
    (args.output_dir / "NDP_LOCAL_EMBEDDING_E2E.md").write_text(report, encoding="utf-8")
    args.repo_report.parent.mkdir(parents=True, exist_ok=True)
    args.repo_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
