#!/usr/bin/env python3
"""Model whether Graph-Bit risk buckets can support W-tile reuse.

The ONNXim sensitivity sweep can show a large benefit when the same W tile is
kept stationary for a larger same-risk node batch.  This script checks whether
that assumption is plausible for real graph workloads:

* how many miss nodes land in each P8/P6/P5/P4 bucket;
* whether each bucket can form 32/64-sized W-tile batches;
* whether a simple SRAM budget can hold W tile + activation plane buffer +
  partial sums + output buffer;
* when buckets or SRAM are too small, the model automatically falls back to the
  conservative baseline W-tile batch, so no extra W-HBM saving is claimed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEPTH_KEYS = ("p8", "p6", "p5", "p4")
DEFAULT_NODE_COUNTS = {
    "cora": 2708,
    "pubmed": 19717,
    "arxiv": 169343,
}


@dataclass(frozen=True)
class BufferModel:
    sram_kb: float
    tile_k: int
    tile_n: int
    weight_bits: int
    fetch_depth: int
    psum_bits: int
    output_bits: int
    buffer_factor: float

    @property
    def sram_bytes(self) -> float:
        return self.sram_kb * 1024.0

    @property
    def weight_bytes(self) -> float:
        return self.tile_k * self.tile_n * self.weight_bits / 8.0

    def activation_bytes(self, batch: int) -> float:
        return batch * self.tile_k * self.fetch_depth / 8.0

    def psum_bytes(self, batch: int) -> float:
        return batch * self.tile_n * self.psum_bits / 8.0

    def output_bytes(self, batch: int) -> float:
        return batch * self.tile_n * self.output_bits / 8.0

    def total_bytes(self, batch: int) -> float:
        data_bytes = (
            self.weight_bytes
            + self.activation_bytes(batch)
            + self.psum_bytes(batch)
            + self.output_bytes(batch)
        )
        return data_bytes * self.buffer_factor

    def max_batch(self) -> int:
        per_node = (
            self.tile_k * self.fetch_depth / 8.0
            + self.tile_n * self.psum_bits / 8.0
            + self.tile_n * self.output_bits / 8.0
        )
        available = self.sram_bytes / self.buffer_factor - self.weight_bytes
        if available <= 0:
            return 0
        return max(0, int(math.floor(available / per_node)))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing workload json: {path}")
    return json.loads(path.read_text())


def profile_text(profile: dict[str, Any]) -> str:
    parts = [str(profile.get("id", ""))]
    route = profile.get("route", {}) or {}
    for key in ("method", "config", "budget", "frontend", "heads", "threshold"):
        parts.append(str(route.get(key, "")))
    return " ".join(parts).lower()


def iter_profiles(workload_paths: list[Path], matches: list[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    needles = [item.lower() for item in matches]
    for path in workload_paths:
        workload = load_json(path)
        for profile in workload.get("profiles", []):
            text = profile_text(profile)
            if any(needle in text for needle in needles):
                copied = dict(profile)
                copied["source_workload"] = str(path)
                selected.append(copied)
    return selected


def parse_manual_profile(spec: str) -> dict[str, Any]:
    fields = spec.split(":")
    if len(fields) != 7:
        raise SystemExit(
            "--manual-profile must be id:dataset:reuse:p8:p6:p5:p4, "
            f"got: {spec}"
        )
    profile_id, dataset, reuse, p8, p6, p5, p4 = fields
    return {
        "id": profile_id,
        "dataset": dataset,
        "model": "llama2_7b",
        "route": {"method": "manual", "config": profile_id},
        "source_workload": "manual",
        "ratios": {
            "reuse": float(reuse),
            "p8": float(p8),
            "p6": float(p6),
            "p5": float(p5),
            "p4": float(p4),
        },
    }


def node_count_for(profile: dict[str, Any], overrides: dict[str, int]) -> int:
    dataset = str(profile.get("dataset", "")).lower()
    if dataset in overrides:
        return overrides[dataset]
    if dataset in DEFAULT_NODE_COUNTS:
        return DEFAULT_NODE_COUNTS[dataset]
    raise SystemExit(f"No node count for dataset '{dataset}'. Pass --node-count {dataset}=N")


def parse_node_overrides(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--node-count expects dataset=N, got {value}")
        dataset, count = value.split("=", 1)
        out[dataset.lower()] = int(count)
    return out


def effective_batch_for_bucket(nodes: int, candidate: int, baseline: int, max_sram_batch: int) -> int:
    if nodes <= baseline:
        return baseline
    return max(baseline, min(candidate, max_sram_batch, nodes))


def batch_loads(nodes: int, batch: int) -> int:
    return math.ceil(nodes / batch) if nodes > 0 else 0


def model_profile(
    profile: dict[str, Any],
    node_count: int,
    tile_batches: list[int],
    baseline_tile_batch: int,
    buffers: BufferModel,
) -> list[dict[str, Any]]:
    ratios = {key: float((profile.get("ratios", {}) or {}).get(key, 0.0) or 0.0) for key in DEPTH_KEYS}
    miss_ratio = sum(ratios.values())
    bucket_nodes = {key: int(round(node_count * ratios[key])) for key in DEPTH_KEYS}
    miss_nodes = sum(bucket_nodes.values())
    max_sram_batch = buffers.max_batch()

    rows: list[dict[str, Any]] = []
    for candidate in tile_batches:
        baseline_loads = sum(batch_loads(bucket_nodes[key], baseline_tile_batch) for key in DEPTH_KEYS)
        graphbit_loads = 0
        bucket_details = {}
        smallest_tail_util = 1.0
        fallback_buckets = []

        for key in DEPTH_KEYS:
            nodes = bucket_nodes[key]
            eff_batch = effective_batch_for_bucket(
                nodes=nodes,
                candidate=candidate,
                baseline=baseline_tile_batch,
                max_sram_batch=max_sram_batch,
            )
            base_load = batch_loads(nodes, baseline_tile_batch)
            gb_load = batch_loads(nodes, eff_batch)
            graphbit_loads += gb_load
            padded = gb_load * eff_batch
            tail_util = (nodes / padded) if padded else 1.0
            smallest_tail_util = min(smallest_tail_util, tail_util)

            fallback_reason = ""
            if nodes == 0:
                fallback_reason = ""
            elif nodes <= baseline_tile_batch and candidate > baseline_tile_batch:
                fallback_reason = "small_bucket"
            elif max_sram_batch < candidate:
                fallback_reason = "sram_limited"
            elif eff_batch <= baseline_tile_batch and candidate > baseline_tile_batch:
                fallback_reason = "baseline"
            if fallback_reason:
                fallback_buckets.append(f"{key}:{fallback_reason}")

            bucket_details[key] = {
                "nodes": nodes,
                "ratio": ratios[key],
                "baseline_loads": base_load,
                "graphbit_loads": gb_load,
                "effective_batch": eff_batch,
                "tail_util": tail_util,
                "fallback": fallback_reason,
            }

        w_hbm_scale = (graphbit_loads / baseline_loads) if baseline_loads else 1.0
        rows.append(
            {
                "profile": profile.get("id", ""),
                "dataset": profile.get("dataset", ""),
                "source": profile.get("source_workload", ""),
                "node_count": node_count,
                "miss_nodes": miss_nodes,
                "miss_ratio": miss_ratio,
                "candidate_batch": candidate,
                "baseline_tile_batch": baseline_tile_batch,
                "max_sram_batch": max_sram_batch,
                "sram_fit": candidate <= max_sram_batch,
                "sram_bytes_at_candidate": buffers.total_bytes(min(candidate, max_sram_batch)),
                "baseline_w_loads": baseline_loads,
                "graphbit_w_loads": graphbit_loads,
                "w_hbm_scale": w_hbm_scale,
                "w_hbm_reduction": 1.0 - w_hbm_scale,
                "min_tail_util": smallest_tail_util,
                "fallback": ",".join(fallback_buckets),
                "buckets": bucket_details,
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]], buffers: BufferModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "graphbit_bucket_scheduler_feasibility.v1",
        "buffer_model": {
            "sram_kb": buffers.sram_kb,
            "tile_k": buffers.tile_k,
            "tile_n": buffers.tile_n,
            "weight_bits": buffers.weight_bits,
            "fetch_depth": buffers.fetch_depth,
            "psum_bits": buffers.psum_bits,
            "output_bits": buffers.output_bits,
            "buffer_factor": buffers.buffer_factor,
            "weight_tile_bytes": buffers.weight_bytes,
            "max_sram_batch": buffers.max_batch(),
        },
        "rows": rows,
    }
    (output_dir / "bucket_scheduler_feasibility.json").write_text(json.dumps(payload, indent=2) + "\n")

    tsv_path = output_dir / "bucket_scheduler_feasibility.tsv"
    with tsv_path.open("w", newline="") as fh:
        fieldnames = [
            "dataset",
            "profile",
            "node_count",
            "miss_nodes",
            "miss_ratio",
            "candidate_batch",
            "sram_fit",
            "max_sram_batch",
            "p8_nodes",
            "p6_nodes",
            "p5_nodes",
            "p4_nodes",
            "baseline_w_loads",
            "graphbit_w_loads",
            "w_hbm_scale",
            "w_hbm_reduction",
            "min_tail_util",
            "fallback",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "profile": row["profile"],
                    "node_count": row["node_count"],
                    "miss_nodes": row["miss_nodes"],
                    "miss_ratio": f"{row['miss_ratio']:.6f}",
                    "candidate_batch": row["candidate_batch"],
                    "sram_fit": int(row["sram_fit"]),
                    "max_sram_batch": row["max_sram_batch"],
                    "p8_nodes": row["buckets"]["p8"]["nodes"],
                    "p6_nodes": row["buckets"]["p6"]["nodes"],
                    "p5_nodes": row["buckets"]["p5"]["nodes"],
                    "p4_nodes": row["buckets"]["p4"]["nodes"],
                    "baseline_w_loads": row["baseline_w_loads"],
                    "graphbit_w_loads": row["graphbit_w_loads"],
                    "w_hbm_scale": f"{row['w_hbm_scale']:.6f}",
                    "w_hbm_reduction": f"{row['w_hbm_reduction']:.6f}",
                    "min_tail_util": f"{row['min_tail_util']:.6f}",
                    "fallback": row["fallback"],
                }
            )

    lines = [
        "Graph-Bit bucket scheduler feasibility",
        "",
        "Buffer model:",
        (
            f"  SRAM={buffers.sram_kb:.0f}KB | tile={buffers.tile_k}x{buffers.tile_n} | "
            f"W{buffers.weight_bits} | fetch_depth={buffers.fetch_depth} | "
            f"psum={buffers.psum_bits}b | out={buffers.output_bits}b | "
            f"buffer_factor={buffers.buffer_factor:.1f}"
        ),
        f"  W tile={buffers.weight_bytes / 1024.0:.1f}KB | max SRAM batch={buffers.max_batch()}",
        "",
        (
            f"{'dataset':<8s} {'profile':<34s} {'miss':>7s} {'tileB':>5s} "
            f"{'SRAM':>5s} {'P8/P6/P5/P4 nodes':>24s} {'Wscale':>8s} "
            f"{'Wred':>7s} {'tail':>7s} {'fallback':<18s}"
        ),
        "-" * 132,
    ]
    for row in rows:
        bucket_text = (
            f"{row['buckets']['p8']['nodes']}/"
            f"{row['buckets']['p6']['nodes']}/"
            f"{row['buckets']['p5']['nodes']}/"
            f"{row['buckets']['p4']['nodes']}"
        )
        lines.append(
            f"{str(row['dataset']):<8s} {str(row['profile'])[:34]:<34s} "
            f"{fmt_pct(row['miss_ratio']):>7s} {row['candidate_batch']:5d} "
            f"{'yes' if row['sram_fit'] else 'no':>5s} {bucket_text:>24s} "
            f"{row['w_hbm_scale']:8.3f} {fmt_pct(row['w_hbm_reduction']):>7s} "
            f"{fmt_pct(row['min_tail_util']):>7s} {row['fallback'][:18]:<18s}"
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "  Wscale=1.000 means no extra W-HBM amortization over the conservative baseline.",
            "  Wscale<1 is only claimed when bucket size and SRAM capacity both support a larger same-risk batch.",
            "  fallback lists buckets that automatically degrade to the conservative baseline.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "bucket_scheduler_feasibility.txt").write_text(text)
    print(text)
    print(f"[GraphBitBucket] wrote {output_dir / 'bucket_scheduler_feasibility.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-json", type=Path, action="append", default=[])
    parser.add_argument("--profile-match", action="append", default=["degree_runtime-bound"])
    parser.add_argument(
        "--manual-profile",
        action="append",
        default=[],
        help="id:dataset:reuse:p8:p6:p5:p4",
    )
    parser.add_argument("--node-count", action="append", default=[], help="dataset=N")
    parser.add_argument("--tile-batches", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--baseline-tile-batch", type=int, default=16)
    parser.add_argument("--sram-kb", type=float, default=512.0)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--fetch-depth", type=int, default=6)
    parser.add_argument("--psum-bits", type=int, default=32)
    parser.add_argument("--output-bits", type=int, default=16)
    parser.add_argument("--buffer-factor", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/onnxim_graphbit/bucket_feasibility"),
    )
    args = parser.parse_args()

    profiles = iter_profiles(args.workload_json, args.profile_match)
    profiles.extend(parse_manual_profile(value) for value in args.manual_profile)
    if not profiles:
        raise SystemExit("No profiles selected")

    overrides = parse_node_overrides(args.node_count)
    buffers = BufferModel(
        sram_kb=args.sram_kb,
        tile_k=args.tile_k,
        tile_n=args.tile_n,
        weight_bits=args.weight_bits,
        fetch_depth=args.fetch_depth,
        psum_bits=args.psum_bits,
        output_bits=args.output_bits,
        buffer_factor=args.buffer_factor,
    )

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        rows.extend(
            model_profile(
                profile=profile,
                node_count=node_count_for(profile, overrides),
                tile_batches=args.tile_batches,
                baseline_tile_batch=args.baseline_tile_batch,
                buffers=buffers,
            )
        )
    write_outputs(rows, buffers, args.output_dir)


if __name__ == "__main__":
    main()
