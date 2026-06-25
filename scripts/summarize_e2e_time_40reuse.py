#!/usr/bin/env python3
"""Summarize normalized end-to-end time for the 40% TSER reuse point.

This script is intentionally a lightweight summarizer. It does not run the
encoder or regenerate pools. It combines:

  1. measured/selected 40%-reuse operating points;
  2. BFP array-trace cycle ratios;
  3. a small configurable online overhead model.

All time numbers are normalized to one full W4BFPA8 encoder pass per task.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent
DEFAULT_INPUT = OFA_ROOT / "output/tser_reuse_drop_tradeoff_40pt_alignment.tsv"
DEFAULT_OUTPUT_DIR = OFA_ROOT / "output/e2e_time_breakdown_40reuse"

TASK_TO_ENCODER_DATASET = {
    "CN": "cora",
    "CL": "cora",
    "PN": "pubmed",
    "PL": "pubmed",
    "AR": "arxiv",
    "WK": "wikics",
}

ARRAY_SUMMARY = {
    "cora": DEFAULT_OUTPUT_DIR / "array_cora_graphstress20/summary.json",
    "pubmed": DEFAULT_OUTPUT_DIR / "array_pubmed_graphstress20/summary.json",
    "wikics": DEFAULT_OUTPUT_DIR / "array_wikics_graphstress20/summary.json",
    "arxiv": DEFAULT_OUTPUT_DIR / "array_arxiv_graphstress10/summary.json",
}

# Measured encoding-only wall time from existing LLaMA2-7B W4BFPA4_B256
# pool-generation logs on the local RTX4090 box. These numbers are used only
# to attach a concrete second-scale to the normalized timing model.
FULL_ENCODER_SECONDS_RTX4090 = {
    "cora": 107.0,
    "pubmed": 1326.0,
    "arxiv": 7796.0,
    "wikics": 791.0,
}


def pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def read_reuse_points(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            task = row["task"]
            rows[task] = {
                "task": task,
                "reuse": float(row["anchor_reuse"]),
                "drop": float(row["target_anchor_drop"]),
            }
    return rows


def read_array_summaries(paths: Dict[str, Path]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        out[dataset] = {
            "dataset": dataset,
            "tag": data.get("tag", "unknown"),
            "metadata": data.get("metadata", ""),
            "refined_ratio": float(data["refined_ratio"]),
            "effective_bits": float(data["effective_bits"]),
            "dynamic_vs_bfpa4": float(data["dynamic_vs_bfpa4_cycles"]),
            "dynamic_vs_bfpa6": float(data["dynamic_vs_bfpa6_cycles"]),
            "dynamic_vs_bfpa8": float(data["dynamic_vs_bfpa8_cycles"]),
        }
    return out


def policy_rows(
    reuse_rows: Dict[str, dict],
    array_rows: Dict[str, dict],
    *,
    filter_tser: float,
    filter_hash: float,
    queue: float,
    backend: float,
) -> List[dict]:
    rows: List[dict] = []
    for task in ["CN", "CL", "PN", "PL", "AR", "WK"]:
        reuse = reuse_rows[task]["reuse"] / 100.0
        miss = 1.0 - reuse
        drop = reuse_rows[task]["drop"]
        enc_dataset = TASK_TO_ENCODER_DATASET[task]
        array = array_rows[enc_dataset]

        policies = [
            ("NoReuse+BFPA8", 0.0, 0.0, 1.0, 0.0),
            ("Hash40+BFPA8", filter_hash, queue, miss, drop),
            ("TSER40+BFPA8", filter_tser, queue, miss, drop),
            ("TSER40+DynBFP", filter_tser, queue, miss * array["dynamic_vs_bfpa8"], drop),
        ]
        base_total = 1.0 + backend
        full_encoder_seconds = FULL_ENCODER_SECONDS_RTX4090[enc_dataset]
        for policy, filter_time, queue_time, encoder_time, policy_drop in policies:
            total = filter_time + queue_time + encoder_time + backend
            rows.append(
                {
                    "task": task,
                    "encoder_dataset": enc_dataset,
                    "policy": policy,
                    "reuse_pct": reuse * 100.0 if policy != "NoReuse+BFPA8" else 0.0,
                    "miss_pct": miss * 100.0 if policy != "NoReuse+BFPA8" else 100.0,
                    "accuracy_drop_pct": policy_drop,
                    "filter_time": filter_time,
                    "queue_time": queue_time,
                    "encoder_time": encoder_time,
                    "backend_time": backend,
                    "total_time": total,
                    "full_encoder_seconds": full_encoder_seconds,
                    "total_seconds": total * full_encoder_seconds,
                    "baseline_seconds": base_total * full_encoder_seconds,
                    "norm_vs_noreuse_bfpa8": total / base_total,
                    "speedup_vs_noreuse_bfpa8": base_total / total,
                    "bfp_tag": array["tag"] if "DynBFP" in policy else "W4BFPA8_B128",
                    "refined_blocks_pct": array["refined_ratio"] * 100.0 if "DynBFP" in policy else 0.0,
                    "effective_bits": array["effective_bits"] if "DynBFP" in policy else 8.0,
                    "dyn_vs_bfpa8": array["dynamic_vs_bfpa8"] if "DynBFP" in policy else 1.0,
                }
            )
    return rows


def write_tsv(path: Path, rows: Iterable[dict], columns: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(
    path: Path,
    rows: List[dict],
    array_rows: Dict[str, dict],
    *,
    filter_tser: float,
    filter_hash: float,
    queue: float,
    backend: float,
) -> None:
    dyn_rows = [r for r in rows if r["policy"] == "TSER40+DynBFP"]
    avg_norm = sum(r["norm_vs_noreuse_bfpa8"] for r in dyn_rows) / len(dyn_rows)
    avg_speedup = sum(r["speedup_vs_noreuse_bfpa8"] for r in dyn_rows) / len(dyn_rows)

    lines = [
        "# 40% Reuse End-to-End Time Summary",
        "",
        "## Experiment Configuration",
        "",
        "- Tasks: CN, CL, PN, PL, AR, WK.",
        "- Frontend reference: LLaMA2-7B `W4BFPA8_B128`; one full no-reuse encoder pass is normalized to `1.0`.",
        "- Reuse point: selected TSER-full operating point near 40% final reuse.",
        "- Dynamic encoder: `W4GraphBFPA4to6_B256`; BFPA4 base path with selected BFPA6 block lift.",
        f"- Online filter overhead: Hash-only `{filter_hash:.3f}`, TSER `{filter_tser:.3f}` normalized encoder-time units.",
        f"- Queue/compaction overhead: `{queue:.3f}`; backend graph head overhead: `{backend:.3f}`.",
        "- CL reuses the Cora encoder trace; PL reuses the PubMed encoder trace because the link task shares the same node-text encoder work.",
        "- AR currently uses the available `graphstress10` array trace; the other tasks use `graphstress20` traces.",
        "- Wall-clock seconds are obtained by multiplying normalized time by the measured full-encoder LLaMA2-7B `W4BFPA4_B256` pool-generation encoding time on the local RTX4090. The normalized speedup is the primary architecture result.",
        "",
        "## Measured Full-Encoder Timing Inputs",
        "",
        "| Encoder Dataset | Full Encoder Time | Shared Tasks |",
        "| --- | ---: | --- |",
        f"| Cora | {FULL_ENCODER_SECONDS_RTX4090['cora']:.1f}s | CN, CL |",
        f"| PubMed | {FULL_ENCODER_SECONDS_RTX4090['pubmed']:.1f}s | PN, PL |",
        f"| OGBN-Arxiv | {FULL_ENCODER_SECONDS_RTX4090['arxiv']:.1f}s | AR |",
        f"| Wiki-CS | {FULL_ENCODER_SECONDS_RTX4090['wikics']:.1f}s | WK |",
        "",
        "## Array Trace Inputs",
        "",
        "| Encoder Dataset | Tag | Lifted Blocks | Eff. Bits | Dyn/BFPA8 Cycles |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in ["cora", "pubmed", "arxiv", "wikics"]:
        a = array_rows[dataset]
        lines.append(
            f"| {dataset} | `{a['tag']}` | {pct(a['refined_ratio'] * 100.0)} | "
            f"{a['effective_bits']:.3f} | {a['dynamic_vs_bfpa8']:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Main Result",
            "",
            "| Task | Reuse | Drop | Dyn Tag | Lifted Blocks | Dyn/BFPA8 | Norm. Time | Est. Time | Speedup |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in dyn_rows:
        lines.append(
            f"| {r['task']} | {pct(r['reuse_pct'])} | {pct(r['accuracy_drop_pct'])} | "
            f"`{r['bfp_tag']}` | {pct(r['refined_blocks_pct'])} | {r['dyn_vs_bfpa8']:.3f}x | "
            f"{r['norm_vs_noreuse_bfpa8']:.3f}x | {r['total_seconds']:.1f}s | "
            f"{r['speedup_vs_noreuse_bfpa8']:.2f}x |"
        )
    lines.extend(
        [
            "",
            f"Average normalized time for `TSER40+DynBFP`: `{avg_norm:.3f}x` of no-reuse BFPA8.",
            f"Average speedup over no-reuse BFPA8: `{avg_speedup:.2f}x`.",
            "",
            "## Full Policy Breakdown",
            "",
            "| Task | Policy | Reuse | Filter | Queue | Encoder | Backend | Total | Seconds | Norm. | Speedup |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['policy']} | {pct(r['reuse_pct'])} | {fmt(r['filter_time'])} | "
            f"{fmt(r['queue_time'])} | {fmt(r['encoder_time'])} | {fmt(r['backend_time'])} | "
            f"{fmt(r['total_time'])} | {r['total_seconds']:.1f}s | {r['norm_vs_noreuse_bfpa8']:.3f}x | "
            f"{r['speedup_vs_noreuse_bfpa8']:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filter-tser", type=float, default=0.020)
    parser.add_argument("--filter-hash", type=float, default=0.015)
    parser.add_argument("--queue", type=float, default=0.005)
    parser.add_argument("--backend", type=float, default=0.010)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reuse_rows = read_reuse_points(args.reuse_input)
    array_rows = read_array_summaries(ARRAY_SUMMARY)
    rows = policy_rows(
        reuse_rows,
        array_rows,
        filter_tser=args.filter_tser,
        filter_hash=args.filter_hash,
        queue=args.queue,
        backend=args.backend,
    )

    columns = [
        "task",
        "encoder_dataset",
        "policy",
        "reuse_pct",
        "miss_pct",
        "accuracy_drop_pct",
        "filter_time",
        "queue_time",
        "encoder_time",
        "backend_time",
        "total_time",
        "full_encoder_seconds",
        "total_seconds",
        "baseline_seconds",
        "norm_vs_noreuse_bfpa8",
        "speedup_vs_noreuse_bfpa8",
        "bfp_tag",
        "refined_blocks_pct",
        "effective_bits",
        "dyn_vs_bfpa8",
    ]
    write_tsv(args.output_dir / "e2e_time_policy_breakdown.tsv", rows, columns)
    write_tsv(
        args.output_dir / "e2e_time_summary.tsv",
        [r for r in rows if r["policy"] == "TSER40+DynBFP"],
        columns,
    )
    write_md(
        args.output_dir / "e2e_time_breakdown.md",
        rows,
        array_rows,
        filter_tser=args.filter_tser,
        filter_hash=args.filter_hash,
        queue=args.queue,
        backend=args.backend,
    )
    print(args.output_dir / "e2e_time_breakdown.md")


if __name__ == "__main__":
    main()
