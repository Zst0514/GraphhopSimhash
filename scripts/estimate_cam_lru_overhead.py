#!/usr/bin/env python3
"""First-order SimHash-CAM/LRU overhead estimator.

This is not a sign-off layout model.  It is a CACTI-style accounting script
that keeps the directory CAM, SRAM metadata, LRU bits, and optional embedding
hot buffer separate.  The constants can be replaced by real CACTI/RTL numbers
later without changing the table format.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Estimate:
    entries: int
    tag_kbit: float
    metadata_kbit: float
    lru_kbit: float
    cam_area_mm2: float
    metadata_area_mm2: float
    lru_area_mm2: float
    directory_area_mm2: float
    cam_search_energy_nj: float
    metadata_access_energy_nj: float
    lru_update_energy_nj: float
    directory_access_energy_nj: float
    active_power_mw: float
    embedding_hot_entries: int
    embedding_hot_area_mm2: float
    embedding_read_energy_nj: float
    total_with_hot_area_mm2: float


def bit_area_mm2(bits: float, bitcell_um2: float, macro_overhead: float, cell_factor: float = 1.0) -> float:
    return bits * bitcell_um2 * macro_overhead * cell_factor * 1e-6


def estimate(args: argparse.Namespace, entries: int, hot_entries: int) -> Estimate:
    tag_bits = entries * args.heads * args.bits_per_head
    metadata_bits = entries * args.metadata_bits_per_entry
    lru_bits = max(entries - 1, 0) if args.lru_policy == "tree_plru" else entries * math.ceil(math.log2(max(entries, 2)))

    cam_area = bit_area_mm2(tag_bits, args.sram_bitcell_um2, args.macro_overhead, args.cam_cell_factor)
    metadata_area = bit_area_mm2(metadata_bits, args.sram_bitcell_um2, args.macro_overhead, 1.0)
    lru_area = bit_area_mm2(lru_bits, args.sram_bitcell_um2, args.logic_overhead, args.logic_cell_factor)
    directory_area = cam_area + metadata_area + lru_area

    cam_search_energy = tag_bits * args.cam_search_pj_per_bit * 1e-3
    metadata_access_energy = args.metadata_read_bits_per_lookup * args.sram_read_pj_per_bit * 1e-3
    lru_update_energy = max(math.ceil(math.log2(max(entries, 2))), 1) * args.logic_pj_per_bit * 1e-3
    directory_energy = cam_search_energy + metadata_access_energy + lru_update_energy
    active_power = directory_energy * args.query_rate_mhz

    embedding_bits_per_entry = args.embedding_dim * args.embedding_bits
    embedding_hot_bits = hot_entries * embedding_bits_per_entry
    embedding_area = bit_area_mm2(embedding_hot_bits, args.sram_bitcell_um2, args.macro_overhead, 1.0)
    embedding_read_energy = embedding_bits_per_entry * args.sram_read_pj_per_bit * 1e-3

    return Estimate(
        entries=entries,
        tag_kbit=tag_bits / 1024.0,
        metadata_kbit=metadata_bits / 1024.0,
        lru_kbit=lru_bits / 1024.0,
        cam_area_mm2=cam_area,
        metadata_area_mm2=metadata_area,
        lru_area_mm2=lru_area,
        directory_area_mm2=directory_area,
        cam_search_energy_nj=cam_search_energy,
        metadata_access_energy_nj=metadata_access_energy,
        lru_update_energy_nj=lru_update_energy,
        directory_access_energy_nj=directory_energy,
        active_power_mw=active_power,
        embedding_hot_entries=hot_entries,
        embedding_hot_area_mm2=embedding_area,
        embedding_read_energy_nj=embedding_read_energy,
        total_with_hot_area_mm2=directory_area + embedding_area,
    )


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def write_outputs(rows: list[Estimate], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.output_dir / "cam_lru_overhead.tsv"
    md = args.output_dir / "CAM_LRU_OVERHEAD_ESTIMATE.md"

    fields = list(asdict(rows[0]).keys())
    with tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    lines = [
        "# SimHash-CAM/LRU Area and Energy Estimate",
        "",
        "This is a first-order CACTI-style estimate, not layout sign-off.",
        "The model separates the SimHash-CAM directory from optional embedding hot storage.",
        "",
        "## Assumptions",
        "",
        f"- Technology node: `{args.tech_nm}nm`",
        f"- SRAM bitcell area: `{args.sram_bitcell_um2}` um^2/bit",
        f"- SRAM macro overhead: `{args.macro_overhead}x`",
        f"- HD-CAM/BCAM cell factor over SRAM: `{args.cam_cell_factor}x`",
        f"- SimHash: `{args.heads}` heads x `{args.bits_per_head}` bits",
        f"- Metadata bits per entry: `{args.metadata_bits_per_entry}`",
        f"- LRU policy: `{args.lru_policy}`",
        f"- CAM search energy: `{args.cam_search_pj_per_bit}` pJ/bit/lookup",
        f"- SRAM read energy: `{args.sram_read_pj_per_bit}` pJ/bit/read",
        f"- Assumed average query rate for active power: `{args.query_rate_mhz}` Mlookup/s",
        f"- Embedding: `{args.embedding_dim}` x `{args.embedding_bits}`b = `{args.embedding_dim * args.embedding_bits / 8 / 1024:.1f}` KB per node",
        "",
        "## Directory-Only Overhead",
        "",
        "| Entries | Tag Kb | Meta Kb | CAM Area | Meta Area | LRU Area | Directory Area | Dir. Energy/Search | Active Power |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.entries} | {fmt(row.tag_kbit)} | {fmt(row.metadata_kbit)} | "
            f"{fmt(row.cam_area_mm2)} mm^2 | {fmt(row.metadata_area_mm2)} mm^2 | "
            f"{fmt(row.lru_area_mm2, 5)} mm^2 | {fmt(row.directory_area_mm2)} mm^2 | "
            f"{fmt(row.directory_access_energy_nj)} nJ | {fmt(row.active_power_mw)} mW |"
        )

    lines.extend(
        [
            "",
            "## Optional Embedding Hot Buffer",
            "",
            "The main directory stores compact SimHash tags and metadata. A full LLaMA-7B embedding is much larger.",
            "If embeddings are cached in a local SRAM hot buffer, that buffer must be reported separately.",
            "",
            "| Entries | Hot Embedding Entries | Hot Buffer Area | One Embedding SRAM Read | Total Area With Hot Buffer |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.entries} | {row.embedding_hot_entries} | "
            f"{fmt(row.embedding_hot_area_mm2)} mm^2 | {fmt(row.embedding_read_energy_nj)} nJ | "
            f"{fmt(row.total_with_hot_area_mm2)} mm^2 |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The CAM/LRU directory is small because each entry stores only compact signatures and metadata.",
            "- Storing many full 4096-d FP16 embeddings on-chip is expensive; it should be modeled as a separate hot buffer or placed in near-memory DRAM.",
            "- Tree-PLRU control bits are negligible compared with the CAM and embedding storage.",
            "",
            f"Raw TSV: `{tsv}`",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/cam_lru_overhead_estimate"))
    parser.add_argument("--tech-nm", type=int, default=28)
    parser.add_argument("--sram-bitcell-um2", type=float, default=0.127)
    parser.add_argument("--macro-overhead", type=float, default=1.6)
    parser.add_argument("--logic-overhead", type=float, default=2.0)
    parser.add_argument("--logic-cell-factor", type=float, default=1.0)
    parser.add_argument("--cam-cell-factor", type=float, default=2.0)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bits-per-head", type=int, default=16)
    parser.add_argument("--metadata-bits-per-entry", type=int, default=64)
    parser.add_argument("--metadata-read-bits-per-lookup", type=int, default=128)
    parser.add_argument("--lru-policy", choices=["tree_plru", "counter"], default="tree_plru")
    parser.add_argument("--cam-search-pj-per-bit", type=float, default=0.20)
    parser.add_argument("--sram-read-pj-per-bit", type=float, default=0.08)
    parser.add_argument("--logic-pj-per-bit", type=float, default=0.02)
    parser.add_argument("--query-rate-mhz", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--embedding-bits", type=int, default=16)
    parser.add_argument("--entries", type=int, nargs="+", default=[1024, 4096, 32768])
    parser.add_argument("--embedding-hot-entries", type=int, nargs="+", default=[64, 64, 64])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.embedding_hot_entries) == 1:
        hot_entries = args.embedding_hot_entries * len(args.entries)
    elif len(args.embedding_hot_entries) == len(args.entries):
        hot_entries = args.embedding_hot_entries
    else:
        raise ValueError("--embedding-hot-entries must have length 1 or match --entries")
    rows = [estimate(args, entries, hot) for entries, hot in zip(args.entries, hot_entries)]
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
