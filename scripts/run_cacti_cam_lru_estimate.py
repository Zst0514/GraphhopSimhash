#!/usr/bin/env python3
"""CACTI-backed SimHash-CAM/LRU overhead estimator.

This script uses CACTI for SRAM macro area/energy and then derives the
SimHash-CAM directory cost from the equivalent SRAM macro with a conservative
CAM multiplier.  The CACTI clone used in this repo crashes on its pure-CAM and
fully-associative paths, so the script keeps raw CACTI outputs and explicitly
reports the SRAM-to-CAM scaling assumption.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CactiMacro:
    name: str
    size_bytes: int
    block_bytes: int
    bus_bits: int
    access_ns: float
    cycle_ns: float
    total_read_nj: float
    data_read_nj: float
    total_leak_mw: float
    data_area_mm2: float
    tag_area_mm2: float
    raw_log: str


@dataclass
class DirectoryEstimate:
    entries: int
    signature_bits: int
    metadata_bits: int
    plru_bits: int
    cam_sram_area_mm2: float
    cam_area_mm2: float
    metadata_area_mm2: float
    plru_area_mm2: float
    directory_area_mm2: float
    cam_search_energy_nj: float
    metadata_read_energy_nj: float
    plru_update_energy_nj: float
    directory_lookup_energy_nj: float
    directory_leak_mw: float
    hot_entries: int
    hot_buffer_area_mm2: float
    hot_embedding_read_energy_nj: float
    hot_buffer_leak_mw: float
    total_area_with_hot_mm2: float


def replace_active_line(text: str, key: str, replacement: str) -> str:
    pattern = re.compile(rf"^(?!//)({re.escape(key)}.*)$", re.MULTILINE)
    text, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise ValueError(f"Could not replace active config line for {key}")
    return text


def make_config(template: str, *, size_bytes: int, block_bytes: int, bus_bits: int, assoc: int) -> str:
    text = template
    text = replace_active_line(text, "-size (bytes)", f"-size (bytes) {size_bytes}")
    text = replace_active_line(text, "-block size (bytes)", f"-block size (bytes) {block_bytes}")
    text = replace_active_line(text, "-associativity", f"-associativity {assoc}")
    text = replace_active_line(text, "-output/input bus width", f"-output/input bus width {bus_bits}")
    text = replace_active_line(text, "-Add ECC", '-Add ECC - "false"')
    text = replace_active_line(text, "-Print input parameters", '-Print input parameters - "false"')
    return text


def parse_first(pattern: str, text: str, default: float | None = None) -> float:
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    if default is not None:
        return default
    raise ValueError(f"Could not parse pattern: {pattern}")


def run_cacti(
    *,
    cacti_dir: Path,
    template: str,
    output_dir: Path,
    name: str,
    size_bytes: int,
    block_bytes: int,
    bus_bits: int,
    assoc: int = 2,
) -> CactiMacro:
    cfg = make_config(template, size_bytes=size_bytes, block_bytes=block_bytes, bus_bits=bus_bits, assoc=assoc)
    input_path = output_dir / "inputs" / f"{name}.cfg"
    raw_path = output_dir / "raw" / f"{name}.txt"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(cfg, encoding="utf-8")

    proc = subprocess.run(
        ["./cacti", "-infile", str(input_path.resolve())],
        cwd=cacti_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    raw_path.write_text(proc.stdout, encoding="utf-8")
    # This CACTI tree often segfaults during cleanup after printing valid results.
    if "Data array: Area (mm2)" not in proc.stdout:
        raise RuntimeError(f"CACTI failed before producing macro results for {name}; see {raw_path}")

    return CactiMacro(
        name=name,
        size_bytes=size_bytes,
        block_bytes=block_bytes,
        bus_bits=bus_bits,
        access_ns=parse_first(r"Access time \(ns\):\s*([0-9.eE+-]+)", proc.stdout),
        cycle_ns=parse_first(r"Cycle time \(ns\):\s*([0-9.eE+-]+)", proc.stdout),
        total_read_nj=parse_first(r"Total dynamic read energy per access \(nJ\):\s*([0-9.eE+-]+)", proc.stdout),
        data_read_nj=parse_first(r"Data array: Total dynamic read energy/access\s*\(nJ\):\s*([0-9.eE+-]+)", proc.stdout),
        total_leak_mw=parse_first(r"Total leakage power of a bank \(mW\):\s*([0-9.eE+-]+)", proc.stdout),
        data_area_mm2=parse_first(r"Data array: Area \(mm2\):\s*([0-9.eE+-]+)", proc.stdout),
        tag_area_mm2=parse_first(r"Tag array: Area \(mm2\):\s*([0-9.eE+-]+)", proc.stdout, default=0.0),
        raw_log=str(raw_path),
    )


def estimate_for_entries(
    *,
    entries: int,
    hot_entries: int,
    args: argparse.Namespace,
    sig_macro: CactiMacro,
    meta_macro: CactiMacro,
    hot_macro: CactiMacro,
) -> DirectoryEstimate:
    signature_bits = entries * args.heads * args.bits_per_head
    metadata_bits = entries * args.metadata_bits_per_entry
    plru_bits = max(entries - 1, 0)

    sig_total_bits = sig_macro.size_bytes * 8
    meta_total_bits = meta_macro.size_bytes * 8
    hot_line_bits = hot_macro.block_bytes * 8

    cam_sram_area = sig_macro.data_area_mm2
    cam_area = cam_sram_area * args.cam_area_factor
    metadata_area = meta_macro.data_area_mm2
    plru_area = metadata_area * (plru_bits / meta_total_bits) * args.logic_area_factor
    directory_area = cam_area + metadata_area + plru_area

    sig_read_nj_per_bit = sig_macro.data_read_nj / sig_macro.block_bytes / 8.0
    cam_search_energy = signature_bits * sig_read_nj_per_bit * args.cam_search_energy_factor
    metadata_read_energy = meta_macro.data_read_nj
    plru_update_energy = max(math.ceil(math.log2(max(entries, 2))), 1) * args.plru_update_pj_per_bit * 1e-3
    directory_energy = cam_search_energy + metadata_read_energy + plru_update_energy
    directory_leak = sig_macro.total_leak_mw * args.cam_leak_factor + meta_macro.total_leak_mw

    embedding_bytes = args.embedding_dim * args.embedding_bits // 8
    hot_reads_per_embedding = math.ceil(embedding_bytes / hot_macro.block_bytes)
    hot_embedding_energy = hot_macro.data_read_nj * hot_reads_per_embedding

    return DirectoryEstimate(
        entries=entries,
        signature_bits=signature_bits,
        metadata_bits=metadata_bits,
        plru_bits=plru_bits,
        cam_sram_area_mm2=cam_sram_area,
        cam_area_mm2=cam_area,
        metadata_area_mm2=metadata_area,
        plru_area_mm2=plru_area,
        directory_area_mm2=directory_area,
        cam_search_energy_nj=cam_search_energy,
        metadata_read_energy_nj=metadata_read_energy,
        plru_update_energy_nj=plru_update_energy,
        directory_lookup_energy_nj=directory_energy,
        directory_leak_mw=directory_leak,
        hot_entries=hot_entries,
        hot_buffer_area_mm2=hot_macro.data_area_mm2,
        hot_embedding_read_energy_nj=hot_embedding_energy,
        hot_buffer_leak_mw=hot_macro.total_leak_mw,
        total_area_with_hot_mm2=directory_area + hot_macro.data_area_mm2,
    )


def write_outputs(output_dir: Path, macros: list[CactiMacro], rows: list[DirectoryEstimate], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_tsv = output_dir / "cacti_sram_macros.tsv"
    est_tsv = output_dir / "cam_lru_cacti_estimate.tsv"
    md = output_dir / "CAM_LRU_CACTI_ESTIMATE.md"

    with macro_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in asdict(macros[0]).keys() if k != "raw_log"], delimiter="\t")
        writer.writeheader()
        for row in macros:
            data = asdict(row)
            data.pop("raw_log")
            writer.writerow(data)

    with est_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    lines = [
        "# CACTI-Backed SimHash-CAM/LRU Overhead Estimate",
        "",
        "## Scope",
        "",
        "This report estimates the near-DIMM SimHash-CAM directory and an optional embedding hot buffer.",
        "CACTI is used for SRAM macro area, read energy, and leakage. The cloned CACTI tree crashes on pure-CAM and fully-associative paths, so the CAM directory is derived from an equivalent CACTI SRAM macro with conservative CAM scaling factors.",
        "",
        "## Assumptions",
        "",
        "- CACTI config base: `cacti/sample_config_files/ddr3_cache.cfg`.",
        "- CACTI technology: native sample configuration, 22 nm ITRS-HP.",
        f"- SimHash signature: `{args.heads}` heads x `{args.bits_per_head}` bits = `{args.heads * args.bits_per_head}` bits/entry.",
        f"- Metadata: `{args.metadata_bits_per_entry}` bits/entry.",
        "- Replacement: tree-PLRU, `entries - 1` bits.",
        f"- CAM area multiplier over equivalent SRAM data array: `{args.cam_area_factor}x`.",
        f"- CAM search-energy multiplier over SRAM bit-read energy: `{args.cam_search_energy_factor}x`.",
        f"- CAM leakage multiplier over equivalent SRAM macro leakage: `{args.cam_leak_factor}x`.",
        f"- Embedding hot buffer entry: `{args.embedding_dim}` x `{args.embedding_bits}`b = `{args.embedding_dim * args.embedding_bits // 8}` B.",
        "",
        "## CACTI SRAM Macros",
        "",
        "| Macro | Size | Block | Bus | Data Area | Data Read | Total Read | Leakage | Raw Log |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for m in macros:
        lines.append(
            f"| {m.name} | {m.size_bytes} B | {m.block_bytes} B | {m.bus_bits} b | "
            f"{m.data_area_mm2:.4f} mm^2 | {m.data_read_nj:.4f} nJ | {m.total_read_nj:.4f} nJ | "
            f"{m.total_leak_mw:.3f} mW | `{m.raw_log}` |"
        )

    lines.extend(
        [
            "",
            "## Directory and Hot-Buffer Estimate",
            "",
            "| Entries | Directory Area | Directory Energy/Search | Directory Leakage | Hot Entries | Hot Buffer Area | Hot Embedding Read | Total Area |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.entries} | {row.directory_area_mm2:.4f} mm^2 | {row.directory_lookup_energy_nj:.2f} nJ | "
            f"{row.directory_leak_mw:.2f} mW | {row.hot_entries} | {row.hot_buffer_area_mm2:.4f} mm^2 | "
            f"{row.hot_embedding_read_energy_nj:.2f} nJ | {row.total_area_with_hot_mm2:.4f} mm^2 |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The compact SimHash-CAM directory remains small because it stores signatures and metadata, not full 4096-d embeddings.",
            "- Full embedding storage dominates if many embeddings are placed in SRAM; this should be treated as a separate hot buffer or kept in near-memory DRAM.",
            "- Tree-PLRU is negligible relative to CAM and embedding storage.",
            "- CACTI exits with a cleanup-time segmentation fault in this clone after printing valid macro results; raw logs are retained for audit.",
            "",
            f"Raw macro TSV: `{macro_tsv}`",
            f"Raw estimate TSV: `{est_tsv}`",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cacti-dir", type=Path, default=Path("cacti"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cacti_cam_lru_estimate"))
    parser.add_argument("--entries", type=int, nargs="+", default=[1024, 4096, 32768])
    parser.add_argument("--hot-entries", type=int, nargs="+", default=[64])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bits-per-head", type=int, default=16)
    parser.add_argument("--metadata-bits-per-entry", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--embedding-bits", type=int, default=16)
    parser.add_argument("--cam-area-factor", type=float, default=2.0)
    parser.add_argument("--cam-search-energy-factor", type=float, default=2.0)
    parser.add_argument("--cam-leak-factor", type=float, default=2.0)
    parser.add_argument("--logic-area-factor", type=float, default=1.0)
    parser.add_argument("--plru-update-pj-per-bit", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.hot_entries) == 1:
        hot_entries = args.hot_entries * len(args.entries)
    elif len(args.hot_entries) == len(args.entries):
        hot_entries = args.hot_entries
    else:
        raise ValueError("--hot-entries must have length 1 or match --entries")

    cacti_dir = args.cacti_dir.resolve()
    template = (cacti_dir / "sample_config_files" / "ddr3_cache.cfg").read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    macros: list[CactiMacro] = []
    rows: list[DirectoryEstimate] = []
    for entries, hot in zip(args.entries, hot_entries):
        sig_bytes = entries * args.heads * args.bits_per_head // 8
        meta_bytes = entries * args.metadata_bits_per_entry // 8
        hot_bytes = hot * args.embedding_dim * args.embedding_bits // 8

        sig_macro = run_cacti(
            cacti_dir=cacti_dir,
            template=template,
            output_dir=args.output_dir,
            name=f"sig_sram_{entries}",
            size_bytes=sig_bytes,
            block_bytes=max(args.heads * args.bits_per_head // 8, 8),
            bus_bits=args.heads * args.bits_per_head,
        )
        meta_macro = run_cacti(
            cacti_dir=cacti_dir,
            template=template,
            output_dir=args.output_dir,
            name=f"meta_sram_{entries}",
            size_bytes=meta_bytes,
            block_bytes=max(args.metadata_bits_per_entry // 8, 8),
            bus_bits=max(args.metadata_bits_per_entry, 64),
        )
        hot_macro = run_cacti(
            cacti_dir=cacti_dir,
            template=template,
            output_dir=args.output_dir,
            name=f"hot_embed_sram_{hot}",
            size_bytes=hot_bytes,
            block_bytes=64,
            bus_bits=512,
        )
        macros.extend([sig_macro, meta_macro, hot_macro])
        rows.append(
            estimate_for_entries(
                entries=entries,
                hot_entries=hot,
                args=args,
                sig_macro=sig_macro,
                meta_macro=meta_macro,
                hot_macro=hot_macro,
            )
        )

    write_outputs(args.output_dir, macros, rows, args)


if __name__ == "__main__":
    main()
