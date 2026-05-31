#!/usr/bin/env python3
"""Profile ONNXim LLaMA GEMM components with a roofline-style breakdown.

The purpose is to answer a prerequisite question before discussing mixed
precision or bit-plane early stop:

    Which LLaMA encoder GEMM parts are memory-bound, and which are compute-bound?

Inputs are ONNXim `summary.tsv` files produced by
`onnxim_graphbit_microbench.py`.  The script reports both per-GEMM and
per-layer weighted totals using the `count_per_layer` column:

    projection: q/k/v/o, count 4
    ffn_up:     gate/up, count 2
    ffn_down:   down, count 1

The classification intentionally includes two views:

* theoretical roofline, using configurable peak GFLOP/s and GB/s;
* observed bottleneck proxy, using ONNXim PE-active ratio and W traffic share.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_tsv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing summary TSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(f(row, key, default))


def achieved_per_s(amount: float, cycles: float, freq_mhz: float) -> float:
    if cycles <= 0.0:
        return 0.0
    return amount * freq_mhz * 1.0e6 / cycles


def classify_theoretical(oi: float, peak_gflops: float, peak_gbps: float) -> str:
    if peak_gbps <= 0.0:
        return "unknown"
    ridge = peak_gflops / peak_gbps
    return "compute" if oi >= ridge else "memory"


def classify_observed(pe_active_ratio: float, weight_read_share: float) -> str:
    if pe_active_ratio >= 0.65:
        return "compute-exposed"
    if weight_read_share >= 0.75:
        return "W-memory"
    return "memory/mixed"


def component_metrics(
    row: dict[str, Any],
    *,
    source: str,
    freq_mhz: float,
    peak_gflops: float,
    peak_gbps: float,
) -> dict[str, Any]:
    cycles = f(row, "cycles")
    gflops = f(row, "gflops")
    gb = f(row, "gb")
    read = f(row, "dram_read_requests")
    write = f(row, "dram_write_requests")
    weight = f(row, "mem_read_weight")
    activation = f(row, "mem_read_input_actual")
    output = f(row, "mem_write_output")
    total_requests = max(1.0, read + write)
    oi = gflops / gb if gb > 0.0 else 0.0
    pe_active_ratio = f(row, "matmul_active_cycles") / cycles if cycles > 0.0 else 0.0
    weight_read_share = weight / total_requests
    activation_share = activation / total_requests
    output_share = output / total_requests
    achieved_gflops = achieved_per_s(gflops, cycles, freq_mhz)
    achieved_gbps = achieved_per_s(gb, cycles, freq_mhz)
    theoretical_bound = classify_theoretical(oi, peak_gflops, peak_gbps)
    observed_bound = classify_observed(pe_active_ratio, weight_read_share)
    roofline_gflops = min(peak_gflops, oi * peak_gbps)
    roofline_util = achieved_gflops / roofline_gflops if roofline_gflops > 0.0 else 0.0
    return {
        "source": source,
        "name": row.get("name", ""),
        "m": i(row, "m"),
        "k": i(row, "k"),
        "n": i(row, "n"),
        "count": i(row, "count_per_layer", 1),
        "cycles": cycles,
        "gflops": gflops,
        "gb": gb,
        "oi_flop_per_byte": oi,
        "achieved_gflops": achieved_gflops,
        "achieved_gbps": achieved_gbps,
        "pe_work_ratio": pe_active_ratio,
        "weight_req_share": weight_read_share,
        "activation_req_share": activation_share,
        "output_req_share": output_share,
        "theoretical_bound": theoretical_bound,
        "observed_bound": observed_bound,
        "roofline_util": roofline_util,
        "weighted_cycles": cycles * i(row, "count_per_layer", 1),
        "weighted_gflops": gflops * i(row, "count_per_layer", 1),
        "weighted_gb": gb * i(row, "count_per_layer", 1),
        "weighted_read_requests": read * i(row, "count_per_layer", 1),
        "weighted_write_requests": write * i(row, "count_per_layer", 1),
        "weighted_weight_requests": weight * i(row, "count_per_layer", 1),
        "weighted_activation_requests": activation * i(row, "count_per_layer", 1),
        "weighted_output_requests": output * i(row, "count_per_layer", 1),
    }


def aggregate_layer(rows: list[dict[str, Any]], *, freq_mhz: float, peak_gflops: float, peak_gbps: float) -> dict[str, Any]:
    cycles = sum(f(row, "weighted_cycles") for row in rows)
    gflops = sum(f(row, "weighted_gflops") for row in rows)
    gb = sum(f(row, "weighted_gb") for row in rows)
    read = sum(f(row, "weighted_read_requests") for row in rows)
    write = sum(f(row, "weighted_write_requests") for row in rows)
    weight = sum(f(row, "weighted_weight_requests") for row in rows)
    activation = sum(f(row, "weighted_activation_requests") for row in rows)
    output = sum(f(row, "weighted_output_requests") for row in rows)
    total_requests = max(1.0, read + write)
    oi = gflops / gb if gb > 0.0 else 0.0
    achieved_gflops = achieved_per_s(gflops, cycles, freq_mhz)
    achieved_gbps = achieved_per_s(gb, cycles, freq_mhz)
    roofline_gflops = min(peak_gflops, oi * peak_gbps)
    return {
        "source": rows[0]["source"] if rows else "",
        "name": "layer_total",
        "m": rows[0]["m"] if rows else 0,
        "k": "-",
        "n": "-",
        "count": sum(i(row, "count") for row in rows),
        "cycles": cycles,
        "gflops": gflops,
        "gb": gb,
        "oi_flop_per_byte": oi,
        "achieved_gflops": achieved_gflops,
        "achieved_gbps": achieved_gbps,
        "pe_work_ratio": sum(f(row, "pe_work_ratio") * f(row, "weighted_cycles") for row in rows) / cycles
        if cycles > 0.0
        else 0.0,
        "weight_req_share": weight / total_requests,
        "activation_req_share": activation / total_requests,
        "output_req_share": output / total_requests,
        "theoretical_bound": classify_theoretical(oi, peak_gflops, peak_gbps),
        "observed_bound": classify_observed(0.0, weight / total_requests),
        "roofline_util": achieved_gflops / roofline_gflops if roofline_gflops > 0.0 else 0.0,
        "weighted_cycles": cycles,
        "weighted_gflops": gflops,
        "weighted_gb": gb,
        "weighted_read_requests": read,
        "weighted_write_requests": write,
        "weighted_weight_requests": weight,
        "weighted_activation_requests": activation,
        "weighted_output_requests": output,
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], *, peak_gflops: float, peak_gbps: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "name",
        "m",
        "k",
        "n",
        "count",
        "cycles",
        "gflops",
        "gb",
        "oi_flop_per_byte",
        "achieved_gflops",
        "achieved_gbps",
        "pe_work_ratio",
        "weight_req_share",
        "activation_req_share",
        "output_req_share",
        "theoretical_bound",
        "observed_bound",
        "roofline_util",
        "weighted_cycles",
    ]
    for path, data in (
        (output_dir / "llama_roofline_components.tsv", rows),
        (output_dir / "llama_roofline_layers.tsv", layer_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in data:
                writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "LLaMA ONNXim roofline profile",
        f"peak_compute={peak_gflops:.1f} GFLOP/s | peak_mem={peak_gbps:.1f} GB/s | ridge={peak_gflops/peak_gbps:.1f} FLOP/byte",
        "",
        "Layer totals",
        (
            f"{'source':<34s} {'M':>5s} {'cycles':>11s} {'OI':>8s} "
            f"{'AchTF':>8s} {'AchGB/s':>8s} {'PEwork':>7s} "
            f"{'Wshare':>7s} {'Ashare':>7s} {'bound':>10s}"
        ),
        "-" * 118,
    ]
    for row in layer_rows:
        lines.append(
            f"{row['source']:<34s} {str(row['m']):>5s} {row['cycles']:11.0f} "
            f"{row['oi_flop_per_byte']:8.1f} {row['achieved_gflops']/1000.0:8.2f} "
            f"{row['achieved_gbps']:8.1f} {pct(row['pe_work_ratio']):>7s} "
            f"{pct(row['weight_req_share']):>7s} {pct(row['activation_req_share']):>7s} "
            f"{row['theoretical_bound']:>10s}/{row['observed_bound']:<12s}"
        )

    lines.extend(
        [
            "",
            "Per-component share inside each layer",
            (
                f"{'source':<34s} {'part':<9s} {'M':>5s} {'count':>5s} "
                f"{'layerC%':>8s} {'OI':>8s} {'PEwork':>7s} "
                f"{'Wshare':>7s} {'bound':>10s}"
            ),
            "-" * 112,
        ]
    )
    totals = {row["source"]: row["cycles"] for row in layer_rows}
    for row in rows:
        layer_share = row["weighted_cycles"] / totals.get(row["source"], 1.0)
        lines.append(
            f"{row['source']:<34s} {row['name']:<9s} {str(row['m']):>5s} "
            f"{row['count']:5d} {pct(layer_share):>8s} "
            f"{row['oi_flop_per_byte']:8.1f} {pct(row['pe_work_ratio']):>7s} "
            f"{pct(row['weight_req_share']):>7s} "
            f"{row['theoretical_bound']:>10s}/{row['observed_bound']:<12s}"
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "- OI is arithmetic intensity from ONNXim GFLOPs/GB.",
            "- AchTF/AchGB/s are achieved throughput at the configured core clock.",
            "- PEwork is ONNXim ideal matmul work divided by wall cycles; it can exceed 100% when work is spread across cores/pipeline.",
            "- Wshare/Ashare are DRAM request shares.  High Wshare with low PEwork means W-memory-bound.",
            "- Theoretical bound uses peak_compute/peak_mem.  Observed bound is a practical proxy from PEwork and Wshare.",
        ]
    )
    (output_dir / "llama_roofline_profile.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "llama_roofline_profile.json").write_text(
        json.dumps({"components": rows, "layers": layer_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print((output_dir / "llama_roofline_profile.txt").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/onnxim_graphbit/llama_roofline"))
    parser.add_argument("--core-freq-mhz", type=float, default=1000.0)
    parser.add_argument("--peak-gflops", type=float, default=131072.0)
    parser.add_argument("--peak-gbps", type=float, default=614.4)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for summary in args.summary:
        raw_rows = read_tsv(summary)
        source = summary.parent.name
        rows = [
            component_metrics(
                row,
                source=source,
                freq_mhz=args.core_freq_mhz,
                peak_gflops=args.peak_gflops,
                peak_gbps=args.peak_gbps,
            )
            for row in raw_rows
        ]
        all_rows.extend(rows)
        layer_rows.append(
            aggregate_layer(
                rows,
                freq_mhz=args.core_freq_mhz,
                peak_gflops=args.peak_gflops,
                peak_gbps=args.peak_gbps,
            )
        )

    write_outputs(args.output_dir, all_rows, layer_rows, peak_gflops=args.peak_gflops, peak_gbps=args.peak_gbps)
    print(f"[LlamaRoofline] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
