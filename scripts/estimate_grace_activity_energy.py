#!/usr/bin/env python3
"""Estimate GRACE energy from the unified activity trace.

This is an activity-based analytical estimator.  It intentionally keeps the
event-energy parameters in a JSON file so that DC/CACTI/published-model numbers
can replace the defaults without changing the trace generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

ROW_COLUMNS = [
    "task",
    "dataset",
    "component",
    "subcomponent",
    "module",
    "source_kind",
    "time_s",
    "bitmac_energy_j",
    "rf_energy_j",
    "psum_energy_j",
    "sram_energy_j",
    "fifo_energy_j",
    "hbm_energy_j",
    "cam_energy_j",
    "leakage_energy_j",
    "total_energy_j",
    "avg_power_w",
    "notes",
]

SUMMARY_COLUMNS = [
    "task",
    "dataset",
    "time_s_sum",
    "bitmac_energy_j",
    "rf_energy_j",
    "psum_energy_j",
    "sram_energy_j",
    "fifo_energy_j",
    "hbm_energy_j",
    "cam_energy_j",
    "leakage_energy_j",
    "total_energy_j",
    "avg_power_w",
]

COMPONENT_COLUMNS = [
    "component",
    "time_s_sum",
    "bitmac_energy_j",
    "rf_energy_j",
    "psum_energy_j",
    "sram_energy_j",
    "fifo_energy_j",
    "hbm_energy_j",
    "cam_energy_j",
    "leakage_energy_j",
    "total_energy_j",
    "avg_power_w",
]

AREA_COLUMNS = ["unit", "components", "area_mm2", "reference_power_w", "source"]


DEFAULT_PARAMS: dict[str, Any] = {
    "schema": "grace_activity_energy_params.v1",
    "profile": "analytical_28nm_placeholder_v0",
    "units": {
        "bit_mac_pj": "pJ per BFP bit-MAC event",
        "rf_read_pj": "pJ per RF/broadcast read event",
        "psum_pj": "pJ per psum read/write/update event",
        "sram_pj_per_byte": "pJ per on-chip SRAM byte access",
        "hbm_pj_per_byte": "pJ per off-chip/local-DRAM byte access",
    },
    "event_energy_pj": {
        "bit_mac": 0.05,
        "rf_read": 0.02,
        "psum_read": 0.02,
        "psum_write": 0.02,
        "psum_update": 0.02,
        "sram_read_per_byte": 0.25,
        "sram_write_per_byte": 0.30,
        "fifo_read": 0.01,
        "fifo_write": 0.01,
        "hbm_read_per_byte": 20.0,
        "hbm_write_per_byte": 20.0,
        "cam_search": 61180.0,
        "cam_hot_read": 8070.0,
        "cam_write": 8070.0,
    },
    "leakage_power_w": {
        "BFPArray": 0.0,
        "BFPLoaderControl": 0.0,
        "CAMFrontend": 0.06175,
        "NDPEmbeddingIO": 0.0,
        "NDPGraphMemory": 0.0,
    },
    "area_power_model": [
        {
            "unit": "BFP NPU systolic arrays",
            "components": "BFPArray,BFPLoaderControl",
            "area_mm2": 48.0,
            "reference_power_w": 34.0,
            "source": "current model input; replace after RTL synthesis",
        },
        {
            "unit": "SH-CAM active directory + hot buffer",
            "components": "CAMFrontend",
            "area_mm2": 0.6726,
            "reference_power_w": 0.67355,
            "source": "CACTI-backed 4K directory plus 64-entry hot buffer at 10M lookup/s",
        },
        {
            "unit": "NDP graph/embedding controller",
            "components": "NDPEmbeddingIO,NDPGraphMemory",
            "area_mm2": 12.0,
            "reference_power_w": 1.32645,
            "source": "current model input; replace after RTL synthesis",
        },
    ],
}


def find_default_activity_trace(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates: list[Path] = []
    env_value = os.environ.get("OFA_OUTPUT_ROOT")
    if env_value:
        candidates.append(Path(env_value) / "grace_activity_trace" / "grace_activity_trace.tsv")
    candidates.extend(
        [
            REPO_ROOT.parent / "Transformer" / "OFA" / "output" / "grace_activity_trace" / "grace_activity_trace.tsv",
            REPO_ROOT.parent / "output" / "grace_activity_trace" / "grace_activity_trace.tsv",
            Path("/home/zhangshangtong/Transformer/OFA/output/grace_activity_trace/grace_activity_trace.tsv"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise SystemExit(f"Cannot find grace_activity_trace.tsv. Searched: {searched}")


def load_params(path: Path | None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_PARAMS))
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = json.loads(json.dumps(DEFAULT_PARAMS))
    for section in ("event_energy_pj", "leakage_power_w"):
        params[section].update(payload.get(section, {}))
    if "area_power_model" in payload:
        params["area_power_model"] = payload["area_power_model"]
    params["profile"] = payload.get("profile", params["profile"])
    return params


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing tsv: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, 0) for col in columns})


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def estimate_row_energy(row: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    e = params["event_energy_pj"]
    component = row["component"]
    bitmac_pj = f(row, "bit_mac_ops") * e["bit_mac"]
    rf_pj = (f(row, "a_rf_reads") + f(row, "w_rf_reads")) * e["rf_read"]
    psum_pj = (
        f(row, "psum_reads") * e["psum_read"]
        + f(row, "psum_writes") * e["psum_write"]
        + f(row, "psum_updates") * e["psum_update"]
    )
    sram_pj = f(row, "sram_read_bytes") * e["sram_read_per_byte"] + f(row, "sram_write_bytes") * e["sram_write_per_byte"]
    fifo_pj = f(row, "fifo_reads") * e["fifo_read"] + f(row, "fifo_writes") * e["fifo_write"]
    hbm_pj = f(row, "hbm_read_bytes") * e["hbm_read_per_byte"] + f(row, "hbm_write_bytes") * e["hbm_write_per_byte"]
    cam_pj = (
        f(row, "cam_lookups") * e["cam_search"]
        + f(row, "cam_hot_reads") * e["cam_hot_read"]
        + f(row, "cam_inserts") * e["cam_write"]
    )
    leakage_w = float(params["leakage_power_w"].get(component, 0.0) or 0.0)
    leakage_pj = leakage_w * f(row, "time_s") * 1.0e12
    total_pj = bitmac_pj + rf_pj + psum_pj + sram_pj + fifo_pj + hbm_pj + cam_pj + leakage_pj
    total_j = total_pj * 1.0e-12
    time_s = f(row, "time_s")
    out = {
        "task": row["task"],
        "dataset": row["dataset"],
        "component": component,
        "subcomponent": row.get("subcomponent", ""),
        "module": row.get("module", ""),
        "source_kind": row.get("source_kind", ""),
        "time_s": time_s,
        "bitmac_energy_j": bitmac_pj * 1.0e-12,
        "rf_energy_j": rf_pj * 1.0e-12,
        "psum_energy_j": psum_pj * 1.0e-12,
        "sram_energy_j": sram_pj * 1.0e-12,
        "fifo_energy_j": fifo_pj * 1.0e-12,
        "hbm_energy_j": hbm_pj * 1.0e-12,
        "cam_energy_j": cam_pj * 1.0e-12,
        "leakage_energy_j": leakage_pj * 1.0e-12,
        "total_energy_j": total_j,
        "avg_power_w": total_j / time_s if time_s > 0.0 else 0.0,
        "notes": row.get("notes", ""),
    }
    return out


def summarize(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    energy_fields = [
        "time_s",
        "bitmac_energy_j",
        "rf_energy_j",
        "psum_energy_j",
        "sram_energy_j",
        "fifo_energy_j",
        "hbm_energy_j",
        "cam_energy_j",
        "leakage_energy_j",
        "total_energy_j",
    ]
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        if key not in buckets:
            buckets[key] = {field: row[field] for field in key_fields}
            for field in energy_fields:
                buckets[key][field] = 0.0
        for field in energy_fields:
            buckets[key][field] += float(row[field])
    out = []
    for bucket in buckets.values():
        bucket["time_s_sum"] = bucket.pop("time_s")
        bucket["avg_power_w"] = bucket["total_energy_j"] / bucket["time_s_sum"] if bucket["time_s_sum"] > 0 else 0.0
        out.append(bucket)
    return out


def area_rows(params: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    total_area = 0.0
    total_power = 0.0
    for row in params["area_power_model"]:
        rows.append(row)
        total_area += float(row.get("area_mm2", 0.0) or 0.0)
        total_power += float(row.get("reference_power_w", 0.0) or 0.0)
    rows.append(
        {
            "unit": "GRACE modeled total",
            "components": "all",
            "area_mm2": total_area,
            "reference_power_w": total_power,
            "source": "sum of model rows; off-chip DRAM area excluded",
        }
    )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    *,
    path: Path,
    activity_trace: Path,
    params: dict[str, Any],
    task_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    areas: list[dict[str, Any]],
) -> None:
    lines = [
        "# GRACE Activity-Based Energy Estimate",
        "",
        f"- Activity trace: `{activity_trace}`",
        f"- Parameter profile: `{params['profile']}`",
        "- Energy equation: `E = sum(activity_count * event_energy) + leakage_power * time`.",
        "- This is an analytical estimate. Replace the default pJ/event values with DC/CACTI/published-model values before making final claims.",
        "",
        "## Task Energy",
        "",
        "| Task | Energy (J) | Time Sum (s) | Avg Power (W) | HBM (J) | CAM (J) | BFP bitmac (J) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(task_rows, key=lambda r: r["task"]):
        lines.append(
            f"| {row['task']} | {row['total_energy_j']:.6f} | {row['time_s_sum']:.6f} | "
            f"{row['avg_power_w']:.3f} | {row['hbm_energy_j']:.6f} | "
            f"{row['cam_energy_j']:.6f} | {row['bitmac_energy_j']:.6f} |"
        )
    lines.extend(["", "## Component Energy", "", "| Component | Energy (J) | Time Sum (s) | Avg Power (W) |", "| --- | ---: | ---: | ---: |"])
    for row in sorted(component_rows, key=lambda r: r["component"]):
        lines.append(
            f"| {row['component']} | {row['total_energy_j']:.6f} | "
            f"{row['time_s_sum']:.6f} | {row['avg_power_w']:.3f} |"
        )
    lines.extend(["", "## Area/Power Model", "", "| Unit | Area (mm^2) | Ref. Power (W) | Source |", "| --- | ---: | ---: | --- |"])
    for row in areas:
        lines.append(
            f"| {row['unit']} | {float(row['area_mm2']):.4f} | "
            f"{float(row['reference_power_w']):.4f} | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "## Default Event Energies",
            "",
            "```json",
            json.dumps(params["event_energy_pj"], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-trace", type=Path, default=None)
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activity_trace = find_default_activity_trace(args.activity_trace)
    output_dir = args.output_dir or activity_trace.parent
    params = load_params(args.params_json)
    trace_rows = read_tsv(activity_trace)
    row_energy = [estimate_row_energy(row, params) for row in trace_rows]
    task_energy = summarize(row_energy, ["task", "dataset"])
    component_energy = summarize(row_energy, ["component"])
    areas = area_rows(params)

    write_tsv(output_dir / "grace_activity_energy_by_row.tsv", row_energy, ROW_COLUMNS)
    write_tsv(output_dir / "grace_activity_energy_by_task.tsv", task_energy, SUMMARY_COLUMNS)
    write_tsv(output_dir / "grace_activity_energy_by_component.tsv", component_energy, COMPONENT_COLUMNS)
    write_tsv(output_dir / "grace_area_power_model.tsv", areas, AREA_COLUMNS)
    write_json(output_dir / "grace_energy_params.default.json", params)
    write_json(
        output_dir / "grace_activity_energy.json",
        {
            "schema": "grace_activity_energy.v1",
            "activity_trace": str(activity_trace),
            "params": params,
            "row_energy": row_energy,
            "task_energy": task_energy,
            "component_energy": component_energy,
            "area_power_model": areas,
        },
    )
    write_report(
        path=output_dir / "GRACE_ACTIVITY_ENERGY.md",
        activity_trace=activity_trace,
        params=params,
        task_rows=task_energy,
        component_rows=component_energy,
        areas=areas,
    )
    print(f"[GRACEEnergy] wrote {output_dir / 'grace_activity_energy_by_task.tsv'}")
    print(f"[GRACEEnergy] wrote {output_dir / 'GRACE_ACTIVITY_ENERGY.md'}")


if __name__ == "__main__":
    main()
