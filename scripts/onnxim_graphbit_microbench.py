#!/usr/bin/env python3
"""Generate and run ONNXim LLaMA GEMM microbenchmarks.

The generated ONNX graphs are shape carriers for ONNXim.  They cover the main
LLaMA-7B encoder GEMMs:

    q/k/v/o projection:  [seq, hidden] x [hidden, hidden]
    gate/up projection: [seq, hidden] x [hidden, intermediate]
    down projection:    [seq, intermediate] x [intermediate, hidden]

The script keeps ONNXim as an external simulator and stores all generated
artifacts under output/onnxim_graphbit/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GemmSpec:
    name: str
    m: int
    k: int
    n: int
    count_per_layer: int

    @property
    def model_name(self) -> str:
        return f"llama7b_{self.name}_m{self.m}_k{self.k}_n{self.n}"


def ofa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_onnxim_home(root: Path, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("ONNXIM_REPO") or os.environ.get("ONNXIM_SRC")
    if env:
        candidates.append(Path(env))
    candidates.extend([root / "ONNXim", root / "GraphhopSimhash" / "ONNXim"])
    for candidate in candidates:
        sim = candidate / "build" / "bin" / "Simulator"
        if sim.exists():
            return candidate.resolve()
    raise SystemExit("Could not find ONNXim build/bin/Simulator. Pass --onnxim-home.")


def llama_specs(seq_len: int, hidden: int, intermediate: int) -> list[GemmSpec]:
    return [
        GemmSpec("proj", seq_len, hidden, hidden, 4),
        GemmSpec("ffn_up", seq_len, hidden, intermediate, 2),
        GemmSpec("ffn_down", seq_len, intermediate, hidden, 1),
    ]


def generate_onnx(spec: GemmSpec, workspace: Path) -> Path:
    import torch

    class LinearNoBias(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            # ONNXim uses the graph as a shape carrier and reads precision from
            # its hardware config.  Keep export in fp32 because PyTorch CPU does
            # not support fp16 Linear tracing in this environment.
            self.fc = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model_dir = workspace / "models" / spec.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / f"{spec.model_name}.onnx"
    if onnx_path.exists():
        return onnx_path

    model = LinearNoBias(spec.k, spec.n).eval()
    dummy = torch.zeros((spec.m, spec.k), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    return onnx_path


def write_model_list(spec: GemmSpec, workspace: Path) -> Path:
    model_list_dir = workspace / "model_lists"
    model_list_dir.mkdir(parents=True, exist_ok=True)
    path = model_list_dir / f"{spec.model_name}.json"
    payload = {"models": [{"name": spec.model_name, "request_time": 0}]}
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def run_simulator(spec: GemmSpec, workspace: Path, onnxim_home: Path, config: Path, log_level: str) -> Path:
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.model_name}.log"
    model_list = write_model_list(spec, workspace)
    simulator = onnxim_home / "build" / "bin" / "Simulator"
    env = os.environ.copy()
    env["ONNXIM_HOME"] = str(workspace.resolve())
    # ONNXim resolves DRAM/NoC configs as $ONNXIM_HOME/configs/<path>.
    # Keep generated models in the workspace while reusing the official configs.
    workspace_configs = workspace / "configs"
    if not workspace_configs.exists():
        try:
            workspace_configs.symlink_to(onnxim_home / "configs", target_is_directory=True)
        except OSError:
            import shutil

            shutil.copytree(onnxim_home / "configs", workspace_configs)
    command = [
        str(simulator),
        "--config",
        str(config.resolve()),
        "--models_list",
        str(model_list.resolve()),
        "--log_level",
        log_level,
    ]
    with log_path.open("w") as handle:
        subprocess.run(command, cwd=onnxim_home, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return log_path


def write_graphbit_config(base_config: Path, workspace: Path, args: argparse.Namespace) -> Path:
    if args.graphbit_depth >= args.graphbit_full_depth and not args.graphbit_bound_enable:
        return base_config

    payload = json.loads(base_config.read_text())
    payload.update(
        {
            "graphbit_enable": True,
            "graphbit_precision_depth": int(args.graphbit_depth),
            "graphbit_full_depth": int(args.graphbit_full_depth),
            "graphbit_min_depth": int(args.graphbit_min_depth),
            "graphbit_bound_enable": bool(args.graphbit_bound_enable),
            "graphbit_bound_tolerance": float(args.graphbit_bound_tolerance),
            "graphbit_bound_scale": float(args.graphbit_bound_scale),
            "graphbit_memory_scale": float(args.graphbit_memory_scale),
        }
    )
    config_dir = workspace / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"gbp{args.graphbit_depth}_min{args.graphbit_min_depth}"
    if args.graphbit_bound_enable:
        suffix += f"_tol{str(args.graphbit_bound_tolerance).replace('.', 'p')}"
    path = config_dir / f"{base_config.stem}_{suffix}.json"
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    cycle_matches = re.findall(r"Simulation Finished at\s+(\d+)\s+cycle", text)
    compute_matches = re.findall(r"Total compute time\s+(\d+)", text)
    gflops_matches = re.findall(r"\[GemmWS\]: total\s+([0-9.eE+-]+)\s+GFLOPs,\s+([0-9.eE+-]+)\s+GB", text)
    read_matches = [int(v) for v in re.findall(r"total_num_read_requests:\s+(\d+)", text)]
    write_matches = [int(v) for v in re.findall(r"total_num_write_requests:\s+(\d+)", text)]
    avg_bw_matches = re.findall(r"avg BW utilization\s+\d+%\s+\((\d+)\s+reads,\s+(\d+)\s+writes\)", text)
    graphbit_matches = re.findall(
        r"GraphBit Inst\s+(\d+)\s+BoundStops\s+(\d+)\s+AvgDepth\s+([0-9.]+)\s+AvgSavedBitplanes\s+([0-9.]+)",
        text,
    )

    bw_reads = sum(int(r) for r, _ in avg_bw_matches)
    bw_writes = sum(int(w) for _, w in avg_bw_matches)
    gflops = sum(float(g) for g, _ in gflops_matches)
    gb = sum(float(gb) for _, gb in gflops_matches)
    graphbit_inst = sum(int(v[0]) for v in graphbit_matches)
    graphbit_bound_stops = sum(int(v[1]) for v in graphbit_matches)
    graphbit_avg_depth = None
    graphbit_avg_saved = None
    if graphbit_inst:
        graphbit_avg_depth = (
            sum(int(inst) * float(avg) for inst, _, avg, _ in graphbit_matches)
            / graphbit_inst
        )
        graphbit_avg_saved = (
            sum(int(inst) * float(avg) for inst, _, _, avg in graphbit_matches)
            / graphbit_inst
        )

    return {
        "cycles": int(cycle_matches[-1]) if cycle_matches else None,
        "compute_cycles_sum": sum(int(v) for v in compute_matches),
        "gflops": gflops,
        "gb": gb,
        "dram_read_requests": sum(read_matches) if read_matches else bw_reads,
        "dram_write_requests": sum(write_matches) if write_matches else bw_writes,
        "avg_bw_read_requests": bw_reads,
        "avg_bw_write_requests": bw_writes,
        "graphbit_inst": graphbit_inst,
        "graphbit_bound_stops": graphbit_bound_stops,
        "graphbit_avg_depth": graphbit_avg_depth,
        "graphbit_avg_saved_bitplanes": graphbit_avg_saved,
        "log": str(path),
    }


def write_summary(specs: list[GemmSpec], workspace: Path, layers: int) -> dict[str, Any]:
    rows = []
    for spec in specs:
        parsed = parse_log(workspace / "logs" / f"{spec.model_name}.log")
        row = {
            "name": spec.name,
            "model_name": spec.model_name,
            "m": spec.m,
            "k": spec.k,
            "n": spec.n,
            "count_per_layer": spec.count_per_layer,
            **parsed,
        }
        rows.append(row)

    summary_tsv = workspace / "summary.tsv"
    with summary_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    def weighted(key: str) -> float:
        return sum((row[key] or 0) * row["count_per_layer"] for row in rows)

    aggregate = {
        "schema": "onnxim_graphbit_microbench.v1",
        "workspace": str(workspace),
        "layers": layers,
        "per_layer": {
            "cycles": weighted("cycles"),
            "compute_cycles_sum": weighted("compute_cycles_sum"),
            "gflops": weighted("gflops"),
            "gb": weighted("gb"),
            "dram_read_requests": weighted("dram_read_requests"),
            "dram_write_requests": weighted("dram_write_requests"),
        },
    }
    aggregate["encoder"] = {
        key: value * layers for key, value in aggregate["per_layer"].items()
    }

    aggregate_path = workspace / "aggregate.json"
    with aggregate_path.open("w") as handle:
        json.dump(aggregate, handle, indent=2)
        handle.write("\n")

    print(f"[ONNXimGraphBit] wrote {summary_tsv}")
    print(f"[ONNXimGraphBit] wrote {aggregate_path}")
    print(
        "[ONNXimGraphBit] encoder baseline "
        f"cycles={aggregate['encoder']['cycles']:.0f} "
        f"read_req={aggregate['encoder']['dram_read_requests']:.0f} "
        f"write_req={aggregate['encoder']['dram_write_requests']:.0f}"
    )
    return aggregate


def main() -> None:
    root = ofa_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnxim-home", default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--graphbit-depth", type=int, default=8)
    parser.add_argument("--graphbit-full-depth", type=int, default=8)
    parser.add_argument("--graphbit-min-depth", type=int, default=4)
    parser.add_argument("--graphbit-bound-enable", action="store_true")
    parser.add_argument("--graphbit-bound-tolerance", type=float, default=0.0)
    parser.add_argument("--graphbit-bound-scale", type=float, default=1.0)
    parser.add_argument("--graphbit-memory-scale", type=float, default=1.0)
    parser.add_argument("--action", choices=["generate", "run", "summarize", "all"], default="all")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    onnxim_home = find_onnxim_home(root, args.onnxim_home)
    workspace = args.workspace or (root / "output" / "onnxim_graphbit" / f"microbench_s{args.seq_len}")
    workspace.mkdir(parents=True, exist_ok=True)
    base_config = args.config or (onnxim_home / "configs" / "systolic_ws_128x128_c4_simple_noc_tpuv4.json")
    config = write_graphbit_config(base_config, workspace, args)
    if config != base_config:
        print(f"[ONNXimGraphBit] using internal Graph-Bit config {config}")
    specs = llama_specs(args.seq_len, args.hidden, args.intermediate)

    if args.action in ("generate", "all"):
        for spec in specs:
            onnx_path = generate_onnx(spec, workspace)
            model_list = write_model_list(spec, workspace)
            print(f"[ONNXimGraphBit] generated {onnx_path}")
            print(f"[ONNXimGraphBit] generated {model_list}")

    if args.action in ("run", "all"):
        for spec in specs:
            if not (workspace / "models" / spec.model_name / f"{spec.model_name}.onnx").exists():
                generate_onnx(spec, workspace)
            log_path = run_simulator(spec, workspace, onnxim_home, config, args.log_level)
            print(f"[ONNXimGraphBit] ran {spec.model_name} -> {log_path}")

    if args.action in ("summarize", "all"):
        write_summary(specs, workspace, args.layers)


if __name__ == "__main__":
    main()
