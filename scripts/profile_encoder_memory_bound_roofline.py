#!/usr/bin/env python3
"""Roofline check for the current LLaMA encoder embedding workload.

The goal is not cycle accuracy.  This script checks whether the measured GPU
encoder wall time can plausibly be explained by HBM bandwidth alone under the
current batch size.  It intentionally uses conservative traffic assumptions:
the full transformer weight set is assumed to be streamed once per encoder
batch.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent
DEFAULT_OUTPUT_DIR = OFA_ROOT / "output" / "encoder_memory_bound_roofline"
DEFAULT_REPO_REPORT = REPO_ROOT / "docs" / "results" / "ENCODER_MEMORY_BOUND_ROOFLINE.md"


@dataclass(frozen=True)
class RooflineResult:
    nodes: int
    batch_size: int
    seq_len: int
    encoder_batches: int
    token_rows_per_batch: int
    layers: int
    hidden: int
    intermediate: int
    linear_params: float
    linear_macs_per_batch: float
    attention_macs_per_batch: float
    total_macs_per_batch: float
    total_macs_all_batches: float
    measured_gpu_time_s: float
    measured_effective_tmac_s: float
    measured_effective_tflop_s: float
    fp16_weight_stream_bytes: float
    w4_weight_stream_bytes: float
    activation_lower_bound_bytes: float
    fp16_weight_only_hbm_s: float
    w4_weight_only_hbm_s: float
    fp16_weight_plus_activation_hbm_s: float
    w4_weight_plus_activation_hbm_s: float
    fp16_weight_only_fraction_of_measured: float
    w4_weight_only_fraction_of_measured: float
    fp16_weight_plus_activation_fraction_of_measured: float
    w4_weight_plus_activation_fraction_of_measured: float
    fp16_arithmetic_intensity_flop_per_byte: float
    w4_arithmetic_intensity_flop_per_byte: float
    gpu_roofline_ridge_flop_per_byte: float
    fp16_compute_lower_bound_s: float
    measured_over_compute_lower_bound: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=2708)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--measured-gpu-time-s", type=float, default=108.0)
    parser.add_argument("--gpu-hbm-gbs", type=float, default=1008.0)
    parser.add_argument("--gpu-fp16-tflops", type=float, default=165.2)
    parser.add_argument("--activation-bytes", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-report", type=Path, default=DEFAULT_REPO_REPORT)
    return parser.parse_args()


def fmt_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    out = float(value)
    idx = 0
    while abs(out) >= 1024.0 and idx < len(units) - 1:
        out /= 1024.0
        idx += 1
    return f"{out:.2f} {units[idx]}"


def fmt_s(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}s"
    if value >= 1.0e-3:
        return f"{value * 1.0e3:.3f}ms"
    return f"{value * 1.0e6:.3f}us"


def build_result(args: argparse.Namespace) -> RooflineResult:
    batches = math.ceil(args.nodes / args.batch_size)
    m = args.batch_size * args.seq_len
    h = args.hidden
    i = args.intermediate
    layers = args.layers

    # LLaMA layer linear projections: Q/K/V/O plus gate/up/down.
    linear_params_per_layer = 4.0 * h * h + 3.0 * h * i
    linear_params = layers * linear_params_per_layer
    linear_macs_per_layer = float(m) * linear_params_per_layer

    # Attention score and value matmuls.  This is small relative to LLaMA FFN
    # at the current M=4*512, but included for completeness.
    attention_macs_per_layer = 2.0 * args.batch_size * args.seq_len * args.seq_len * h
    total_macs_per_batch = layers * (linear_macs_per_layer + attention_macs_per_layer)
    linear_macs_per_batch = layers * linear_macs_per_layer
    attention_macs_per_batch = layers * attention_macs_per_layer
    total_macs_all_batches = total_macs_per_batch * batches

    fp16_weight_stream_bytes = linear_params * 2.0 * batches
    w4_weight_stream_bytes = linear_params * 0.5 * batches

    # Conservative activation lower bound.  It counts only large tensor reads
    # and writes around projections/FFN, not every temporary tensor in PyTorch.
    activation_bytes_per_layer = (
        # QKV input, attention output/O input, O output.
        (3.0 * m * h)
        # gate/up inputs and intermediate tensors, down output.
        + (2.0 * m * h)
        + (3.0 * m * i)
        + (1.0 * m * h)
    ) * args.activation_bytes
    activation_lower_bound_bytes = activation_bytes_per_layer * layers * batches

    measured_effective_tmac_s = total_macs_all_batches / args.measured_gpu_time_s / 1.0e12
    measured_effective_tflop_s = 2.0 * measured_effective_tmac_s
    fp16_weight_only_hbm_s = fp16_weight_stream_bytes / (args.gpu_hbm_gbs * 1.0e9)
    w4_weight_only_hbm_s = w4_weight_stream_bytes / (args.gpu_hbm_gbs * 1.0e9)
    fp16_weight_plus_activation_hbm_s = (
        fp16_weight_stream_bytes + activation_lower_bound_bytes
    ) / (args.gpu_hbm_gbs * 1.0e9)
    w4_weight_plus_activation_hbm_s = (
        w4_weight_stream_bytes + activation_lower_bound_bytes
    ) / (args.gpu_hbm_gbs * 1.0e9)

    total_flops = 2.0 * total_macs_all_batches
    fp16_arithmetic_intensity = total_flops / max(
        1.0, fp16_weight_stream_bytes + activation_lower_bound_bytes
    )
    w4_arithmetic_intensity = total_flops / max(
        1.0, w4_weight_stream_bytes + activation_lower_bound_bytes
    )
    ridge = args.gpu_fp16_tflops * 1.0e12 / (args.gpu_hbm_gbs * 1.0e9)
    fp16_compute_lower_bound_s = total_flops / (args.gpu_fp16_tflops * 1.0e12)

    return RooflineResult(
        nodes=args.nodes,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        encoder_batches=batches,
        token_rows_per_batch=m,
        layers=layers,
        hidden=h,
        intermediate=i,
        linear_params=linear_params,
        linear_macs_per_batch=linear_macs_per_batch,
        attention_macs_per_batch=attention_macs_per_batch,
        total_macs_per_batch=total_macs_per_batch,
        total_macs_all_batches=total_macs_all_batches,
        measured_gpu_time_s=args.measured_gpu_time_s,
        measured_effective_tmac_s=measured_effective_tmac_s,
        measured_effective_tflop_s=measured_effective_tflop_s,
        fp16_weight_stream_bytes=fp16_weight_stream_bytes,
        w4_weight_stream_bytes=w4_weight_stream_bytes,
        activation_lower_bound_bytes=activation_lower_bound_bytes,
        fp16_weight_only_hbm_s=fp16_weight_only_hbm_s,
        w4_weight_only_hbm_s=w4_weight_only_hbm_s,
        fp16_weight_plus_activation_hbm_s=fp16_weight_plus_activation_hbm_s,
        w4_weight_plus_activation_hbm_s=w4_weight_plus_activation_hbm_s,
        fp16_weight_only_fraction_of_measured=fp16_weight_only_hbm_s / args.measured_gpu_time_s,
        w4_weight_only_fraction_of_measured=w4_weight_only_hbm_s / args.measured_gpu_time_s,
        fp16_weight_plus_activation_fraction_of_measured=(
            fp16_weight_plus_activation_hbm_s / args.measured_gpu_time_s
        ),
        w4_weight_plus_activation_fraction_of_measured=(
            w4_weight_plus_activation_hbm_s / args.measured_gpu_time_s
        ),
        fp16_arithmetic_intensity_flop_per_byte=fp16_arithmetic_intensity,
        w4_arithmetic_intensity_flop_per_byte=w4_arithmetic_intensity,
        gpu_roofline_ridge_flop_per_byte=ridge,
        fp16_compute_lower_bound_s=fp16_compute_lower_bound_s,
        measured_over_compute_lower_bound=args.measured_gpu_time_s / fp16_compute_lower_bound_s,
    )


def render_report(result: RooflineResult, args: argparse.Namespace) -> str:
    lines = [
        "# Encoder Memory-Bound Roofline Check",
        "",
        "## Workload",
        "",
        f"- Nodes: `{result.nodes}`.",
        f"- Batch size: `{result.batch_size}` nodes.",
        f"- Sequence length: `{result.seq_len}` tokens.",
        f"- Token rows per encoder batch: `{result.token_rows_per_batch}`.",
        f"- Encoder batches: `{result.encoder_batches}`.",
        f"- Model shape: LLaMA-style `{result.layers}` layers, hidden `{result.hidden}`, intermediate `{result.intermediate}`.",
        f"- Measured GPU BFPA5 encoding time: `{result.measured_gpu_time_s:.3f}s`.",
        f"- GPU HBM bandwidth used for the check: `{args.gpu_hbm_gbs:.1f} GB/s`.",
        f"- GPU FP16 Tensor throughput used for compute lower bound: `{args.gpu_fp16_tflops:.1f} TFLOP/s`.",
        "",
        "## Operation Count",
        "",
        f"- Linear parameters counted in transformer blocks: `{result.linear_params / 1.0e9:.3f}B`.",
        f"- Linear MACs per encoder batch: `{result.linear_macs_per_batch / 1.0e12:.3f}T`.",
        f"- Attention MACs per encoder batch: `{result.attention_macs_per_batch / 1.0e12:.3f}T`.",
        f"- Total MACs over Cora: `{result.total_macs_all_batches / 1.0e15:.3f}P`.",
        f"- Measured effective throughput: `{result.measured_effective_tmac_s:.2f} TMAC/s` (`{result.measured_effective_tflop_s:.2f} TFLOP/s` if one MAC is two FLOPs).",
        "",
        "## HBM Lower Bounds",
        "",
        "| Traffic model | Total bytes | HBM-only lower bound | Fraction of measured 108s |",
        "| --- | ---: | ---: | ---: |",
        f"| FP16 weights only | {fmt_bytes(result.fp16_weight_stream_bytes)} | {fmt_s(result.fp16_weight_only_hbm_s)} | {100.0 * result.fp16_weight_only_fraction_of_measured:.2f}% |",
        f"| W4-packed weights only | {fmt_bytes(result.w4_weight_stream_bytes)} | {fmt_s(result.w4_weight_only_hbm_s)} | {100.0 * result.w4_weight_only_fraction_of_measured:.2f}% |",
        f"| FP16 weights + activation lower bound | {fmt_bytes(result.fp16_weight_stream_bytes + result.activation_lower_bound_bytes)} | {fmt_s(result.fp16_weight_plus_activation_hbm_s)} | {100.0 * result.fp16_weight_plus_activation_fraction_of_measured:.2f}% |",
        f"| W4 weights + activation lower bound | {fmt_bytes(result.w4_weight_stream_bytes + result.activation_lower_bound_bytes)} | {fmt_s(result.w4_weight_plus_activation_hbm_s)} | {100.0 * result.w4_weight_plus_activation_fraction_of_measured:.2f}% |",
        "",
        "## Roofline Read",
        "",
        f"- FP16-weight arithmetic intensity: `{result.fp16_arithmetic_intensity_flop_per_byte:.1f}` FLOP/byte.",
        f"- W4-weight arithmetic intensity: `{result.w4_arithmetic_intensity_flop_per_byte:.1f}` FLOP/byte.",
        f"- GPU ridge point: `{result.gpu_roofline_ridge_flop_per_byte:.1f}` FLOP/byte.",
        f"- FP16 compute lower bound: `{fmt_s(result.fp16_compute_lower_bound_s)}`; measured time is `{result.measured_over_compute_lower_bound:.2f}x` this lower bound.",
        "",
        "## Conclusion",
        "",
        "- Even the conservative FP16 weight-stream assumption gives an HBM-only lower bound far below the measured time.",
        "- The workload has high arithmetic intensity because `batch_size * seq_len = 2048` token rows reuse the same layer weights.",
        "- The current encoder run is therefore not plausibly explained by HBM bandwidth alone; it is compute/kernel-overhead dominated rather than decoder-style memory-bound.",
        "- GPU and NPU comparisons should keep the same node batch size and sequence length, then align full BFPA5 throughput before enabling TSER.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = build_result(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {"args": payload_args, "result": asdict(result)}
    (args.output_dir / "encoder_memory_bound_roofline.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(result, args)
    (args.output_dir / "ENCODER_MEMORY_BOUND_ROOFLINE.md").write_text(report, encoding="utf-8")
    args.repo_report.parent.mkdir(parents=True, exist_ok=True)
    args.repo_report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.output_dir / 'ENCODER_MEMORY_BOUND_ROOFLINE.md'}")
    print(f"Wrote {args.repo_report}")


if __name__ == "__main__":
    main()
