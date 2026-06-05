"""Export GraphhopSimhash hash heads as a binary hardware trace."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import torch

from ...cli import build_parser, validate_args
from ...data import load_run_state
from ...runner import build_route_bundle, make_run_args, train_baseline_model


MAGIC = b"GHSIMTRACE"
TRACE_VERSION = 1
NUM_HEADS = 8
HASH_BITS = 16


def _bits_to_uint16(bits: torch.Tensor) -> torch.Tensor:
    if bits.size(1) != HASH_BITS:
        raise ValueError(f"hardware trace exporter requires {HASH_BITS}-bit heads, got {bits.size(1)}")
    weights = (2 ** torch.arange(HASH_BITS - 1, -1, -1, device=bits.device, dtype=torch.long)).view(1, -1)
    return (bits.to(torch.long) * weights).sum(dim=1).to(torch.int64).cpu()


def _compute_hash_values(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    features = features.to(device=matrix.device)
    bits = torch.matmul(features, matrix) > 0
    return _bits_to_uint16(bits)


def _degree_buckets(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    row, col = edge_index
    sym_nodes = torch.cat([row, col], dim=0)
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, sym_nodes, torch.ones(sym_nodes.numel(), dtype=torch.float32, device=device))
    buckets = torch.floor(torch.log2(degree + 1.0)).clamp(0, 255).to(torch.uint8)
    return buckets.cpu()


def _write_trace(path: str, hashes_by_head, degree_bucket, radius: int, support_threshold: int) -> None:
    num_nodes = int(hashes_by_head[0].numel())
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as out:
        out.write(struct.pack(
            "<16sIIIIII",
            MAGIC.ljust(16, b"\0"),
            TRACE_VERSION,
            num_nodes,
            NUM_HEADS,
            HASH_BITS,
            int(radius),
            int(support_threshold),
        ))
        for node_id in range(num_nodes):
            head_values = [int(head_hashes[node_id].item()) for head_hashes in hashes_by_head]
            out.write(struct.pack(
                "<I8HHBB",
                int(node_id),
                *head_values,
                0,  # optional sensitivity_q, reserved for future score-gate metadata
                int(degree_bucket[node_id].item()),
                0,
            ))


def main() -> None:
    parser = build_parser()
    parser.description = "Export 8-head 16-bit GraphhopSimhash hashes for C++ hardware simulators."
    parser.add_argument("--output", required=True, help="Output .trace path")
    args = parser.parse_args()
    validate_args(parser, args)

    if len(args.datasets) != 1:
        parser.error("hardware trace exporter expects exactly one dataset")
    ds_key = args.datasets[0]

    run_args = make_run_args(args, int(args.seed))
    if run_args.controller_seed is None:
        run_args.controller_seed = int(args.seed)
    if run_args.hash_head_seed is None:
        run_args.hash_head_seed = int(args.seed)
    if run_args.topology_sketch_seed is None:
        run_args.topology_sketch_seed = int(args.seed)

    def log_important(msg: str) -> None:
        print(msg)

    _conf, data, verify_features, device = load_run_state(ds_key, run_args, int(args.seed))
    model, base_acc, baseline_embs, baseline_logits = train_baseline_model(data, run_args, device)
    del model
    log_important(f"[TraceExport] baseline_acc={base_acc:.4f}")

    route_bundle = build_route_bundle(
        verify_features=verify_features,
        data=data,
        oracle_embs=baseline_embs,
        oracle_logits=baseline_logits,
        args=run_args,
        log_important=log_important,
        device=device,
    )

    hash_route_bits = [int(bits) for bits in route_bundle["hash_route_bits"][:NUM_HEADS]]
    if len(hash_route_bits) != NUM_HEADS or any(bits != HASH_BITS for bits in hash_route_bits):
        raise ValueError(
            "hardware trace exporter requires the first 8 hash heads to be 16-bit; "
            f"got {hash_route_bits}"
        )
    matrices = route_bundle["hash_route_matrices"]
    if matrices is None or len(matrices) < NUM_HEADS:
        raise ValueError("route bundle did not provide enough hash matrices for 8 heads")

    hashes_by_head = []
    for head_idx in range(NUM_HEADS):
        hashes_by_head.append(
            _compute_hash_values(
                route_bundle["hash_route_features"][head_idx],
                matrices[head_idx].to(device),
            )
        )

    degree_bucket = _degree_buckets(data.edge_index, int(data.num_nodes), device)
    support_threshold = int(args.route_min_support_hits[0] if args.route_min_support_hits else 3)
    _write_trace(args.output, hashes_by_head, degree_bucket, int(args.radius), support_threshold)

    size_bytes = os.path.getsize(args.output)
    print(
        f"[TraceExport] wrote {args.output} | nodes={int(data.num_nodes)} "
        f"| heads={NUM_HEADS} | bits={HASH_BITS} | radius={int(args.radius)} "
        f"| support_threshold={support_threshold} | bytes={size_bytes}"
    )


if __name__ == "__main__":
    main()
