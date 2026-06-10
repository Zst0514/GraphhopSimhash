"""Export GraphhopSimhash hash heads as a binary hardware trace."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import struct
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ONEFORALL_ROOT = PACKAGE_ROOT.parent


def _resolve_user_path(path: str, original_cwd: Path) -> str:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return str(raw)
    if str(raw).startswith("CAM_sim/"):
        return str(PACKAGE_ROOT / raw)
    return str(original_cwd / raw)


def _ensure_project_cwd() -> None:
    os.chdir(ONEFORALL_ROOT)


try:
    from ...cli import build_parser, validate_args
    from ...data import load_run_state
    from ...features import _compute_neighbor_mean
    from ...runner import build_route_bundle, make_run_args, train_baseline_model
    from ...scoring import build_node_risk_scores
    from .unified_frontend_support import compute_tser_scores
except ImportError:
    PACKAGE_ALIAS = "_graphhopsimhash_repo"
    init_py = PACKAGE_ROOT / "__init__.py"

    cleaned_sys_path = []
    for entry in sys.path:
        entry_path = Path(entry or os.getcwd()).resolve()
        if entry_path == PACKAGE_ROOT:
            continue
        cleaned_sys_path.append(entry)
    sys.path[:] = cleaned_sys_path
    if str(ONEFORALL_ROOT) not in sys.path:
        sys.path.insert(0, str(ONEFORALL_ROOT))

    def register_namespace_package(name: str, package_dir: Path) -> None:
        package = types.ModuleType(name)
        package.__path__ = [str(package_dir)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [str(package_dir)]
        sys.modules[name] = package

    register_namespace_package("models", ONEFORALL_ROOT / "models")
    register_namespace_package("data", ONEFORALL_ROOT / "data")

    spec = importlib.util.spec_from_file_location(
        PACKAGE_ALIAS,
        init_py,
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to create package alias for repo root: {PACKAGE_ROOT}")
    if PACKAGE_ALIAS not in sys.modules:
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE_ALIAS] = module
        spec.loader.exec_module(module)

    cli_mod = importlib.import_module(f"{PACKAGE_ALIAS}.cli")
    data_mod = importlib.import_module(f"{PACKAGE_ALIAS}.data")
    runner_mod = importlib.import_module(f"{PACKAGE_ALIAS}.runner")
    features_mod = importlib.import_module(f"{PACKAGE_ALIAS}.features")
    scoring_mod = importlib.import_module(f"{PACKAGE_ALIAS}.scoring")
    support_mod = importlib.import_module(f"{PACKAGE_ALIAS}.CAM_sim.tools.unified_frontend_support")
    build_parser = cli_mod.build_parser
    validate_args = cli_mod.validate_args
    load_run_state = data_mod.load_run_state
    build_route_bundle = runner_mod.build_route_bundle
    make_run_args = runner_mod.make_run_args
    train_baseline_model = runner_mod.train_baseline_model
    _compute_neighbor_mean = features_mod._compute_neighbor_mean
    build_node_risk_scores = scoring_mod.build_node_risk_scores
    compute_tser_scores = support_mod.compute_tser_scores


MAGIC = b"GHSIMTRACE"
TRACE_VERSION = 2
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


def _write_trace(path: str, hashes_by_head, score_bundle, degree_bucket, radius: int, support_threshold: int) -> None:
    num_nodes = int(hashes_by_head[0].numel())
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_q = score_bundle["sensitivity_q"].to(device="cpu", dtype=torch.int64)
    propagation_q = score_bundle["propagation_q"].to(device="cpu", dtype=torch.int64)
    graph_context_q = score_bundle["graph_context_q"].to(device="cpu", dtype=torch.int64)
    low_unique_q = score_bundle["low_degree_unique_q"].to(device="cpu", dtype=torch.int64)
    rarity_q = score_bundle["rarity_q"].to(device="cpu", dtype=torch.int64)

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
                "<I8HHBBBBBB",
                int(node_id),
                *head_values,
                int(sensitivity_q[node_id].item()),
                int(propagation_q[node_id].item()),
                int(graph_context_q[node_id].item()),
                int(low_unique_q[node_id].item()),
                int(rarity_q[node_id].item()),
                int(degree_bucket[node_id].item()),
                0,
            ))


def main() -> None:
    original_cwd = Path.cwd()
    parser = build_parser()
    parser.description = "Export 8-head 16-bit GraphhopSimhash hashes for C++ hardware simulators."
    parser.add_argument("--output", required=True, help="Output .trace path")
    args = parser.parse_args()
    validate_args(parser, args)
    args.output = _resolve_user_path(args.output, original_cwd)
    _ensure_project_cwd()

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
    if route_bundle["hash_route_matrices"] is None or route_bundle["hash_route_matrices"][0] is None:
        raise ValueError("trace exporter requires a concrete first hash matrix for TSER sensitivity")
    hash_features = route_bundle["hash_features"].to(device=device, dtype=torch.float32)
    score_bundle = compute_tser_scores(
        build_node_risk_scores=build_node_risk_scores,
        compute_neighbor_mean=_compute_neighbor_mean,
        verify_features=verify_features.to(device=device, dtype=torch.float32),
        hash_features=hash_features,
        hash_matrix=route_bundle["hash_route_matrices"][0].to(device=device),
        edge_index=data.edge_index.to(device),
        rarity_bits=int(args.score_rarity_bits),
        rarity_seed=int(args.score_rarity_seed),
        propagation_weight=int(args.score_propagation_weight),
        graph_context_weight=int(args.score_graph_context_weight),
        low_unique_weight=int(args.score_low_unique_weight),
    )
    sensitivity_q = score_bundle["sensitivity_q"].to(device="cpu", dtype=torch.int64)
    if int(sensitivity_q.max().item()) > 65535:
        raise ValueError("sensitivity_q overflowed trace field; expected weighted TSER score <= 65535")
    for key in ("propagation_q", "graph_context_q", "low_degree_unique_q", "rarity_q"):
        values = score_bundle[key].to(device="cpu", dtype=torch.int64)
        if int(values.min().item()) < 0 or int(values.max().item()) > 255:
            raise ValueError(f"{key} overflowed trace field; expected uint8")
    support_threshold = int(args.route_min_support_hits[0] if args.route_min_support_hits else 3)
    _write_trace(args.output, hashes_by_head, score_bundle, degree_bucket, int(args.radius), support_threshold)

    size_bytes = os.path.getsize(args.output)
    sensitivity_stats = (
        int(sensitivity_q.min().item()),
        float(sensitivity_q.float().mean().item()),
        int(sensitivity_q.max().item()),
    )
    print(
        f"[TraceExport] wrote {args.output} | nodes={int(data.num_nodes)} "
        f"| heads={NUM_HEADS} | bits={HASH_BITS} | radius={int(args.radius)} "
        f"| support_threshold={support_threshold} | bytes={size_bytes}"
    )
    print(
        "[TraceExport] "
        f"TSER weights={int(args.score_propagation_weight)}/"
        f"{int(args.score_graph_context_weight)}/"
        f"{int(args.score_low_unique_weight)} "
        f"| sensitivity_q min/mean/max={sensitivity_stats[0]}/{sensitivity_stats[1]:.2f}/{sensitivity_stats[2]}"
    )


if __name__ == "__main__":
    main()
