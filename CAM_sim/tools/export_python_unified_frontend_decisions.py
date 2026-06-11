"""Export native Python full-stack unified front-end decisions as CAM_sim-compatible CSV."""

from __future__ import annotations

import csv
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch

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
    from ...real_quant import default_pool_path, load_tensor_pool
    from ...runner import (
        apply_soft_cosine_gate,
        build_controller,
        build_residual_correction_mask,
        build_route_bundle,
        build_support_split_masks,
        make_run_args,
        replace_reuse_anchors_with_random,
        train_baseline_model,
    )
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
    real_quant_mod = importlib.import_module(f"{PACKAGE_ALIAS}.real_quant")
    runner_mod = importlib.import_module(f"{PACKAGE_ALIAS}.runner")
    build_parser = cli_mod.build_parser
    validate_args = cli_mod.validate_args
    load_run_state = data_mod.load_run_state
    default_pool_path = real_quant_mod.default_pool_path
    load_tensor_pool = real_quant_mod.load_tensor_pool
    apply_soft_cosine_gate = runner_mod.apply_soft_cosine_gate
    build_controller = runner_mod.build_controller
    build_residual_correction_mask = runner_mod.build_residual_correction_mask
    build_route_bundle = runner_mod.build_route_bundle
    build_support_split_masks = runner_mod.build_support_split_masks
    make_run_args = runner_mod.make_run_args
    replace_reuse_anchors_with_random = runner_mod.replace_reuse_anchors_with_random
    train_baseline_model = runner_mod.train_baseline_model


DECISION_FIELDS = [
    "node_id",
    "hit",
    "candidate_found",
    "source_id",
    "support",
    "route_hit_count",
    "base_route_hit_count",
    "winning_base_table_hit_count",
    "min_dist",
    "kind",
    "route",
    "sensitivity_q",
    "propagation_q",
    "graph_context_q",
    "low_unique_q",
    "rarity_q",
    "score_gate_checked",
    "score_gate_allow",
    "score_error_q",
    "score_risk",
    "score_reason",
    "route_idx",
    "route_name",
    "base_route_idx",
    "base_route_name",
    "route_weight",
    "route_score",
    "timestamp",
]


def _load_reference_pool(ds_key: str, args, device) -> tuple[torch.Tensor, str, bool]:
    primary_path = default_pool_path(ds_key, args.reference_model_name, args.reference_tag)
    if os.path.exists(primary_path):
        return load_tensor_pool(primary_path, device), primary_path, False

    fallback_path = default_pool_path(ds_key, args.reference_model_name, args.reference_fallback_tag)
    if os.path.exists(fallback_path):
        return load_tensor_pool(fallback_path, device), fallback_path, True

    raise FileNotFoundError(
        "Neither primary nor fallback reference pool exists:\n"
        f"  - {primary_path}\n"
        f"  - {fallback_path}"
    )


def _prepare_run_args(args, dataset: str, seed: int):
    run_args = make_run_args(args, seed)
    run_args.radius = 2
    run_args.hash_heads_per_route = 8
    run_args.main_hash_head_bits = [16] * 8
    run_args.learned_hash_epochs = 10
    run_args.learned_hash_dim = 128
    run_args.hamming_only_acceptor = True
    run_args.disable_structure_check = True
    run_args.disable_score_gate = False
    run_args.route_min_support_hits = [int(args.unified_soft_support)]
    run_args.score_propagation_weight = int(args.unified_score_weights[0])
    run_args.score_graph_context_weight = int(args.unified_score_weights[1])
    run_args.score_low_unique_weight = int(args.unified_score_weights[2])
    run_args.score_reuse_threshold = float(args.unified_threshold)
    run_args.residual_hard_min_support_hits = int(args.unified_hard_support)
    run_args.residual_soft_min_support_hits = int(args.unified_soft_support)
    run_args.residual_direct_threshold = float(args.unified_direct_threshold)
    run_args.residual_min_route_hits = int(args.unified_min_route_hits)
    run_args.residual_min_base_hits = int(args.unified_min_base_hits)
    run_args.residual_soft_min_cosine = float(args.unified_soft_min_cosine)
    run_args.real_quant_model_name = str(args.reference_model_name)
    run_args.precision_depth_reference_tag = str(args.reference_tag)
    run_args.residual_anchor_mode = str(getattr(args, "residual_anchor_mode", "cam"))
    if dataset in {"pubmed", "arxiv"}:
        run_args.allow_rare_fuzzy = True
    return run_args


def _support_split_enabled(run_args) -> bool:
    return (
        int(getattr(run_args, "residual_hard_min_support_hits", -1)) > 0
        and int(getattr(run_args, "residual_soft_min_support_hits", -1)) > 0
    )


def _build_effective_route_masks(trace, controller, run_args, device):
    correction_mask, correction_info = build_residual_correction_mask(
        trace,
        controller.risk_gate,
        run_args.residual_direct_threshold,
        device,
        min_route_hits=run_args.residual_min_route_hits,
        min_base_hits=run_args.residual_min_base_hits,
    )
    split_info = None
    if _support_split_enabled(run_args):
        split_info = build_support_split_masks(
            trace,
            run_args.residual_soft_min_support_hits,
            run_args.residual_hard_min_support_hits,
            device,
        )
        split_info = apply_soft_cosine_gate(
            trace,
            split_info,
            run_args.residual_soft_min_cosine,
            device,
        )
        correction_mask = correction_mask & split_info["soft_mask"]
        correction_info = dict(correction_info)
        correction_info["soft_cos_filtered"] = int(split_info.get("cos_filtered", 0))
        correction_info["residual_candidates"] = int(correction_mask.sum().item())

    hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
    source_ok = trace["source_ids"].to(device=device, dtype=torch.long) >= 0
    residual_mask = correction_mask.to(device=device, dtype=torch.bool) & hit_mask & source_ok
    direct_mask = hit_mask & source_ok & ~residual_mask
    return direct_mask, residual_mask, split_info, correction_info


def _write_decisions_csv(path: str, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in DECISION_FIELDS:
                value = row.get(key, "")
                if isinstance(value, bool):
                    encoded[key] = int(value)
                else:
                    encoded[key] = value
            writer.writerow(encoded)


def main() -> None:
    original_cwd = Path.cwd()
    parser = build_parser()
    parser.description = "Export native Python unified front-end decisions as CAM_sim-compatible CSV."
    parser.add_argument("--out", required=True, help="Output decisions.csv path")
    parser.add_argument("--summary_out", default="", help="Optional JSON summary path")
    parser.add_argument("--reference_model_name", default="llama2_7b")
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--reference_fallback_tag", default="W4A8")
    parser.add_argument("--unified_threshold", type=int, required=True, help="Dataset-level TSER threshold T")
    parser.add_argument("--unified_score_weights", nargs=3, type=int, default=[3, 1, 1])
    parser.add_argument("--unified_soft_support", type=int, default=3)
    parser.add_argument("--unified_hard_support", type=int, default=5)
    parser.add_argument(
        "--unified_direct_threshold",
        type=float,
        default=-1.0,
        help="If non-negative, only higher-risk soft hits stay in residual; lower-risk soft hits remain direct.",
    )
    parser.add_argument("--unified_min_route_hits", type=int, default=1)
    parser.add_argument("--unified_min_base_hits", type=int, default=1)
    parser.add_argument("--unified_soft_min_cosine", type=float, default=-1.0)
    args = parser.parse_args()
    validate_args(parser, args)
    args.out = _resolve_user_path(args.out, original_cwd)
    if args.summary_out:
        args.summary_out = _resolve_user_path(args.summary_out, original_cwd)
    _ensure_project_cwd()

    if len(args.datasets) != 1:
        parser.error("python unified frontend exporter expects exactly one dataset")
    ds_key = args.datasets[0]
    seed = int(args.seed)
    run_args = _prepare_run_args(args, ds_key, seed)

    def log_important(msg: str) -> None:
        print(msg)

    _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
    reference_raw, reference_path, used_fallback = _load_reference_pool(ds_key, args, device)
    data.x = reference_raw
    model, base_acc, ref_embs, ref_logits = train_baseline_model(data, run_args, device)
    route_bundle = build_route_bundle(
        verify_features=verify_features,
        data=data,
        oracle_embs=ref_embs,
        oracle_logits=ref_logits,
        args=run_args,
        log_important=log_important,
        device=device,
    )
    controller = build_controller(
        data,
        verify_features,
        route_bundle,
        {"name": "PythonExport", "overrides": {}},
        run_args,
        device,
    )
    direct_raw, _trace_hits = controller.query_full_batch(
        route_bundle["hash_route_features"],
        verify_features,
        reference_raw,
    )
    trace = controller.last_query_trace
    if run_args.residual_anchor_mode == "random":
        trace, direct_raw, anchor_info = replace_reuse_anchors_with_random(
            trace,
            direct_raw,
            reference_raw,
            verify_features,
            seed,
        )
    else:
        anchor_info = {"randomized": 0}

    base_rows = list(controller.last_query_decisions or [])
    if len(base_rows) != int(data.num_nodes):
        raise ValueError(
            f"expected {int(data.num_nodes)} decision rows from controller, got {len(base_rows)}"
        )

    direct_mask, residual_mask, split_info, correction_info = _build_effective_route_masks(
        trace,
        controller,
        run_args,
        device,
    )

    exported_rows = []
    score_reject_count = 0
    for node_id, row in enumerate(base_rows):
        out_row = dict(row)
        if bool(residual_mask[node_id].item()):
            out_row["route"] = "residual"
            out_row["hit"] = True
            out_row["candidate_found"] = True
        elif bool(direct_mask[node_id].item()):
            out_row["route"] = "direct"
            out_row["hit"] = True
            out_row["candidate_found"] = True
        else:
            out_row["route"] = "compute"
            out_row["hit"] = False
        if (
            str(out_row.get("route", "compute")) == "compute"
            and int(out_row.get("candidate_found", 0)) != 0
            and str(out_row.get("score_reason", "none")) in {"risk", "hub_protect", "rare_leaf"}
        ):
            score_reject_count += 1
        exported_rows.append(out_row)

    _write_decisions_csv(args.out, exported_rows)

    total = int(data.num_nodes)
    direct_count = int(direct_mask.sum().item())
    residual_count = int(residual_mask.sum().item())
    summary = {
        "dataset": ds_key,
        "seed": seed,
        "decisions_path": args.out,
        "summary_path": args.summary_out,
        "reference_model_name": args.reference_model_name,
        "reference_tag_requested": args.reference_tag,
        "reference_path_used": reference_path,
        "reference_fallback_used": bool(used_fallback),
        "reference_fallback_tag": args.reference_fallback_tag,
        "baseline_acc": float(base_acc),
        "ts_threshold": int(args.unified_threshold),
        "ts_weights": [int(v) for v in args.unified_score_weights],
        "hard_support_threshold": int(args.unified_hard_support),
        "soft_support_threshold": int(args.unified_soft_support),
        "direct_threshold": float(args.unified_direct_threshold),
        "min_route_hits": int(args.unified_min_route_hits),
        "min_base_hits": int(args.unified_min_base_hits),
        "soft_min_cosine": float(args.unified_soft_min_cosine),
        "residual_anchor_mode": str(run_args.residual_anchor_mode),
        "randomized_anchors": int(anchor_info.get("randomized", 0)),
        "reuse_rate": float((direct_count + residual_count) / max(1, total)),
        "direct_rate": float(direct_count / max(1, total)),
        "residual_rate": float(residual_count / max(1, total)),
        "compute_rate": float((total - direct_count - residual_count) / max(1, total)),
        "score_reject_rate": float(score_reject_count / max(1, total)),
        "counts": {
            "total": total,
            "direct": direct_count,
            "residual": residual_count,
            "compute": int(total - direct_count - residual_count),
            "score_reject": int(score_reject_count),
        },
        "trace_stats": dict(controller.stats),
        "support_split": None if split_info is None else {
            "soft_min_hits": int(split_info["soft_min_hits"]),
            "hard_min_hits": int(split_info["hard_min_hits"]),
            "hard_count": int(split_info["hard_count"]),
            "soft_count": int(split_info["soft_count"]),
            "residual_count": int(split_info["residual_count"]),
            "support_hist": {str(k): int(v) for k, v in split_info["support_hist"].items()},
            "cos_filtered": int(split_info.get("cos_filtered", 0)),
            "cos_threshold": float(split_info.get("cos_threshold", -1.0)),
        },
        "correction_info": correction_info,
    }

    summary_path = args.summary_out or (args.out + ".summary.json")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"[PythonUnifiedFrontendExport] dataset={ds_key} "
        f"reuse={summary['reuse_rate']:.1%} "
        f"direct={summary['direct_rate']:.1%} "
        f"residual={summary['residual_rate']:.1%} "
        f"compute={summary['compute_rate']:.1%} "
        f"score_reject={summary['score_reject_rate']:.1%}"
    )
    if used_fallback:
        print(
            f"[PythonUnifiedFrontendExport] fallback reference used: requested={args.reference_tag} "
            f"missing, actual={args.reference_fallback_tag}"
        )
    print(f"[PythonUnifiedFrontendExport] decisions={args.out}")
    print(f"[PythonUnifiedFrontendExport] summary={summary_path}")


if __name__ == "__main__":
    main()
