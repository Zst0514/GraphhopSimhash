"""Evaluate CAM_sim unified-front-end decisions with the existing residual/full-pool stack."""

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
    from ...real_quant import default_pool_path, load_tensor_pool
    from ...residual_reuse import (
        apply_residual_adapter,
        compute_bucket_values_from_trace,
        format_bucket_label,
        train_residual_adapter,
    )
    from ...runner import (
        build_controller,
        build_route_bundle,
        evaluate_raw_node_features,
        make_run_args,
        resolve_residual_alpha_grid,
        resolve_residual_fit_config,
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
    residual_mod = importlib.import_module(f"{PACKAGE_ALIAS}.residual_reuse")
    runner_mod = importlib.import_module(f"{PACKAGE_ALIAS}.runner")
    build_parser = cli_mod.build_parser
    validate_args = cli_mod.validate_args
    load_run_state = data_mod.load_run_state
    default_pool_path = real_quant_mod.default_pool_path
    load_tensor_pool = real_quant_mod.load_tensor_pool
    apply_residual_adapter = residual_mod.apply_residual_adapter
    compute_bucket_values_from_trace = residual_mod.compute_bucket_values_from_trace
    train_residual_adapter = residual_mod.train_residual_adapter
    build_controller = runner_mod.build_controller
    build_route_bundle = runner_mod.build_route_bundle
    evaluate_raw_node_features = runner_mod.evaluate_raw_node_features
    format_bucket_label = residual_mod.format_bucket_label
    make_run_args = runner_mod.make_run_args
    resolve_residual_alpha_grid = runner_mod.resolve_residual_alpha_grid
    resolve_residual_fit_config = runner_mod.resolve_residual_fit_config
    train_baseline_model = runner_mod.train_baseline_model


def _load_decisions(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"decision file is empty: {path}")
    return rows


def _int_field(row: dict, key: str, default: int) -> int:
    value = row.get(key, None)
    if value is None or value == "":
        return int(default)
    return int(value)


def _move_adapter_to_device(adapter, device):
    if adapter is None:
        return None
    if hasattr(adapter, "to"):
        adapter = adapter.to(device)
    if hasattr(adapter, "global_adapter") and hasattr(adapter.global_adapter, "to"):
        adapter.global_adapter = adapter.global_adapter.to(device)
    if hasattr(adapter, "adapters_by_support"):
        adapter.adapters_by_support = {
            int(key): value.to(device) if hasattr(value, "to") else value
            for key, value in dict(adapter.adapters_by_support).items()
        }
    return adapter


def _load_residual_adapter_checkpoint(path: str, device):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"residual adapter checkpoint not found: {path}")
    try:
        adapter = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        adapter = torch.load(path, map_location=device)
    return _move_adapter_to_device(adapter, device)


def _save_residual_adapter_checkpoint(path: str, adapter) -> None:
    if not path or adapter is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter, output_path)


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
    run_args.residual_min_dist = 1.0
    run_args.residual_soft_min_cosine = -1.0
    run_args.residual_fit_profile = "llama"
    run_args.residual_rank = 64
    run_args.residual_epochs = 200
    run_args.residual_max_train_pairs = 4096
    run_args.residual_alpha_grid = [0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5]
    run_args.residual_support_aware_alpha = True
    run_args.residual_adapter_type = "mlp"
    run_args.residual_dropout = 0.05
    run_args.residual_loss_cosine_weight = 1.0
    run_args.residual_loss_mse_weight = 0.5
    run_args.residual_loss_delta_weight = 0.75
    run_args.residual_bucket_mode = "support_dist"
    run_args.residual_offline_extra_anchors_per_node = 8
    run_args.residual_offline_extra_query_nodes = 4096
    run_args.residual_train_split = "train_val"
    run_args.residual_gate_loss_weight = 0.5
    run_args.residual_gate_error_scale = 0.25
    run_args.residual_gate_error_max = 0.45
    run_args.residual_gate_accept_threshold = float(args.unified_gate_threshold)
    run_args.residual_alpha = -1.0
    run_args.real_quant_model_name = str(args.reference_model_name)
    run_args.precision_depth_reference_tag = str(args.reference_tag)

    if dataset == "cora":
        run_args.residual_accept_mode = "separate"
        run_args.residual_positive_error_max = -1.0
        run_args.residual_offline_negative_anchors_per_node = 0
        run_args.residual_negative_gate_weight = 0.0
        run_args.residual_accept_loss_weight = 1.0
        run_args.residual_gate_sparsity_weight = 0.0
    else:
        run_args.residual_accept_mode = "shared"
        run_args.residual_positive_error_max = 0.40
        run_args.residual_offline_negative_anchors_per_node = 4
        run_args.residual_negative_gate_weight = 1.0
        run_args.residual_accept_loss_weight = 0.0
        run_args.residual_gate_sparsity_weight = 0.02
    return run_args


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


def _build_trace_from_decisions(decisions: list[dict], verify_features: torch.Tensor, num_nodes: int, device):
    hit_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    source_ids = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    best_dists = torch.zeros(num_nodes, dtype=torch.long, device=device)
    route_hit_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    base_route_hit_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    winning_hits = torch.zeros(num_nodes, dtype=torch.long, device=device)
    best_cosines = torch.full((num_nodes,), -1.0, dtype=torch.float32, device=device)
    direct_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    residual_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    score_reject_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    hit_kinds = ["none"] * num_nodes

    for row in decisions:
        node_id = int(row["node_id"])
        route = str(row.get("route", "compute"))
        candidate_found = int(row.get("candidate_found", row.get("hit", "0"))) != 0
        support = _int_field(row, "support", 0)
        route_hit_count = _int_field(row, "route_hit_count", support)
        base_route_hit_count = _int_field(
            row,
            "base_route_hit_count",
            1 if candidate_found and support > 0 else 0,
        )
        winning_base_table_hit_count = _int_field(row, "winning_base_table_hit_count", support)
        min_dist = _int_field(row, "min_dist", -1)
        source_id = _int_field(row, "source_id", -1)
        if source_id >= (1 << 31):
            source_id = -1

        if node_id < 0 or node_id >= num_nodes:
            raise ValueError(f"node_id {node_id} out of range 0..{num_nodes - 1}")

        if candidate_found and source_id >= 0:
            source_ids[node_id] = source_id
            best_dists[node_id] = max(-1, min_dist)
            route_hit_counts[node_id] = max(0, route_hit_count)
            base_route_hit_counts[node_id] = max(0, base_route_hit_count)
            winning_hits[node_id] = max(0, winning_base_table_hit_count)
            best_cosines[node_id] = F.cosine_similarity(
                verify_features[node_id : node_id + 1],
                verify_features[source_id : source_id + 1],
                dim=1,
            )[0]

        if route == "direct":
            hit_mask[node_id] = True
            direct_mask[node_id] = True
            hit_kinds[node_id] = str(row.get("kind", "exact"))
        elif route == "residual":
            hit_mask[node_id] = True
            residual_mask[node_id] = True
            hit_kinds[node_id] = str(row.get("kind", "fuzzy"))
        elif candidate_found and str(row.get("score_reason", "")) in {"risk", "hub_protect", "rare_leaf"}:
            score_reject_mask[node_id] = True

    trace = {
        "hit_mask": hit_mask,
        "source_ids": source_ids,
        "best_dists": best_dists,
        "route_hit_counts": route_hit_counts,
        "base_route_hit_counts": base_route_hit_counts,
        "winning_base_table_hit_counts": winning_hits,
        "best_cosines": best_cosines,
        "hit_kinds": hit_kinds,
    }
    return trace, direct_mask, residual_mask, score_reject_mask


def _apply_direct_anchors(reference_raw: torch.Tensor, trace) -> torch.Tensor:
    out = reference_raw.clone()
    hit_mask = trace["hit_mask"].to(device=reference_raw.device, dtype=torch.bool)
    source_ids = trace["source_ids"].to(device=reference_raw.device, dtype=torch.long)
    anchor_mask = hit_mask & (source_ids >= 0)
    if bool(anchor_mask.any()):
        out[anchor_mask] = reference_raw[source_ids[anchor_mask]]
    return out


def _jsonify(obj):
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(key): _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(value) for value in obj]
    return obj


def _select_residual_apply_config(
    *,
    adapter,
    model,
    data,
    verify_features,
    reference_raw,
    trace,
    risk_scores,
    correction_mask,
    run_args,
):
    if adapter is None:
        return 0.0, float(run_args.residual_gate_accept_threshold), {"mode": "adapter_none"}

    fit_cfg = resolve_residual_fit_config(run_args, int(reference_raw.size(1)))
    alpha_grid = resolve_residual_alpha_grid(run_args, fit_cfg)
    residual_base_raw = _apply_direct_anchors(reference_raw, trace)
    selected_alpha = float(alpha_grid[0])
    selected_val_acc = -1.0
    for alpha in alpha_grid:
        candidate_raw, _ = apply_residual_adapter(
            direct_embeddings=residual_base_raw,
            target_embeddings=reference_raw,
            verify_features=verify_features,
            edge_index=data.edge_index,
            trace=trace,
            adapter=adapter,
            risk_scores=risk_scores,
            alpha=float(alpha),
            gate_accept_threshold=float(run_args.residual_gate_accept_threshold),
            min_dist=run_args.residual_min_dist,
            correction_mask=correction_mask,
            bucket_mode=run_args.residual_bucket_mode,
        )
        val_acc = evaluate_raw_node_features(model, data, candidate_raw, mask=data.val_mask)
        if val_acc > selected_val_acc + 1e-12:
            selected_val_acc = val_acc
            selected_alpha = float(alpha)

    if not bool(getattr(run_args, "residual_support_aware_alpha", False)):
        return selected_alpha, float(run_args.residual_gate_accept_threshold), {
            "mode": "global",
            "selected_val_acc": float(selected_val_acc),
        }

    full_bucket_values = compute_bucket_values_from_trace(
        trace,
        torch.arange(trace["hit_mask"].numel(), device=verify_features.device),
        bucket_mode=run_args.residual_bucket_mode,
        device=verify_features.device,
    )
    active_bucket_values = compute_bucket_values_from_trace(
        trace,
        correction_mask.nonzero(as_tuple=False).view(-1),
        bucket_mode=run_args.residual_bucket_mode,
        device=verify_features.device,
    )
    alpha_by_support = {}
    support_notes = {}
    residual_base_raw = _apply_direct_anchors(reference_raw, trace)
    for bucket_value in sorted(int(v) for v in active_bucket_values.detach().cpu().unique().tolist()):
        support_mask = correction_mask & (full_bucket_values == int(bucket_value))
        val_support_mask = data.val_mask.to(device=verify_features.device, dtype=torch.bool) & support_mask
        best_alpha = float(selected_alpha)
        best_val_acc = -1.0
        if int(val_support_mask.sum().item()) > 0:
            for alpha in alpha_grid:
                candidate_raw, _ = apply_residual_adapter(
                    direct_embeddings=residual_base_raw,
                    target_embeddings=reference_raw,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    adapter=adapter,
                    risk_scores=risk_scores,
                    alpha=float(alpha),
                    gate_accept_threshold=float(run_args.residual_gate_accept_threshold),
                    min_dist=run_args.residual_min_dist,
                    correction_mask=support_mask,
                    bucket_mode=run_args.residual_bucket_mode,
                )
                val_acc = evaluate_raw_node_features(model, data, candidate_raw, mask=val_support_mask)
                if val_acc > best_val_acc + 1e-12:
                    best_val_acc = val_acc
                    best_alpha = float(alpha)
        alpha_by_support[int(bucket_value)] = float(best_alpha)
        support_notes[format_bucket_label(int(bucket_value), run_args.residual_bucket_mode)] = {
            "alpha": float(best_alpha),
            "val_acc": None if best_val_acc < 0.0 else float(best_val_acc),
        }
    return {"default": float(selected_alpha), "by_support": alpha_by_support}, float(
        run_args.residual_gate_accept_threshold
    ), {
        "mode": "support_aware",
        "selected_val_acc": float(selected_val_acc),
        "alpha_by_support": support_notes,
    }


def main() -> None:
    original_cwd = Path.cwd()
    parser = build_parser()
    parser.description = "Evaluate CAM_sim unified front-end decisions against the FullP8-style reference path."
    parser.add_argument("--decisions", required=True, help="Path to CAM_sim decisions.csv")
    parser.add_argument("--out", required=True, help="Output JSON summary path")
    parser.add_argument("--reference_model_name", default="llama2_7b")
    parser.add_argument("--reference_tag", default="W4BFPA8_B128")
    parser.add_argument("--reference_fallback_tag", default="W4A8")
    parser.add_argument("--unified_threshold", type=int, required=True, help="Dataset-level TSER threshold T")
    parser.add_argument("--unified_score_weights", nargs=3, type=int, default=[3, 1, 1])
    parser.add_argument("--unified_soft_support", type=int, default=3)
    parser.add_argument("--unified_hard_support", type=int, default=5)
    parser.add_argument("--unified_gate_threshold", type=float, default=0.575)
    parser.add_argument(
        "--load_residual_adapter",
        default="",
        help="Load a previously saved residual adapter checkpoint instead of retraining from replay decisions.",
    )
    parser.add_argument(
        "--save_residual_adapter",
        default="",
        help="Save the residual adapter trained from this replay trace for later fixed-adapter comparisons.",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    args.decisions = _resolve_user_path(args.decisions, original_cwd)
    args.out = _resolve_user_path(args.out, original_cwd)
    if args.load_residual_adapter:
        args.load_residual_adapter = _resolve_user_path(args.load_residual_adapter, original_cwd)
    if args.save_residual_adapter:
        args.save_residual_adapter = _resolve_user_path(args.save_residual_adapter, original_cwd)
    _ensure_project_cwd()

    if len(args.datasets) != 1:
        parser.error("unified frontend evaluator expects exactly one dataset")
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
        {"name": "CAMSimReplay", "overrides": {}},
        run_args,
        device,
    )
    decisions = _load_decisions(args.decisions)
    trace, direct_mask, residual_mask, score_reject_mask = _build_trace_from_decisions(
        decisions,
        verify_features.to(device=device, dtype=torch.float32),
        int(data.num_nodes),
        device,
    )
    correction_mask = residual_mask.clone()

    if args.load_residual_adapter:
        adapter = _load_residual_adapter_checkpoint(args.load_residual_adapter, device)
        train_info = {
            "source": "loaded_checkpoint",
            "loaded_from": str(args.load_residual_adapter),
        }
        adapter_source = "loaded_checkpoint"
    else:
        adapter, train_info = train_residual_adapter(
            target_embeddings=reference_raw,
            verify_features=verify_features,
            edge_index=data.edge_index,
            trace=trace,
            data=data,
            risk_scores=controller.node_risk_scores,
            rank=64,
            epochs=200,
            lr=run_args.residual_lr,
            weight_decay=run_args.residual_weight_decay,
            residual_l2=run_args.residual_l2,
            train_split=run_args.residual_train_split,
            max_pairs=4096,
            correction_mask=correction_mask,
            min_dist=run_args.residual_min_dist,
            controller=controller,
            hash_route_features=route_bundle["hash_route_features"],
            extra_anchors_per_node=run_args.residual_offline_extra_anchors_per_node,
            extra_query_nodes=run_args.residual_offline_extra_query_nodes,
            positive_error_max=run_args.residual_positive_error_max,
            adapter_type=run_args.residual_adapter_type,
            hidden_dim=getattr(run_args, "residual_hidden_dim", 0),
            hidden_layers=getattr(run_args, "residual_hidden_layers", 2),
            dropout=run_args.residual_dropout,
            accept_mode=run_args.residual_accept_mode,
            cosine_weight=run_args.residual_loss_cosine_weight,
            mse_weight=run_args.residual_loss_mse_weight,
            delta_weight=run_args.residual_loss_delta_weight,
            bucket_mode=run_args.residual_bucket_mode,
            gate_loss_weight=run_args.residual_gate_loss_weight,
            accept_loss_weight=run_args.residual_accept_loss_weight,
            gate_error_scale=run_args.residual_gate_error_scale,
            gate_error_max=run_args.residual_gate_error_max,
            gate_sparsity_weight=run_args.residual_gate_sparsity_weight,
            extra_negative_anchors_per_node=run_args.residual_offline_negative_anchors_per_node,
            negative_error_min=getattr(run_args, "residual_negative_error_min", 0.45),
            negative_gate_weight=run_args.residual_negative_gate_weight,
        )
        adapter_source = "trained_from_replay_trace"
        if args.save_residual_adapter:
            _save_residual_adapter_checkpoint(args.save_residual_adapter, adapter)
            train_info = dict(train_info)
            train_info["saved_to"] = str(args.save_residual_adapter)

    alpha_for_apply, gate_threshold_for_apply, alpha_meta = _select_residual_apply_config(
        adapter=adapter,
        model=model,
        data=data,
        verify_features=verify_features,
        reference_raw=reference_raw,
        trace=trace,
        risk_scores=controller.node_risk_scores,
        correction_mask=correction_mask,
        run_args=run_args,
    )

    residual_base_raw = _apply_direct_anchors(reference_raw, trace)
    final_raw, apply_info = apply_residual_adapter(
        direct_embeddings=residual_base_raw,
        target_embeddings=reference_raw,
        verify_features=verify_features,
        edge_index=data.edge_index,
        trace=trace,
        adapter=adapter,
        risk_scores=controller.node_risk_scores,
        alpha=alpha_for_apply,
        gate_accept_threshold=gate_threshold_for_apply,
        min_dist=run_args.residual_min_dist,
        correction_mask=correction_mask,
        bucket_mode=run_args.residual_bucket_mode,
    )
    final_acc = evaluate_raw_node_features(model, data, final_raw)
    fullp8_drop = float(base_acc - final_acc)

    effective_residual_mask = residual_mask.clone()
    rejected_nodes = apply_info.get("rejected_nodes", None)
    if rejected_nodes is not None and int(rejected_nodes.numel()) > 0:
        effective_residual_mask[rejected_nodes] = False
    accepted_hit_mask = direct_mask | effective_residual_mask
    compute_mask = ~accepted_hit_mask

    summary = {
        "dataset": ds_key,
        "seed": seed,
        "decisions_path": args.decisions,
        "reference_model_name": args.reference_model_name,
        "reference_tag_requested": args.reference_tag,
        "reference_path_used": reference_path,
        "reference_fallback_used": bool(used_fallback),
        "reference_fallback_tag": args.reference_fallback_tag,
        "decision_schema_has_full_support_fields": all(
            key in decisions[0]
            for key in (
                "route_hit_count",
                "base_route_hit_count",
                "winning_base_table_hit_count",
            )
        ),
        "residual_adapter_source": adapter_source,
        "residual_adapter_loaded_from": str(args.load_residual_adapter) if args.load_residual_adapter else "",
        "residual_adapter_saved_to": str(args.save_residual_adapter) if args.save_residual_adapter else "",
        "ts_threshold": int(args.unified_threshold),
        "ts_weights": [int(v) for v in args.unified_score_weights],
        "hard_support_threshold": int(args.unified_hard_support),
        "soft_support_threshold": int(args.unified_soft_support),
        "residual_gate_threshold": float(args.unified_gate_threshold),
        "baseline_acc": float(base_acc),
        "fullp8_acc": float(final_acc),
        "fullp8_drop": float(fullp8_drop),
        "reuse_rate": float(accepted_hit_mask.float().mean().item()),
        "direct_rate": float(direct_mask.float().mean().item()),
        "residual_rate": float(effective_residual_mask.float().mean().item()),
        "compute_rate": float(compute_mask.float().mean().item()),
        "score_reject_rate": float(score_reject_mask.float().mean().item()),
        "counts": {
            "total": int(data.num_nodes),
            "direct": int(direct_mask.sum().item()),
            "residual_pre_gate": int(residual_mask.sum().item()),
            "residual": int(effective_residual_mask.sum().item()),
            "residual_rejected": int(0 if rejected_nodes is None else rejected_nodes.numel()),
            "compute": int(compute_mask.sum().item()),
            "score_reject": int(score_reject_mask.sum().item()),
        },
        "alpha_apply": alpha_for_apply,
        "alpha_meta": _jsonify(alpha_meta),
        "train_info": _jsonify(train_info),
        "apply_info": _jsonify(apply_info),
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"[UnifiedFrontendEval] dataset={ds_key} reuse={summary['reuse_rate']:.1%} "
        f"direct={summary['direct_rate']:.1%} residual={summary['residual_rate']:.1%} "
        f"compute={summary['compute_rate']:.1%} fullp8_drop={summary['fullp8_drop']:.2%}"
    )
    if used_fallback:
        print(
            f"[UnifiedFrontendEval] fallback reference used: requested={args.reference_tag} "
            f"missing, actual={args.reference_fallback_tag}"
        )


if __name__ == "__main__":
    main()
