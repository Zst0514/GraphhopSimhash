import os
import json
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .config import DATASET_CONFIGS
from .controller import PaperHashReuseController
from .data import load_raw_texts, load_run_state, maybe_limit_test_mask
from .features import build_hash_feature_routes, build_topology_hash_features, format_hash_route_specs
from .internal_split_calibration import (
    build_internal_split_calibration,
    save_internal_split_calibration,
)
from .models import GNN_LLM_Model
from .projections import fit_multihead_hash_projection
from .real_quant import (
    assemble_real_quant_embeddings,
    build_real_quant_scores,
    compute_real_quant_errors,
    default_pool_path,
    load_tensor_pool,
    load_real_quant_pools,
    regenerate_real_quant_pools,
    select_real_quant_policy_actions,
    summarize_real_quant_policy,
)
from .residual_reuse import (
    apply_residual_adapter,
    compute_bucket_values_from_trace,
    embedding_error,
    format_bucket_label,
    train_residual_adapter,
)
from .routing import (
    apply_table_weight_decay,
    build_log_path,
    expand_route_values_by_base_specs,
    list_retrieval_route_tags,
    resolve_route_accept_tau_offsets,
    resolve_route_min_accept_votes,
    resolve_route_min_support_hits,
    resolve_route_score_weights,
)

def build_adaptive_configs(args):
    radius_name = f"R{int(args.radius)}"
    if args.experiment_suite == "score_ablation":
        return [
            {
                "name": f"{radius_name}_NoScore",
                "overrides": {
                    "disable_score_gate": True,
                    "enable_quant_policy": False,
                },
            },
            {
                "name": f"{radius_name}_DegreeOnly",
                "overrides": {
                    "disable_score_gate": False,
                    "enable_quant_policy": False,
                    "score_graph_context_weight": 0,
                    "score_low_unique_weight": 0,
                    "allow_rare_fuzzy": True,
                },
            },
            {
                "name": f"{radius_name}_TSER",
                "overrides": {
                    "disable_score_gate": False,
                    "enable_quant_policy": False,
                },
            },
        ]
    if args.experiment_suite == "quant_ablation":
        return [
            {
                "name": f"{radius_name}_NoScoreReuse",
                "overrides": {
                    "disable_score_gate": True,
                    "enable_quant_policy": False,
                },
            },
            {
                "name": f"{radius_name}_TSERReuse",
                "overrides": {
                    "disable_score_gate": False,
                    "enable_quant_policy": False,
                },
            },
            {
                "name": f"{radius_name}_DegreeQuant",
                "overrides": {
                    "disable_score_gate": False,
                    "enable_quant_policy": True,
                    "score_graph_context_weight": 0,
                    "score_low_unique_weight": 0,
                    "allow_rare_fuzzy": True,
                },
            },
            {
                "name": f"{radius_name}_TSERQuant",
                "overrides": {
                    "disable_score_gate": False,
                    "enable_quant_policy": True,
                },
            },
        ]
    return [{"name": radius_name, "overrides": {}}]


def apply_config_overrides(args, cfg):
    cfg_args = deepcopy(args)
    for key, value in cfg.get("overrides", {}).items():
        setattr(cfg_args, key, value)
    return cfg_args


def make_run_args(args, run_seed):
    run_args = deepcopy(args)
    run_args.run_seed = int(run_seed)
    if run_args.controller_seed is None:
        run_args.controller_seed = int(run_seed)
    if run_args.hash_head_seed is None:
        run_args.hash_head_seed = int(run_seed)
    if run_args.topology_sketch_seed is None:
        run_args.topology_sketch_seed = int(run_seed)
    return run_args


def train_baseline_model(data, args, device):
    num_classes = int(data.y.max().item()) + 1
    model = GNN_LLM_Model(data.x.shape[1], 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    if data.y.dim() > 1:
        data.y = data.y.squeeze()

    maybe_limit_test_mask(data, args.max_test)
    raw_features = data.x

    best_val_acc = 0.0
    best_model_w = None

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        emb = model.encoder(raw_features)
        out = model.forward_gnn_only(emb, data.edge_index, data.edge_type, data.edge_attr)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                if getattr(args, "standard_eval_baseline", False):
                    model.eval()
                    val_emb = model.encoder(raw_features)
                    val_logits = model.forward_gnn_only(
                        val_emb,
                        data.edge_index,
                        data.edge_type,
                        data.edge_attr,
                    )
                    pred = val_logits.argmax(dim=1)
                else:
                    pred = model.forward_gnn_only(
                        emb,
                        data.edge_index,
                        data.edge_type,
                        data.edge_attr,
                    ).argmax(dim=1)
                val_acc = (
                    (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
                    / data.val_mask.sum().item()
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_w = deepcopy(model.state_dict())

    if best_model_w is not None:
        model.load_state_dict(best_model_w)

    model.eval()
    with torch.no_grad():
        baseline_embs = model.encoder(raw_features)
        baseline_logits = model.forward_gnn_only(
            baseline_embs,
            data.edge_index,
            data.edge_type,
            data.edge_attr,
        )
        pred = baseline_logits.argmax(dim=1)
        base_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    return model, float(base_acc), baseline_embs.detach(), baseline_logits.detach()


def build_route_bundle(verify_features, data, oracle_embs, oracle_logits, args, log_important, device):
    base_route_specs = build_hash_feature_routes(
        verify_features,
        data.edge_index,
        args.hash_view,
        args.hash_mix_weights,
        args.union_hash_views,
    )
    if args.enable_topology_retrieval_route:
        topology_features = build_topology_hash_features(verify_features, data.edge_index)
        base_route_specs.append(
            {
                "name": "topology",
                "view": "topology",
                "features": topology_features,
                "weights": None,
                "base_route_idx": len(base_route_specs),
                "base_name": "topology",
                "table_idx": 0,
                "table_count": 1,
                "route_role": "topology",
            }
        )
    base_route_tags = list_retrieval_route_tags(
        args.hash_view,
        args.union_hash_views,
        include_topology=args.enable_topology_retrieval_route,
    )
    base_route_score_weights = resolve_route_score_weights(
        base_route_tags,
        configured_weights=args.route_score_weights,
        default_union_weight=args.union_route_weight,
    )
    base_route_accept_tau_offsets = resolve_route_accept_tau_offsets(
        base_route_tags,
        configured_offsets=args.route_accept_tau_offsets,
        default_union_bonus=args.union_accept_tau_bonus,
    )
    base_route_min_accept_votes = resolve_route_min_accept_votes(
        base_route_tags,
        configured_votes=args.route_min_accept_votes,
        default_union_votes=args.union_min_accept_votes,
    )
    base_route_min_support_hits = resolve_route_min_support_hits(
        base_route_tags,
        configured_hits=args.route_min_support_hits,
        default_main_hits=2,
        default_union_hits=args.union_min_support_hits,
    )

    route_specs = base_route_specs
    projection_stats = None
    supervision_mask = None
    if args.learned_hash_projection:
        if args.learned_hash_supervision == "train":
            supervision_mask = data.train_mask
        else:
            supervision_mask = data.train_mask | data.val_mask

    route_specs, projection_stats = fit_multihead_hash_projection(
        route_specs,
        oracle_embs,
        oracle_logits,
        supervision_mask=supervision_mask,
        args=args,
        device=device,
    )

    hash_route_features = [spec["features"] for spec in route_specs]
    hash_route_matrices = [spec.get("hash_matrix") for spec in route_specs]
    hash_route_bits = [int(spec.get("hash_bits", matrix.size(1) if matrix is not None else args.sketch_bits)) for spec, matrix in zip(route_specs, hash_route_matrices)]
    if all(matrix is None for matrix in hash_route_matrices):
        hash_route_matrices = None
    elif any(matrix is None for matrix in hash_route_matrices):
        raise ValueError("Either every route must provide a hash_matrix or none of them should")
    hash_route_names = [spec["name"] for spec in route_specs]
    route_base_indices = [int(spec.get("base_route_idx", idx)) for idx, spec in enumerate(route_specs)]
    route_base_names = []
    seen_base_names = set()
    for spec in route_specs:
        base_name = spec.get("base_name", spec["name"])
        if base_name in seen_base_names:
            continue
        route_base_names.append(base_name)
        seen_base_names.add(base_name)

    route_score_weights = expand_route_values_by_base_specs(route_specs, base_route_score_weights)
    route_score_weights = apply_table_weight_decay(
        route_specs,
        route_score_weights,
        table_weight_decay=args.table_route_weight_decay,
    )
    route_accept_tau_offsets = expand_route_values_by_base_specs(route_specs, base_route_accept_tau_offsets)
    route_min_accept_votes = expand_route_values_by_base_specs(route_specs, base_route_min_accept_votes)

    route_bundle = {
        "hash_features": route_specs[0]["features"],
        "hash_route_features": hash_route_features,
        "hash_route_matrices": hash_route_matrices,
        "hash_route_bits": hash_route_bits,
        "hash_route_names": hash_route_names,
        "route_base_indices": route_base_indices,
        "route_base_names": route_base_names,
        "route_score_weights": route_score_weights,
        "route_accept_tau_offsets": route_accept_tau_offsets,
        "route_min_accept_votes": route_min_accept_votes,
        "route_min_support_hits": base_route_min_support_hits,
    }

    log_important(f"[HashRoutes] {format_hash_route_specs(route_specs, route_score_weights)}")
    log_important(
        "[RouteAccept] "
        + ", ".join(
            f"{route_name}[tau+={tau_offset:.2f}, min_votes={min_votes}, min_hits={min_hits}]"
            for route_name, tau_offset, min_votes, min_hits in zip(
                hash_route_names,
                route_accept_tau_offsets,
                route_min_accept_votes,
                expand_route_values_by_base_specs(route_specs, base_route_min_support_hits),
            )
        )
    )
    log_important(
        f"[Retrieval] per_route={args.max_candidates_per_route} "
        f"| total={args.max_total_candidates} "
        f"| struct_checks={args.max_structure_checks if args.max_structure_checks is not None else 'auto'} "
        f"| table_weight_decay={args.table_route_weight_decay:.2f} "
        f"| coarse_union_bits_max={args.coarse_union_bits_max if args.coarse_union_bits_max is not None else 'none'}"
    )

    if projection_stats is not None:
        log_important(
            f"[MultiHeadHash] dim={projection_stats['output_dim']} "
            f"| total_heads={projection_stats['total_heads']} "
            f"| trained_encoders={projection_stats['trained_encoders']}/{projection_stats['total_encoders']}"
        )
        for route_name, route_schedule in projection_stats["route_head_schedules"].items():
            log_important(
                f"  [HeadSchedule] {route_name} | bits={'/'.join(str(bit) for bit in route_schedule)}"
            )
        for route_stat in projection_stats["routes"]:
            if route_stat["trained"]:
                log_important(
                    f"  [ProjectionHead] {route_stat['route_name']} "
                    f"| head={route_stat['head_idx']} "
                    f"| bits={route_stat['hash_bits']} "
                    f"| support_nodes={route_stat['support_nodes']} "
                    f"| pairs={route_stat['pair_count']} "
                    f"| positive_rate={route_stat['positive_rate']:.1%} "
                    f"| train_loss={route_stat['train_loss']:.6f} "
                    f"| seed={route_stat['seed']} "
                    f"| hash_seed={route_stat['hash_seed']}"
                )
            else:
                log_important(
                    f"  [ProjectionHead] {route_stat['route_name']} "
                    f"| head={route_stat['head_idx']} "
                    f"| bits={route_stat['hash_bits']} "
                    f"| fallback=orthogonal_init "
                    f"| seed={route_stat['seed']} "
                    f"| hash_seed={route_stat['hash_seed']}"
                )
    return route_bundle


def build_controller(
    data,
    verify_features,
    route_bundle,
    cfg,
    args,
    device,
):
    return PaperHashReuseController(
        input_dim=route_bundle["hash_features"].shape[1],
        sketch_bits=args.sketch_bits,
        device=device,
        hamming_radius=args.radius,
        full_verify_features=verify_features,
        full_hash_features=route_bundle["hash_features"],
        full_hash_feature_routes=route_bundle["hash_route_features"],
        hash_route_matrices=route_bundle["hash_route_matrices"],
        hash_route_bits=route_bundle["hash_route_bits"],
        hash_route_names=route_bundle["hash_route_names"],
        route_base_indices=route_bundle["route_base_indices"],
        route_base_names=route_bundle["route_base_names"],
        route_score_weights=route_bundle["route_score_weights"],
        route_accept_tau_offsets=route_bundle["route_accept_tau_offsets"],
        route_min_accept_votes=route_bundle["route_min_accept_votes"],
        route_min_support_hits=route_bundle["route_min_support_hits"],
        min_base_route_hits=args.min_base_route_hits,
        max_candidates_per_route=args.max_candidates_per_route,
        max_total_candidates=args.max_total_candidates,
        max_structure_checks=args.max_structure_checks,
        coarse_union_bits_max=args.coarse_union_bits_max,
        edge_index=data.edge_index,
        max_cache_size=args.cache_size,
        second_stage_tau=args.cosine_tau,
        memo_k=args.memo_k,
        vote_top_m=args.vote_top_m,
        vote_relax_margin=args.vote_relax_margin,
        structure_neighbor_tau=args.structure_neighbor_tau,
        structure_degree_ratio_max=args.structure_degree_ratio_max,
        structure_homophily_gap_max=args.structure_homophily_gap_max,
        structure_check_mode=args.structure_check_mode,
        enable_homophily_bucket_guard=args.enable_homophily_bucket_guard,
        topology_sketch_bits=args.topology_sketch_bits,
        topology_sketch_radius=args.topology_sketch_radius,
        enable_topology_sketch_guard=args.enable_topology_sketch_guard,
        topology_degree_bucket_gap=args.topology_degree_bucket_gap,
        topology_homophily_bins=args.topology_homophily_bins,
        topology_homophily_bucket_gap=args.topology_homophily_bucket_gap,
        topology_sketch_seed=args.topology_sketch_seed,
        exact_guard_low_bits=args.exact_guard_low_bits,
        exact_guard_min_bucket_size=args.exact_guard_min_bucket_size,
        exact_guard_large_bucket_size=args.exact_guard_large_bucket_size,
        exact_guard_min_margin=args.exact_guard_min_margin,
        exact_guard_cosine_bonus=args.exact_guard_cosine_bonus,
        hamming_only_acceptor=args.hamming_only_acceptor,
        disable_structure_check=args.disable_structure_check,
        score_gate_enabled=not args.disable_score_gate,
        score_reuse_threshold=args.score_reuse_threshold,
        score_hub_threshold=args.score_hub_threshold,
        score_rare_threshold=args.score_rare_threshold,
        score_protect_hub_exact=args.score_protect_hub_exact,
        score_protect_hub_fuzzy=not args.allow_hub_fuzzy,
        score_forbid_rare_fuzzy=not args.allow_rare_fuzzy,
        score_support_discount=not args.disable_score_support_discount,
        score_rare_gate_mode=args.score_rare_gate_mode,
        score_rare_min_dist=args.score_rare_min_dist,
        score_rare_min_route_hits=args.score_rare_min_route_hits,
        score_rare_min_base_hits=args.score_rare_min_base_hits,
        score_pair_confidence_discount=args.score_pair_confidence_discount,
        score_pair_confidence_max_dist=args.score_pair_confidence_max_dist,
        score_pair_confidence_min_route_hits=args.score_pair_confidence_min_route_hits,
        score_pair_confidence_min_base_hits=args.score_pair_confidence_min_base_hits,
        score_pair_confidence_min_cos_margin=args.score_pair_confidence_min_cos_margin,
        score_rarity_bits=args.score_rarity_bits,
        score_rarity_seed=args.score_rarity_seed,
        score_propagation_weight=args.score_propagation_weight,
        score_graph_context_weight=args.score_graph_context_weight,
        score_low_unique_weight=args.score_low_unique_weight,
        score_use_continuous_risk=args.score_use_continuous_risk,
        quant_policy_enabled=args.enable_quant_policy,
        quant_int4_threshold=args.quant_int4_threshold,
        quant_int8_threshold=args.quant_int8_threshold,
        quant_int4_error=args.quant_int4_error,
        quant_int8_error=args.quant_int8_error,
        quant_int4_bits=args.quant_int4_bits,
        quant_int8_bits=args.quant_int8_bits,
        hash_init_seed=args.controller_seed,
    )


def evaluate_with_controller(model, data, controller, route_bundle, verify_features, oracle_embs, args):
    reconstructed_embs, _hits = controller.query_full_batch(
        route_bundle["hash_route_features"],
        verify_features,
        oracle_embs,
    )

    with torch.no_grad():
        out_adapt = model.forward_gnn_only(
            reconstructed_embs,
            data.edge_index,
            data.edge_type,
            data.edge_attr,
        )
        pred_adapt = out_adapt.argmax(dim=1)
        acc = (pred_adapt[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    stats = controller.stats
    reuse_rate = stats["reuse"] / stats["total_queries"] if stats["total_queries"] > 0 else 0.0
    total_queries = stats["total_queries"] if stats["total_queries"] > 0 else 1
    return {
        "acc": float(acc),
        "drop": None,
        "reuse_rate": float(reuse_rate),
        "reuse_num": int(stats["reuse"]),
        "reuse_den": int(stats["reuse_denominator"]),
        "int4_rate": float(stats.get("quant_int4", 0) / total_queries),
        "int8_rate": float(stats.get("quant_int8", 0) / total_queries),
        "full_rate": float(stats.get("full_precision", 0) / total_queries),
        "protected_rate": float(stats.get("protected", 0) / total_queries),
        "stats": stats,
    }


def run_single_config(model, data, verify_features, oracle_embs, route_bundle, cfg, args, device, log_important=None):
    cfg_args = apply_config_overrides(args, cfg)
    controller = build_controller(
        data,
        verify_features,
        route_bundle,
        cfg,
        cfg_args,
        device,
    )
    return evaluate_with_controller(
        model,
        data,
        controller,
        route_bundle,
        verify_features,
        oracle_embs,
        cfg_args,
    )


def evaluate_gnn_embeddings(model, data, node_embs, mask=None):
    if mask is None:
        mask = data.test_mask
    with torch.no_grad():
        out = forward_gnn_logits(model, data, node_embs)
        pred = out.argmax(dim=1)
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
    return float(acc)


def forward_gnn_logits(model, data, node_embs):
    return model.forward_gnn_only(
        node_embs,
        data.edge_index,
        data.edge_type,
        data.edge_attr,
    )


def node_logit_margin(logits):
    if logits.size(1) <= 1:
        return torch.zeros(logits.size(0), dtype=torch.float32, device=logits.device)
    top2 = torch.topk(logits, k=2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).to(dtype=torch.float32)


def evaluate_raw_node_features(model, data, raw_features, mask=None):
    with torch.no_grad():
        hidden = model.encoder(raw_features)
    return evaluate_gnn_embeddings(model, data, hidden, mask=mask)


def load_graph_eager_token_pools(ds_key, args, data, device, log_important):
    model_name = args.real_quant_model_name
    reference_tag = str(args.graph_eager_reference_tag)
    full_tag = str(args.graph_eager_full_tag)
    token_prefix = str(args.graph_eager_token_tag_prefix)
    lengths = sorted(int(length) for length in args.graph_eager_token_lengths)

    reference_path = default_pool_path(ds_key, model_name, reference_tag)
    full_path = default_pool_path(ds_key, model_name, full_tag)
    token_paths = {
        length: default_pool_path(ds_key, model_name, f"{token_prefix}{length}")
        for length in lengths
    }
    missing = [
        path
        for path in [reference_path, full_path, *token_paths.values()]
        if not os.path.exists(path)
    ]
    if missing:
        msg = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Graph-eager token pools are missing:\n"
            f"{msg}\n"
            "Generate shortened pools with generate_real_quant_pools using --max_length "
            "and --output_path, e.g. W4A8_S128/W4A8_S256."
        )

    reference = load_tensor_pool(reference_path, device)
    full = load_tensor_pool(full_path, device)
    token = {length: load_tensor_pool(path, device) for length, path in token_paths.items()}

    expected_nodes = int(data.num_nodes)
    expected_shape = tuple(reference.shape)
    for label, tensor in [("full", full), *[(f"S{length}", tensor) for length, tensor in token.items()]]:
        if int(tensor.size(0)) != expected_nodes:
            raise ValueError(f"{label} pool has wrong node count: {tuple(tensor.shape)} vs {expected_nodes}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{label} pool shape {tuple(tensor.shape)} must match reference {expected_shape}")

    log_important(
        "[GraphEagerPools] "
        f"reference={reference_path} | full={full_path} | "
        f"tokens={', '.join(f'S{k}:{v}' for k, v in token_paths.items())}"
    )
    log_important(f"[GraphEagerPools] shape={expected_shape} | lengths={lengths}")
    return {
        "reference": reference,
        "full": full,
        "token": token,
        "lengths": lengths,
        "reference_path": reference_path,
        "full_path": full_path,
        "token_paths": token_paths,
    }


def graph_eager_token_cost(action, args):
    if int(action) == -1:
        ratio = 1.0
    else:
        ratio = float(action) / float(args.graph_eager_full_length)
    denom = float(args.graph_eager_attn_weight) + float(args.graph_eager_ffn_weight)
    attn_w = float(args.graph_eager_attn_weight) / denom
    ffn_w = float(args.graph_eager_ffn_weight) / denom
    return float(args.graph_eager_cost_scale) * (attn_w * ratio * ratio + ffn_w * ratio)


def build_graph_eager_token_policy_configs(args, lengths):
    lengths = sorted(int(length) for length in lengths)
    configs = [("FullW4A8", {"kind": "all_full"})]
    for length in lengths:
        configs.append((f"AllS{length}", {"kind": "all_token", "length": length}))
    configs.extend(
        [
            ("RandomTokenBudget", {"kind": "budget", "priority": "random"}),
            ("DegreeTokenBudget", {"kind": "budget", "priority": "degree"}),
            ("TSERTokenBudget", {"kind": "budget", "priority": "tser"}),
            ("ContextTokenBudget", {"kind": "budget", "priority": "context"}),
            ("PredictorTokenBudget", {"kind": "budget", "priority": "predictor"}),
            ("OracleDamageBudget", {"kind": "budget", "priority": "oracle_damage"}),
        ]
    )
    return configs


def select_graph_eager_token_actions(
    policy,
    scores,
    args,
    num_nodes,
    lengths,
    seed,
    device,
    oracle_priority=None,
    predictor_priority=None,
):
    lengths = sorted(int(length) for length in lengths)
    full_action = -1
    if policy["kind"] == "all_full":
        return torch.full((num_nodes,), full_action, dtype=torch.int64, device=device)
    if policy["kind"] == "all_token":
        return torch.full((num_nodes,), int(policy["length"]), dtype=torch.int64, device=device)
    if policy["kind"] != "budget":
        raise ValueError(f"Unknown graph-eager token policy: {policy}")

    priority_name = policy["priority"]
    if priority_name == "degree":
        priority = scores["propagation_q"].to(dtype=torch.float32)
    elif priority_name == "tser":
        priority = scores["sensitivity_q"].to(dtype=torch.float32)
    elif priority_name == "context":
        priority = scores["graph_context_q"].to(dtype=torch.float32)
    elif priority_name == "oracle_damage":
        if oracle_priority is None:
            raise ValueError("oracle_damage policy requires oracle_priority")
        priority = oracle_priority.to(dtype=torch.float32)
    elif priority_name == "predictor":
        if predictor_priority is None:
            raise ValueError("predictor policy requires predictor_priority")
        priority = predictor_priority.to(dtype=torch.float32)
    elif priority_name == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 49999)
        priority = torch.rand(num_nodes, generator=generator, device="cpu").to(device=device)
    else:
        raise ValueError(f"Unknown graph-eager priority: {priority_name}")

    actions = torch.full((num_nodes,), lengths[0], dtype=torch.int64, device=device)
    order = torch.argsort(priority, descending=True)
    full_count = int(round(float(args.graph_eager_full_ratio) * num_nodes))
    mid_count = int(round(float(args.graph_eager_mid_ratio) * num_nodes))
    full_count = max(0, min(full_count, num_nodes))
    mid_count = max(0, min(mid_count, num_nodes - full_count))
    actions[order[:full_count]] = full_action
    if len(lengths) >= 2:
        actions[order[full_count : full_count + mid_count]] = lengths[-1]
    return actions


def assemble_graph_eager_token_embeddings(actions, full_embs, token_embs):
    mixed = full_embs.clone()
    for length, embs in token_embs.items():
        mask = actions == int(length)
        if bool(mask.any()):
            mixed[mask] = embs[mask]
    return mixed


def summarize_graph_eager_token_policy(actions, errors_by_action, lengths, args):
    total = max(1, int(actions.numel()))
    rates = {"full": float((actions == -1).float().mean().item())}
    cost = rates["full"] * graph_eager_token_cost(-1, args)
    selected_err = torch.zeros(total, dtype=torch.float32, device=actions.device)
    selected_err[actions == -1] = errors_by_action[-1][actions == -1]
    for length in lengths:
        mask = actions == int(length)
        rate = float(mask.float().mean().item())
        rates[int(length)] = rate
        cost += rate * graph_eager_token_cost(int(length), args)
        selected_err[mask] = errors_by_action[int(length)][mask]
    return {
        "rates": rates,
        "cost": float(cost),
        "avg_err": float(selected_err.mean().item()),
    }


def pearson_corr(x, y):
    x = x.to(dtype=torch.float32).view(-1)
    y = y.to(dtype=torch.float32).view(-1)
    if x.numel() <= 1:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp(min=1e-12)
    return float(((x * y).sum() / denom).item())


def spearman_corr(x, y):
    return pearson_corr(rank01(x, descending=False), rank01(y, descending=False))


def binary_auc(score, label):
    score = score.to(dtype=torch.float32).view(-1)
    label = label.to(dtype=torch.bool).view(-1)
    pos = int(label.sum().item())
    neg = int(label.numel() - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(score, descending=False)
    ranks = torch.empty_like(score, dtype=torch.float32)
    ranks[order] = torch.arange(1, int(score.numel()) + 1, device=score.device, dtype=torch.float32)
    pos_rank_sum = ranks[label].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc.item())


def build_graph_eager_text_features(ds_key, num_nodes, device):
    try:
        texts = load_raw_texts(ds_key)
    except Exception:
        texts = [""] * int(num_nodes)
    if len(texts) != int(num_nodes):
        texts = list(texts[: int(num_nodes)]) + [""] * max(0, int(num_nodes) - len(texts))
    word_counts = torch.tensor(
        [len(str(text).split()) for text in texts],
        dtype=torch.float32,
        device=device,
    )
    char_counts = torch.tensor(
        [len(str(text)) for text in texts],
        dtype=torch.float32,
        device=device,
    )
    return word_counts, char_counts


def build_graph_eager_predictor_features(ds_key, scores, num_nodes, device):
    word_counts, char_counts = build_graph_eager_text_features(ds_key, num_nodes, device)
    prop = scores["propagation_q"].to(dtype=torch.float32)
    context = scores["graph_context_q"].to(dtype=torch.float32)
    low_unique = scores["low_degree_unique_q"].to(dtype=torch.float32)
    rarity = scores["rarity_q"].to(dtype=torch.float32)
    similar = torch.log1p(scores["similar_count"].to(dtype=torch.float32))
    tser = scores["sensitivity_q"].to(dtype=torch.float32)
    word_rank = rank01(word_counts, descending=False)
    char_rank = rank01(char_counts, descending=False)
    features = torch.stack(
        [
            prop / 15.0,
            context / 15.0,
            low_unique / 15.0,
            rarity / 15.0,
            similar / similar.max().clamp(min=1.0),
            tser / tser.max().clamp(min=1.0),
            word_rank,
            char_rank,
            (context / 15.0) * word_rank,
            (prop / 15.0) * word_rank,
        ],
        dim=1,
    )
    return features


def fit_graph_eager_damage_predictor(ds_key, scores, data, target, args, seed, device):
    num_nodes = int(data.num_nodes)
    features = build_graph_eager_predictor_features(ds_key, scores, num_nodes, device)
    eligible = (data.train_mask | data.val_mask).to(device=device, dtype=torch.bool)
    if int(eligible.sum().item()) == 0:
        eligible = torch.ones(num_nodes, dtype=torch.bool, device=device)
    candidate_idx = eligible.nonzero(as_tuple=False).view(-1)
    sample_count = min(int(args.graph_eager_predictor_calib_samples), int(candidate_idx.numel()))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 811)
    perm = torch.randperm(int(candidate_idx.numel()), generator=generator, device="cpu")[:sample_count].to(device=device)
    calib_idx = candidate_idx[perm]

    x_cal = features[calib_idx]
    y_cal = target[calib_idx].to(dtype=torch.float32)
    mean = x_cal.mean(dim=0, keepdim=True)
    std = x_cal.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (features - mean) / std
    x_cal = x_all[calib_idx]
    ones_cal = torch.ones(x_cal.size(0), 1, dtype=x_cal.dtype, device=device)
    x_design = torch.cat([ones_cal, x_cal], dim=1)
    ridge = float(args.graph_eager_predictor_ridge)
    eye = torch.eye(x_design.size(1), dtype=x_design.dtype, device=device)
    eye[0, 0] = 0.0
    xtx = x_design.t().matmul(x_design) + ridge * eye
    xty = x_design.t().matmul(y_cal.view(-1, 1))
    try:
        coef = torch.linalg.solve(xtx, xty).view(-1)
    except RuntimeError:
        coef = torch.linalg.lstsq(xtx, xty).solution.view(-1)
    ones_all = torch.ones(x_all.size(0), 1, dtype=x_all.dtype, device=device)
    pred = torch.cat([ones_all, x_all], dim=1).matmul(coef).to(dtype=torch.float32)
    return pred, {
        "calib_samples": int(sample_count),
        "train_rho": spearman_corr(pred[calib_idx], y_cal),
        "all_rho": spearman_corr(pred, target),
    }


def rank01(values, descending=False):
    values = values.to(dtype=torch.float32)
    if values.numel() <= 1:
        return torch.ones_like(values)
    order = torch.argsort(values, descending=descending)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, steps=int(values.numel()), device=values.device)
    return ranks


def build_residual_correction_mask(
    trace,
    risk_gate,
    direct_threshold,
    device,
    min_route_hits=1,
    min_base_hits=1,
):
    hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
    source_ok = trace["source_ids"].to(device=device) >= 0
    correction_mask = hit_mask & source_ok
    support_mask = (
        (trace["route_hit_counts"].to(device=device) >= int(min_route_hits))
        | (trace["base_route_hit_counts"].to(device=device) >= int(min_base_hits))
    )
    support_filtered = int((correction_mask & ~support_mask).sum().item())
    correction_mask = correction_mask & support_mask
    if float(direct_threshold) < 0.0 or risk_gate is None:
        return correction_mask, {
            "direct_threshold": float(direct_threshold),
            "direct_low_risk": 0,
            "support_filtered": support_filtered,
            "residual_candidates": int(correction_mask.sum().item()),
        }

    hit_nodes = correction_mask.nonzero(as_tuple=False).view(-1)
    active = torch.zeros_like(correction_mask)
    direct_low_risk = 0
    for node_idx in hit_nodes.detach().cpu().tolist():
        decision = risk_gate.evaluate(
            int(node_idx),
            int(trace["best_dists"][node_idx].item()),
            route_hit_count=int(trace["route_hit_counts"][node_idx].item()),
            base_route_hit_count=int(trace["base_route_hit_counts"][node_idx].item()),
            cos_margin=None,
        )
        if float(decision["risk"]) > float(direct_threshold):
            active[node_idx] = True
        else:
            direct_low_risk += 1

    return active, {
        "direct_threshold": float(direct_threshold),
        "direct_low_risk": int(direct_low_risk),
        "support_filtered": support_filtered,
        "residual_candidates": int(active.sum().item()),
    }


def replace_reuse_anchors_with_random(trace, direct_features, target_features, verify_features, seed):
    random_trace = {}
    for key, value in trace.items():
        if torch.is_tensor(value):
            random_trace[key] = value.clone()
        elif isinstance(value, list):
            random_trace[key] = list(value)
        else:
            random_trace[key] = value

    hit_nodes = (random_trace["hit_mask"] & (random_trace["source_ids"] >= 0)).nonzero(as_tuple=False).view(-1)
    if hit_nodes.numel() == 0:
        return random_trace, direct_features, {"randomized": 0}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 7919)
    random_sources = torch.randint(
        low=0,
        high=int(target_features.size(0)),
        size=(int(hit_nodes.numel()),),
        generator=generator,
        device="cpu",
    ).to(device=target_features.device, dtype=torch.long)
    random_sources = torch.where(
        random_sources == hit_nodes.to(random_sources.device),
        (random_sources + 1) % int(target_features.size(0)),
        random_sources,
    )

    random_trace["source_ids"][hit_nodes] = random_sources
    random_trace["best_dists"][hit_nodes] = torch.clamp(random_trace["best_dists"][hit_nodes], min=1)
    random_trace["best_cosines"][hit_nodes] = F.cosine_similarity(
        verify_features[hit_nodes],
        verify_features[random_sources],
        dim=1,
    )
    for node_idx in hit_nodes.detach().cpu().tolist():
        random_trace["hit_kinds"][node_idx] = "random"

    random_direct = direct_features.clone()
    random_direct[hit_nodes] = target_features[random_sources]
    return random_trace, random_direct, {"randomized": int(hit_nodes.numel())}


def load_residual_target_features(ds_key, data, args, device, log_important):
    source = str(getattr(args, "residual_embedding_source", "data_x"))
    explicit_path = getattr(args, "residual_embedding_path", None)
    if explicit_path:
        path = explicit_path
        target = load_tensor_pool(path, device)
        source_label = "explicit"
    elif source == "real_quant_fp":
        path = (
            getattr(args, "real_quant_fp_path", None)
            or default_pool_path(ds_key, args.real_quant_model_name, args.real_quant_fp_tag)
        )
        target = load_tensor_pool(path, device)
        source_label = f"{args.real_quant_model_name}:{args.real_quant_fp_tag}"
    elif source == "data_x":
        path = "<data.x>"
        target = data.x.detach().to(device=device, dtype=torch.float32)
        source_label = "data_x"
    else:
        raise ValueError(f"Unknown residual embedding source: {source}")

    if target.size(0) != int(data.num_nodes):
        raise ValueError(
            f"Residual target node count mismatch for {ds_key}: "
            f"target={tuple(target.shape)}, data.num_nodes={int(data.num_nodes)}"
        )
    log_important(
        "[ResidualTarget] "
        f"source={source_label} | path={path} | shape={tuple(target.shape)}"
    )
    return target


def resolve_residual_fit_config(args, target_dim):
    profile = str(getattr(args, "residual_fit_profile", "manual"))
    model_name = str(getattr(args, "real_quant_model_name", ""))
    target_dim = int(target_dim)
    effective_profile = profile
    if profile == "auto":
        effective_profile = "llama" if target_dim >= 2048 or model_name.startswith("llama") else "st"

    rank = int(getattr(args, "residual_rank", 32))
    epochs = int(getattr(args, "residual_epochs", 200))
    max_pairs = int(getattr(args, "residual_max_train_pairs", 4096))

    if effective_profile == "llama":
        rank = max(rank, 64)
        epochs = max(epochs, 120)
        max_pairs = max(max_pairs, 4096)
    elif effective_profile == "st":
        rank = max(rank, 32)
        epochs = max(epochs, 100)
        max_pairs = max(max_pairs, 1024)

    return {
        "profile": profile,
        "effective_profile": effective_profile,
        "target_dim": target_dim,
        "rank": rank,
        "epochs": epochs,
        "max_pairs": max_pairs,
    }


def resolve_residual_alpha_grid(args, fit_cfg):
    grid = sorted(float(value) for value in getattr(args, "residual_alpha_grid", [0.0, 0.125, 0.25, 0.5]))
    if fit_cfg["effective_profile"] == "llama":
        grid = [value for value in grid if value <= 0.5 + 1e-12]
    return grid if grid else [0.0]


def log_residual_fit_config(log_important, fit_cfg):
    log_important(
        "[ResidualFitConfig] "
        f"profile={fit_cfg['profile']}->{fit_cfg['effective_profile']} "
        f"| target_dim={fit_cfg['target_dim']} "
        f"| rank={fit_cfg['rank']} "
        f"| epochs={fit_cfg['epochs']} "
        f"| max_pairs={fit_cfg['max_pairs']}"
    )


def build_support_split_masks(trace, soft_min_hits, hard_min_hits, device):
    hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
    source_ok = trace["source_ids"].to(device=device) >= 0
    support_hits = trace["winning_base_table_hit_counts"].to(device=device, dtype=torch.long)
    soft_min_hits = int(soft_min_hits)
    hard_min_hits = int(hard_min_hits)
    residual_hit_mask = hit_mask & source_ok & (support_hits >= soft_min_hits)
    hard_mask = residual_hit_mask & (support_hits >= hard_min_hits)
    soft_mask = residual_hit_mask & (support_hits < hard_min_hits)
    support_hist = {
        int(hits): int((residual_hit_mask & (support_hits == int(hits))).sum().item())
        for hits in range(soft_min_hits, hard_min_hits)
    }
    return {
        "enabled": True,
        "soft_min_hits": soft_min_hits,
        "hard_min_hits": hard_min_hits,
        "hard_mask": hard_mask,
        "soft_mask": soft_mask,
        "residual_hit_mask": residual_hit_mask,
        "support_hist": support_hist,
        "hard_count": int(hard_mask.sum().item()),
        "soft_count": int(soft_mask.sum().item()),
        "residual_count": int(residual_hit_mask.sum().item()),
    }


def apply_soft_cosine_gate(trace, split_info, min_cosine, device):
    threshold = float(min_cosine)
    if threshold < 0.0:
        split_info["cos_filtered"] = 0
        split_info["cos_threshold"] = threshold
        return split_info

    best_cos = trace["best_cosines"].to(device=device, dtype=torch.float32)
    original_soft = split_info["soft_mask"].to(device=device, dtype=torch.bool)
    keep_soft = original_soft & (best_cos >= threshold)
    filtered = original_soft & ~keep_soft
    hard_mask = split_info["hard_mask"].to(device=device, dtype=torch.bool)
    residual_hit_mask = hard_mask | keep_soft

    updated = dict(split_info)
    updated["soft_mask"] = keep_soft
    updated["residual_hit_mask"] = residual_hit_mask
    updated["soft_count"] = int(keep_soft.sum().item())
    updated["residual_count"] = int(residual_hit_mask.sum().item())
    updated["cos_filtered"] = int(filtered.sum().item())
    updated["cos_threshold"] = threshold
    return updated


def format_trace_dist_hist(trace, mask, max_dist=6):
    if mask is None or int(mask.sum().item()) <= 0:
        return "empty"
    dists = trace["best_dists"].to(device=mask.device)[mask]
    parts = []
    for dist in range(int(max_dist) + 1):
        count = int((dists == dist).sum().item())
        if count > 0:
            parts.append(f"d{dist}={count}")
    other = int((dists > int(max_dist)).sum().item())
    if other > 0:
        parts.append(f"d>{int(max_dist)}={other}")
    return ", ".join(parts) if parts else "empty"


def summarize_reuse_real_quant_execution(actions, hit_mask, errors, fp_embs, final_embs):
    total = max(1, int(actions.numel()))
    hit_mask = hit_mask.to(dtype=torch.bool, device=actions.device)
    miss_mask = ~hit_mask

    int4_mask = miss_mask & (actions == 4)
    int8_mask = miss_mask & (actions == 8)
    fp_mask = miss_mask & (actions == 16)

    selected_err = torch.zeros(total, dtype=torch.float32, device=actions.device)
    selected_err[int4_mask] = errors["int4_err"][int4_mask]
    selected_err[int8_mask] = errors["int8_err"][int8_mask]
    miss_count = max(1, int(miss_mask.sum().item()))

    final_err = 1.0 - F.cosine_similarity(fp_embs, final_embs, dim=1)
    final_err = final_err.clamp(min=0.0)

    int4_rate = float(int4_mask.float().mean().item())
    int8_rate = float(int8_mask.float().mean().item())
    fp_rate = float(fp_mask.float().mean().item())
    reuse_rate = float(hit_mask.float().mean().item())
    cost = fp_rate + 0.50 * int8_rate + 0.25 * int4_rate

    return {
        "reuse_num": int(hit_mask.sum().item()),
        "reuse_den": total,
        "reuse_rate": reuse_rate,
        "int4_num": int(int4_mask.sum().item()),
        "int8_num": int(int8_mask.sum().item()),
        "fp_num": int(fp_mask.sum().item()),
        "int4_rate": int4_rate,
        "int8_rate": int8_rate,
        "fp_rate": fp_rate,
        "cost": float(cost),
        "miss_avg_selected_error": float(selected_err[miss_mask].sum().item() / miss_count),
        "final_avg_error": float(final_err.mean().item()),
    }


def run_residual_reuse_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "residual_reuse")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)

        results = {
            "baseline": [],
            "direct_acc": [],
            "direct_drop": [],
            "residual_acc": [],
            "residual_drop": [],
            "direct_reuse": [],
            "direct_reuse_num": [],
            "direct_reuse_den": [],
            "soft_direct_acc": [],
            "soft_direct_drop": [],
            "soft_direct_reuse": [],
            "soft_direct_reuse_num": [],
            "soft_direct_reuse_den": [],
            "soft_direct_err": [],
            "soft_direct_hit_err": [],
            "residual_reuse": [],
            "residual_reuse_num": [],
            "residual_reuse_den": [],
            "soft_reuse_num": [],
            "train_pairs": [],
            "direct_err": [],
            "direct_hit_err": [],
            "residual_err": [],
            "residual_hit_err": [],
            "residual_alpha": [],
        }

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Low-Rank Residual Reuse on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[ResidualReuse] "
                f"rank={int(args.residual_rank)} | epochs={int(args.residual_epochs)} "
                f"| train_split={args.residual_train_split} "
                f"| max_pairs={int(args.residual_max_train_pairs)} "
                f"| alpha={'auto' if float(args.residual_alpha) < 0.0 else float(args.residual_alpha)} "
                f"| min_dist={float(args.residual_min_dist):.1f} "
                f"| min_route_hits={int(args.residual_min_route_hits)} "
                f"| min_base_hits={int(args.residual_min_base_hits)} "
                f"| T_direct={'none' if float(args.residual_direct_threshold) < 0.0 else float(args.residual_direct_threshold)} "
                f"| anchor={args.residual_anchor_mode}"
            )
            support_split_enabled = (
                int(getattr(args, "residual_hard_min_support_hits", -1)) > 0
                and int(getattr(args, "residual_soft_min_support_hits", -1)) > 0
            )
            if support_split_enabled:
                log_important(
                    "[ResidualSupportSplit] "
                    f"hard_direct>= {int(args.residual_hard_min_support_hits)} heads "
                    f"| residual_soft= {int(args.residual_soft_min_support_hits)}.."
                    f"{int(args.residual_hard_min_support_hits) - 1} heads "
                    f"| compute< {int(args.residual_soft_min_support_hits)} heads"
                )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                if support_split_enabled:
                    run_args.route_min_support_hits = [int(args.residual_soft_min_support_hits)]
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                log_important(
                    f"[Seed] run={int(seed)} | controller={int(run_args.controller_seed)} "
                    f"| hash_head={int(run_args.hash_head_seed)} "
                    f"| topology_sketch={int(run_args.topology_sketch_seed)}"
                )

                target_features = load_residual_target_features(ds_key, data, run_args, device, log_important)
                residual_fit_cfg = resolve_residual_fit_config(run_args, target_features.size(1))
                log_residual_fit_config(log_important, residual_fit_cfg)
                data.x = target_features
                model, base_acc, baseline_embs, oracle_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline] Acc: {base_acc:.4f}")

                route_bundle = build_route_bundle(
                    verify_features,
                    data,
                    baseline_embs,
                    oracle_logits,
                    run_args,
                    log_important,
                    device,
                )
                controller = build_controller(
                    data,
                    verify_features,
                    route_bundle,
                    {"name": "ResidualReuse", "overrides": {}},
                    run_args,
                    device,
                )

                direct_features, _hits = controller.query_full_batch(
                    route_bundle["hash_route_features"],
                    verify_features,
                    target_features,
                )
                trace = controller.last_query_trace
                if args.residual_anchor_mode == "random":
                    trace, direct_features, anchor_info = replace_reuse_anchors_with_random(
                        trace,
                        direct_features,
                        target_features,
                        verify_features,
                        seed,
                    )
                else:
                    anchor_info = {"randomized": 0}
                stats = controller.stats
                hit_mask = trace["hit_mask"]
                correction_mask, correction_info = build_residual_correction_mask(
                    trace,
                    controller.risk_gate,
                    args.residual_direct_threshold,
                    device,
                    min_route_hits=args.residual_min_route_hits,
                    min_base_hits=args.residual_min_base_hits,
                )
                if support_split_enabled:
                    split_info = build_support_split_masks(
                        trace,
                        args.residual_soft_min_support_hits,
                        args.residual_hard_min_support_hits,
                        device,
                    )
                    split_info = apply_soft_cosine_gate(
                        trace,
                        split_info,
                        args.residual_soft_min_cosine,
                        device,
                    )
                    if int(split_info.get("cos_filtered", 0)) > 0:
                        log_important(
                            "[ResidualSoftCosGate] "
                            f"threshold={float(split_info['cos_threshold']):.3f} "
                            f"| filtered={int(split_info['cos_filtered'])} "
                            f"| soft_kept={int(split_info['soft_count'])}"
                        )
                    direct_hit_mask = split_info["hard_mask"]
                    residual_hit_mask = split_info["residual_hit_mask"]
                    soft_mask = split_info["soft_mask"]
                    direct_eval_features = target_features.clone()
                    direct_eval_features[direct_hit_mask] = direct_features[direct_hit_mask]
                    residual_base_features = target_features.clone()
                    residual_base_features[residual_hit_mask] = direct_features[residual_hit_mask]
                    soft_direct_features = residual_base_features
                    correction_mask = correction_mask & soft_mask
                    correction_info["soft_cos_filtered"] = int(split_info.get("cos_filtered", 0))
                    correction_info["residual_candidates"] = int(correction_mask.sum().item())
                else:
                    split_info = None
                    direct_hit_mask = hit_mask
                    residual_hit_mask = hit_mask
                    direct_eval_features = direct_features
                    residual_base_features = direct_features
                    soft_direct_features = None

                direct_acc = evaluate_raw_node_features(model, data, direct_eval_features)
                direct_drop = base_acc - direct_acc
                direct_err = embedding_error(target_features, direct_eval_features)
                if soft_direct_features is not None:
                    soft_direct_acc = evaluate_raw_node_features(model, data, soft_direct_features)
                    soft_direct_drop = base_acc - soft_direct_acc
                    soft_direct_err = embedding_error(target_features, soft_direct_features)
                else:
                    soft_direct_acc = None
                    soft_direct_drop = None
                    soft_direct_err = None

                adapter, train_info = train_residual_adapter(
                    target_embeddings=target_features,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    data=data,
                    risk_scores=controller.node_risk_scores,
                    rank=residual_fit_cfg["rank"],
                    epochs=residual_fit_cfg["epochs"],
                    lr=args.residual_lr,
                    weight_decay=args.residual_weight_decay,
                    residual_l2=args.residual_l2,
                    train_split=args.residual_train_split,
                    max_pairs=residual_fit_cfg["max_pairs"],
                    correction_mask=correction_mask,
                    min_dist=args.residual_min_dist,
                    controller=controller,
                    hash_route_features=route_bundle["hash_route_features"],
                    extra_anchors_per_node=args.residual_offline_extra_anchors_per_node,
                    extra_query_nodes=args.residual_offline_extra_query_nodes,
                    positive_error_max=args.residual_positive_error_max,
                    extra_negative_anchors_per_node=args.residual_offline_negative_anchors_per_node,
                    negative_error_min=args.residual_negative_error_min,
                    negative_gate_weight=args.residual_negative_gate_weight,
                    adapter_type=args.residual_adapter_type,
                    accept_mode=args.residual_accept_mode,
                    hidden_dim=args.residual_hidden_dim,
                    hidden_layers=args.residual_hidden_layers,
                    dropout=args.residual_dropout,
                    cosine_weight=args.residual_loss_cosine_weight,
                    mse_weight=args.residual_loss_mse_weight,
                    delta_weight=args.residual_loss_delta_weight,
                    bucket_mode=args.residual_bucket_mode,
                    gate_loss_weight=args.residual_gate_loss_weight,
                    accept_loss_weight=args.residual_accept_loss_weight,
                    gate_error_scale=args.residual_gate_error_scale,
                    gate_error_max=args.residual_gate_error_max,
                    gate_sparsity_weight=args.residual_gate_sparsity_weight,
                    class_aware_accept=args.residual_class_aware_accept,
                    classifier_accept_gate=args.residual_classifier_accept_gate,
                    classifier_model=model,
                    classifier_reference_logits=oracle_logits,
                    classifier_accept_mode=args.residual_classifier_accept_mode,
                    classifier_accept_scope=args.residual_classifier_accept_scope,
                    classifier_accept_after_residual=args.residual_classifier_accept_after_residual,
                    classifier_accept_probe_alpha=args.residual_classifier_accept_probe_alpha,
                    classifier_accept_max_kl=args.residual_classifier_accept_max_kl,
                )

                if adapter is not None:
                    support_alpha_enabled = bool(args.residual_support_aware_alpha)
                    full_bucket_values = compute_bucket_values_from_trace(
                        trace,
                        torch.arange(trace["hit_mask"].numel(), device=device),
                        bucket_mode=args.residual_bucket_mode,
                        device=device,
                    )
                    support_hits = compute_bucket_values_from_trace(
                        trace,
                        correction_mask.nonzero(as_tuple=False).view(-1),
                        bucket_mode=args.residual_bucket_mode,
                        device=device,
                    )
                    alpha_grid = (
                        resolve_residual_alpha_grid(args, residual_fit_cfg)
                        if float(args.residual_alpha) < 0.0
                        else [max(0.0, float(args.residual_alpha))]
                    )
                    gate_grid = (
                        sorted(float(value) for value in args.residual_gate_accept_grid)
                        if float(args.residual_gate_accept_threshold) < 0.0
                        else [max(0.0, float(args.residual_gate_accept_threshold))]
                    )

                    selected_alpha = float(alpha_grid[0])
                    selected_gate_threshold = float(gate_grid[0])
                    selected_val_acc = -1.0
                    for alpha in alpha_grid:
                        for gate_threshold in gate_grid:
                            candidate_features, _candidate_info = apply_residual_adapter(
                                direct_embeddings=residual_base_features,
                                target_embeddings=target_features,
                                verify_features=verify_features,
                                edge_index=data.edge_index,
                                trace=trace,
                                adapter=adapter,
                                risk_scores=controller.node_risk_scores,
                                alpha=alpha,
                                gate_accept_threshold=gate_threshold,
                                min_dist=args.residual_min_dist,
                                correction_mask=correction_mask,
                            )
                            val_acc = evaluate_raw_node_features(model, data, candidate_features, mask=data.val_mask)
                            if val_acc > selected_val_acc + 1e-12:
                                selected_val_acc = val_acc
                                selected_alpha = float(alpha)
                                selected_gate_threshold = float(gate_threshold)
                    if support_alpha_enabled:
                        alpha_by_support = {}
                        support_note_parts = []
                        active_supports = sorted(int(v) for v in support_hits.detach().cpu().unique().tolist())
                        for support_value in active_supports:
                            support_mask = correction_mask & (full_bucket_values == int(support_value))
                            val_support_mask = data.val_mask.to(device=device, dtype=torch.bool) & support_mask
                            best_alpha = float(selected_alpha)
                            best_val_acc = -1.0
                            if int(val_support_mask.sum().item()) > 0:
                                for alpha in alpha_grid:
                                    candidate_features, _candidate_info = apply_residual_adapter(
                                        direct_embeddings=residual_base_features,
                                        target_embeddings=target_features,
                                        verify_features=verify_features,
                                        edge_index=data.edge_index,
                                        trace=trace,
                                        adapter=adapter,
                                        risk_scores=controller.node_risk_scores,
                                        alpha=alpha,
                                        gate_accept_threshold=selected_gate_threshold,
                                        min_dist=args.residual_min_dist,
                                        correction_mask=support_mask,
                                    )
                                    val_acc = evaluate_raw_node_features(
                                        model,
                                        data,
                                        candidate_features,
                                        mask=val_support_mask,
                                    )
                                    if val_acc > best_val_acc + 1e-12:
                                        best_val_acc = val_acc
                                        best_alpha = float(alpha)
                            alpha_by_support[int(support_value)] = float(best_alpha)
                            bucket_label = format_bucket_label(int(support_value), args.residual_bucket_mode)
                            if best_val_acc >= 0.0:
                                support_note_parts.append(f"{bucket_label}={float(best_alpha):.3f}@{best_val_acc:.4f}")
                            else:
                                support_note_parts.append(f"{bucket_label}={float(best_alpha):.3f}@na")
                        alpha_for_apply = {"default": float(selected_alpha), "by_support": alpha_by_support}
                        alpha_note = (
                            f"support-auto(global={selected_val_acc:.4f}; " + ", ".join(support_note_parts) + ")"
                        )
                    else:
                        alpha_for_apply = float(selected_alpha)
                        alpha_note = f"auto(val_acc={selected_val_acc:.4f})"
                    gate_threshold_for_apply = float(selected_gate_threshold)
                    gate_note = (
                        f"auto({selected_val_acc:.4f}, tau={selected_gate_threshold:.3f})"
                        if float(args.residual_gate_accept_threshold) < 0.0
                        else "fixed"
                    )
                else:
                    alpha_for_apply = max(0.0, float(args.residual_alpha))
                    alpha_note = "fixed"
                    gate_threshold_for_apply = max(0.0, float(args.residual_gate_accept_threshold))
                    gate_note = "fixed"

                residual_features, apply_info = apply_residual_adapter(
                    direct_embeddings=residual_base_features,
                    target_embeddings=target_features,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    adapter=adapter,
                    risk_scores=controller.node_risk_scores,
                    alpha=alpha_for_apply,
                    gate_accept_threshold=gate_threshold_for_apply,
                    min_dist=args.residual_min_dist,
                    correction_mask=correction_mask,
                )
                residual_acc = evaluate_raw_node_features(model, data, residual_features)
                residual_drop = base_acc - residual_acc
                residual_err = embedding_error(target_features, residual_features)

                direct_hit_count = int(direct_hit_mask.sum().item())
                soft_residual_hit_count = int(residual_hit_mask.sum().item())
                effective_residual_hit_mask = residual_hit_mask.clone()
                rejected_nodes = apply_info.get("rejected_nodes", None)
                if rejected_nodes is not None and int(rejected_nodes.numel()) > 0:
                    effective_residual_hit_mask[rejected_nodes] = False
                residual_hit_count = int(effective_residual_hit_mask.sum().item())
                direct_hit_err = float(direct_err[direct_hit_mask].mean().item()) if direct_hit_count > 0 else 0.0
                residual_hit_err = (
                    float(residual_err[effective_residual_hit_mask].mean().item()) if residual_hit_count > 0 else 0.0
                )
                direct_reuse_rate = direct_hit_count / max(1, stats["total_queries"])
                residual_reuse_rate = residual_hit_count / max(1, stats["total_queries"])
                if soft_direct_err is not None:
                    soft_direct_hit_count = soft_residual_hit_count
                    soft_direct_hit_err = (
                        float(soft_direct_err[residual_hit_mask].mean().item())
                        if soft_direct_hit_count > 0
                        else 0.0
                    )
                    soft_direct_reuse_rate = soft_direct_hit_count / max(1, stats["total_queries"])
                else:
                    soft_direct_hit_count = 0
                    soft_direct_hit_err = 0.0
                    soft_direct_reuse_rate = 0.0

                results["direct_acc"].append(direct_acc)
                results["direct_drop"].append(direct_drop)
                results["residual_acc"].append(residual_acc)
                results["residual_drop"].append(residual_drop)
                results["direct_reuse"].append(float(direct_reuse_rate))
                results["direct_reuse_num"].append(int(direct_hit_count))
                results["direct_reuse_den"].append(int(stats["reuse_denominator"]))
                if soft_direct_acc is not None:
                    results["soft_direct_acc"].append(float(soft_direct_acc))
                    results["soft_direct_drop"].append(float(soft_direct_drop))
                    results["soft_direct_reuse"].append(float(soft_direct_reuse_rate))
                    results["soft_direct_reuse_num"].append(int(soft_direct_hit_count))
                    results["soft_direct_reuse_den"].append(int(stats["reuse_denominator"]))
                    results["soft_direct_err"].append(float(soft_direct_err.mean().item()))
                    results["soft_direct_hit_err"].append(float(soft_direct_hit_err))
                results["residual_reuse"].append(float(residual_reuse_rate))
                results["residual_reuse_num"].append(int(residual_hit_count))
                results["residual_reuse_den"].append(int(stats["reuse_denominator"]))
                results["soft_reuse_num"].append(int(split_info["soft_count"] if split_info is not None else 0))
                results["train_pairs"].append(int(train_info["train_pairs"]))
                results["direct_err"].append(float(direct_err.mean().item()))
                results["direct_hit_err"].append(direct_hit_err)
                results["residual_err"].append(float(residual_err.mean().item()))
                results["residual_hit_err"].append(residual_hit_err)
                results["residual_alpha"].append(float(apply_info["alpha"]))

                log_important(
                    f"[DirectReuse] Reuse={direct_reuse_rate:.1%} "
                    f"| AnchorMode={args.residual_anchor_mode} "
                    f"| Randomized={anchor_info['randomized']} "
                    f"| Acc={direct_acc:.4f} | Drop={direct_drop:.2%} "
                    f"| AvgErr={float(direct_err.mean().item()):.5f} "
                    f"| HitErr={direct_hit_err:.5f}"
                )
                if soft_direct_acc is not None:
                    log_important(
                        f"[SoftDirectReuse] Reuse={soft_direct_reuse_rate:.1%} "
                        f"| Acc={soft_direct_acc:.4f} | Drop={soft_direct_drop:.2%} "
                        f"| AvgErr={float(soft_direct_err.mean().item()):.5f} "
                        f"| HitErr={soft_direct_hit_err:.5f}"
                    )
                log_important(
                    f"[ResidualReuse] Reuse={residual_reuse_rate:.1%} "
                    f"| Corrected={apply_info['corrected']} "
                    f"| DirectLowRisk={correction_info['direct_low_risk']} "
                    f"| SupportFiltered={correction_info['support_filtered']} "
                    f"| SoftCosFiltered={correction_info.get('soft_cos_filtered', 0)} "
                    f"| ResidualCand={correction_info['residual_candidates']} "
                    f"| TrainPairs={train_info['train_pairs']} "
                    f"| BaseNodes={train_info.get('base_train_nodes', train_info['train_pairs'])} "
                    f"| BaseKeep={train_info.get('base_pairs_kept', train_info.get('base_train_nodes', train_info['train_pairs']))}/{train_info.get('base_pairs_total', train_info.get('base_train_nodes', train_info['train_pairs']))} "
                    f"| ExtraPairs={train_info.get('extra_pairs', 0)} "
                    f"| NegPairs={train_info.get('negative_pairs', 0)} "
                    f"| Alpha={apply_info['alpha']:.3f} ({alpha_note}) "
                    f"| Tau={apply_info.get('gate_accept_threshold', 0.0):.3f} ({gate_note}) "
                    f"| Gate={apply_info.get('gate', 1.0):.3f} "
                    f"| AcceptGate={apply_info.get('accept_gate', 1.0):.3f} "
                    f"| Rejected={apply_info.get('rejected', 0)} "
                    f"| TrainLoss={train_info['loss']:.6f} "
                    f"| TrainGate={train_info.get('gate_mean', 1.0):.3f} "
                    f"| TrainAcceptGate={train_info.get('accept_gate_mean', 1.0):.3f} "
                    f"| Acc={residual_acc:.4f} | Drop={residual_drop:.2%} "
                    f"| AvgErr={float(residual_err.mean().item()):.5f} "
                    f"| HitErr={residual_hit_err:.5f}"
                )
                if train_info.get("classifier_accept_gate"):
                    log_important(
                        "  ClassifierAccept: "
                        f"positive={int(train_info.get('classifier_accept_positive', 0))}/"
                        f"{int(train_info.get('classifier_accept_evaluated', 0))} "
                        f"| mean_kl={float(train_info.get('classifier_accept_mean_kl', 0.0)):.4f}"
                    )
                if "alpha_by_support" in apply_info:
                    alpha_support_text = ", ".join(
                        f"{format_bucket_label(int(k), args.residual_bucket_mode)}={float(v):.3f}"
                        for k, v in sorted(apply_info["alpha_by_support"].items())
                    )
                    log_important(f"  AlphaBySupport: {alpha_support_text}")
                if "gate_by_support" in apply_info:
                    gate_support_text = ", ".join(
                        f"{format_bucket_label(int(k), args.residual_bucket_mode)}={float(v):.3f}"
                        for k, v in sorted(apply_info["gate_by_support"].items())
                    )
                    log_important(f"  GateBySupport: {gate_support_text}")
                if "accept_gate_by_support" in apply_info:
                    gate_support_text = ", ".join(
                        f"{format_bucket_label(int(k), args.residual_bucket_mode)}={float(v):.3f}"
                        for k, v in sorted(apply_info["accept_gate_by_support"].items())
                    )
                    log_important(f"  AcceptGateBySupport: {gate_support_text}")
                if train_info.get("support_aware"):
                    support_pair_text = ", ".join(
                        f"{format_bucket_label(int(k), train_info.get('bucket_mode', args.residual_bucket_mode))}={int(v)}"
                        for k, v in sorted(train_info.get("support_pairs", {}).items())
                    )
                    if support_pair_text:
                        log_important(f"  TrainSupportPairs: {support_pair_text}")
                    support_gate_text = ", ".join(
                        f"{format_bucket_label(int(k), train_info.get('bucket_mode', args.residual_bucket_mode))}={float(v):.3f}"
                        for k, v in sorted(train_info.get("support_gate_means", {}).items())
                    )
                    if support_gate_text:
                        log_important(f"  TrainSupportGates: {support_gate_text}")
                    support_gate_text = ", ".join(
                        f"{format_bucket_label(int(k), train_info.get('bucket_mode', args.residual_bucket_mode))}={float(v):.3f}"
                        for k, v in sorted(train_info.get("support_accept_gate_means", {}).items())
                    )
                    if support_gate_text:
                        log_important(f"  TrainSupportAcceptGates: {support_gate_text}")
                if split_info is not None:
                    hist_text = ", ".join(
                        f"{hits}head={count}" for hits, count in split_info["support_hist"].items()
                    )
                    log_important(
                        f"  SupportSplit: hard_direct={split_info['hard_count']} "
                        f"| soft_residual={split_info['soft_count']} "
                        f"| residual_total={split_info['residual_count']} "
                        f"| compute={stats['total_queries'] - split_info['residual_count']} "
                        f"| soft_hist: {hist_text}"
                    )
                    log_important(
                        "  DistHist: "
                        f"hard[{format_trace_dist_hist(trace, direct_hit_mask)}] "
                        f"| soft[{format_trace_dist_hist(trace, soft_mask)}] "
                        f"| total[{format_trace_dist_hist(trace, residual_hit_mask)}]"
                    )
                log_important(
                    f"  ReuseDetail: numerator={stats['reuse']} "
                    f"(exact={stats['exact_reuse']}, fuzzy={stats['fuzzy_reuse']}) "
                    f"/ denominator={stats['reuse_denominator']} "
                    f"| computed={stats['computed']} "
                    f"| score_reject={stats['score_reject']} "
                    f"(hub={stats['score_reject_hub_protect']}, "
                    f"rare={stats['score_reject_rare_leaf']}, "
                    f"risk={stats['score_reject_risk']})"
                )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL RESIDUAL REUSE SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            log_important("-" * 112)
            log_important(
                f"{'Config':<18} | {'Reuse %':<9} | {'TrainPairs':<10} | {'Acc':<10} | "
                f"{'Drop %':<10} | {'AvgErr':<10} | {'HitErr':<10} | {'Alpha':<7} | {'Reuse n/d':<15}"
            )
            log_important("-" * 112)
            direct_reuse_mean = float(np.mean(results["direct_reuse"]))
            direct_reuse_num = float(np.mean(results["direct_reuse_num"]))
            direct_reuse_den = float(np.mean(results["direct_reuse_den"]))
            direct_reuse_frac = (
                f"{direct_reuse_num:.1f}/{direct_reuse_den:.1f}"
                if args.runs > 1
                else f"{int(direct_reuse_num)}/{int(direct_reuse_den)}"
            )
            residual_reuse_mean = float(np.mean(results["residual_reuse"]))
            residual_reuse_num = float(np.mean(results["residual_reuse_num"]))
            residual_reuse_den = float(np.mean(results["residual_reuse_den"]))
            residual_reuse_frac = (
                f"{residual_reuse_num:.1f}/{residual_reuse_den:.1f}"
                if args.runs > 1
                else f"{int(residual_reuse_num)}/{int(residual_reuse_den)}"
            )
            train_pairs = float(np.mean(results["train_pairs"]))
            log_important(
                f"{'DirectReuse':<18} | {direct_reuse_mean:<9.1%} | {'-':<10} | "
                f"{float(np.mean(results['direct_acc'])):<10.4f} | "
                f"{float(np.mean(results['direct_drop'])):<10.2%} | "
                f"{float(np.mean(results['direct_err'])):<10.5f} | "
                f"{float(np.mean(results['direct_hit_err'])):<10.5f} | {'-':<7} | {direct_reuse_frac:<15}"
            )
            if results["soft_direct_acc"]:
                soft_direct_reuse_mean = float(np.mean(results["soft_direct_reuse"]))
                soft_direct_reuse_num = float(np.mean(results["soft_direct_reuse_num"]))
                soft_direct_reuse_den = float(np.mean(results["soft_direct_reuse_den"]))
                soft_direct_reuse_frac = (
                    f"{soft_direct_reuse_num:.1f}/{soft_direct_reuse_den:.1f}"
                    if args.runs > 1
                    else f"{int(soft_direct_reuse_num)}/{int(soft_direct_reuse_den)}"
                )
                log_important(
                    f"{'SoftDirectReuse':<18} | {soft_direct_reuse_mean:<9.1%} | {'-':<10} | "
                    f"{float(np.mean(results['soft_direct_acc'])):<10.4f} | "
                    f"{float(np.mean(results['soft_direct_drop'])):<10.2%} | "
                    f"{float(np.mean(results['soft_direct_err'])):<10.5f} | "
                    f"{float(np.mean(results['soft_direct_hit_err'])):<10.5f} | {'-':<7} | "
                    f"{soft_direct_reuse_frac:<15}"
                )
            log_important(
                f"{'ResidualReuse':<18} | {residual_reuse_mean:<9.1%} | {train_pairs:<10.1f} | "
                f"{float(np.mean(results['residual_acc'])):<10.4f} | "
                f"{float(np.mean(results['residual_drop'])):<10.2%} | "
                f"{float(np.mean(results['residual_err'])):<10.5f} | "
                f"{float(np.mean(results['residual_hit_err'])):<10.5f} | "
                f"{float(np.mean(results['residual_alpha'])):<7.3f} | {residual_reuse_frac:<15}"
            )
            log_important(f"{'=' * 72}\n")


def evaluate_with_controller_real_quant(
    model,
    data,
    controller,
    route_bundle,
    verify_features,
    selected_embs,
    actions,
    errors,
    fp_embs,
):
    reconstructed_embs, hits = controller.query_full_batch(
        route_bundle["hash_route_features"],
        verify_features,
        selected_embs,
    )

    acc = evaluate_gnn_embeddings(model, data, reconstructed_embs)
    exec_stats = summarize_reuse_real_quant_execution(
        actions,
        hits,
        errors,
        fp_embs,
        reconstructed_embs,
    )
    return {
        "acc": float(acc),
        "drop": None,
        "hits": hits,
        "stats": controller.stats,
        **exec_stats,
    }


def run_internal_split_calibration_step(
    ds_key,
    data,
    verify_features,
    args,
    device,
    dataset_log_dir,
    log_important,
):
    if not bool(args.internal_split_calibration):
        return None, None

    bundle, report = build_internal_split_calibration(ds_key, data, verify_features, args, device)
    report_dir = args.internal_calib_report_dir or dataset_log_dir
    report_path = save_internal_split_calibration(
        bundle=bundle,
        report=report,
        out_dir=report_dir,
        ds_key=ds_key,
        seed=getattr(args, "run_seed", args.seed),
    )

    assignment = report["assignment"]
    sampling = report["sampling"]
    high_sampling = sampling["high_sampling"]
    low_sampling = sampling["low_sampling"]
    log_important(
        "[InternalSplitCalib] "
        f"priority={assignment['priority_policy']} "
        f"| high={assignment['high_count']}/{assignment['node_count']} "
        f"({assignment['topk_ratio']:.1%}) "
        f"| degree_mass={assignment['degree_mass']:.1%} "
        f"| priority_mass={assignment['priority_mass']:.1%} "
        f"| bits={assignment['high_bit']}/{assignment['low_bit']}"
    )
    log_important(
        "[InternalSplitCalib] "
        f"budget={report['node_budget']} "
        f"| prompt=0 "
        f"| high_samples={high_sampling['selected_count']}/{report['high_node_budget']} "
        f"| low_samples={low_sampling['selected_count']}/{report['low_node_budget']} "
        f"| strategy={sampling['base_strategy']} "
        f"| report={report_path}"
    )
    if report.get("text_error"):
        log_important(f"[InternalSplitCalib] Warning: raw texts unavailable: {report['text_error']}")
    return bundle, report


def build_real_quant_policy_configs(args):
    if bool(getattr(args, "reuse_real_quant_allfp_only", False)):
        return [("AllFP", "all_fp")]

    if args.real_quant_policy_suite == "w4a8_budget":
        int8_tag = str(args.real_quant_int8_tag)
        int4_tag = str(args.real_quant_int4_tag)
        # Keep the default table deployable: no route may use per-node
        # FP-vs-quantized embedding error as an input signal.
        configs = [
            ("AllFP", "all_fp"),
            (f"Uniform{int8_tag}", "all_int8"),
            (f"Uniform{int4_tag}", "all_int4"),
            (f"RandomTopK_{int8_tag}", "random_int8_budget"),
            (f"DegreeTopK_{int8_tag}", "degree_int8_budget"),
            (f"TSERTopK_{int8_tag}", "tser_int8_budget"),
        ]
        if bool(args.internal_split_calibration):
            configs.append((f"InternalSplitCalib_{int8_tag}", "internal_split"))
        return configs

    # The standard suite is also kept deployable by default. Error-aware and
    # QuantTSER variants remain implemented as research/debug helpers, but they
    # are intentionally excluded from the main printed policy table.
    configs = [
        ("AllFP", "all_fp"),
        ("AllINT8", "all_int8"),
        ("AllINT4", "all_int4"),
        ("DegreeTopK", "degree_topk"),
        ("TSERTopK", "tser_topk"),
        ("DegreeCascade", "degree_cascade"),
        ("TSERCascade", "tser_cascade"),
        ("DegreeRiskThreshold", "degree"),
        ("TSERRiskThreshold", "tser"),
    ]
    if bool(args.internal_split_calibration):
        configs.append(("InternalSplitCalib", "internal_split"))
    return configs


def run_internal_split_calibration_only(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "graph_simhash")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = os.path.join(dataset_log_dir, f"{ds_key}_internal_split_calibration.log")

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Internal Split Calibration on {ds_key.upper()}")
            log_important(f"{'=' * 72}")

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                run_internal_split_calibration_step(
                    ds_key=ds_key,
                    data=data,
                    verify_features=verify_features,
                    args=run_args,
                    device=device,
                    dataset_log_dir=dataset_log_dir,
                    log_important=log_important,
                )

            log_important(f"{'=' * 72}\n")


def load_precision_depth_pools(ds_key, args, data, device, log_important):
    model_name = str(args.real_quant_model_name)
    reference_tag = str(args.precision_depth_reference_tag)
    tags = [str(tag) for tag in args.precision_depth_tags]
    bits = [int(bit) for bit in args.precision_depth_bits]
    tag_by_bit = {int(bit): str(tag) for bit, tag in zip(bits, tags)}

    reference_path = default_pool_path(ds_key, model_name, reference_tag)
    depth_paths = {
        int(bit): default_pool_path(ds_key, model_name, tag)
        for bit, tag in tag_by_bit.items()
    }
    missing = [
        path
        for path in [reference_path, *depth_paths.values()]
        if not os.path.exists(path)
    ]
    if missing:
        msg = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Precision-depth pools are missing:\n"
            f"{msg}\n"
            "Generate them with generate_real_quant_pools, e.g. --configs W4A8 W4A6 W4A5 W4A4."
        )

    reference = load_tensor_pool(reference_path, device)
    depth = {bit: load_tensor_pool(path, device) for bit, path in depth_paths.items()}

    expected_nodes = int(data.num_nodes)
    expected_shape = tuple(reference.shape)
    for bit, tensor in depth.items():
        if int(tensor.size(0)) != expected_nodes:
            raise ValueError(f"P{bit} pool has wrong node count: {tuple(tensor.shape)} vs {expected_nodes}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"P{bit} pool shape {tuple(tensor.shape)} must match reference {expected_shape}")

    log_important(
        "[PrecisionDepthPools] "
        f"reference=P{int(args.precision_depth_reference_bits)}:{reference_path} | "
        f"depths={', '.join(f'P{bit}:{path}' for bit, path in depth_paths.items())}"
    )
    log_important(f"[PrecisionDepthPools] shape={expected_shape} | bits={sorted(bits, reverse=True)}")
    return {
        "reference": reference,
        "depth": depth,
        "bits": sorted(bits),
        "reference_path": reference_path,
        "depth_paths": depth_paths,
        "tag_by_bit": tag_by_bit,
    }


def precision_depth_cost(bit, args):
    ref_bits = float(args.precision_depth_reference_bits)
    ratio = 1.0 if int(bit) == int(args.precision_depth_reference_bits) else float(bit) / ref_bits
    fixed = float(args.precision_depth_fixed_cost)
    variable = 1.0 - fixed
    return float(args.precision_depth_cost_scale) * (fixed + variable * ratio)


def build_precision_depth_policy_configs(args, bits):
    bits = sorted(int(bit) for bit in bits)
    ref_bit = int(args.precision_depth_reference_bits)
    configs = [(f"FullP{ref_bit}", {"kind": "all_ref"})]
    for bit in sorted(bits, reverse=True):
        configs.append((f"AllP{bit}", {"kind": "all_depth", "bit": int(bit)}))
    configs.extend(
        [
            ("Rand", {"kind": "budget", "priority": "random"}),
            ("Deg", {"kind": "budget", "priority": "degree"}),
            ("TSER", {"kind": "budget", "priority": "tser"}),
            ("Ctx", {"kind": "budget", "priority": "context"}),
            ("Uniq", {"kind": "budget", "priority": "low_unique"}),
        ]
    )
    if bool(getattr(args, "precision_depth_include_predictor", False)):
        configs.append(("Pred", {"kind": "budget", "priority": "predictor"}))
    if bool(getattr(args, "precision_depth_include_oracle", False)):
        configs.append(("Oracle", {"kind": "budget", "priority": "oracle_damage"}))
    if bool(getattr(args, "precision_depth_bound_enable", False)):
        name_by_priority = {
            "degree": "DegBound",
            "tser": "TSERBound",
            "context": "CtxBound",
            "low_unique": "UniqBound",
            "random": "RandBound",
        }
        for priority in getattr(args, "precision_depth_bound_priorities", ["degree", "tser"]):
            configs.append((name_by_priority[priority], {"kind": "bound_budget", "priority": priority}))
    return configs


def precision_depth_remaining_bit_bound(depth, ref_bit, args):
    """Predictor-free bound for omitted low activation bit-planes.

    This is a simulation-side proxy for the hardware bound estimator: if execution
    stops at `depth`, the omitted low bits are bounded by their maximum normalized
    contribution. The tile factor keeps the bound conservative for larger K tiles.
    """
    depth = int(depth)
    ref_bit = int(ref_bit)
    if depth >= ref_bit:
        return 0.0
    omitted = (2 ** (ref_bit - depth)) - 1
    denom = max(1, (2 ** ref_bit) - 1)
    tile_k = max(1, int(getattr(args, "precision_depth_bound_tile_k", 128)))
    tile_scale = float(np.sqrt(tile_k / 128.0))
    return float(getattr(args, "precision_depth_bound_scale", 1.0)) * (float(omitted) / float(denom)) * tile_scale


def select_runtime_bound_depth(min_depth, tolerance, ref_bit, args):
    """Return the first depth accepted by runtime bound, respecting min_depth."""
    min_depth = max(1, min(int(min_depth), int(ref_bit)))
    tolerance = max(0.0, float(tolerance))
    for depth in range(min_depth, int(ref_bit) + 1):
        if precision_depth_remaining_bit_bound(depth, ref_bit, args) <= tolerance:
            return int(depth)
    return int(ref_bit)


def nearest_available_precision_depth(requested_depth, bits, ref_bit):
    """Map a runtime depth to the nearest generated embedding pool not below it."""
    available = sorted({int(ref_bit), *[int(bit) for bit in bits]})
    requested_depth = int(requested_depth)
    for bit in available:
        if bit >= requested_depth:
            return int(bit)
    return int(ref_bit)


def bound_policy_bucket_bits(bits, ref_bit, args):
    buckets = [
        (
            "high",
            int(getattr(args, "precision_depth_bound_high_min_depth", ref_bit)),
            float(getattr(args, "precision_depth_bound_high_tolerance", 0.0)),
        ),
        (
            "mid",
            int(getattr(args, "precision_depth_bound_mid_min_depth", 6)),
            float(getattr(args, "precision_depth_bound_mid_tolerance", 0.02)),
        ),
        (
            "low",
            int(getattr(args, "precision_depth_bound_low_min_depth", 4)),
            float(getattr(args, "precision_depth_bound_low_tolerance", 0.04)),
        ),
    ]
    out = {}
    for name, min_depth, tolerance in buckets:
        runtime_depth = select_runtime_bound_depth(min_depth, tolerance, ref_bit, args)
        pool_bit = nearest_available_precision_depth(runtime_depth, bits, ref_bit)
        out[name] = {
            "min_depth": int(min_depth),
            "tolerance": float(tolerance),
            "runtime_depth": int(runtime_depth),
            "pool_bit": int(pool_bit),
            "bound": precision_depth_remaining_bit_bound(runtime_depth, ref_bit, args),
        }
    return out


def select_precision_depth_actions(
    policy,
    scores,
    args,
    num_nodes,
    bits,
    seed,
    device,
    eligible_mask=None,
    oracle_priority=None,
    predictor_priority=None,
):
    bits = sorted(int(bit) for bit in bits)
    ref_bit = int(args.precision_depth_reference_bits)
    if policy["kind"] == "all_ref":
        return torch.full((num_nodes,), ref_bit, dtype=torch.int64, device=device)
    if policy["kind"] == "all_depth":
        return torch.full((num_nodes,), int(policy["bit"]), dtype=torch.int64, device=device)
    if policy["kind"] not in {"budget", "bound_budget"}:
        raise ValueError(f"Unknown precision-depth policy: {policy}")

    priority_name = policy["priority"]
    if priority_name == "degree":
        priority = scores["propagation_q"].to(dtype=torch.float32)
    elif priority_name == "tser":
        priority = scores["sensitivity_q"].to(dtype=torch.float32)
    elif priority_name == "context":
        priority = scores["graph_context_q"].to(dtype=torch.float32)
    elif priority_name == "low_unique":
        priority = scores["low_degree_unique_q"].to(dtype=torch.float32)
    elif priority_name == "oracle_damage":
        if oracle_priority is None:
            raise ValueError("oracle_damage policy requires oracle_priority")
        priority = oracle_priority.to(dtype=torch.float32)
    elif priority_name == "predictor":
        if predictor_priority is None:
            raise ValueError("predictor policy requires predictor_priority")
        priority = predictor_priority.to(dtype=torch.float32)
    elif priority_name == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 59999)
        priority = torch.rand(num_nodes, generator=generator, device="cpu").to(device=device)
    else:
        raise ValueError(f"Unknown precision-depth priority: {priority_name}")

    cheapest_bit = int(bits[0])
    low_to_high_bits = sorted(bits, reverse=True)
    safest_low_bit = int(low_to_high_bits[0])
    second_low_bit = int(low_to_high_bits[1]) if len(low_to_high_bits) > 1 else safest_low_bit

    actions = torch.full((num_nodes,), ref_bit, dtype=torch.int64, device=device)
    if eligible_mask is None:
        eligible = torch.ones(num_nodes, dtype=torch.bool, device=device)
    else:
        eligible = torch.as_tensor(eligible_mask, dtype=torch.bool, device=device)
        if eligible.numel() != num_nodes:
            raise ValueError(f"eligible_mask length must match node count {num_nodes}, got {eligible.numel()}")

    eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
    eligible_count = int(eligible_idx.numel())
    if eligible_count <= 0:
        return actions

    actions[eligible_idx] = cheapest_bit
    order = eligible_idx[torch.argsort(priority[eligible_idx], descending=True)]
    high_count = int(round(float(args.precision_depth_high_ratio) * eligible_count))
    mid_count = int(round(float(args.precision_depth_mid_ratio) * eligible_count))
    low_count = int(round(float(getattr(args, "precision_depth_low_ratio", 0.0)) * eligible_count))
    high_count = max(0, min(high_count, eligible_count))
    mid_count = max(0, min(mid_count, eligible_count - high_count))
    low_count = max(0, min(low_count, eligible_count - high_count - mid_count))
    if policy["kind"] == "bound_budget":
        bucket_bits = bound_policy_bucket_bits(bits, ref_bit, args)
        actions[eligible_idx] = bucket_bits["low"]["pool_bit"]
        actions[order[:high_count]] = bucket_bits["high"]["pool_bit"]
        actions[order[high_count : high_count + mid_count]] = bucket_bits["mid"]["pool_bit"]
        actions[order[high_count + mid_count : high_count + mid_count + low_count]] = bucket_bits["low"]["pool_bit"]
    else:
        actions[order[:high_count]] = ref_bit
        actions[order[high_count : high_count + mid_count]] = safest_low_bit
        actions[order[high_count + mid_count : high_count + mid_count + low_count]] = second_low_bit
    return actions


def assemble_precision_depth_embeddings(actions, reference_embs, depth_embs):
    mixed = reference_embs.clone()
    for bit, embs in depth_embs.items():
        mask = actions == int(bit)
        if bool(mask.any()):
            mixed[mask] = embs[mask]
    return mixed


def summarize_precision_depth_policy(actions, errors_by_bit, bits, args):
    total = max(1, int(actions.numel()))
    ref_bit = int(args.precision_depth_reference_bits)
    rates = {ref_bit: float((actions == ref_bit).float().mean().item())}
    cost = rates[ref_bit] * precision_depth_cost(ref_bit, args)
    selected_err = torch.zeros(total, dtype=torch.float32, device=actions.device)
    selected_err[actions == ref_bit] = errors_by_bit[ref_bit][actions == ref_bit]
    for bit in bits:
        mask = actions == int(bit)
        rate = float(mask.float().mean().item())
        rates[int(bit)] = rate
        cost += rate * precision_depth_cost(int(bit), args)
        selected_err[mask] = errors_by_bit[int(bit)][mask]
    return {
        "rates": rates,
        "cost": float(cost),
        "avg_err": float(selected_err.mean().item()),
    }


def summarize_reuse_precision_depth_execution(actions, hit_mask, errors_by_bit, bits, args, fp_embs, final_embs):
    total = max(1, int(actions.numel()))
    hit_mask = hit_mask.to(dtype=torch.bool, device=actions.device)
    miss_mask = ~hit_mask
    ref_bit = int(args.precision_depth_reference_bits)
    all_bits = [ref_bit, *sorted((int(bit) for bit in bits), reverse=True)]

    selected_err = torch.zeros(total, dtype=torch.float32, device=actions.device)
    rates = {}
    cost = 0.0
    for bit in all_bits:
        mask = miss_mask & (actions == int(bit))
        rate = float(mask.float().mean().item())
        rates[int(bit)] = rate
        cost += rate * precision_depth_cost(int(bit), args)
        selected_err[mask] = errors_by_bit[int(bit)][mask]

    final_err = 1.0 - F.cosine_similarity(fp_embs, final_embs, dim=1)
    final_err = final_err.clamp(min=0.0)
    miss_count = max(1, int(miss_mask.sum().item()))
    return {
        "reuse_num": int(hit_mask.sum().item()),
        "reuse_den": total,
        "reuse_rate": float(hit_mask.float().mean().item()),
        "rates": rates,
        "cost": float(cost),
        "miss_avg_selected_error": float(selected_err[miss_mask].sum().item() / miss_count),
        "final_avg_error": float(final_err.mean().item()),
    }


def apply_reuse_trace_to_embeddings(trace, selected_embs):
    """Apply a fixed CAM/hash reuse trace to a candidate embedding pool."""
    final_embs = selected_embs.clone()
    hit_mask = trace["hit_mask"].to(device=selected_embs.device, dtype=torch.bool)
    source_ids = trace["source_ids"].to(device=selected_embs.device, dtype=torch.long)
    reusable = hit_mask & (source_ids >= 0)
    if bool(reusable.any()):
        final_embs[reusable] = selected_embs[source_ids[reusable]]
    return final_embs, hit_mask


def build_active_residual_mask(trace, correction_mask, min_dist, device):
    hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
    source_ok = trace["source_ids"].to(device=device, dtype=torch.long) >= 0
    active = hit_mask & source_ok & correction_mask.to(device=device, dtype=torch.bool)
    if float(min_dist) > 0.0:
        active = active & (
            trace["best_dists"].to(device=device, dtype=torch.float32) >= float(min_dist)
        )
    return active


def apply_residual_precision_depth_trace(
    trace,
    selected_raw,
    reference_raw,
    residual_raw,
    residual_mask,
    accepted_hit_mask=None,
):
    """Compose direct reuse, residual reuse, and precision-depth miss execution."""
    final_raw = selected_raw.clone()
    device = selected_raw.device
    trace_hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
    hit_mask = (
        trace_hit_mask
        if accepted_hit_mask is None
        else accepted_hit_mask.to(device=device, dtype=torch.bool) & trace_hit_mask
    )
    source_ids = trace["source_ids"].to(device=device, dtype=torch.long)
    source_ok = source_ids >= 0
    residual_mask = residual_mask.to(device=device, dtype=torch.bool) & hit_mask & source_ok
    direct_mask = hit_mask & source_ok & ~residual_mask
    if bool(direct_mask.any()):
        final_raw[direct_mask] = reference_raw[source_ids[direct_mask]]
    if bool(residual_mask.any()):
        final_raw[residual_mask] = residual_raw[residual_mask]
    return final_raw, direct_mask, residual_mask, hit_mask


def summarize_residual_precision_depth_execution(
    actions,
    trace,
    residual_mask,
    errors_by_bit,
    bits,
    args,
    fp_embs,
    final_embs,
    accepted_hit_mask=None,
):
    total = max(1, int(actions.numel()))
    device = actions.device
    trace_hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
    hit_mask = (
        trace_hit_mask
        if accepted_hit_mask is None
        else accepted_hit_mask.to(device=device, dtype=torch.bool) & trace_hit_mask
    )
    source_ok = trace["source_ids"].to(device=device, dtype=torch.long) >= 0
    residual_mask = residual_mask.to(device=device, dtype=torch.bool) & hit_mask & source_ok
    direct_mask = hit_mask & source_ok & ~residual_mask
    miss_mask = ~hit_mask
    ref_bit = int(args.precision_depth_reference_bits)
    all_bits = [ref_bit, *sorted((int(bit) for bit in bits), reverse=True)]

    selected_err = torch.zeros(total, dtype=torch.float32, device=device)
    rates = {}
    cost = float(residual_mask.float().mean().item()) * float(args.hierarchical_residual_cost)
    for bit in all_bits:
        mask = miss_mask & (actions == int(bit))
        rate = float(mask.float().mean().item())
        rates[int(bit)] = rate
        cost += rate * precision_depth_cost(int(bit), args)
        selected_err[mask] = errors_by_bit[int(bit)][mask]

    final_err = (1.0 - F.cosine_similarity(fp_embs, final_embs, dim=1)).clamp(min=0.0)
    miss_count = max(1, int(miss_mask.sum().item()))
    return {
        "reuse_num": int(hit_mask.sum().item()),
        "reuse_den": total,
        "reuse_rate": float(hit_mask.float().mean().item()),
        "direct_rate": float(direct_mask.float().mean().item()),
        "residual_rate": float(residual_mask.float().mean().item()),
        "residual_num": int(residual_mask.sum().item()),
        "rates": rates,
        "cost": float(cost),
        "miss_avg_selected_error": float(selected_err[miss_mask].sum().item() / miss_count),
        "final_avg_error": float(final_err.mean().item()),
    }


def export_residual_precision_depth_node_trace(
    path,
    *,
    dataset,
    seed,
    config_name,
    policy,
    actions,
    trace,
    accepted_hit_mask,
    direct_mask,
    residual_mask,
    scores,
    args,
):
    """Export a per-node Graph-Bit replay trace.

    The trace is intentionally small: one JSON object per graph node with the
    online reuse decision and the miss-node precision-depth action.  It does not
    include embeddings.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    device = actions.device
    num_nodes = int(actions.numel())
    accepted_hit_mask = accepted_hit_mask.to(device=device, dtype=torch.bool)
    direct_mask = direct_mask.to(device=device, dtype=torch.bool)
    residual_mask = residual_mask.to(device=device, dtype=torch.bool)
    source_ids = trace["source_ids"].to(device=device, dtype=torch.long)
    best_dists = trace["best_dists"].to(device=device, dtype=torch.long)
    route_hits = trace["route_hit_counts"].to(device=device, dtype=torch.long)
    base_hits = trace["base_route_hit_counts"].to(device=device, dtype=torch.long)
    winning_hits = trace.get("winning_base_table_hit_counts", base_hits)
    winning_hits = winning_hits.to(device=device, dtype=torch.long)
    hit_kinds = list(trace.get("hit_kinds", ["none"] * num_nodes))
    ref_bit = int(args.precision_depth_reference_bits)
    bucket_by_bit = {ref_bit: "p8", 8: "p8", 6: "p6", 5: "p5", 4: "p4"}
    priority = str(policy.get("priority", ""))
    ratios = {
        "high": float(getattr(args, "precision_depth_high_ratio", 0.0)),
        "mid": float(getattr(args, "precision_depth_mid_ratio", 0.0)),
        "low": float(getattr(args, "precision_depth_low_ratio", 0.0)),
    }
    meta = {
        "schema": "graphbit_node_trace.v1",
        "dataset": str(dataset),
        "seed": int(seed),
        "config": str(config_name),
        "policy": dict(policy),
        "priority": priority,
        "reference_bit": ref_bit,
        "ratios": ratios,
        "bound": {
            "enabled": bool(getattr(args, "precision_depth_bound_enable", False)),
            "high_min_depth": int(getattr(args, "precision_depth_bound_high_min_depth", ref_bit)),
            "mid_min_depth": int(getattr(args, "precision_depth_bound_mid_min_depth", 6)),
            "low_min_depth": int(getattr(args, "precision_depth_bound_low_min_depth", 4)),
            "high_tolerance": float(getattr(args, "precision_depth_bound_high_tolerance", 0.0)),
            "mid_tolerance": float(getattr(args, "precision_depth_bound_mid_tolerance", 0.02)),
            "low_tolerance": float(getattr(args, "precision_depth_bound_low_tolerance", 0.04)),
            "scale": float(getattr(args, "precision_depth_bound_scale", 1.0)),
            "tile_k": int(getattr(args, "precision_depth_bound_tile_k", 128)),
        },
    }
    score_tensors = {
        "degree_q": scores["propagation_q"].to(device=device, dtype=torch.float32),
        "tser_q": scores["sensitivity_q"].to(device=device, dtype=torch.float32),
        "context_q": scores["graph_context_q"].to(device=device, dtype=torch.float32),
        "low_unique_q": scores["low_degree_unique_q"].to(device=device, dtype=torch.float32),
    }
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"meta": meta}, sort_keys=True) + "\n")
        for node_id in range(num_nodes):
            bit = int(actions[node_id].item())
            if bool(direct_mask[node_id].item()):
                role = "direct"
            elif bool(residual_mask[node_id].item()):
                role = "residual"
            elif bool(accepted_hit_mask[node_id].item()):
                role = "reuse_other"
            else:
                role = "miss"
            row = {
                "node_id": int(node_id),
                "role": role,
                "is_miss": role == "miss",
                "source_id": int(source_ids[node_id].item()),
                "hit_kind": str(hit_kinds[node_id]) if node_id < len(hit_kinds) else "none",
                "best_dist": int(best_dists[node_id].item()),
                "route_hits": int(route_hits[node_id].item()),
                "base_hits": int(base_hits[node_id].item()),
                "support_hits": int(winning_hits[node_id].item()),
                "action_bit": bit,
                "depth_bucket": bucket_by_bit.get(bit, f"p{bit}"),
                "stop_depth": bit if role == "miss" else 0,
            }
            for key, tensor in score_tensors.items():
                row[key] = float(tensor[node_id].item())
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def fit_precision_depth_damage_predictor(ds_key, scores, data, target, args, seed, device):
    original_samples = getattr(args, "graph_eager_predictor_calib_samples", None)
    original_ridge = getattr(args, "graph_eager_predictor_ridge", None)
    args.graph_eager_predictor_calib_samples = int(args.precision_depth_predictor_calib_samples)
    args.graph_eager_predictor_ridge = float(args.precision_depth_predictor_ridge)
    try:
        return fit_graph_eager_damage_predictor(ds_key, scores, data, target, args, seed, device)
    finally:
        if original_samples is not None:
            args.graph_eager_predictor_calib_samples = original_samples
        if original_ridge is not None:
            args.graph_eager_predictor_ridge = original_ridge


def run_precision_depth_ablation(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "precision_depth")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        bits = sorted(int(bit) for bit in args.precision_depth_bits)
        ref_bit = int(args.precision_depth_reference_bits)
        configs = build_precision_depth_policy_configs(args, bits)

        results = {
            name: {
                "acc": [],
                "drop": [],
                "avg_err": [],
                "cost": [],
                f"P{ref_bit}": [],
                **{f"P{bit}": [] for bit in bits},
            }
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Graph-Conditioned Precision-Depth Ablation on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[PrecisionDepthCost] "
                f"model={args.real_quant_model_name} | reference={args.precision_depth_reference_tag}/P{ref_bit} "
                f"| tags={list(zip(args.precision_depth_tags, args.precision_depth_bits))} "
                f"| high_ratio={args.precision_depth_high_ratio:.2f} "
                f"| mid_ratio={args.precision_depth_mid_ratio:.2f} "
                f"| low_ratio={args.precision_depth_low_ratio:.2f} "
                f"| cost_scale={args.precision_depth_cost_scale:.2f} "
                f"| fixed_cost={args.precision_depth_fixed_cost:.2f}"
            )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                pools = load_precision_depth_pools(ds_key, run_args, data, device, log_important)

                data.x = pools["reference"]
                model, base_acc, ref_embs, ref_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:P{ref_bit}] Acc: {base_acc:.4f}")

                model.eval()
                with torch.no_grad():
                    depth_embs = {
                        bit: model.encoder(pool)
                        for bit, pool in pools["depth"].items()
                    }

                scores = build_real_quant_scores(verify_features, data, run_args, device)
                errors_by_bit = {ref_bit: embedding_error(ref_embs, ref_embs)}
                for bit, embs in depth_embs.items():
                    errors_by_bit[int(bit)] = embedding_error(ref_embs, embs)

                ref_pred = ref_logits.argmax(dim=1)
                ref_margin = node_logit_margin(ref_logits)
                proxy_items = [
                    ("Degree", scores["propagation_q"]),
                    ("Context", scores["graph_context_q"]),
                    ("LowUnique", scores["low_degree_unique_q"]),
                    ("TSER", scores["sensitivity_q"]),
                ]
                margin_drop_by_bit = {}
                for bit in sorted(bits, reverse=True):
                    with torch.no_grad():
                        depth_logits = forward_gnn_logits(model, data, depth_embs[int(bit)])
                    depth_pred = depth_logits.argmax(dim=1)
                    depth_margin = node_logit_margin(depth_logits)
                    margin_drop = (ref_margin - depth_margin).clamp(min=0.0)
                    margin_drop_by_bit[int(bit)] = margin_drop
                    flip = depth_pred != ref_pred
                    err = errors_by_bit[int(bit)]
                    proxy_text = []
                    for proxy_name, proxy_score in proxy_items:
                        proxy_text.append(
                            f"{proxy_name}:rho_err={spearman_corr(proxy_score, err):.3f},"
                            f"rho_margin={spearman_corr(proxy_score, margin_drop):.3f},"
                            f"auc_flip={binary_auc(proxy_score, flip):.3f}"
                        )
                    log_important(f"[PrecisionDepthCorr] P{bit} " + " | ".join(proxy_text))

                log_important(
                    "[PrecisionDepthScore] "
                    f"prop_mean={scores['propagation_q'].float().mean().item():.1f} | "
                    f"context_mean={scores['graph_context_q'].float().mean().item():.1f} | "
                    f"low_unique_mean={scores['low_degree_unique_q'].float().mean().item():.1f} | "
                    f"tser_mean={scores['sensitivity_q'].float().mean().item():.1f}"
                )
                err_parts = [f"P{ref_bit}=0.00000"] + [
                    f"P{bit}={errors_by_bit[int(bit)].mean().item():.5f}"
                    for bit in sorted(bits, reverse=True)
                ]
                log_important("[PrecisionDepthError] " + " | ".join(err_parts))

                cheapest_bit = int(bits[0])
                oracle_priority = errors_by_bit[cheapest_bit]
                if run_args.precision_depth_predictor_target == "margin":
                    predictor_target = margin_drop_by_bit[cheapest_bit]
                else:
                    predictor_target = oracle_priority
                predictor_priority = None
                if any(policy.get("priority") == "predictor" for _name, policy in configs):
                    predictor_priority, predictor_info = fit_precision_depth_damage_predictor(
                        ds_key,
                        scores,
                        data,
                        predictor_target,
                        run_args,
                        seed,
                        device,
                    )
                    log_important(
                        "[PrecisionDepthPredictor] "
                        f"target={run_args.precision_depth_predictor_target} | "
                        f"calib={predictor_info['calib_samples']} | "
                        f"rho_train={predictor_info['train_rho']:.3f} | "
                        f"rho_all={predictor_info['all_rho']:.3f}"
                    )

                for name, policy in configs:
                    actions = select_precision_depth_actions(
                        policy,
                        scores,
                        run_args,
                        int(data.num_nodes),
                        bits,
                        seed,
                        device,
                        oracle_priority=oracle_priority,
                        predictor_priority=predictor_priority,
                    )
                    mixed_embs = assemble_precision_depth_embeddings(actions, ref_embs, depth_embs)
                    acc = evaluate_gnn_embeddings(model, data, mixed_embs)
                    drop = base_acc - acc
                    stats = summarize_precision_depth_policy(actions, errors_by_bit, bits, run_args)

                    results[name]["acc"].append(acc)
                    results[name]["drop"].append(drop)
                    results[name]["avg_err"].append(stats["avg_err"])
                    results[name]["cost"].append(stats["cost"])
                    results[name][f"P{ref_bit}"].append(stats["rates"][ref_bit])
                    for bit in bits:
                        results[name][f"P{bit}"].append(stats["rates"][int(bit)])

                    bit_rate_text = " | ".join(
                        f"P{bit}={stats['rates'][int(bit)]:.1%}"
                        for bit in sorted(bits, reverse=True)
                    )
                    log_important(
                        f"[{name}] P{ref_bit}={stats['rates'][ref_bit]:.1%} | {bit_rate_text} "
                        f"| Cost={stats['cost']:.3f} | Acc={acc:.4f} "
                        f"| Drop={drop:.2%} | AvgErr={stats['avg_err']:.5f}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL PRECISION-DEPTH SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            depth_headers = [f"P{bit} %" for bit in [ref_bit, *sorted(bits, reverse=True)]]
            log_important("-" * 144)
            log_important(
                f"{'Config':<24} | "
                + " | ".join(f"{header:<8}" for header in depth_headers)
                + f" | {'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | {'AvgErr':<10}"
            )
            log_important("-" * 144)
            for name, _policy in configs:
                rate_values = [
                    float(np.mean(results[name][f"P{bit}"]))
                    for bit in [ref_bit, *sorted(bits, reverse=True)]
                ]
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                avg_err = float(np.mean(results[name]["avg_err"]))
                log_important(
                    f"{name:<24} | "
                    + " | ".join(f"{rate:<8.1%}" for rate in rate_values)
                    + f" | {cost:<8.3f} | {acc:<10.4f} | {drop:<10.2%} | {avg_err:<10.5f}"
                )
            log_important(f"{'=' * 72}\n")


def run_reuse_precision_depth_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "reuse_precision_depth")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        bits = sorted(int(bit) for bit in args.precision_depth_bits)
        ref_bit = int(args.precision_depth_reference_bits)
        configs = build_precision_depth_policy_configs(args, bits)

        results = {
            name: {
                "reuse": [],
                "reuse_num": [],
                "reuse_den": [],
                "cost": [],
                "acc": [],
                "drop": [],
                "final_err": [],
                "miss_err": [],
                f"P{ref_bit}": [],
                **{f"P{bit}": [] for bit in bits},
            }
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Hash Reuse + Graph-Bit Precision-Depth on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[ReuseGraphBit] Reuse hits are free cache reads; hash misses are routed to "
                f"P{ref_bit}/" + "/".join(f"P{bit}" for bit in sorted(bits, reverse=True)) + "."
            )
            log_important(
                "[PrecisionDepthCost] "
                f"model={args.real_quant_model_name} | reference={args.precision_depth_reference_tag}/P{ref_bit} "
                f"| tags={list(zip(args.precision_depth_tags, args.precision_depth_bits))} "
                f"| high_ratio={args.precision_depth_high_ratio:.2f} "
                f"| mid_ratio={args.precision_depth_mid_ratio:.2f} "
                f"| low_ratio={args.precision_depth_low_ratio:.2f} "
                f"| cost_scale={args.precision_depth_cost_scale:.2f} "
                f"| fixed_cost={args.precision_depth_fixed_cost:.2f}"
            )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                run_args.enable_quant_policy = False
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                pools = load_precision_depth_pools(ds_key, run_args, data, device, log_important)

                data.x = pools["reference"]
                model, base_acc, ref_embs, ref_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:P{ref_bit}] Acc: {base_acc:.4f}")

                model.eval()
                with torch.no_grad():
                    depth_embs = {
                        int(bit): model.encoder(pool)
                        for bit, pool in pools["depth"].items()
                    }

                scores = build_real_quant_scores(verify_features, data, run_args, device)
                errors_by_bit = {ref_bit: embedding_error(ref_embs, ref_embs)}
                for bit, embs in depth_embs.items():
                    errors_by_bit[int(bit)] = embedding_error(ref_embs, embs)
                log_important(
                    "[PrecisionDepthScore] "
                    f"prop_mean={scores['propagation_q'].float().mean().item():.1f} | "
                    f"context_mean={scores['graph_context_q'].float().mean().item():.1f} | "
                    f"low_unique_mean={scores['low_degree_unique_q'].float().mean().item():.1f} | "
                    f"tser_mean={scores['sensitivity_q'].float().mean().item():.1f}"
                )
                err_parts = [f"P{ref_bit}=0.00000"] + [
                    f"P{bit}={errors_by_bit[int(bit)].mean().item():.5f}"
                    for bit in sorted(bits, reverse=True)
                ]
                log_important("[PrecisionDepthError] " + " | ".join(err_parts))

                route_bundle = build_route_bundle(
                    verify_features,
                    data,
                    ref_embs,
                    ref_logits,
                    run_args,
                    log_important,
                    device,
                )
                trace_controller = build_controller(
                    data,
                    verify_features,
                    route_bundle,
                    {"name": "Trace", "overrides": {}},
                    run_args,
                    device,
                )
                _trace_embs, trace_hits = trace_controller.query_full_batch(
                    route_bundle["hash_route_features"],
                    verify_features,
                    ref_embs,
                )
                trace = trace_controller.last_query_trace
                trace_stats = trace_controller.stats
                trace_reuse = trace_stats["reuse"] / max(1, trace_stats["total_queries"])
                log_important(
                    f"[ReuseTrace] Reuse={trace_reuse:.1%} "
                    f"| miss={(~trace_hits).float().mean().item():.1%} "
                    f"| reuse n/d={trace_stats['reuse']}/{trace_stats['reuse_denominator']}"
                )

                miss_mask = ~trace_hits
                oracle_priority = errors_by_bit[int(bits[0])]
                predictor_priority = None
                if any(policy.get("priority") == "predictor" for _name, policy in configs):
                    predictor_priority, predictor_info = fit_precision_depth_damage_predictor(
                        ds_key,
                        scores,
                        data,
                        oracle_priority,
                        run_args,
                        seed,
                        device,
                    )
                    log_important(
                        "[PrecisionDepthPredictor] "
                        f"calib={predictor_info['calib_samples']} | "
                        f"rho_train={predictor_info['train_rho']:.3f} | "
                        f"rho_all={predictor_info['all_rho']:.3f}"
                    )

                for name, policy in configs:
                    actions = select_precision_depth_actions(
                        policy,
                        scores,
                        run_args,
                        int(data.num_nodes),
                        bits,
                        seed,
                        device,
                        eligible_mask=miss_mask,
                        oracle_priority=oracle_priority,
                        predictor_priority=predictor_priority,
                    )
                    selected_embs = assemble_precision_depth_embeddings(actions, ref_embs, depth_embs)
                    reconstructed_embs, hits = apply_reuse_trace_to_embeddings(trace, selected_embs)
                    acc = evaluate_gnn_embeddings(model, data, reconstructed_embs)
                    drop = base_acc - acc
                    stats = summarize_reuse_precision_depth_execution(
                        actions,
                        hits,
                        errors_by_bit,
                        bits,
                        run_args,
                        ref_embs,
                        reconstructed_embs,
                    )

                    results[name]["reuse"].append(stats["reuse_rate"])
                    results[name]["reuse_num"].append(stats["reuse_num"])
                    results[name]["reuse_den"].append(stats["reuse_den"])
                    results[name]["cost"].append(stats["cost"])
                    results[name]["acc"].append(acc)
                    results[name]["drop"].append(drop)
                    results[name]["final_err"].append(stats["final_avg_error"])
                    results[name]["miss_err"].append(stats["miss_avg_selected_error"])
                    for bit in [ref_bit, *bits]:
                        results[name][f"P{int(bit)}"].append(stats["rates"].get(int(bit), 0.0))

                    bit_rate_text = " | ".join(
                        f"P{bit}={stats['rates'].get(int(bit), 0.0):.1%}"
                        for bit in [ref_bit, *sorted(bits, reverse=True)]
                    )
                    log_important(
                        f"[{name}] Reuse={stats['reuse_rate']:.1%} | {bit_rate_text} "
                        f"| Cost={stats['cost']:.3f} | Acc={acc:.4f} "
                        f"| Drop={drop:.2%} | FinalErr={stats['final_avg_error']:.5f} "
                        f"| MissErr={stats['miss_avg_selected_error']:.5f}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL REUSE + GRAPH-BIT SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            depth_order = [ref_bit, *sorted(bits, reverse=True)]
            log_important("-" * 168)
            log_important(
                f"{'Config':<24} | {'Reuse %':<9} | "
                + " | ".join(f"P{bit} %".ljust(8) for bit in depth_order)
                + f" | {'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | {'FinalErr':<10} | {'Reuse n/d':<15}"
            )
            log_important("-" * 168)
            for name, _policy in configs:
                reuse = float(np.mean(results[name]["reuse"]))
                reuse_num = float(np.mean(results[name]["reuse_num"]))
                reuse_den = float(np.mean(results[name]["reuse_den"]))
                reuse_frac = (
                    f"{reuse_num:.1f}/{reuse_den:.1f}"
                    if args.runs > 1
                    else f"{int(reuse_num)}/{int(reuse_den)}"
                )
                rate_values = [
                    float(np.mean(results[name][f"P{bit}"]))
                    for bit in depth_order
                ]
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                final_err = float(np.mean(results[name]["final_err"]))
                log_important(
                    f"{name:<24} | {reuse:<9.1%} | "
                    + " | ".join(f"{rate:<8.1%}" for rate in rate_values)
                    + f" | {cost:<8.3f} | {acc:<10.4f} | {drop:<10.2%} | "
                    f"{final_err:<10.5f} | {reuse_frac:<15}"
                )
            log_important(f"{'=' * 72}\n")


def run_residual_precision_depth_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "residual_precision_depth")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        bits = sorted(int(bit) for bit in args.precision_depth_bits)
        ref_bit = int(args.precision_depth_reference_bits)
        configs = build_precision_depth_policy_configs(args, bits)
        depth_order = [ref_bit, *sorted(bits, reverse=True)]

        results = {
            name: {
                "reuse": [],
                "direct": [],
                "residual": [],
                "reuse_num": [],
                "reuse_den": [],
                "cost": [],
                "acc": [],
                "drop": [],
                "final_err": [],
                "miss_err": [],
                "train_pairs": [],
                "alpha": [],
                **{f"P{bit}": [] for bit in depth_order},
            }
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Residual Reuse + Graph-Bit Precision-Depth on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[FullStack] Exact hit -> direct cache reuse | fuzzy hit -> residual correction | "
                f"miss -> P{ref_bit}/" + "/".join(f"P{bit}" for bit in sorted(bits, reverse=True))
            )
            support_split_enabled = (
                int(getattr(args, "residual_hard_min_support_hits", -1)) > 0
                and int(getattr(args, "residual_soft_min_support_hits", -1)) > 0
            )
            if support_split_enabled:
                log_important(
                    "[FullStackSupportSplit] "
                    f"hard_direct>= {int(args.residual_hard_min_support_hits)} heads | "
                    f"residual_soft= {int(args.residual_soft_min_support_hits)}.."
                    f"{int(args.residual_hard_min_support_hits) - 1} heads | "
                    f"compute< {int(args.residual_soft_min_support_hits)} heads"
                )
            log_important(
                "[PrecisionDepthCost] "
                f"model={args.real_quant_model_name} | reference={args.precision_depth_reference_tag}/P{ref_bit} "
                f"| tags={list(zip(args.precision_depth_tags, args.precision_depth_bits))} "
                f"| high_ratio={args.precision_depth_high_ratio:.2f} "
                f"| mid_ratio={args.precision_depth_mid_ratio:.2f} "
                f"| low_ratio={args.precision_depth_low_ratio:.2f} "
                f"| cost_scale={args.precision_depth_cost_scale:.2f} "
                f"| fixed_cost={args.precision_depth_fixed_cost:.2f} "
                f"| residual_cost={args.hierarchical_residual_cost:.4f}"
            )
            if bool(getattr(args, "precision_depth_bound_enable", False)):
                bucket_bits = bound_policy_bucket_bits(bits, ref_bit, args)
                bucket_text = " | ".join(
                    f"{name}:min={info['min_depth']},tau={info['tolerance']:.4f},"
                    f"runtime=P{info['runtime_depth']},pool=P{info['pool_bit']},"
                    f"bound={info['bound']:.5f}"
                    for name, info in bucket_bits.items()
                )
                log_important(
                    "[GraphBitBound] "
                    f"priorities={list(getattr(args, 'precision_depth_bound_priorities', []))} | "
                    f"tile_k={int(getattr(args, 'precision_depth_bound_tile_k', 128))} | "
                    f"scale={float(getattr(args, 'precision_depth_bound_scale', 1.0)):.3f} | "
                    f"{bucket_text}"
                )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                run_args.enable_quant_policy = False
                if support_split_enabled:
                    run_args.route_min_support_hits = [int(args.residual_soft_min_support_hits)]
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                pools = load_precision_depth_pools(ds_key, run_args, data, device, log_important)

                reference_raw = pools["reference"]
                depth_raw = {int(bit): pool for bit, pool in pools["depth"].items()}
                residual_fit_cfg = resolve_residual_fit_config(run_args, reference_raw.size(1))
                log_residual_fit_config(log_important, residual_fit_cfg)
                data.x = reference_raw
                model, base_acc, ref_embs, ref_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:P{ref_bit}] Acc: {base_acc:.4f}")

                model.eval()
                with torch.no_grad():
                    depth_embs = {
                        int(bit): model.encoder(pool)
                        for bit, pool in depth_raw.items()
                    }

                scores = build_real_quant_scores(verify_features, data, run_args, device)
                errors_by_bit = {ref_bit: embedding_error(ref_embs, ref_embs)}
                for bit, embs in depth_embs.items():
                    errors_by_bit[int(bit)] = embedding_error(ref_embs, embs)
                log_important(
                    "[PrecisionDepthScore] "
                    f"prop_mean={scores['propagation_q'].float().mean().item():.1f} | "
                    f"context_mean={scores['graph_context_q'].float().mean().item():.1f} | "
                    f"low_unique_mean={scores['low_degree_unique_q'].float().mean().item():.1f} | "
                    f"tser_mean={scores['sensitivity_q'].float().mean().item():.1f}"
                )
                err_parts = [f"P{ref_bit}=0.00000"] + [
                    f"P{bit}={errors_by_bit[int(bit)].mean().item():.5f}"
                    for bit in sorted(bits, reverse=True)
                ]
                log_important("[PrecisionDepthError] " + " | ".join(err_parts))

                route_bundle = build_route_bundle(
                    verify_features,
                    data,
                    ref_embs,
                    ref_logits,
                    run_args,
                    log_important,
                    device,
                )
                controller = build_controller(
                    data,
                    verify_features,
                    route_bundle,
                    {"name": "FullStack", "overrides": {}},
                    run_args,
                    device,
                )
                direct_raw, trace_hits = controller.query_full_batch(
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
                trace_stats = controller.stats
                hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
                miss_mask = ~hit_mask
                exact_mask = hit_mask & (
                    torch.tensor([kind == "exact" for kind in trace["hit_kinds"]], device=device)
                )
                fuzzy_mask = hit_mask & (
                    torch.tensor([kind == "fuzzy" for kind in trace["hit_kinds"]], device=device)
                )
                log_important(
                    f"[ReuseTrace] Reuse={hit_mask.float().mean().item():.1%} "
                    f"(exact={exact_mask.float().mean().item():.1%}, fuzzy={fuzzy_mask.float().mean().item():.1%}) "
                    f"| miss={miss_mask.float().mean().item():.1%} "
                    f"| reuse n/d={trace_stats['reuse']}/{trace_stats['reuse_denominator']} "
                    f"| randomized={anchor_info['randomized']}"
                )

                correction_mask, correction_info = build_residual_correction_mask(
                    trace,
                    controller.risk_gate,
                    run_args.residual_direct_threshold,
                    device,
                    min_route_hits=run_args.residual_min_route_hits,
                    min_base_hits=run_args.residual_min_base_hits,
                )
                if support_split_enabled:
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
                    correction_info["soft_cos_filtered"] = int(split_info.get("cos_filtered", 0))
                    correction_info["residual_candidates"] = int(correction_mask.sum().item())
                    log_important(
                        "[FullStackSupportTrace] "
                        f"hard={split_info['hard_count']} | "
                        f"soft={split_info['soft_count']} | "
                        f"accepted={split_info['residual_count']} | "
                        f"soft_hist={split_info['support_hist']} | "
                        f"cos_filtered={int(split_info.get('cos_filtered', 0))}"
                    )
                adapter, train_info = train_residual_adapter(
                    target_embeddings=reference_raw,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    data=data,
                    risk_scores=controller.node_risk_scores,
                    rank=residual_fit_cfg["rank"],
                    epochs=residual_fit_cfg["epochs"],
                    lr=run_args.residual_lr,
                    weight_decay=run_args.residual_weight_decay,
                    residual_l2=run_args.residual_l2,
                    train_split=run_args.residual_train_split,
                    max_pairs=residual_fit_cfg["max_pairs"],
                    correction_mask=correction_mask,
                    min_dist=run_args.residual_min_dist,
                    controller=controller,
                    hash_route_features=route_bundle["hash_route_features"],
                    extra_anchors_per_node=run_args.residual_offline_extra_anchors_per_node,
                    extra_query_nodes=run_args.residual_offline_extra_query_nodes,
                    positive_error_max=run_args.residual_positive_error_max,
                    extra_negative_anchors_per_node=run_args.residual_offline_negative_anchors_per_node,
                    negative_error_min=run_args.residual_negative_error_min,
                    negative_gate_weight=run_args.residual_negative_gate_weight,
                    adapter_type=run_args.residual_adapter_type,
                    accept_mode=run_args.residual_accept_mode,
                    hidden_dim=run_args.residual_hidden_dim,
                    hidden_layers=run_args.residual_hidden_layers,
                    dropout=run_args.residual_dropout,
                    cosine_weight=run_args.residual_loss_cosine_weight,
                    mse_weight=run_args.residual_loss_mse_weight,
                    delta_weight=run_args.residual_loss_delta_weight,
                    bucket_mode=run_args.residual_bucket_mode,
                    gate_loss_weight=run_args.residual_gate_loss_weight,
                    accept_loss_weight=run_args.residual_accept_loss_weight,
                    gate_error_scale=run_args.residual_gate_error_scale,
                    gate_error_max=run_args.residual_gate_error_max,
                    gate_sparsity_weight=run_args.residual_gate_sparsity_weight,
                    class_aware_accept=run_args.residual_class_aware_accept,
                    classifier_accept_gate=run_args.residual_classifier_accept_gate,
                    classifier_model=model,
                    classifier_reference_logits=ref_logits,
                    classifier_accept_mode=run_args.residual_classifier_accept_mode,
                    classifier_accept_scope=run_args.residual_classifier_accept_scope,
                    classifier_accept_after_residual=run_args.residual_classifier_accept_after_residual,
                    classifier_accept_probe_alpha=run_args.residual_classifier_accept_probe_alpha,
                    classifier_accept_max_kl=run_args.residual_classifier_accept_max_kl,
                )

                if adapter is not None:
                    support_alpha_enabled = bool(run_args.residual_support_aware_alpha)
                    full_bucket_values = compute_bucket_values_from_trace(
                        trace,
                        torch.arange(trace["hit_mask"].numel(), device=device),
                        bucket_mode=run_args.residual_bucket_mode,
                        device=device,
                    )
                    support_hits = compute_bucket_values_from_trace(
                        trace,
                        correction_mask.nonzero(as_tuple=False).view(-1),
                        bucket_mode=run_args.residual_bucket_mode,
                        device=device,
                    )
                    alpha_grid = (
                        resolve_residual_alpha_grid(run_args, residual_fit_cfg)
                        if float(run_args.residual_alpha) < 0.0
                        else [max(0.0, float(run_args.residual_alpha))]
                    )
                    gate_grid = (
                        sorted(float(value) for value in run_args.residual_gate_accept_grid)
                        if float(run_args.residual_gate_accept_threshold) < 0.0
                        else [max(0.0, float(run_args.residual_gate_accept_threshold))]
                    )
                    selected_alpha = float(alpha_grid[0])
                    selected_gate_threshold = float(gate_grid[0])
                    selected_val_acc = -1.0
                    for alpha in alpha_grid:
                        for gate_threshold in gate_grid:
                            candidate_raw, _candidate_info = apply_residual_adapter(
                                direct_embeddings=direct_raw,
                                target_embeddings=reference_raw,
                                verify_features=verify_features,
                                edge_index=data.edge_index,
                                trace=trace,
                                adapter=adapter,
                                risk_scores=controller.node_risk_scores,
                                alpha=alpha,
                                gate_accept_threshold=gate_threshold,
                                min_dist=run_args.residual_min_dist,
                                correction_mask=correction_mask,
                            )
                            val_acc = evaluate_raw_node_features(model, data, candidate_raw, mask=data.val_mask)
                            if val_acc > selected_val_acc + 1e-12:
                                selected_val_acc = val_acc
                                selected_alpha = float(alpha)
                                selected_gate_threshold = float(gate_threshold)
                    if support_alpha_enabled:
                        alpha_by_support = {}
                        support_note_parts = []
                        active_supports = sorted(int(v) for v in support_hits.detach().cpu().unique().tolist())
                        for support_value in active_supports:
                            support_mask = correction_mask & (full_bucket_values == int(support_value))
                            val_support_mask = data.val_mask.to(device=device, dtype=torch.bool) & support_mask
                            best_alpha = float(selected_alpha)
                            best_val_acc = -1.0
                            if int(val_support_mask.sum().item()) > 0:
                                for alpha in alpha_grid:
                                    candidate_raw, _candidate_info = apply_residual_adapter(
                                        direct_embeddings=direct_raw,
                                        target_embeddings=reference_raw,
                                        verify_features=verify_features,
                                        edge_index=data.edge_index,
                                        trace=trace,
                                        adapter=adapter,
                                        risk_scores=controller.node_risk_scores,
                                        alpha=alpha,
                                        gate_accept_threshold=selected_gate_threshold,
                                        min_dist=run_args.residual_min_dist,
                                        correction_mask=support_mask,
                                    )
                                    val_acc = evaluate_raw_node_features(
                                        model,
                                        data,
                                        candidate_raw,
                                        mask=val_support_mask,
                                    )
                                    if val_acc > best_val_acc + 1e-12:
                                        best_val_acc = val_acc
                                        best_alpha = float(alpha)
                            alpha_by_support[int(support_value)] = float(best_alpha)
                            bucket_label = format_bucket_label(int(support_value), run_args.residual_bucket_mode)
                            if best_val_acc >= 0.0:
                                support_note_parts.append(f"{bucket_label}={float(best_alpha):.3f}@{best_val_acc:.4f}")
                            else:
                                support_note_parts.append(f"{bucket_label}={float(best_alpha):.3f}@na")
                        alpha_for_apply = {"default": float(selected_alpha), "by_support": alpha_by_support}
                        alpha_note = (
                            f"support-auto(global={selected_val_acc:.4f}; "
                            + ", ".join(support_note_parts)
                            + ")"
                        )
                    else:
                        alpha_for_apply = float(selected_alpha)
                        alpha_note = f"auto(val_acc={selected_val_acc:.4f})"
                    gate_threshold_for_apply = float(selected_gate_threshold)
                    gate_note = (
                        f"auto({selected_val_acc:.4f}, tau={selected_gate_threshold:.3f})"
                        if float(run_args.residual_gate_accept_threshold) < 0.0
                        else "fixed"
                    )
                else:
                    alpha_for_apply = max(0.0, float(run_args.residual_alpha))
                    alpha_note = "fixed"
                    gate_threshold_for_apply = max(0.0, float(run_args.residual_gate_accept_threshold))
                    gate_note = "fixed"

                residual_raw, apply_info = apply_residual_adapter(
                    direct_embeddings=direct_raw,
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
                )
                active_residual_mask = (
                    build_active_residual_mask(trace, correction_mask, run_args.residual_min_dist, device)
                    if adapter is not None
                    else torch.zeros(int(data.num_nodes), dtype=torch.bool, device=device)
                )
                accepted_hit_mask = hit_mask.clone()
                rejected_nodes = apply_info.get("rejected_nodes", None)
                if rejected_nodes is not None and int(rejected_nodes.numel()) > 0:
                    active_residual_mask[rejected_nodes] = False
                    accepted_hit_mask[rejected_nodes] = False
                effective_miss_mask = ~accepted_hit_mask
                log_important(
                    "[ResidualFit] "
                    f"candidates={correction_info['residual_candidates']} | "
                    f"active={int(active_residual_mask.sum().item())} | "
                    f"train_pairs={int(train_info['train_pairs'])} | "
                    f"loss={float(train_info['loss']):.6f} | "
                    f"alpha={float(apply_info['alpha']):.3f} ({alpha_note}) | "
                    f"gate={float(apply_info.get('accept_gate', 1.0)):.3f}/{gate_note} | "
                    f"corrected={int(apply_info['corrected'])} | "
                    f"accepted={int(apply_info.get('accepted', 0))} | "
                    f"rejected={int(apply_info.get('rejected', 0))}"
                )
                if train_info.get("classifier_accept_gate"):
                    log_important(
                        "  ClassifierAccept: "
                        f"positive={int(train_info.get('classifier_accept_positive', 0))}/"
                        f"{int(train_info.get('classifier_accept_evaluated', 0))} "
                        f"| mean_kl={float(train_info.get('classifier_accept_mean_kl', 0.0)):.4f}"
                    )

                oracle_priority = errors_by_bit[int(bits[0])]
                predictor_priority = None
                if any(policy.get("priority") == "predictor" for _name, policy in configs):
                    predictor_priority, predictor_info = fit_precision_depth_damage_predictor(
                        ds_key,
                        scores,
                        data,
                        oracle_priority,
                        run_args,
                        seed,
                        device,
                    )
                    log_important(
                        "[PrecisionDepthPredictor] "
                        f"calib={predictor_info['calib_samples']} | "
                        f"rho_train={predictor_info['train_rho']:.3f} | "
                        f"rho_all={predictor_info['all_rho']:.3f}"
                    )

                for name, policy in configs:
                    actions = select_precision_depth_actions(
                        policy,
                        scores,
                        run_args,
                        int(data.num_nodes),
                        bits,
                        seed,
                        device,
                        eligible_mask=effective_miss_mask,
                        oracle_priority=oracle_priority,
                        predictor_priority=predictor_priority,
                    )
                    selected_raw = assemble_precision_depth_embeddings(actions, reference_raw, depth_raw)
                    final_raw, direct_mask, residual_mask, _hits = apply_residual_precision_depth_trace(
                        trace,
                        selected_raw,
                        reference_raw,
                        residual_raw,
                        active_residual_mask,
                        accepted_hit_mask=accepted_hit_mask,
                    )
                    with torch.no_grad():
                        final_embs = model.encoder(final_raw)
                    acc = evaluate_gnn_embeddings(model, data, final_embs)
                    drop = base_acc - acc
                    stats = summarize_residual_precision_depth_execution(
                        actions,
                        trace,
                        residual_mask,
                        errors_by_bit,
                        bits,
                        run_args,
                        ref_embs,
                        final_embs,
                        accepted_hit_mask=accepted_hit_mask,
                    )

                    results[name]["reuse"].append(stats["reuse_rate"])
                    results[name]["direct"].append(stats["direct_rate"])
                    results[name]["residual"].append(stats["residual_rate"])
                    results[name]["reuse_num"].append(stats["reuse_num"])
                    results[name]["reuse_den"].append(stats["reuse_den"])
                    results[name]["cost"].append(stats["cost"])
                    results[name]["acc"].append(acc)
                    results[name]["drop"].append(drop)
                    results[name]["final_err"].append(stats["final_avg_error"])
                    results[name]["miss_err"].append(stats["miss_avg_selected_error"])
                    results[name]["train_pairs"].append(int(train_info["train_pairs"]))
                    results[name]["alpha"].append(float(apply_info["alpha"]))
                    for bit in depth_order:
                        results[name][f"P{int(bit)}"].append(stats["rates"].get(int(bit), 0.0))

                    bit_rate_text = " | ".join(
                        f"P{bit}={stats['rates'].get(int(bit), 0.0):.1%}"
                        for bit in depth_order
                    )
                    log_important(
                        f"[{name}] Reuse={stats['reuse_rate']:.1%} "
                        f"(direct={stats['direct_rate']:.1%}, residual={stats['residual_rate']:.1%}) "
                        f"| {bit_rate_text} | Cost={stats['cost']:.3f} | Acc={acc:.4f} "
                        f"| Drop={drop:.2%} | FinalErr={stats['final_avg_error']:.5f} "
                        f"| MissErr={stats['miss_avg_selected_error']:.5f}"
                    )
                    trace_export_dir = str(getattr(run_args, "precision_depth_trace_export_dir", "") or "")
                    export_configs = [
                        str(item)
                        for item in getattr(run_args, "precision_depth_trace_export_configs", ["DegBound"])
                    ]
                    if trace_export_dir and ("all" in export_configs or name in export_configs):
                        trace_name = f"{ds_key}_seed{seed}_{name}.jsonl"
                        trace_path = os.path.join(trace_export_dir, trace_name)
                        export_residual_precision_depth_node_trace(
                            trace_path,
                            dataset=ds_key,
                            seed=seed,
                            config_name=name,
                            policy=policy,
                            actions=actions,
                            trace=trace,
                            accepted_hit_mask=accepted_hit_mask,
                            direct_mask=direct_mask,
                            residual_mask=residual_mask,
                            scores=scores,
                            args=run_args,
                        )
                        log_important(f"[GraphBitTrace] wrote {trace_path}")

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL RESIDUAL + GRAPH-BIT SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            log_important("-" * 134)
            log_important(
                f"{'Cfg':<8} | {'Reuse':<7} | {'Dir':<7} | {'Res':<7} | "
                + " | ".join(f"P{bit}".ljust(6) for bit in depth_order)
                + f" | {'Cost':<6} | {'Acc':<7} | {'Drop':<7} | {'Err':<8} | {'Tr':<6} | {'A':<5}"
            )
            log_important("-" * 134)
            for name, _policy in configs:
                reuse = float(np.mean(results[name]["reuse"]))
                direct = float(np.mean(results[name]["direct"]))
                residual = float(np.mean(results[name]["residual"]))
                rate_values = [
                    float(np.mean(results[name][f"P{bit}"]))
                    for bit in depth_order
                ]
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                final_err = float(np.mean(results[name]["final_err"]))
                train_pairs = float(np.mean(results[name]["train_pairs"]))
                alpha = float(np.mean(results[name]["alpha"]))
                log_important(
                    f"{name:<8} | {reuse:<7.1%} | {direct:<7.1%} | {residual:<7.1%} | "
                    + " | ".join(f"{rate:<6.1%}" for rate in rate_values)
                    + f" | {cost:<6.3f} | {acc:<7.4f} | {drop:<7.2%} | "
                    f"{final_err:<8.5f} | {train_pairs:<6.1f} | {alpha:<5.3f}"
                )
            log_important(f"{'=' * 72}\n")


def run_graph_eager_token_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "graph_eager_token")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        lengths = sorted(int(length) for length in args.graph_eager_token_lengths)
        configs = build_graph_eager_token_policy_configs(args, lengths)

        results = {
            name: {
                "acc": [],
                "drop": [],
                "avg_err": [],
                "cost": [],
                "full": [],
                **{f"S{length}": [] for length in lengths},
            }
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Graph-Eager Token Routing on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[GraphEagerCost] "
                f"model={args.real_quant_model_name} | reference={args.graph_eager_reference_tag} "
                f"| full={args.graph_eager_full_tag} | token_prefix={args.graph_eager_token_tag_prefix} "
                f"| lengths={lengths} | full_length={args.graph_eager_full_length} "
                f"| full_ratio={args.graph_eager_full_ratio:.2f} "
                f"| mid_ratio={args.graph_eager_mid_ratio:.2f} "
                f"| cost_scale={args.graph_eager_cost_scale:.2f} "
                f"| attn_weight={args.graph_eager_attn_weight:.2f} "
                f"| ffn_weight={args.graph_eager_ffn_weight:.2f}"
            )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                pools = load_graph_eager_token_pools(ds_key, run_args, data, device, log_important)

                data.x = pools["reference"]
                model, base_acc, ref_embs, ref_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:{run_args.graph_eager_reference_tag}] Acc: {base_acc:.4f}")

                model.eval()
                with torch.no_grad():
                    full_embs = model.encoder(pools["full"])
                    token_embs = {
                        length: model.encoder(token_pool)
                        for length, token_pool in pools["token"].items()
                    }

                scores = build_real_quant_scores(verify_features, data, run_args, device)
                errors_by_action = {-1: embedding_error(ref_embs, full_embs)}
                for length, embs in token_embs.items():
                    errors_by_action[int(length)] = embedding_error(ref_embs, embs)

                ref_pred = ref_logits.argmax(dim=1)
                ref_margin = node_logit_margin(ref_logits)
                proxy_items = [
                    ("Degree", scores["propagation_q"]),
                    ("Context", scores["graph_context_q"]),
                    ("LowUnique", scores["low_degree_unique_q"]),
                    ("TSER", scores["sensitivity_q"]),
                ]
                corr_lines = []
                margin_drop_by_length = {}
                for length in lengths:
                    with torch.no_grad():
                        token_logits = forward_gnn_logits(model, data, token_embs[int(length)])
                    token_pred = token_logits.argmax(dim=1)
                    token_margin = node_logit_margin(token_logits)
                    margin_drop = (ref_margin - token_margin).clamp(min=0.0)
                    margin_drop_by_length[int(length)] = margin_drop
                    flip = token_pred != ref_pred
                    err = errors_by_action[int(length)]
                    proxy_text = []
                    for proxy_name, proxy_score in proxy_items:
                        proxy_text.append(
                            f"{proxy_name}:rho_err={spearman_corr(proxy_score, err):.3f},"
                            f"rho_margin={spearman_corr(proxy_score, margin_drop):.3f},"
                            f"auc_flip={binary_auc(proxy_score, flip):.3f}"
                        )
                    corr_lines.append(f"S{length} " + " | ".join(proxy_text))

                log_important(
                    "[GraphEagerScore] "
                    f"prop_mean={scores['propagation_q'].float().mean().item():.1f} | "
                    f"context_mean={scores['graph_context_q'].float().mean().item():.1f} | "
                    f"low_unique_mean={scores['low_degree_unique_q'].float().mean().item():.1f} | "
                    f"tser_mean={scores['sensitivity_q'].float().mean().item():.1f}"
                )
                err_parts = [
                    f"Full={errors_by_action[-1].mean().item():.5f}",
                    *[
                        f"S{length}={errors_by_action[int(length)].mean().item():.5f}"
                        for length in lengths
                    ],
                ]
                log_important("[GraphEagerError] " + " | ".join(err_parts))
                for corr_line in corr_lines:
                    log_important("[GraphEagerCorr] " + corr_line)

                shortest_length = int(lengths[0])
                oracle_priority = errors_by_action[shortest_length] if lengths else errors_by_action[-1]
                if run_args.graph_eager_predictor_target == "margin":
                    predictor_target = margin_drop_by_length[shortest_length]
                else:
                    predictor_target = oracle_priority
                predictor_priority, predictor_info = fit_graph_eager_damage_predictor(
                    ds_key,
                    scores,
                    data,
                    predictor_target,
                    run_args,
                    seed,
                    device,
                )
                log_important(
                    "[GraphEagerPredictor] "
                    f"target={run_args.graph_eager_predictor_target} | "
                    f"calib={predictor_info['calib_samples']} | "
                    f"rho_train={predictor_info['train_rho']:.3f} | "
                    f"rho_all={predictor_info['all_rho']:.3f}"
                )

                for name, policy in configs:
                    actions = select_graph_eager_token_actions(
                        policy,
                        scores,
                        run_args,
                        int(data.num_nodes),
                        lengths,
                        seed,
                        device,
                        oracle_priority=oracle_priority,
                        predictor_priority=predictor_priority,
                    )
                    mixed_embs = assemble_graph_eager_token_embeddings(actions, full_embs, token_embs)
                    acc = evaluate_gnn_embeddings(model, data, mixed_embs)
                    drop = base_acc - acc
                    stats = summarize_graph_eager_token_policy(
                        actions,
                        errors_by_action,
                        lengths,
                        run_args,
                    )

                    results[name]["acc"].append(acc)
                    results[name]["drop"].append(drop)
                    results[name]["avg_err"].append(stats["avg_err"])
                    results[name]["cost"].append(stats["cost"])
                    results[name]["full"].append(stats["rates"]["full"])
                    for length in lengths:
                        results[name][f"S{length}"].append(stats["rates"][int(length)])

                    token_rate_text = " | ".join(
                        f"S{length}={stats['rates'][int(length)]:.1%}"
                        for length in lengths
                    )
                    log_important(
                        f"[{name}] Full={stats['rates']['full']:.1%} | {token_rate_text} "
                        f"| Cost={stats['cost']:.3f} | Acc={acc:.4f} "
                        f"| Drop={drop:.2%} | AvgErr={stats['avg_err']:.5f}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL GRAPH-EAGER TOKEN SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            token_headers = [f"S{length} %" for length in lengths]
            log_important("-" * 132)
            log_important(
                f"{'Config':<24} | {'Full %':<8} | "
                + " | ".join(f"{header:<8}" for header in token_headers)
                + f" | {'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | {'AvgErr':<10}"
            )
            log_important("-" * 132)
            for name, _policy in configs:
                full_rate = float(np.mean(results[name]["full"]))
                token_rate_values = [float(np.mean(results[name][f"S{length}"])) for length in lengths]
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                avg_err = float(np.mean(results[name]["avg_err"]))
                log_important(
                    f"{name:<24} | {full_rate:<8.1%} | "
                    + " | ".join(f"{rate:<8.1%}" for rate in token_rate_values)
                    + f" | {cost:<8.3f} | {acc:<10.4f} | {drop:<10.2%} | {avg_err:<10.5f}"
                )
            log_important(f"{'=' * 72}\n")


def run_token_compaction_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "token_compaction")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        tags = [str(tag) for tag in args.token_compaction_tags]
        if args.token_compaction_names is None:
            names = [tag.replace("W4A8_", "") for tag in tags]
        else:
            names = [str(name) for name in args.token_compaction_names]

        configs = [("FullW4A8", {"tag": str(args.token_compaction_full_tag), "kind": "full"})]
        configs.extend((name, {"tag": tag, "kind": "compact"}) for name, tag in zip(names, tags))

        results = {
            name: {"acc": [], "drop": [], "avg_err": [], "cost": []}
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Token/Chunk Compaction Validation on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[TokenCompaction] "
                f"model={args.real_quant_model_name} | reference={args.token_compaction_reference_tag} "
                f"| full={args.token_compaction_full_tag} | length={args.token_compaction_length} "
                f"| tags={', '.join(tags)}"
            )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, _verify_features, device = load_run_state(ds_key, run_args, seed)

                ref_path = default_pool_path(ds_key, run_args.real_quant_model_name, run_args.token_compaction_reference_tag)
                paths = {
                    name: default_pool_path(ds_key, run_args.real_quant_model_name, policy["tag"])
                    for name, policy in configs
                }
                missing = [path for path in [ref_path, *paths.values()] if not os.path.exists(path)]
                if missing:
                    msg = "\n".join(f"  - {path}" for path in missing)
                    raise FileNotFoundError(
                        "Token compaction pools are missing:\n"
                        f"{msg}\n"
                        "Generate them with generate_real_quant_pools --text_compaction_strategy ..."
                    )

                reference = load_tensor_pool(ref_path, device)
                pool_by_name = {name: load_tensor_pool(path, device) for name, path in paths.items()}
                data.x = reference
                model, base_acc, ref_embs, _ref_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:{run_args.token_compaction_reference_tag}] Acc: {base_acc:.4f}")
                log_important("[TokenCompactionPools] reference=" + ref_path)
                for name, path in paths.items():
                    log_important(f"  {name}: {path}")

                model.eval()
                for name, policy in configs:
                    with torch.no_grad():
                        embs = model.encoder(pool_by_name[name])
                    acc = evaluate_gnn_embeddings(model, data, embs)
                    drop = base_acc - acc
                    err = embedding_error(ref_embs, embs)
                    if policy["kind"] == "full":
                        cost = graph_eager_token_cost(-1, run_args)
                    else:
                        cost = graph_eager_token_cost(int(run_args.token_compaction_length), run_args)

                    results[name]["acc"].append(acc)
                    results[name]["drop"].append(drop)
                    results[name]["avg_err"].append(float(err.mean().item()))
                    results[name]["cost"].append(cost)
                    log_important(
                        f"[{name}] Cost={cost:.3f} | Acc={acc:.4f} | "
                        f"Drop={drop:.2%} | AvgErr={err.mean().item():.5f}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL TOKEN COMPACTION SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(f"Baseline Acc: {float(np.mean(results['baseline'])):.4f}")
            log_important("-" * 82)
            log_important(f"{'Config':<22} | {'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | {'AvgErr':<10}")
            log_important("-" * 82)
            for name, _policy in configs:
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                avg_err = float(np.mean(results[name]["avg_err"]))
                log_important(
                    f"{name:<22} | {cost:<8.3f} | {acc:<10.4f} | "
                    f"{drop:<10.2%} | {avg_err:<10.5f}"
                )
            log_important(f"{'=' * 72}\n")


def ffn_channel_gating_cost(keep_ratio, args):
    attn_weight = float(args.ffn_gating_attn_weight)
    ffn_weight = float(args.ffn_gating_ffn_weight)
    denom = max(1e-12, attn_weight + ffn_weight)
    return float(args.ffn_gating_cost_scale) * (attn_weight + ffn_weight * float(keep_ratio)) / denom


def select_ffn_gating_route_mask(policy_name, route_ratio, scores, num_nodes, seed, device):
    route_ratio = float(route_ratio)
    route_count = int(round(max(0.0, min(1.0, route_ratio)) * int(num_nodes)))
    mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    if route_count <= 0:
        return mask
    if route_count >= int(num_nodes):
        mask[:] = True
        return mask

    policy_name = str(policy_name).lower()
    if policy_name == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 6151)
        idx = torch.randperm(int(num_nodes), generator=generator, device="cpu")[:route_count].to(device=device)
    elif policy_name == "degree":
        idx = torch.argsort(scores["propagation_q"].to(device=device, dtype=torch.float32), descending=False)[:route_count]
    elif policy_name == "tser":
        idx = torch.argsort(scores["sensitivity_q"].to(device=device, dtype=torch.float32), descending=False)[:route_count]
    else:
        raise ValueError(f"Unknown FFN gating route policy: {policy_name}")
    mask[idx] = True
    return mask


def run_ffn_channel_gating_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "ffn_channel_gating")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        tags = [str(tag) for tag in args.ffn_gating_tags]
        keep_ratios = [float(ratio) for ratio in args.ffn_gating_keep_ratios]
        if args.ffn_gating_names is None:
            names = [tag.replace("W4A8_", "") for tag in tags]
        else:
            names = [str(name) for name in args.ffn_gating_names]

        configs = [("FullW4A8", {"tag": str(args.ffn_gating_full_tag), "keep_ratio": 1.0, "kind": "full"})]
        configs.extend(
            (name, {"tag": tag, "keep_ratio": keep_ratio, "kind": "uniform"})
            for name, tag, keep_ratio in zip(names, tags, keep_ratios)
        )
        for name, tag, keep_ratio in zip(names, tags, keep_ratios):
            for route_ratio in [float(value) for value in args.ffn_gating_route_ratios]:
                pct = int(round(route_ratio * 100))
                configs.extend(
                    [
                        (
                            f"Degree{pct}_{name}",
                            {
                                "tag": tag,
                                "keep_ratio": keep_ratio,
                                "kind": "routed",
                                "route_policy": "degree",
                                "route_ratio": route_ratio,
                            },
                        ),
                        (
                            f"TSER{pct}_{name}",
                            {
                                "tag": tag,
                                "keep_ratio": keep_ratio,
                                "kind": "routed",
                                "route_policy": "tser",
                                "route_ratio": route_ratio,
                            },
                        ),
                        (
                            f"Random{pct}_{name}",
                            {
                                "tag": tag,
                                "keep_ratio": keep_ratio,
                                "kind": "routed",
                                "route_policy": "random",
                                "route_ratio": route_ratio,
                            },
                        ),
                    ]
                )

        results = {
            name: {"acc": [], "drop": [], "avg_err": [], "cost": [], "keep_ratio": [], "gated_rate": []}
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running FFN Channel-Gating Validation on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[FFNGating] "
                f"model={args.real_quant_model_name} | reference={args.ffn_gating_reference_tag} "
                f"| full={args.ffn_gating_full_tag} | tags={', '.join(tags)} "
                f"| keep_ratios={', '.join(f'{ratio:.2f}' for ratio in keep_ratios)} "
                f"| cost_scale={args.ffn_gating_cost_scale:.2f} "
                f"| attn_weight={args.ffn_gating_attn_weight:.2f} "
                f"| ffn_weight={args.ffn_gating_ffn_weight:.2f}"
            )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)

                use_data_x_reference = str(run_args.ffn_gating_reference_tag).upper() in {"DATA_X", "DATAX", "CLEAN"}
                ref_path = (
                    "<data.x>"
                    if use_data_x_reference
                    else default_pool_path(ds_key, run_args.real_quant_model_name, run_args.ffn_gating_reference_tag)
                )
                paths = {
                    name: default_pool_path(ds_key, run_args.real_quant_model_name, policy["tag"])
                    for name, policy in configs
                }
                missing = [path for path in ([] if use_data_x_reference else [ref_path]) + list(paths.values()) if not os.path.exists(path)]
                if missing:
                    msg = "\n".join(f"  - {path}" for path in missing)
                    raise FileNotFoundError(
                        "FFN channel-gating pools are missing:\n"
                        f"{msg}\n"
                        "Generate them with generate_real_quant_pools --ffn_channel_gating --tag_suffix ..."
                    )

                reference = (
                    data.x.detach().to(device=device, dtype=torch.float32)
                    if use_data_x_reference
                    else load_tensor_pool(ref_path, device)
                )
                pool_by_name = {name: load_tensor_pool(path, device) for name, path in paths.items()}
                for name, pool in pool_by_name.items():
                    if pool.shape[1:] != reference.shape[1:]:
                        raise ValueError(
                            f"FFN gating reference shape {tuple(reference.shape)} does not match "
                            f"{name} pool shape {tuple(pool.shape)}. Use a matching reference tag, "
                            "for example FP16 for LLaMA pools or DATA_X for ST/data.x pools."
                        )
                data.x = reference
                model, base_acc, ref_embs, _ref_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:{run_args.ffn_gating_reference_tag}] Acc: {base_acc:.4f}")
                log_important("[FFNGatingPools] reference=" + ref_path)
                for name, path in paths.items():
                    log_important(f"  {name}: {path}")

                model.eval()
                with torch.no_grad():
                    encoded_by_name = {
                        name: model.encoder(pool)
                        for name, pool in pool_by_name.items()
                    }
                scores = build_real_quant_scores(verify_features, data, run_args, device)
                for name, policy in configs:
                    full_embs = encoded_by_name["FullW4A8"]
                    if policy["kind"] == "routed":
                        gated_embs = encoded_by_name[name]
                        route_mask = select_ffn_gating_route_mask(
                            policy["route_policy"],
                            policy["route_ratio"],
                            scores,
                            int(data.num_nodes),
                            seed,
                            device,
                        )
                        embs = full_embs.clone()
                        embs[route_mask] = gated_embs[route_mask]
                        gated_rate = float(route_mask.float().mean().item())
                        full_cost = ffn_channel_gating_cost(1.0, run_args)
                        gated_cost = ffn_channel_gating_cost(float(policy["keep_ratio"]), run_args)
                        cost = (1.0 - gated_rate) * full_cost + gated_rate * gated_cost
                    else:
                        embs = encoded_by_name[name]
                        gated_rate = 0.0 if policy["kind"] == "full" else 1.0
                        cost = ffn_channel_gating_cost(float(policy["keep_ratio"]), run_args)
                    acc = evaluate_gnn_embeddings(model, data, embs)
                    drop = base_acc - acc
                    err = embedding_error(ref_embs, embs)
                    keep_ratio = float(policy["keep_ratio"])

                    results[name]["acc"].append(acc)
                    results[name]["drop"].append(drop)
                    results[name]["avg_err"].append(float(err.mean().item()))
                    results[name]["cost"].append(cost)
                    results[name]["keep_ratio"].append(keep_ratio)
                    results[name]["gated_rate"].append(gated_rate)
                    log_important(
                        f"[{name}] Gate={gated_rate:.1%} | Keep={keep_ratio:.1%} | Cost={cost:.3f} | "
                        f"Acc={acc:.4f} | Drop={drop:.2%} | AvgErr={err.mean().item():.5f}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL FFN CHANNEL-GATING SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(f"Baseline Acc: {float(np.mean(results['baseline'])):.4f}")
            log_important("-" * 96)
            log_important(
                f"{'Config':<22} | {'Gate %':<8} | {'FFN Keep':<8} | {'Cost':<8} | "
                f"{'Acc':<10} | {'Drop %':<10} | {'AvgErr':<10}"
            )
            log_important("-" * 96)
            for name, _policy in configs:
                gated_rate = float(np.mean(results[name]["gated_rate"]))
                keep_ratio = float(np.mean(results[name]["keep_ratio"]))
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                avg_err = float(np.mean(results[name]["avg_err"]))
                log_important(
                    f"{name:<22} | {gated_rate:<8.1%} | {keep_ratio:<8.1%} | {cost:<8.3f} | "
                    f"{acc:<10.4f} | {drop:<10.2%} | {avg_err:<10.5f}"
                )
            log_important(f"{'=' * 72}\n")


def summarize_hierarchical_features(final_raw, reference_raw, model, data, base_acc, ref_embs):
    acc = evaluate_raw_node_features(model, data, final_raw)
    with torch.no_grad():
        final_embs = model.encoder(final_raw)
    err = embedding_error(ref_embs, final_embs)
    return {
        "acc": float(acc),
        "drop": float(base_acc - acc),
        "avg_err": float(err.mean().item()),
    }


def run_hierarchical_encoder_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "hierarchical_encoder")
    os.makedirs(log_dir, exist_ok=True)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)

        config_names = [
            "FullW4A8",
            "DirectReuse",
            "ResidualReuse",
            f"{args.hierarchical_gated_route_policy.title()}FFNGatingOnly",
            "FullHierarchy",
        ]
        results = {
            name: {
                "reuse": [],
                "direct": [],
                "residual": [],
                "gated": [],
                "full": [],
                "cost": [],
                "acc": [],
                "drop": [],
                "avg_err": [],
                "train_pairs": [],
                "alpha": [],
            }
            for name in config_names
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Hierarchical Encoder Validation on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[Hierarchy] "
                f"model={args.real_quant_model_name} | reference={args.hierarchical_reference_tag} "
                f"| full={args.hierarchical_full_tag} | gated={args.hierarchical_gated_tag} "
                f"| router_ref={args.hierarchical_router_reference} "
                f"| gated_keep={args.hierarchical_gated_keep_ratio:.2f} "
                f"| gated_route={args.hierarchical_gated_route_policy}:{args.hierarchical_gated_route_ratio:.1%} "
                f"| residual_cost={args.hierarchical_residual_cost:.4f}"
            )

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                clean_data_x = data.x.detach().to(device=device, dtype=torch.float32)

                use_data_x_reference = str(run_args.hierarchical_reference_tag).upper() in {"DATA_X", "DATAX", "CLEAN"}
                ref_path = (
                    "<data.x>"
                    if use_data_x_reference
                    else default_pool_path(ds_key, run_args.real_quant_model_name, run_args.hierarchical_reference_tag)
                )
                full_path = default_pool_path(ds_key, run_args.real_quant_model_name, run_args.hierarchical_full_tag)
                gated_path = default_pool_path(ds_key, run_args.real_quant_model_name, run_args.hierarchical_gated_tag)
                missing = [path for path in ((full_path, gated_path) if use_data_x_reference else (ref_path, full_path, gated_path)) if not os.path.exists(path)]
                if missing:
                    msg = "\n".join(f"  - {path}" for path in missing)
                    raise FileNotFoundError(
                        "Hierarchical encoder pools are missing:\n"
                        f"{msg}\n"
                        "Generate full/gated pools first with generate_real_quant_pools."
                    )

                reference_raw = (
                    data.x.detach().to(device=device, dtype=torch.float32)
                    if use_data_x_reference
                    else load_tensor_pool(ref_path, device)
                )
                full_raw = load_tensor_pool(full_path, device)
                gated_raw = load_tensor_pool(gated_path, device)
                for name, pool in (("full", full_raw), ("gated", gated_raw)):
                    if pool.shape[1:] != reference_raw.shape[1:]:
                        raise ValueError(
                            f"Hierarchical reference shape {tuple(reference_raw.shape)} does not match "
                            f"{name} pool shape {tuple(pool.shape)}. Use a matching reference tag, "
                            "for example FP16 for LLaMA pools or DATA_X for ST/data.x pools."
                        )

                data.x = reference_raw
                model, base_acc, ref_embs, oracle_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:{run_args.hierarchical_reference_tag}] Acc: {base_acc:.4f}")
                log_important(
                    "[HierarchyPools] "
                    f"reference={ref_path} | full={full_path} | gated={gated_path}"
                )

                model.eval()
                with torch.no_grad():
                    full_embs = model.encoder(full_raw)
                full_stats = {
                    "acc": evaluate_gnn_embeddings(model, data, full_embs),
                    "drop": base_acc - evaluate_gnn_embeddings(model, data, full_embs),
                    "avg_err": float(embedding_error(ref_embs, full_embs).mean().item()),
                }

                if str(run_args.hierarchical_router_reference) == "data_x":
                    data.x = clean_data_x
                    _router_model, router_acc, router_embs, router_logits = train_baseline_model(data, run_args, device)
                    data.x = reference_raw
                    log_important(
                        "[HierarchyRouter] reference=data_x "
                        f"| shape={tuple(clean_data_x.shape)} | route_acc={router_acc:.4f}"
                    )
                else:
                    router_embs = ref_embs
                    router_logits = oracle_logits
                    log_important(
                        "[HierarchyRouter] reference=execution_reference "
                        f"| shape={tuple(reference_raw.shape)}"
                    )

                route_bundle = build_route_bundle(
                    verify_features,
                    data,
                    router_embs,
                    router_logits,
                    run_args,
                    log_important,
                    device,
                )
                controller = build_controller(
                    data,
                    verify_features,
                    route_bundle,
                    {"name": "Hierarchy", "overrides": {}},
                    run_args,
                    device,
                )
                direct_raw, _hits = controller.query_full_batch(
                    route_bundle["hash_route_features"],
                    verify_features,
                    full_raw,
                )
                trace = controller.last_query_trace
                stats = controller.stats
                hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
                exact_mask = hit_mask & (torch.tensor([kind == "exact" for kind in trace["hit_kinds"]], device=device))
                fuzzy_mask = hit_mask & (torch.tensor([kind == "fuzzy" for kind in trace["hit_kinds"]], device=device))
                miss_mask = ~hit_mask

                correction_mask, correction_info = build_residual_correction_mask(
                    trace,
                    controller.risk_gate,
                    run_args.residual_direct_threshold,
                    device,
                    min_route_hits=run_args.residual_min_route_hits,
                    min_base_hits=run_args.residual_min_base_hits,
                )
                residual_fit_cfg = resolve_residual_fit_config(run_args, full_raw.size(1))
                log_residual_fit_config(log_important, residual_fit_cfg)
                adapter, train_info = train_residual_adapter(
                    target_embeddings=full_raw,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    data=data,
                    risk_scores=controller.node_risk_scores,
                    rank=residual_fit_cfg["rank"],
                    epochs=residual_fit_cfg["epochs"],
                    lr=run_args.residual_lr,
                    weight_decay=run_args.residual_weight_decay,
                    residual_l2=run_args.residual_l2,
                    train_split=run_args.residual_train_split,
                    max_pairs=residual_fit_cfg["max_pairs"],
                    correction_mask=correction_mask,
                    min_dist=run_args.residual_min_dist,
                    class_aware_accept=run_args.residual_class_aware_accept,
                    classifier_accept_gate=run_args.residual_classifier_accept_gate,
                    classifier_model=model,
                    classifier_reference_logits=oracle_logits,
                    classifier_accept_mode=run_args.residual_classifier_accept_mode,
                    classifier_accept_scope=run_args.residual_classifier_accept_scope,
                    classifier_accept_after_residual=run_args.residual_classifier_accept_after_residual,
                    classifier_accept_probe_alpha=run_args.residual_classifier_accept_probe_alpha,
                    classifier_accept_max_kl=run_args.residual_classifier_accept_max_kl,
                )

                if adapter is not None and float(run_args.residual_alpha) < 0.0:
                    alpha_grid = resolve_residual_alpha_grid(run_args, residual_fit_cfg)
                    selected_alpha = float(alpha_grid[0])
                    selected_val_acc = -1.0
                    for alpha in alpha_grid:
                        candidate_raw, _candidate_info = apply_residual_adapter(
                            direct_embeddings=direct_raw,
                            target_embeddings=full_raw,
                            verify_features=verify_features,
                            edge_index=data.edge_index,
                            trace=trace,
                            adapter=adapter,
                            risk_scores=controller.node_risk_scores,
                            alpha=alpha,
                            min_dist=run_args.residual_min_dist,
                            correction_mask=correction_mask,
                        )
                        val_acc = evaluate_raw_node_features(model, data, candidate_raw, mask=data.val_mask)
                        if val_acc > selected_val_acc + 1e-12:
                            selected_val_acc = val_acc
                            selected_alpha = alpha
                    alpha_for_apply = selected_alpha
                else:
                    alpha_for_apply = max(0.0, float(run_args.residual_alpha))

                residual_raw, apply_info = apply_residual_adapter(
                    direct_embeddings=direct_raw,
                    target_embeddings=full_raw,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    adapter=adapter,
                    risk_scores=controller.node_risk_scores,
                    alpha=alpha_for_apply,
                    min_dist=run_args.residual_min_dist,
                    correction_mask=correction_mask,
                )

                scores = build_real_quant_scores(verify_features, data, run_args, device)
                miss_indices = miss_mask.nonzero(as_tuple=False).view(-1)
                gated_miss_mask = torch.zeros(int(data.num_nodes), dtype=torch.bool, device=device)
                if miss_indices.numel() > 0 and float(run_args.hierarchical_gated_route_ratio) > 0:
                    route_ratio = float(run_args.hierarchical_gated_route_ratio)
                    select_count = int(round(route_ratio * int(miss_indices.numel())))
                    select_count = max(0, min(int(miss_indices.numel()), select_count))
                    if select_count > 0:
                        policy = str(run_args.hierarchical_gated_route_policy).lower()
                        if policy == "random":
                            generator = torch.Generator(device="cpu")
                            generator.manual_seed(int(seed) + 9323)
                            local = torch.randperm(int(miss_indices.numel()), generator=generator, device="cpu")[:select_count].to(device=device)
                            chosen = miss_indices[local]
                        elif policy == "degree":
                            local_score = scores["propagation_q"][miss_indices].to(dtype=torch.float32)
                            chosen = miss_indices[torch.argsort(local_score, descending=False)[:select_count]]
                        elif policy == "tser":
                            local_score = scores["sensitivity_q"][miss_indices].to(dtype=torch.float32)
                            chosen = miss_indices[torch.argsort(local_score, descending=False)[:select_count]]
                        else:
                            raise ValueError(f"Unknown hierarchical route policy: {policy}")
                        gated_miss_mask[chosen] = True

                full_cost = ffn_channel_gating_cost(1.0, run_args)
                gated_cost = ffn_channel_gating_cost(run_args.hierarchical_gated_keep_ratio, run_args)
                residual_cost = float(run_args.hierarchical_residual_cost)
                total_nodes = max(1, int(data.num_nodes))

                def record(name, raw, direct_rate=0.0, residual_rate=0.0, gated_rate=0.0, full_rate=0.0, train_pairs=0, alpha=0.0):
                    item = summarize_hierarchical_features(raw, reference_raw, model, data, base_acc, ref_embs)
                    cost = residual_rate * residual_cost + gated_rate * gated_cost + full_rate * full_cost
                    results[name]["reuse"].append(float(direct_rate + residual_rate))
                    results[name]["direct"].append(float(direct_rate))
                    results[name]["residual"].append(float(residual_rate))
                    results[name]["gated"].append(float(gated_rate))
                    results[name]["full"].append(float(full_rate))
                    results[name]["cost"].append(float(cost))
                    results[name]["acc"].append(item["acc"])
                    results[name]["drop"].append(item["drop"])
                    results[name]["avg_err"].append(item["avg_err"])
                    results[name]["train_pairs"].append(int(train_pairs))
                    results[name]["alpha"].append(float(alpha))
                    log_important(
                        f"[{name}] Reuse={direct_rate + residual_rate:.1%} "
                        f"(direct={direct_rate:.1%}, residual={residual_rate:.1%}) "
                        f"| FFNGated={gated_rate:.1%} | Full={full_rate:.1%} "
                        f"| Cost={cost:.3f} | Acc={item['acc']:.4f} "
                        f"| Drop={item['drop']:.2%} | AvgErr={item['avg_err']:.5f}"
                    )

                record(
                    "FullW4A8",
                    full_raw,
                    full_rate=1.0,
                )
                record(
                    "DirectReuse",
                    direct_raw,
                    direct_rate=float(hit_mask.float().mean().item()),
                    full_rate=float(miss_mask.float().mean().item()),
                )
                record(
                    "ResidualReuse",
                    residual_raw,
                    direct_rate=float(exact_mask.float().mean().item()),
                    residual_rate=float(fuzzy_mask.float().mean().item()),
                    full_rate=float(miss_mask.float().mean().item()),
                    train_pairs=int(train_info["train_pairs"]),
                    alpha=float(apply_info["alpha"]),
                )
                ffn_only_raw = full_raw.clone()
                ffn_only_raw[gated_miss_mask] = gated_raw[gated_miss_mask]
                record(
                    f"{run_args.hierarchical_gated_route_policy.title()}FFNGatingOnly",
                    ffn_only_raw,
                    gated_rate=float(gated_miss_mask.float().mean().item()),
                    full_rate=float((~gated_miss_mask).float().mean().item()),
                )
                hierarchy_raw = residual_raw.clone()
                hierarchy_raw[gated_miss_mask] = gated_raw[gated_miss_mask]
                hierarchy_full_mask = miss_mask & ~gated_miss_mask
                record(
                    "FullHierarchy",
                    hierarchy_raw,
                    direct_rate=float(exact_mask.float().mean().item()),
                    residual_rate=float(fuzzy_mask.float().mean().item()),
                    gated_rate=float(gated_miss_mask.float().mean().item()),
                    full_rate=float(hierarchy_full_mask.float().mean().item()),
                    train_pairs=int(train_info["train_pairs"]),
                    alpha=float(apply_info["alpha"]),
                )
                log_important(
                    "  ReuseDetail: "
                    f"reuse={stats['reuse']}/{stats['reuse_denominator']} "
                    f"(exact={stats['exact_reuse']}, fuzzy={stats['fuzzy_reuse']}) "
                    f"| computed={stats['computed']} "
                    f"| gated_miss={int(gated_miss_mask.sum().item())}/{int(miss_mask.sum().item())} "
                    f"| residual_candidates={correction_info['residual_candidates']} "
                    f"| corrected={apply_info['corrected']} | alpha={apply_info['alpha']:.3f}"
                )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL HIERARCHICAL ENCODER SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(f"Baseline Acc: {float(np.mean(results['baseline'])):.4f}")
            log_important("-" * 132)
            log_important(
                f"{'Config':<24} | {'Reuse %':<8} | {'Direct %':<8} | {'Residual %':<10} | "
                f"{'FFN %':<8} | {'Full %':<8} | {'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | {'AvgErr':<10}"
            )
            log_important("-" * 132)
            for name in config_names:
                log_important(
                    f"{name:<24} | "
                    f"{float(np.mean(results[name]['reuse'])):<8.1%} | "
                    f"{float(np.mean(results[name]['direct'])):<8.1%} | "
                    f"{float(np.mean(results[name]['residual'])):<10.1%} | "
                    f"{float(np.mean(results[name]['gated'])):<8.1%} | "
                    f"{float(np.mean(results[name]['full'])):<8.1%} | "
                    f"{float(np.mean(results[name]['cost'])):<8.3f} | "
                    f"{float(np.mean(results[name]['acc'])):<10.4f} | "
                    f"{float(np.mean(results[name]['drop'])):<10.2%} | "
                    f"{float(np.mean(results[name]['avg_err'])):<10.5f}"
                )
            log_important(f"{'=' * 72}\n")


def run_real_quant_ablation(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "graph_simhash")
    os.makedirs(log_dir, exist_ok=True)
    configs = build_real_quant_policy_configs(args)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        results = {
            name: {
                "int4": [],
                "int8": [],
                "fp": [],
                "acc": [],
                "drop": [],
                "avg_err": [],
                "cost": [],
            }
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Real Quantization Policy Ablation on {ds_key.upper()}")
            log_important(f"{'=' * 72}")

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                internal_bundle, internal_report = run_internal_split_calibration_step(
                    ds_key=ds_key,
                    data=data,
                    verify_features=verify_features,
                    args=run_args,
                    device=device,
                    dataset_log_dir=dataset_log_dir,
                    log_important=log_important,
                )
                if internal_bundle is not None:
                    run_args._internal_split_calibration_bundle = internal_bundle
                    run_args._internal_split_calibration_report = internal_report
                pools = load_real_quant_pools(ds_key, run_args, data, device)
                log_important(
                    "[RealQuantPools] "
                    f"FP={pools.fp_path} | INT8={pools.int8_path} | INT4={pools.int4_path}"
                )
                data.x = pools.fp
                log_important(
                    f"[RealQuantPools] shape={tuple(pools.fp.shape)} "
                    f"| error_space={run_args.real_quant_error_space} "
                    f"| error_norm={run_args.real_quant_error_norm} "
                    f"| fp_ratio={run_args.real_quant_fp_ratio:.2f} "
                    f"| int8_ratio={run_args.real_quant_int8_ratio:.2f} "
                    f"| tail={run_args.real_quant_tail_precision}"
                )

                model, base_acc, fp_embs, _oracle_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:AllFP] Acc: {base_acc:.4f}")

                model.eval()
                with torch.no_grad():
                    int8_embs = model.encoder(pools.int8)
                    int4_embs = model.encoder(pools.int4)

                if run_args.real_quant_error_space == "raw":
                    errors = compute_real_quant_errors(pools.fp, pools.int8, pools.int4, run_args)
                else:
                    errors = compute_real_quant_errors(fp_embs, int8_embs, int4_embs, run_args)

                scores = build_real_quant_scores(verify_features, data, run_args, device)
                log_important(
                    "[RealQuantScore] "
                    f"sensitivity min={scores['sensitivity_q'].float().min().item():.1f}, "
                    f"mean={scores['sensitivity_q'].float().mean().item():.1f}, "
                    f"max={scores['sensitivity_q'].float().max().item():.1f}"
                )
                log_important(
                    "[RealQuantError] "
                    f"INT8 mean={errors['int8_err'].mean().item():.5f}, "
                    f"max={errors['int8_err'].max().item():.5f} | "
                    f"INT4 mean={errors['int4_err'].mean().item():.5f}, "
                    f"max={errors['int4_err'].max().item():.5f}"
                )

                for name, policy in configs:
                    actions = select_real_quant_policy_actions(policy, scores, errors, run_args)
                    mixed_embs = assemble_real_quant_embeddings(actions, fp_embs, int8_embs, int4_embs)
                    acc = evaluate_gnn_embeddings(model, data, mixed_embs)
                    drop = base_acc - acc
                    stats = summarize_real_quant_policy(actions, errors, scores)
                    rel_cost = stats["fp_rate"] + 0.50 * stats["int8_rate"] + 0.25 * stats["int4_rate"]

                    results[name]["int4"].append(stats["int4_rate"])
                    results[name]["int8"].append(stats["int8_rate"])
                    results[name]["fp"].append(stats["fp_rate"])
                    results[name]["acc"].append(acc)
                    results[name]["drop"].append(drop)
                    results[name]["avg_err"].append(stats["avg_selected_error"])
                    results[name]["cost"].append(rel_cost)

                    log_important(
                        f"[{name}] {run_args.real_quant_int4_tag}={stats['int4_rate']:.1%} "
                        f"| {run_args.real_quant_int8_tag}={stats['int8_rate']:.1%} "
                        f"| FP={stats['fp_rate']:.1%} | Cost={rel_cost:.3f} "
                        f"| Acc={acc:.4f} | Drop={drop:.2%} "
                        f"| AvgErr={stats['avg_selected_error']:.5f}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL REAL QUANT SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            int4_header = f"{args.real_quant_int4_tag} %"
            int8_header = f"{args.real_quant_int8_tag} %"
            log_important("-" * 120)
            log_important(
                f"{'Config':<26} | {int4_header:<8} | {int8_header:<8} | {'FP %':<8} | "
                f"{'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | {'AvgErr':<10}"
            )
            log_important("-" * 120)
            for name, _policy in configs:
                i4 = float(np.mean(results[name]["int4"]))
                i8 = float(np.mean(results[name]["int8"]))
                fp = float(np.mean(results[name]["fp"]))
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                avg_err = float(np.mean(results[name]["avg_err"]))
                log_important(
                    f"{name:<26} | {i4:<8.1%} | {i8:<8.1%} | {fp:<8.1%} | "
                    f"{cost:<8.3f} | {acc:<10.4f} | {drop:<10.2%} | {avg_err:<10.5f}"
                )
            log_important(f"{'=' * 72}\n")


def run_reuse_real_quant_experiment(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "graph_simhash")
    os.makedirs(log_dir, exist_ok=True)
    configs = build_real_quant_policy_configs(args)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)
        results = {
            name: {
                "reuse": [],
                "reuse_num": [],
                "reuse_den": [],
                "int4": [],
                "int8": [],
                "fp": [],
                "cost": [],
                "acc": [],
                "drop": [],
                "final_err": [],
                "miss_err": [],
            }
            for name, _policy in configs
        }
        results["baseline"] = []

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 72}")
            log_important(f"Running Hash Reuse + Real Quant Feature Pools on {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            log_important(
                "[JointPolicy] Reuse hits are free cache reads; "
                "real-quant budgets are allocated only over hash-miss nodes."
            )
            if bool(getattr(args, "disable_real_quant_autogen", False)):
                log_important(
                    "[RealQuantAutoGen] disabled; using existing real-quant feature pools."
                )
            else:
                regenerate_real_quant_pools(ds_key, args, log_important)

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                run_args.enable_quant_policy = False
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                log_important(
                    f"[Seed] run={int(seed)} | controller={int(run_args.controller_seed)} "
                    f"| hash_head={int(run_args.hash_head_seed)} "
                    f"| topology_sketch={int(run_args.topology_sketch_seed)}"
                )

                internal_bundle, internal_report = run_internal_split_calibration_step(
                    ds_key=ds_key,
                    data=data,
                    verify_features=verify_features,
                    args=run_args,
                    device=device,
                    dataset_log_dir=dataset_log_dir,
                    log_important=log_important,
                )
                if internal_bundle is not None:
                    run_args._internal_split_calibration_bundle = internal_bundle
                    run_args._internal_split_calibration_report = internal_report

                pools = load_real_quant_pools(ds_key, run_args, data, device)
                log_important(
                    "[RealQuantPools] "
                    f"FP={pools.fp_path} | INT8={pools.int8_path} | INT4={pools.int4_path}"
                )
                log_important(
                    f"[RealQuantPools] shape={tuple(pools.fp.shape)} "
                    f"| error_space={run_args.real_quant_error_space} "
                    f"| error_norm={run_args.real_quant_error_norm} "
                    f"| miss_fp_ratio={run_args.real_quant_fp_ratio:.2f} "
                    f"| miss_int8_ratio={run_args.real_quant_int8_ratio:.2f} "
                    f"| tail={run_args.real_quant_tail_precision}"
                )

                data.x = pools.fp
                model, base_acc, fp_embs, oracle_logits = train_baseline_model(data, run_args, device)
                results["baseline"].append(base_acc)
                log_important(f"[Baseline:AllFP] Acc: {base_acc:.4f}")

                model.eval()
                with torch.no_grad():
                    int8_embs = model.encoder(pools.int8)
                    int4_embs = model.encoder(pools.int4)

                if run_args.real_quant_error_space == "raw":
                    errors = compute_real_quant_errors(pools.fp, pools.int8, pools.int4, run_args)
                else:
                    errors = compute_real_quant_errors(fp_embs, int8_embs, int4_embs, run_args)

                scores = build_real_quant_scores(verify_features, data, run_args, device)
                log_important(
                    "[RealQuantScore] "
                    f"sensitivity min={scores['sensitivity_q'].float().min().item():.1f}, "
                    f"mean={scores['sensitivity_q'].float().mean().item():.1f}, "
                    f"max={scores['sensitivity_q'].float().max().item():.1f}"
                )
                log_important(
                    "[RealQuantError] "
                    f"INT8 mean={errors['int8_err'].mean().item():.5f}, "
                    f"max={errors['int8_err'].max().item():.5f} | "
                    f"INT4 mean={errors['int4_err'].mean().item():.5f}, "
                    f"max={errors['int4_err'].max().item():.5f}"
                )

                route_bundle = build_route_bundle(
                    verify_features,
                    data,
                    fp_embs,
                    oracle_logits,
                    run_args,
                    log_important,
                    device,
                )

                trace_controller = build_controller(
                    data,
                    verify_features,
                    route_bundle,
                    {"name": "Trace", "overrides": {}},
                    run_args,
                    device,
                )
                _trace_embs, trace_hits = trace_controller.query_full_batch(
                    route_bundle["hash_route_features"],
                    verify_features,
                    fp_embs,
                )
                trace_stats = trace_controller.stats
                trace_reuse = trace_stats["reuse"] / max(1, trace_stats["total_queries"])
                log_important(
                    f"[ReuseTrace] Reuse={trace_reuse:.1%} "
                    f"| miss={(~trace_hits).float().mean().item():.1%} "
                    f"| reuse n/d={trace_stats['reuse']}/{trace_stats['reuse_denominator']}"
                )

                miss_mask = ~trace_hits
                for name, policy in configs:
                    actions = select_real_quant_policy_actions(
                        policy,
                        scores,
                        errors,
                        run_args,
                        eligible_mask=miss_mask,
                    )
                    selected_embs = assemble_real_quant_embeddings(actions, fp_embs, int8_embs, int4_embs)
                    controller = build_controller(
                        data,
                        verify_features,
                        route_bundle,
                        {"name": name, "overrides": {}},
                        run_args,
                        device,
                    )
                    result = evaluate_with_controller_real_quant(
                        model,
                        data,
                        controller,
                        route_bundle,
                        verify_features,
                        selected_embs,
                        actions,
                        errors,
                        fp_embs,
                    )
                    result["drop"] = base_acc - result["acc"]
                    hit_mismatch = int((result["hits"] != trace_hits).sum().item())
                    if hit_mismatch > 0:
                        log_important(
                            f"[{name}] Warning: reuse trace changed for {hit_mismatch} nodes; "
                            "using the policy run's actual hit mask for reporting."
                        )

                    results[name]["reuse"].append(result["reuse_rate"])
                    results[name]["reuse_num"].append(result["reuse_num"])
                    results[name]["reuse_den"].append(result["reuse_den"])
                    results[name]["int4"].append(result["int4_rate"])
                    results[name]["int8"].append(result["int8_rate"])
                    results[name]["fp"].append(result["fp_rate"])
                    results[name]["cost"].append(result["cost"])
                    results[name]["acc"].append(result["acc"])
                    results[name]["drop"].append(result["drop"])
                    results[name]["final_err"].append(result["final_avg_error"])
                    results[name]["miss_err"].append(result["miss_avg_selected_error"])

                    stats = result["stats"]
                    log_important(
                        f"[{name}] Reuse={result['reuse_rate']:.1%} "
                        f"| {run_args.real_quant_int4_tag}={result['int4_rate']:.1%} "
                        f"| {run_args.real_quant_int8_tag}={result['int8_rate']:.1%} "
                        f"| FP={result['fp_rate']:.1%} "
                        f"| Cost={result['cost']:.3f} "
                        f"| Acc={result['acc']:.4f} "
                        f"| Drop={result['drop']:.2%} "
                        f"| FinalErr={result['final_avg_error']:.5f} "
                        f"| MissErr={result['miss_avg_selected_error']:.5f}"
                    )
                    log_important(
                        f"  ReuseDetail: numerator={stats['reuse']} "
                        f"(exact={stats['exact_reuse']}, fuzzy={stats['fuzzy_reuse']}) "
                        f"/ denominator={stats['reuse_denominator']} "
                        f"| computed={stats['computed']} "
                        f"| exact_guarded={stats['exact_guarded']} "
                        f"| exact_guard_reject={stats['exact_guard_reject']} "
                        f"| struct_checked={stats['structure_checked']} "
                        f"| struct_reject={stats['structure_reject']} "
                        f"| score_checked={stats['score_checked']} "
                        f"| score_reject={stats['score_reject']} "
                        f"(hub={stats['score_reject_hub_protect']}, "
                        f"rare={stats['score_reject_rare_leaf']}, "
                        f"risk={stats['score_reject_risk']}) "
                        f"| avg_score_risk={stats['avg_score_risk']:.1f} "
                        f"| checks: reuse=exact+fuzzy {stats['reuse_consistency_ok']}, "
                        f"reuse+computed=total {stats['query_consistency_ok']}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL REUSE + REAL QUANT SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            int4_header = f"{args.real_quant_int4_tag} %"
            int8_header = f"{args.real_quant_int8_tag} %"
            log_important("-" * 140)
            log_important(
                f"{'Config':<26} | {'Reuse %':<9} | {int4_header:<8} | {int8_header:<8} | "
                f"{'FP %':<8} | {'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | "
                f"{'FinalErr':<10} | {'Reuse n/d':<15}"
            )
            log_important("-" * 140)
            for name, _policy in configs:
                reuse = float(np.mean(results[name]["reuse"]))
                reuse_num = float(np.mean(results[name]["reuse_num"]))
                reuse_den = float(np.mean(results[name]["reuse_den"]))
                int4 = float(np.mean(results[name]["int4"]))
                int8 = float(np.mean(results[name]["int8"]))
                fp = float(np.mean(results[name]["fp"]))
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                final_err = float(np.mean(results[name]["final_err"]))
                if args.runs > 1:
                    reuse_frac = f"{reuse_num:.1f}/{reuse_den:.1f}"
                else:
                    reuse_frac = f"{int(reuse_num)}/{int(reuse_den)}"
                log_important(
                    f"{name:<26} | {reuse:<9.1%} | {int4:<8.1%} | {int8:<8.1%} | "
                    f"{fp:<8.1%} | {cost:<8.3f} | {acc:<10.4f} | {drop:<10.2%} | "
                    f"{final_err:<10.5f} | {reuse_frac:<15}"
                )
            log_important(f"{'=' * 72}\n")


def run_adaptive_simulation(args):
    if bool(args.internal_split_calibration_only):
        return run_internal_split_calibration_only(args)

    if args.experiment_suite == "real_quant_ablation":
        return run_real_quant_ablation(args)
    if args.experiment_suite == "reuse_real_quant":
        return run_reuse_real_quant_experiment(args)
    if args.experiment_suite == "residual_reuse":
        return run_residual_reuse_experiment(args)
    if args.experiment_suite == "graph_eager_token":
        return run_graph_eager_token_experiment(args)
    if args.experiment_suite == "token_compaction":
        return run_token_compaction_experiment(args)
    if args.experiment_suite == "ffn_channel_gating":
        return run_ffn_channel_gating_experiment(args)
    if args.experiment_suite == "hierarchical_encoder":
        return run_hierarchical_encoder_experiment(args)
    if args.experiment_suite == "precision_depth_ablation":
        return run_precision_depth_ablation(args)
    if args.experiment_suite == "reuse_precision_depth":
        return run_reuse_precision_depth_experiment(args)
    if args.experiment_suite == "residual_precision_depth":
        return run_residual_precision_depth_experiment(args)

    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "graph_simhash")
    os.makedirs(log_dir, exist_ok=True)

    configs_to_test = build_adaptive_configs(args)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            continue

        dataset_log_dir = os.path.join(log_dir, ds_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        log_path = build_log_path(dataset_log_dir, ds_key, args)

        results_collector = {
            cfg["name"]: {
                "reuse": [],
                "reuse_num": [],
                "reuse_den": [],
                "int4": [],
                "int8": [],
                "full": [],
                "protected": [],
                "acc": [],
                "drop": [],
            }
            for cfg in configs_to_test
        }
        results_collector["baseline"] = []
        show_execution_breakdown = any(
            bool(cfg.get("overrides", {}).get("enable_quant_policy", args.enable_quant_policy))
            for cfg in configs_to_test
        )

        with open(log_path, "w", encoding="utf-8") as summary_file:
            def log_important(msg):
                print(msg)
                summary_file.write(msg + "\n")
                summary_file.flush()

            log_important(f"\n{'=' * 60}")
            log_important(f"Running Paper Hash Reuse on {ds_key.upper()}")
            log_important(f"{'=' * 60}")

            seeds = [int(args.seed) + run_idx for run_idx in range(args.runs)]
            for run_idx, seed in enumerate(seeds):
                log_important(f"\n--- Run {run_idx + 1}/{args.runs} (Seed {seed}) ---")
                run_args = make_run_args(args, seed)
                _conf, data, verify_features, device = load_run_state(ds_key, run_args, seed)
                log_important(
                    f"[Seed] run={int(seed)} | controller={int(run_args.controller_seed)} "
                    f"| hash_head={int(run_args.hash_head_seed)} "
                    f"| topology_sketch={int(run_args.topology_sketch_seed)}"
                )
                run_internal_split_calibration_step(
                    ds_key=ds_key,
                    data=data,
                    verify_features=verify_features,
                    args=run_args,
                    device=device,
                    dataset_log_dir=dataset_log_dir,
                    log_important=log_important,
                )
                model, base_acc, oracle_embs, oracle_logits = train_baseline_model(data, run_args, device)
                results_collector["baseline"].append(base_acc)
                log_important(f"[Baseline] Acc: {base_acc:.4f}")

                route_bundle = build_route_bundle(
                    verify_features,
                    data,
                    oracle_embs,
                    oracle_logits,
                    run_args,
                    log_important,
                    device,
                )

                for cfg in configs_to_test:
                    result = run_single_config(
                        model,
                        data,
                        verify_features,
                        oracle_embs,
                        route_bundle,
                        cfg,
                        run_args,
                        device,
                        log_important=log_important,
                    )
                    result["drop"] = base_acc - result["acc"]

                    results_collector[cfg["name"]]["reuse"].append(result["reuse_rate"])
                    results_collector[cfg["name"]]["reuse_num"].append(result["reuse_num"])
                    results_collector[cfg["name"]]["reuse_den"].append(result["reuse_den"])
                    results_collector[cfg["name"]]["int4"].append(result["int4_rate"])
                    results_collector[cfg["name"]]["int8"].append(result["int8_rate"])
                    results_collector[cfg["name"]]["full"].append(result["full_rate"])
                    results_collector[cfg["name"]]["protected"].append(result["protected_rate"])
                    results_collector[cfg["name"]]["acc"].append(result["acc"])
                    results_collector[cfg["name"]]["drop"].append(result["drop"])

                    stats = result["stats"]
                    if show_execution_breakdown:
                        log_important(
                            f"[{cfg['name']}] Reuse={result['reuse_rate']:.1%} "
                            f"| I4={result['int4_rate']:.1%} "
                            f"| I8={result['int8_rate']:.1%} "
                            f"| Full={result['full_rate']:.1%} "
                            f"| Prot={result['protected_rate']:.1%} "
                            f"| Acc={result['acc']:.4f} "
                            f"| Drop={result['drop']:.2%}"
                        )
                    else:
                        log_important(
                            f"[{cfg['name']}] Reuse={result['reuse_rate']:.1%} "
                            f"| Acc={result['acc']:.4f} "
                            f"| Drop={result['drop']:.2%}"
                        )
                    detail_msg = (
                        f"  ReuseDetail: numerator={stats['reuse']} "
                        f"(exact={stats['exact_reuse']}, fuzzy={stats['fuzzy_reuse']}) "
                        f"/ denominator={stats['reuse_denominator']} "
                        f"| computed={stats['computed']} "
                        f"| exact_guarded={stats['exact_guarded']} "
                        f"| exact_guard_reject={stats['exact_guard_reject']} "
                        f"| struct_checked={stats['structure_checked']} "
                        f"| struct_reject={stats['structure_reject']} "
                        f"| score_checked={stats['score_checked']} "
                        f"| score_reject={stats['score_reject']} "
                        f"(hub={stats['score_reject_hub_protect']}, "
                        f"rare={stats['score_reject_rare_leaf']}, "
                        f"risk={stats['score_reject_risk']}) "
                        f"| avg_score_risk={stats['avg_score_risk']:.1f} "
                    )
                    if show_execution_breakdown:
                        detail_msg += (
                            f"| exec=int4:{stats['quant_int4']}, "
                            f"int8:{stats['quant_int8']}, "
                            f"full:{stats['full_precision']}, "
                            f"protected:{stats['protected']} "
                            f"| avg_quant_risk={stats['avg_quant_risk']:.1f} "
                        )
                    detail_msg += (
                        f"| checks: reuse=exact+fuzzy {stats['reuse_consistency_ok']}, "
                        f"reuse+computed=total {stats['query_consistency_ok']}"
                    )
                    if show_execution_breakdown:
                        detail_msg += f", exec_total=total {stats['execution_consistency_ok']}"
                    log_important(detail_msg)

            log_important(f"\n{'=' * 60}")
            log_important(f"FINAL SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 60}")
            base_mean = float(np.mean(results_collector["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            summary_width = 120 if show_execution_breakdown else 60
            log_important("-" * summary_width)
            if show_execution_breakdown:
                log_important(
                    f"{'Config':<24} | {'Reuse %':<9} | {'I4 %':<8} | {'I8 %':<8} | "
                    f"{'Full %':<8} | {'Prot %':<8} | {'Acc':<10} | {'Drop %':<10} | {'Reuse n/d':<15}"
                )
            else:
                log_important(f"{'Config':<20} | {'Reuse %':<10} | {'Reuse n/d':<15} | {'Acc':<10} | {'Drop %':<10}")
            log_important("-" * summary_width)

            for cfg in configs_to_test:
                r_mean = float(np.mean(results_collector[cfg["name"]]["reuse"]))
                rn_mean = float(np.mean(results_collector[cfg["name"]]["reuse_num"]))
                rd_mean = float(np.mean(results_collector[cfg["name"]]["reuse_den"]))
                i4_mean = float(np.mean(results_collector[cfg["name"]]["int4"]))
                i8_mean = float(np.mean(results_collector[cfg["name"]]["int8"]))
                full_mean = float(np.mean(results_collector[cfg["name"]]["full"]))
                prot_mean = float(np.mean(results_collector[cfg["name"]]["protected"]))
                a_mean = float(np.mean(results_collector[cfg["name"]]["acc"]))
                d_mean = float(np.mean(results_collector[cfg["name"]]["drop"]))
                if args.runs > 1:
                    reuse_frac = f"{rn_mean:.1f}/{rd_mean:.1f}"
                else:
                    reuse_frac = f"{int(rn_mean)}/{int(rd_mean)}"
                if show_execution_breakdown:
                    log_important(
                        f"{cfg['name']:<24} | {r_mean:<9.1%} | {i4_mean:<8.1%} | {i8_mean:<8.1%} | "
                        f"{full_mean:<8.1%} | {prot_mean:<8.1%} | {a_mean:<10.4f} | {d_mean:<10.2%} | {reuse_frac:<15}"
                    )
                else:
                    log_important(
                        f"{cfg['name']:<20} | {r_mean:<10.1%} | {reuse_frac:<15} | {a_mean:<10.4f} | {d_mean:<10.2%}"
                    )
            log_important(f"{'=' * 60}\n")
