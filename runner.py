import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .config import DATASET_CONFIGS
from .controller import PaperHashReuseController
from .data import load_run_state, maybe_limit_test_mask
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
    load_real_quant_pools,
    regenerate_real_quant_pools,
    select_real_quant_policy_actions,
    summarize_real_quant_policy,
)
from .residual_reuse import (
    apply_residual_adapter,
    embedding_error,
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
        out = model.forward_gnn_only(
            node_embs,
            data.edge_index,
            data.edge_type,
            data.edge_attr,
        )
        pred = out.argmax(dim=1)
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
    return float(acc)


def evaluate_raw_node_features(model, data, raw_features, mask=None):
    with torch.no_grad():
        hidden = model.encoder(raw_features)
    return evaluate_gnn_embeddings(model, data, hidden, mask=mask)


def build_residual_correction_mask(trace, risk_gate, direct_threshold, device):
    hit_mask = trace["hit_mask"].to(device=device, dtype=torch.bool)
    source_ok = trace["source_ids"].to(device=device) >= 0
    correction_mask = hit_mask & source_ok
    if float(direct_threshold) < 0.0 or risk_gate is None:
        return correction_mask, {
            "direct_threshold": float(direct_threshold),
            "direct_low_risk": 0,
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
            "reuse": [],
            "reuse_num": [],
            "reuse_den": [],
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
                f"| T_direct={'none' if float(args.residual_direct_threshold) < 0.0 else float(args.residual_direct_threshold)} "
                f"| anchor={args.residual_anchor_mode}"
            )

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

                target_features = data.x.detach()
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
                )

                direct_acc = evaluate_raw_node_features(model, data, direct_features)
                direct_drop = base_acc - direct_acc
                direct_err = embedding_error(target_features, direct_features)

                adapter, train_info = train_residual_adapter(
                    target_embeddings=target_features,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    data=data,
                    risk_scores=controller.node_risk_scores,
                    rank=args.residual_rank,
                    epochs=args.residual_epochs,
                    lr=args.residual_lr,
                    weight_decay=args.residual_weight_decay,
                    residual_l2=args.residual_l2,
                    train_split=args.residual_train_split,
                    max_pairs=args.residual_max_train_pairs,
                    correction_mask=correction_mask,
                )

                if adapter is not None and float(args.residual_alpha) < 0.0:
                    selected_alpha = float(args.residual_alpha_grid[0])
                    selected_val_acc = -1.0
                    for alpha in sorted(float(value) for value in args.residual_alpha_grid):
                        candidate_features, _candidate_info = apply_residual_adapter(
                            direct_embeddings=direct_features,
                            target_embeddings=target_features,
                            verify_features=verify_features,
                            edge_index=data.edge_index,
                            trace=trace,
                            adapter=adapter,
                            risk_scores=controller.node_risk_scores,
                            alpha=alpha,
                            min_dist=args.residual_min_dist,
                            correction_mask=correction_mask,
                        )
                        val_acc = evaluate_raw_node_features(model, data, candidate_features, mask=data.val_mask)
                        if val_acc > selected_val_acc + 1e-12:
                            selected_val_acc = val_acc
                            selected_alpha = alpha
                    alpha_for_apply = selected_alpha
                    alpha_note = f"auto(val_acc={selected_val_acc:.4f})"
                else:
                    alpha_for_apply = max(0.0, float(args.residual_alpha))
                    alpha_note = "fixed"

                residual_features, apply_info = apply_residual_adapter(
                    direct_embeddings=direct_features,
                    target_embeddings=target_features,
                    verify_features=verify_features,
                    edge_index=data.edge_index,
                    trace=trace,
                    adapter=adapter,
                    risk_scores=controller.node_risk_scores,
                    alpha=alpha_for_apply,
                    min_dist=args.residual_min_dist,
                    correction_mask=correction_mask,
                )
                residual_acc = evaluate_raw_node_features(model, data, residual_features)
                residual_drop = base_acc - residual_acc
                residual_err = embedding_error(target_features, residual_features)

                hit_den = max(1, int(hit_mask.sum().item()))
                direct_hit_err = float(direct_err[hit_mask].mean().item()) if hit_den > 0 else 0.0
                residual_hit_err = float(residual_err[hit_mask].mean().item()) if hit_den > 0 else 0.0
                reuse_rate = stats["reuse"] / max(1, stats["total_queries"])

                results["direct_acc"].append(direct_acc)
                results["direct_drop"].append(direct_drop)
                results["residual_acc"].append(residual_acc)
                results["residual_drop"].append(residual_drop)
                results["reuse"].append(float(reuse_rate))
                results["reuse_num"].append(int(stats["reuse"]))
                results["reuse_den"].append(int(stats["reuse_denominator"]))
                results["train_pairs"].append(int(train_info["train_pairs"]))
                results["direct_err"].append(float(direct_err.mean().item()))
                results["direct_hit_err"].append(direct_hit_err)
                results["residual_err"].append(float(residual_err.mean().item()))
                results["residual_hit_err"].append(residual_hit_err)
                results["residual_alpha"].append(float(apply_info["alpha"]))

                log_important(
                    f"[DirectReuse] Reuse={reuse_rate:.1%} "
                    f"| AnchorMode={args.residual_anchor_mode} "
                    f"| Randomized={anchor_info['randomized']} "
                    f"| Acc={direct_acc:.4f} | Drop={direct_drop:.2%} "
                    f"| AvgErr={float(direct_err.mean().item()):.5f} "
                    f"| HitErr={direct_hit_err:.5f}"
                )
                log_important(
                    f"[ResidualReuse] Corrected={apply_info['corrected']} "
                    f"| DirectLowRisk={correction_info['direct_low_risk']} "
                    f"| ResidualCand={correction_info['residual_candidates']} "
                    f"| TrainPairs={train_info['train_pairs']} "
                    f"| Alpha={apply_info['alpha']:.3f} ({alpha_note}) "
                    f"| TrainLoss={train_info['loss']:.6f} "
                    f"| Acc={residual_acc:.4f} | Drop={residual_drop:.2%} "
                    f"| AvgErr={float(residual_err.mean().item()):.5f} "
                    f"| HitErr={residual_hit_err:.5f}"
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
            reuse_mean = float(np.mean(results["reuse"]))
            reuse_num = float(np.mean(results["reuse_num"]))
            reuse_den = float(np.mean(results["reuse_den"]))
            reuse_frac = f"{reuse_num:.1f}/{reuse_den:.1f}" if args.runs > 1 else f"{int(reuse_num)}/{int(reuse_den)}"
            train_pairs = float(np.mean(results["train_pairs"]))
            log_important(
                f"{'DirectReuse':<18} | {reuse_mean:<9.1%} | {'-':<10} | "
                f"{float(np.mean(results['direct_acc'])):<10.4f} | "
                f"{float(np.mean(results['direct_drop'])):<10.2%} | "
                f"{float(np.mean(results['direct_err'])):<10.5f} | "
                f"{float(np.mean(results['direct_hit_err'])):<10.5f} | {'-':<7} | {reuse_frac:<15}"
            )
            log_important(
                f"{'ResidualReuse':<18} | {reuse_mean:<9.1%} | {train_pairs:<10.1f} | "
                f"{float(np.mean(results['residual_acc'])):<10.4f} | "
                f"{float(np.mean(results['residual_drop'])):<10.2%} | "
                f"{float(np.mean(results['residual_err'])):<10.5f} | "
                f"{float(np.mean(results['residual_hit_err'])):<10.5f} | "
                f"{float(np.mean(results['residual_alpha'])):<7.3f} | {reuse_frac:<15}"
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
