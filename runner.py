import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .config import DATASET_CONFIGS
from .controller import PaperHashReuseController
from .data import load_run_state, maybe_limit_test_mask
from .features import build_hash_feature_routes, build_topology_hash_features, format_hash_route_specs
from .models import GNN_LLM_Model
from .projections import fit_multihead_hash_projection
from .real_quant import (
    assemble_real_quant_embeddings,
    build_real_quant_scores,
    compute_real_quant_errors,
    load_real_quant_pools,
    select_real_quant_policy_actions,
    summarize_real_quant_policy,
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


def evaluate_gnn_embeddings(model, data, node_embs):
    with torch.no_grad():
        out = model.forward_gnn_only(
            node_embs,
            data.edge_index,
            data.edge_type,
            data.edge_attr,
        )
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return float(acc)


def run_real_quant_ablation(args):
    target_datasets = args.datasets if args.datasets else ["cora"]
    log_dir = os.path.join("output", "graph_simhash")
    os.makedirs(log_dir, exist_ok=True)
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
                        f"[{name}] I4={stats['int4_rate']:.1%} | I8={stats['int8_rate']:.1%} "
                        f"| FP={stats['fp_rate']:.1%} | Cost={rel_cost:.3f} "
                        f"| Acc={acc:.4f} | Drop={drop:.2%} "
                        f"| AvgErr={stats['avg_selected_error']:.5f}"
                    )

            log_important(f"\n{'=' * 72}")
            log_important(f"FINAL REAL QUANT SUMMARY ({args.runs} Runs) | {ds_key.upper()}")
            log_important(f"{'=' * 72}")
            base_mean = float(np.mean(results["baseline"]))
            log_important(f"Baseline Acc: {base_mean:.4f}")
            log_important("-" * 104)
            log_important(
                f"{'Config':<18} | {'I4 %':<8} | {'I8 %':<8} | {'FP %':<8} | "
                f"{'Cost':<8} | {'Acc':<10} | {'Drop %':<10} | {'AvgErr':<10}"
            )
            log_important("-" * 104)
            for name, _policy in configs:
                i4 = float(np.mean(results[name]["int4"]))
                i8 = float(np.mean(results[name]["int8"]))
                fp = float(np.mean(results[name]["fp"]))
                cost = float(np.mean(results[name]["cost"]))
                acc = float(np.mean(results[name]["acc"]))
                drop = float(np.mean(results[name]["drop"]))
                avg_err = float(np.mean(results[name]["avg_err"]))
                log_important(
                    f"{name:<18} | {i4:<8.1%} | {i8:<8.1%} | {fp:<8.1%} | "
                    f"{cost:<8.3f} | {acc:<10.4f} | {drop:<10.2%} | {avg_err:<10.5f}"
                )
            log_important(f"{'=' * 72}\n")


def run_adaptive_simulation(args):
    if args.experiment_suite == "real_quant_ablation":
        return run_real_quant_ablation(args)

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
