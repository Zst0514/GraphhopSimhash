import hashlib
import os

def list_hash_route_tags(hash_view, union_hash_views=None):
    tags = []
    seen = set()
    for route_tag in [hash_view] + list(union_hash_views or []):
        if route_tag in seen:
            continue
        tags.append(route_tag)
        seen.add(route_tag)
    return tags


def list_retrieval_route_tags(hash_view, union_hash_views=None, include_topology=False):
    tags = list_hash_route_tags(hash_view, union_hash_views)
    if include_topology and "topology" not in tags:
        tags.append("topology")
    return tags


def make_hash_route_tag(hash_view, union_hash_views=None, include_topology=False):
    return "__union__".join(
        list_retrieval_route_tags(
            hash_view,
            union_hash_views,
            include_topology=include_topology,
        )
    )


def make_tau_tag(cosine_tau):
    return f"tau_{cosine_tau:.2f}".replace(".", "p")


def make_projection_tag(args):
    if not args.learned_hash_projection:
        return "proj_raw"
    base_tag = f"proj_mhh{int(args.learned_hash_dim)}"
    if args.shared_hash_projection_heads:
        base_tag += "_shared"
    if args.main_hash_head_bits is not None:
        base_tag += "_mhb" + "-".join(str(int(bit)) for bit in args.main_hash_head_bits)
    if args.union_hash_views and args.union_hash_head_bits is not None:
        base_tag += "_uhb" + "-".join(str(int(bit)) for bit in args.union_hash_head_bits)
    if args.enable_topology_retrieval_route and args.topology_hash_head_bits is not None:
        base_tag += "_thb" + "-".join(str(int(bit)) for bit in args.topology_hash_head_bits)
    elif args.hash_head_bits is not None:
        bits = [int(bit) for bit in args.hash_head_bits]
        if len(bits) == 1 and args.hash_heads_per_route > 1:
            bits = bits * int(args.hash_heads_per_route)
        base_tag += "_hb" + "-".join(str(bit) for bit in bits)
    else:
        base_tag += f"_tbl{int(args.hash_heads_per_route)}"
    return base_tag


def resolve_route_score_weights(route_tags, configured_weights=None, default_union_weight=0.94):
    if configured_weights is not None:
        if len(configured_weights) != len(route_tags):
            raise ValueError(
                f"route_score_weights expects {len(route_tags)} values, got {len(configured_weights)}"
            )
        weights = [float(weight) for weight in configured_weights]
    else:
        weights = [1.0] + [float(default_union_weight)] * max(0, len(route_tags) - 1)

    if any(weight <= 0.0 for weight in weights):
        raise ValueError("route score weights must be positive")
    return weights


def resolve_route_accept_tau_offsets(route_tags, configured_offsets=None, default_union_bonus=0.0):
    if configured_offsets is not None:
        if len(configured_offsets) != len(route_tags):
            raise ValueError(
                f"route_accept_tau_offsets expects {len(route_tags)} values, got {len(configured_offsets)}"
            )
        return [float(offset) for offset in configured_offsets]
    return [0.0] + [float(default_union_bonus)] * max(0, len(route_tags) - 1)


def resolve_route_min_accept_votes(route_tags, configured_votes=None, default_union_votes=1):
    if configured_votes is not None:
        if len(configured_votes) != len(route_tags):
            raise ValueError(
                f"route_min_accept_votes expects {len(route_tags)} values, got {len(configured_votes)}"
            )
        votes = [int(vote) for vote in configured_votes]
    else:
        votes = [1] + [int(default_union_votes)] * max(0, len(route_tags) - 1)

    if any(vote < 1 for vote in votes):
        raise ValueError("route_min_accept_votes values must be at least 1")
    return votes


def resolve_route_min_support_hits(
    route_tags,
    configured_hits=None,
    default_main_hits=2,
    default_union_hits=1,
):
    if configured_hits is not None:
        if len(configured_hits) == 1 and len(route_tags) > 1:
            hits = [int(configured_hits[0])] + [int(default_union_hits)] * max(0, len(route_tags) - 1)
        elif len(configured_hits) == 1 and len(route_tags) == 1:
            hits = [int(configured_hits[0])]
        elif len(configured_hits) != len(route_tags):
            raise ValueError(
                f"route_min_support_hits expects 1 or {len(route_tags)} values, got {len(configured_hits)}"
            )
        else:
            hits = [int(hit) for hit in configured_hits]
    else:
        hits = [int(default_main_hits)] + [int(default_union_hits)] * max(0, len(route_tags) - 1)

    if any(hit < 1 for hit in hits):
        raise ValueError("route_min_support_hits values must be at least 1")
    return hits


def expand_route_values_by_base_specs(route_specs, base_values):
    expanded = []
    for spec in route_specs:
        base_route_idx = int(spec.get("base_route_idx", 0))
        if base_route_idx < 0 or base_route_idx >= len(base_values):
            raise ValueError(
                f"base_route_idx {base_route_idx} is out of range for {len(base_values)} base route values"
            )
        expanded.append(base_values[base_route_idx])
    return expanded


def apply_table_weight_decay(route_specs, expanded_weights, table_weight_decay=1.0):
    adjusted = []
    decay = float(table_weight_decay)
    for spec, weight in zip(route_specs, expanded_weights):
        table_idx = int(spec.get("table_idx", 0))
        adjusted.append(float(weight) * (decay ** max(0, table_idx)))
    return adjusted


def make_route_weight_tag(route_tags, route_score_weights, union_route_weight):
    weights = resolve_route_score_weights(
        route_tags,
        configured_weights=route_score_weights,
        default_union_weight=union_route_weight,
    )
    return "rw_" + "__".join(f"{weight:.2f}".replace(".", "p") for weight in weights)


def make_route_accept_tag(
    route_tags,
    route_accept_tau_offsets,
    union_accept_tau_bonus,
    route_min_accept_votes,
    union_min_accept_votes,
    route_min_support_hits,
    union_min_support_hits,
):
    tau_offsets = resolve_route_accept_tau_offsets(
        route_tags,
        configured_offsets=route_accept_tau_offsets,
        default_union_bonus=union_accept_tau_bonus,
    )
    min_votes = resolve_route_min_accept_votes(
        route_tags,
        configured_votes=route_min_accept_votes,
        default_union_votes=union_min_accept_votes,
    )
    min_support_hits = resolve_route_min_support_hits(
        route_tags,
        configured_hits=route_min_support_hits,
        default_main_hits=2,
        default_union_hits=union_min_support_hits,
    )
    parts = []
    for tau_offset, min_vote, min_support_hit in zip(tau_offsets, min_votes, min_support_hits):
        parts.append(f"{tau_offset:.2f}".replace(".", "p") + f"v{min_vote}s{min_support_hit}")
    return "ra_" + "__".join(parts)


def make_retrieval_tag(args):
    parts = [
        f"cand{int(args.max_candidates_per_route)}x{int(args.max_total_candidates)}",
        f"chk{('auto' if args.max_structure_checks is None else int(args.max_structure_checks))}",
        f"twd{args.table_route_weight_decay:.2f}".replace(".", "p"),
        f"cub{('none' if args.coarse_union_bits_max is None else int(args.coarse_union_bits_max))}",
        f"sm{args.structure_check_mode}",
        f"honly{int(bool(args.hamming_only_acceptor))}",
        f"tret{int(bool(args.enable_topology_retrieval_route))}",
        f"mbr{int(args.min_base_route_hits)}",
        f"struct{int(not bool(args.disable_structure_check))}",
    ]
    if args.structure_check_mode == "sketch":
        parts.append(f"hbg{int(bool(args.enable_homophily_bucket_guard))}")
        parts.append(f"tsg{int(bool(args.enable_topology_sketch_guard))}")
        if args.enable_topology_sketch_guard:
            parts.extend(
                [
                    f"tsb{int(args.topology_sketch_bits)}",
                    f"tsr{int(args.topology_sketch_radius)}",
                ]
            )
        parts.extend(
            [
                f"tdg{('none' if args.topology_degree_bucket_gap is None else int(args.topology_degree_bucket_gap))}",
                f"thb{int(args.topology_homophily_bins)}",
                f"thg{int(args.topology_homophily_bucket_gap)}",
            ]
        )
    return "_".join(parts)


def build_adaptive_configs(args):
    return [{"name": f"R{int(args.radius)}"}]


def build_log_path(dataset_log_dir, ds_key, args):
    route_tags = list_retrieval_route_tags(
        args.hash_view,
        args.union_hash_views,
        include_topology=args.enable_topology_retrieval_route,
    )
    hash_route_tag = make_hash_route_tag(
        args.hash_view,
        args.union_hash_views,
        include_topology=args.enable_topology_retrieval_route,
    )
    projection_tag = make_projection_tag(args)
    route_weight_tag = make_route_weight_tag(route_tags, args.route_score_weights, args.union_route_weight)
    route_accept_tag = make_route_accept_tag(
        route_tags,
        args.route_accept_tau_offsets,
        args.union_accept_tau_bonus,
        args.route_min_accept_votes,
        args.union_min_accept_votes,
        args.route_min_support_hits,
        args.union_min_support_hits,
    )
    retrieval_tag = make_retrieval_tag(args)
    tau_tag = make_tau_tag(args.cosine_tau)
    config_tag = f"r{int(args.radius)}"
    if getattr(args, "experiment_suite", None) in ("real_quant_ablation", "reuse_real_quant"):
        if getattr(args, "experiment_suite", None) == "reuse_real_quant":
            config_tag += "_reuse"
        config_tag += (
            f"_rq{getattr(args, 'real_quant_policy_suite', 'standard')}"
            f"_{getattr(args, 'real_quant_fp_tag', 'FP')}"
            f"_{getattr(args, 'real_quant_int8_tag', 'INT8')}"
            f"_{getattr(args, 'real_quant_int4_tag', 'INT4')}"
            f"_i8r{float(getattr(args, 'real_quant_int8_ratio', 0.0)):.2f}".replace(".", "p")
        )
        if bool(getattr(args, "internal_split_calibration", False)):
            config_tag += (
                f"_is{getattr(args, 'internal_split_priority', 'degree')}"
                f"{float(getattr(args, 'internal_split_topk_ratio', 0.0)):.2f}".replace(".", "p")
            )
        config_tag += (
            f"_qts{float(getattr(args, 'quant_tser_propagation_weight', 4.0)):.1f}"
            f"-{float(getattr(args, 'quant_tser_graph_context_weight', 1.0)):.1f}"
            f"-{float(getattr(args, 'quant_tser_low_unique_weight', 0.0)):.1f}"
            f"_qeb{float(getattr(args, 'quant_error_bias', 1.0)):.1f}"
            f"_{getattr(args, 'quant_error_rank_source', 'continuous')}"
        ).replace(".", "p")
    if getattr(args, "experiment_suite", None) == "residual_reuse":
        embedding_tag = getattr(args, "residual_embedding_source", "data_x")
        if embedding_tag == "real_quant_fp":
            embedding_tag = (
                f"{getattr(args, 'real_quant_model_name', 'model')}"
                f"_{getattr(args, 'real_quant_fp_tag', 'fp')}"
            )
        config_tag += (
            f"_resr{int(getattr(args, 'residual_rank', 32))}"
            f"_e{int(getattr(args, 'residual_epochs', 200))}"
            f"_md{float(getattr(args, 'residual_min_dist', 1.0)):.1f}"
            f"_td{float(getattr(args, 'residual_direct_threshold', -1.0)):.1f}"
            f"_{getattr(args, 'residual_anchor_mode', 'cam')}"
            f"_{embedding_tag}"
            f"_mx{int(getattr(args, 'residual_max_train_pairs', 4096))}"
        ).replace(".", "p").replace("-", "m")
    if getattr(args, "experiment_suite", None) == "graph_eager_token":
        lengths_tag = "-".join(
            str(int(length)) for length in getattr(args, "graph_eager_token_lengths", [128, 256])
        )
        config_tag += (
            f"_get{getattr(args, 'real_quant_model_name', 'model')}"
            f"_{getattr(args, 'graph_eager_reference_tag', 'ref')}"
            f"_{getattr(args, 'graph_eager_full_tag', 'full')}"
            f"_{getattr(args, 'graph_eager_token_tag_prefix', 'tok')}"
            f"_S{lengths_tag}"
            f"_fl{int(getattr(args, 'graph_eager_full_length', 512))}"
            f"_fr{float(getattr(args, 'graph_eager_full_ratio', 0.2)):.2f}"
            f"_mr{float(getattr(args, 'graph_eager_mid_ratio', 0.3)):.2f}"
        ).replace(".", "p")
    if getattr(args, "experiment_suite", None) == "token_compaction":
        tags = "-".join(str(tag) for tag in getattr(args, "token_compaction_tags", ["W4A8_S128"]))
        if len(tags) > 64:
            tags = tags[:64]
        config_tag += (
            f"_tc{getattr(args, 'real_quant_model_name', 'model')}"
            f"_{getattr(args, 'token_compaction_reference_tag', 'ref')}"
            f"_{getattr(args, 'token_compaction_full_tag', 'full')}"
            f"_L{int(getattr(args, 'token_compaction_length', 128))}"
            f"_{tags}"
        ).replace(".", "p").replace("/", "_")
    if not bool(getattr(args, "disable_score_gate", True)):
        config_tag += (
            f"_sg{int(getattr(args, 'score_reuse_threshold', 45))}"
            f"_{getattr(args, 'score_rare_gate_mode', 'support')}"
            f"_rb{int(getattr(args, 'score_rarity_bits', 16))}"
            f"_sw{int(getattr(args, 'score_propagation_weight', 3))}"
            f"-{int(getattr(args, 'score_graph_context_weight', 2))}"
            f"-{int(getattr(args, 'score_low_unique_weight', 2))}"
        )
    filename = (
        f"{ds_key}_paper_R{args.radius}_bits_{args.sketch_bits}_"
        f"{hash_route_tag}_{projection_tag}_{route_weight_tag}_"
        f"{route_accept_tag}_{retrieval_tag}_{tau_tag}_{config_tag}.log"
    )
    if len(filename) > 220:
        digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
        filename = (
            f"{ds_key}_paper_R{args.radius}_bits_{args.sketch_bits}_"
            f"{hash_route_tag}_{projection_tag}_{tau_tag}_{config_tag}_{digest}.log"
        )
    return os.path.join(dataset_log_dir, filename)
