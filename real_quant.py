from dataclasses import dataclass
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .features import _compute_neighbor_mean, build_hash_features
from .scoring import build_node_risk_scores, build_random_matrix


@dataclass
class RealQuantPools:
    fp: torch.Tensor
    int8: torch.Tensor
    int4: torch.Tensor
    fp_path: str
    int8_path: str
    int4_path: str


def default_pool_path(ds_key, model_name, tag):
    return os.path.join("cache_data", f"{ds_key}_{model_name}_oracle_{tag}.pt")


def resolve_pool_paths(ds_key, args):
    model_name = args.real_quant_model_name
    fp_path = args.real_quant_fp_path or default_pool_path(ds_key, model_name, args.real_quant_fp_tag)
    int8_path = args.real_quant_int8_path or default_pool_path(ds_key, model_name, args.real_quant_int8_tag)
    int4_path = args.real_quant_int4_path or default_pool_path(ds_key, model_name, args.real_quant_int4_tag)
    return fp_path, int8_path, int4_path


def _resolve_generation_config_name(tag):
    from .generate_real_quant_pools import CONFIG_SPECS

    normalized = str(tag).strip()
    if normalized in CONFIG_SPECS:
        return normalized

    bnb_aliases = {
        "FP16": "fp16",
        "INT8": "int8",
        "INT4": "int4",
    }
    upper = normalized.upper()
    if upper in bnb_aliases:
        return bnb_aliases[upper]

    for config_name, spec in CONFIG_SPECS.items():
        if upper == str(spec["tag"]).upper():
            return config_name

    raise ValueError(
        f"Cannot regenerate real-quant pool for tag {tag!r}. "
        f"Available generator configs: {sorted(CONFIG_SPECS)}"
    )


def regenerate_real_quant_pools(ds_key, args, log_fn=print):
    from .generate_real_quant_pools import generate_pool

    fp_path, int8_path, int4_path = resolve_pool_paths(ds_key, args)
    is_st_model = str(args.real_quant_model_name).upper() == "ST"
    awq_calib_samples = 16 if is_st_model else 128
    awq_seqlen = 128 if is_st_model else 512
    jobs = [
        ("FP", args.real_quant_fp_tag, fp_path),
        ("INT8", args.real_quant_int8_tag, int8_path),
        ("INT4", args.real_quant_int4_tag, int4_path),
    ]
    gen_args = SimpleNamespace(
        batch_size=64,
        max_length=500,
        cache_dir="cache_data/model",
        output_path=None,
        w4a_calib_samples=64,
        w4a_awq_grid=21,
        ptq_align_output=True,
        ptq_align_samples=512,
        ptq_align_reference_path=None,
        awq_calib_samples=awq_calib_samples,
        awq_seqlen=awq_seqlen,
        awq_q_group_size=128,
        awq_no_zero_point=False,
        awq_disable_auto_scale=False,
        awq_disable_mse_clip=False,
        awq_force_mse_clip=False,
        awq_results_path=None,
        awq_overwrite_results=False,
        overwrite=True,
    )

    log_fn(
        "[RealQuantAutoGen] Regenerating real-quant feature pools before reuse_real_quant; "
        "existing cache files will be overwritten."
    )
    for role, tag, out_path in jobs:
        config_name = _resolve_generation_config_name(tag)
        gen_args.output_path = out_path
        log_fn(
            f"[RealQuantAutoGen] {role}: tag={tag} | config={config_name} "
            f"| batch_size=64 | w4a_calib_samples=64 "
            f"| awq_calib_samples={awq_calib_samples} | awq_seqlen={awq_seqlen} "
            f"| output={out_path}"
        )
        generate_pool(ds_key, args.real_quant_model_name, config_name, gen_args)


def load_tensor_pool(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tensor = torch.load(path, map_location=device)
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} did not contain a torch.Tensor")
    return tensor.to(device=device, dtype=torch.float32)


def load_real_quant_pools(ds_key, args, data, device):
    fp_path, int8_path, int4_path = resolve_pool_paths(ds_key, args)
    missing = [path for path in (fp_path, int8_path, int4_path) if not os.path.exists(path)]
    if missing:
        msg = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Real quantization pools are missing:\n"
            f"{msg}\n"
            "Generate real FP/INT8/INT4 embeddings first, or pass explicit paths with "
            "--real_quant_fp_path/--real_quant_int8_path/--real_quant_int4_path. "
            "Use tags that match the generated cache files, for example "
            "--real_quant_fp_tag W4A16 --real_quant_int8_tag W4A8 --real_quant_int4_tag W4A4 "
            "for the AWQ-family path, or --real_quant_fp_tag FP16 for a clean full-precision reference."
        )

    pools = RealQuantPools(
        fp=load_tensor_pool(fp_path, device),
        int8=load_tensor_pool(int8_path, device),
        int4=load_tensor_pool(int4_path, device),
        fp_path=fp_path,
        int8_path=int8_path,
        int4_path=int4_path,
    )
    num_nodes = int(data.num_nodes)
    shapes = {
        "fp": tuple(pools.fp.shape),
        "int8": tuple(pools.int8.shape),
        "int4": tuple(pools.int4.shape),
    }
    if pools.fp.size(0) != num_nodes or pools.int8.size(0) != num_nodes or pools.int4.size(0) != num_nodes:
        raise ValueError(f"Pool node counts must match data.num_nodes={num_nodes}, got {shapes}")
    if pools.fp.shape != pools.int8.shape or pools.fp.shape != pools.int4.shape:
        raise ValueError(f"FP/INT8/INT4 pools must have identical shapes, got {shapes}")
    return pools


def build_real_quant_scores(verify_features, data, args, device):
    hash_features, _weights = build_hash_features(
        verify_features,
        data.edge_index,
        args.hash_view,
        args.hash_mix_weights,
    )
    neighbor_mean = _compute_neighbor_mean(verify_features, data.edge_index)
    context_signature = F.normalize(0.5 * verify_features + 0.5 * neighbor_mean, p=2, dim=1)

    row, col = data.edge_index
    sym_row = torch.cat([row, col], dim=0)
    total_degree = torch.zeros(verify_features.size(0), device=device)
    total_degree.index_add_(0, sym_row, torch.ones(sym_row.size(0), device=device))

    hash_matrix = build_random_matrix(
        hash_features.size(1),
        args.sketch_bits,
        device,
        args.controller_seed if args.controller_seed is not None else args.run_seed,
    )
    return build_node_risk_scores(
        verify_features=verify_features,
        hash_features=hash_features,
        edge_index=data.edge_index,
        total_degree=total_degree,
        context_signature=context_signature,
        hash_matrix=hash_matrix,
        rarity_bits=args.score_rarity_bits,
        rarity_seed=args.score_rarity_seed,
        propagation_weight=args.score_propagation_weight,
        graph_context_weight=args.score_graph_context_weight,
        low_unique_weight=args.score_low_unique_weight,
    )


def cosine_error_q(fp_embs, quant_embs, norm):
    err = 1.0 - F.cosine_similarity(fp_embs, quant_embs, dim=1)
    err = err.clamp(min=0.0)
    q = torch.round((err / float(norm)).clamp(0.0, 1.0) * 15.0).to(torch.int64)
    return err, q


def compute_real_quant_errors(fp_embs, int8_embs, int4_embs, args):
    int8_err, int8_err_q = cosine_error_q(fp_embs, int8_embs, args.real_quant_error_norm)
    int4_err, int4_err_q = cosine_error_q(fp_embs, int4_embs, args.real_quant_error_norm)
    return {
        "int8_err": int8_err,
        "int4_err": int4_err,
        "int8_err_q": int8_err_q,
        "int4_err_q": int4_err_q,
    }


def resolve_eligible_mask(scores, eligible_mask=None):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    if eligible_mask is None:
        return torch.ones(num_nodes, dtype=torch.bool, device=device)

    mask = torch.as_tensor(eligible_mask, dtype=torch.bool, device=device)
    if mask.numel() != num_nodes:
        raise ValueError(
            f"eligible_mask length must match node count {num_nodes}, got {mask.numel()}"
        )
    return mask


def select_real_quant_actions(policy_name, scores, errors, args, eligible_mask=None):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    eligible = resolve_eligible_mask(scores, eligible_mask)
    actions = torch.full((num_nodes,), 16, dtype=torch.int64, device=device)
    if policy_name == "all_fp":
        return actions
    if policy_name == "all_int8":
        actions[eligible] = 8
        return actions
    if policy_name == "all_int4":
        actions[eligible] = 4
        return actions

    if policy_name == "degree":
        sensitivity = args.score_propagation_weight * scores["propagation_q"]
    elif policy_name == "tser":
        sensitivity = scores["sensitivity_q"]
    elif policy_name == "error_only":
        sensitivity = torch.ones_like(scores["sensitivity_q"])
    else:
        raise ValueError(f"Unknown real quant policy: {policy_name}")

    int4_risk = sensitivity * errors["int4_err_q"]
    int8_risk = sensitivity * errors["int8_err_q"]
    int4_mask = eligible & (int4_risk <= int(args.real_quant_int4_threshold))
    int8_mask = eligible & (~int4_mask) & (int8_risk <= int(args.real_quant_int8_threshold))
    actions[int4_mask] = 4
    actions[int8_mask] = 8
    return actions


def get_quant_tser_score(scores, args):
    propagation_weight = float(getattr(args, "quant_tser_propagation_weight", 4.0))
    graph_context_weight = float(getattr(args, "quant_tser_graph_context_weight", 1.0))
    low_unique_weight = float(getattr(args, "quant_tser_low_unique_weight", 0.0))
    return (
        propagation_weight * scores["propagation_q"].float()
        + graph_context_weight * scores["graph_context_q"].float()
        + low_unique_weight * scores["low_degree_unique_q"].float()
    )


def get_error_multiplier(errors, args):
    bias = float(getattr(args, "quant_error_bias", 1.0))
    source = str(getattr(args, "quant_error_rank_source", "continuous")).lower()
    if source == "quantized":
        return errors["int4_err_q"].float() + bias
    if source != "continuous":
        raise ValueError(f"Unknown quant_error_rank_source={source}")

    err = errors["int4_err"].float()
    scale = err.mean().clamp(min=1e-8)
    return err / scale + bias


def get_rank_score(policy_name, scores, args=None, errors=None):
    if policy_name.startswith("degree_error"):
        if errors is None:
            raise ValueError("degree_error ranking requires quantization errors")
        return scores["propagation_q"].float() * get_error_multiplier(errors, args)
    if policy_name.startswith("tser_error"):
        if errors is None:
            raise ValueError("tser_error ranking requires quantization errors")
        return get_quant_tser_score(scores, args) * get_error_multiplier(errors, args)
    if policy_name.startswith("quant_tser"):
        return get_quant_tser_score(scores, args)
    if policy_name.startswith("degree"):
        return scores["propagation_q"].float()
    if policy_name.startswith("tser"):
        return scores["sensitivity_q"].float()
    raise ValueError(f"Unknown ranking policy: {policy_name}")


def _map_calibration_bit_to_action(bit):
    bit = int(bit)
    if bit >= 16:
        return 16
    if bit >= 8:
        return 8
    return 4


def select_random_int8_budget_actions(scores, args, eligible_mask=None):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    eligible = resolve_eligible_mask(scores, eligible_mask)
    eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
    actions = torch.full((num_nodes,), 16, dtype=torch.int64, device=device)
    actions[eligible_idx] = 4
    eligible_count = int(eligible_idx.numel())
    int8_count = int(round(float(args.real_quant_int8_ratio) * eligible_count))
    int8_count = max(0, min(eligible_count, int8_count))
    if int8_count <= 0:
        return actions

    seed = int(getattr(args, "run_seed", getattr(args, "seed", 0)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(eligible_count, generator=generator, device="cpu").to(device)
    actions[eligible_idx[order[:int8_count]]] = 8
    return actions


def select_ranked_int8_budget_actions(policy_name, scores, args, eligible_mask=None):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    eligible = resolve_eligible_mask(scores, eligible_mask)
    eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
    actions = torch.full((num_nodes,), 16, dtype=torch.int64, device=device)
    actions[eligible_idx] = 4
    rank_score = get_rank_score(policy_name, scores, args=args)
    eligible_count = int(eligible_idx.numel())
    if eligible_count <= 0:
        return actions
    order = eligible_idx[torch.argsort(rank_score[eligible_idx], descending=True)]

    int8_count = int(round(float(args.real_quant_int8_ratio) * eligible_count))
    int8_count = max(0, min(eligible_count, int8_count))
    if int8_count > 0:
        actions[order[:int8_count]] = 8
    return actions


def select_internal_split_actions(scores, args, eligible_mask=None):
    bundle = getattr(args, "_internal_split_calibration_bundle", None)
    if bundle is None:
        raise ValueError("internal_split policy requires --internal_split_calibration")

    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    eligible = resolve_eligible_mask(scores, eligible_mask)
    low_bit = int(getattr(args, "internal_calib_low_a_bit", 4))
    actions = torch.full(
        (num_nodes,),
        16,
        dtype=torch.int64,
        device=device,
    )
    actions[eligible] = _map_calibration_bit_to_action(low_bit)

    assignment = bundle.get("assignment", {})
    activation_bits = assignment.get("activation_bits")
    if activation_bits is not None:
        bit_tensor = torch.as_tensor(activation_bits, dtype=torch.long, device=device)
        if bit_tensor.numel() != num_nodes:
            raise ValueError(
                f"internal_split activation_bits length must match node count {num_nodes}, got {bit_tensor.numel()}"
            )
        actions[eligible] = 4
        actions[eligible & (bit_tensor >= 8)] = 8
        actions[eligible & (bit_tensor >= 16)] = 16
        return actions

    for pass_spec in bundle.get("passes", []):
        node_indices = pass_spec.get("node_indices", [])
        if not node_indices:
            continue
        idx = torch.as_tensor(node_indices, dtype=torch.long, device=device)
        idx = idx[(idx >= 0) & (idx < num_nodes)]
        idx = idx[eligible[idx]]
        if idx.numel() == 0:
            continue
        bit = int(pass_spec.get("bit", low_bit))
        actions[idx] = _map_calibration_bit_to_action(bit)
    return actions


def select_ranked_int8_budget_actions_by_score(rank_score, scores, args, eligible_mask=None):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    eligible = resolve_eligible_mask(scores, eligible_mask)
    eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
    actions = torch.full((num_nodes,), 16, dtype=torch.int64, device=device)
    actions[eligible_idx] = 4
    eligible_count = int(eligible_idx.numel())
    if eligible_count <= 0:
        return actions
    order = eligible_idx[torch.argsort(rank_score[eligible_idx], descending=True)]

    int8_count = int(round(float(args.real_quant_int8_ratio) * eligible_count))
    int8_count = max(0, min(eligible_count, int8_count))
    if int8_count > 0:
        actions[order[:int8_count]] = 8
    return actions


def select_ranked_budget_actions(policy_name, scores, errors, args, mode, eligible_mask=None):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    eligible = resolve_eligible_mask(scores, eligible_mask)
    eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
    actions = torch.full((num_nodes,), 16, dtype=torch.int64, device=device)
    actions[eligible_idx] = 4
    rank_score = get_rank_score(policy_name, scores, args=args, errors=errors)
    eligible_count = int(eligible_idx.numel())
    if eligible_count <= 0:
        return actions
    order = eligible_idx[torch.argsort(rank_score[eligible_idx], descending=True)]

    fp_count = int(round(float(args.real_quant_fp_ratio) * eligible_count))
    fp_count = max(0, min(eligible_count, fp_count))
    if fp_count > 0:
        actions[order[:fp_count]] = 16

    if mode == "topk":
        if args.real_quant_tail_precision == "int8":
            actions[order[fp_count:]] = 8
        return actions

    if mode != "cascade":
        raise ValueError(f"Unknown ranked budget mode: {mode}")

    int8_count = int(round(float(args.real_quant_int8_ratio) * eligible_count))
    int8_count = max(0, min(eligible_count - fp_count, int8_count))
    if int8_count > 0:
        actions[order[fp_count : fp_count + int8_count]] = 8
    return actions


def select_real_quant_policy_actions(policy_name, scores, errors, args, eligible_mask=None):
    if policy_name in ("degree", "tser", "all_fp", "all_int8", "all_int4"):
        return select_real_quant_actions(policy_name, scores, errors, args, eligible_mask=eligible_mask)
    if policy_name == "random_int8_budget":
        return select_random_int8_budget_actions(scores, args, eligible_mask=eligible_mask)
    if policy_name == "degree_int8_budget":
        return select_ranked_int8_budget_actions("degree", scores, args, eligible_mask=eligible_mask)
    if policy_name == "tser_int8_budget":
        return select_ranked_int8_budget_actions("tser", scores, args, eligible_mask=eligible_mask)
    if policy_name == "quant_tser_int8_budget":
        return select_ranked_int8_budget_actions("quant_tser", scores, args, eligible_mask=eligible_mask)
    if policy_name == "degree_error_int8_budget":
        rank_score = get_rank_score("degree_error", scores, args=args, errors=errors)
        return select_ranked_int8_budget_actions_by_score(rank_score, scores, args, eligible_mask=eligible_mask)
    if policy_name == "tser_error_int8_budget":
        rank_score = get_rank_score("tser_error", scores, args=args, errors=errors)
        return select_ranked_int8_budget_actions_by_score(rank_score, scores, args, eligible_mask=eligible_mask)
    if policy_name == "internal_split":
        return select_internal_split_actions(scores, args, eligible_mask=eligible_mask)
    if policy_name == "degree_topk":
        return select_ranked_budget_actions("degree", scores, errors, args, mode="topk", eligible_mask=eligible_mask)
    if policy_name == "tser_topk":
        return select_ranked_budget_actions("tser", scores, errors, args, mode="topk", eligible_mask=eligible_mask)
    if policy_name == "quant_tser_topk":
        return select_ranked_budget_actions("quant_tser", scores, errors, args, mode="topk", eligible_mask=eligible_mask)
    if policy_name == "degree_error_topk":
        return select_ranked_budget_actions("degree_error", scores, errors, args, mode="topk", eligible_mask=eligible_mask)
    if policy_name == "tser_error_topk":
        return select_ranked_budget_actions("tser_error", scores, errors, args, mode="topk", eligible_mask=eligible_mask)
    if policy_name == "degree_cascade":
        return select_ranked_budget_actions("degree", scores, errors, args, mode="cascade", eligible_mask=eligible_mask)
    if policy_name == "tser_cascade":
        return select_ranked_budget_actions("tser", scores, errors, args, mode="cascade", eligible_mask=eligible_mask)
    raise ValueError(f"Unknown real quant policy: {policy_name}")


def assemble_real_quant_embeddings(actions, fp_embs, int8_embs, int4_embs):
    mixed = fp_embs.clone()
    int8_mask = actions == 8
    int4_mask = actions == 4
    if int8_mask.any():
        mixed[int8_mask] = int8_embs[int8_mask]
    if int4_mask.any():
        mixed[int4_mask] = int4_embs[int4_mask]
    return mixed


def summarize_real_quant_policy(actions, errors, scores):
    total = max(1, int(actions.numel()))
    int4_mask = actions == 4
    int8_mask = actions == 8
    fp_mask = actions == 16
    selected_err = torch.zeros(total, device=actions.device)
    selected_err[int4_mask] = errors["int4_err"][int4_mask]
    selected_err[int8_mask] = errors["int8_err"][int8_mask]
    return {
        "int4_num": int(int4_mask.sum().item()),
        "int8_num": int(int8_mask.sum().item()),
        "fp_num": int(fp_mask.sum().item()),
        "int4_rate": float(int4_mask.float().mean().item()),
        "int8_rate": float(int8_mask.float().mean().item()),
        "fp_rate": float(fp_mask.float().mean().item()),
        "avg_selected_error": float(selected_err.mean().item()),
        "avg_int4_error": float(errors["int4_err"].mean().item()),
        "avg_int8_error": float(errors["int8_err"].mean().item()),
        "avg_sensitivity": float(scores["sensitivity_q"].float().mean().item()),
    }
