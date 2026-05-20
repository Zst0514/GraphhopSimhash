from dataclasses import dataclass
import os

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
            "If you only have the older W4A16/W4A8/W4A4 pools, run with "
            "--real_quant_fp_tag W4A16 --real_quant_int8_tag W4A8 --real_quant_int4_tag W4A4."
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


def select_real_quant_actions(policy_name, scores, errors, args):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    actions = torch.full((num_nodes,), 16, dtype=torch.int64, device=device)
    if policy_name == "all_fp":
        return actions
    if policy_name == "all_int8":
        actions.fill_(8)
        return actions
    if policy_name == "all_int4":
        actions.fill_(4)
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
    int4_mask = int4_risk <= int(args.real_quant_int4_threshold)
    int8_mask = (~int4_mask) & (int8_risk <= int(args.real_quant_int8_threshold))
    actions[int4_mask] = 4
    actions[int8_mask] = 8
    return actions


def get_rank_score(policy_name, scores):
    if policy_name.startswith("degree"):
        return scores["propagation_q"].float()
    if policy_name.startswith("tser"):
        return scores["sensitivity_q"].float()
    raise ValueError(f"Unknown ranking policy: {policy_name}")


def select_ranked_budget_actions(policy_name, scores, args, mode):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    actions = torch.full((num_nodes,), 4, dtype=torch.int64, device=device)
    rank_score = get_rank_score(policy_name, scores)
    order = torch.argsort(rank_score, descending=True)

    fp_count = int(round(float(args.real_quant_fp_ratio) * num_nodes))
    fp_count = max(0, min(num_nodes, fp_count))
    if fp_count > 0:
        actions[order[:fp_count]] = 16

    if mode == "topk":
        if args.real_quant_tail_precision == "int8":
            actions[order[fp_count:]] = 8
        return actions

    if mode != "cascade":
        raise ValueError(f"Unknown ranked budget mode: {mode}")

    int8_count = int(round(float(args.real_quant_int8_ratio) * num_nodes))
    int8_count = max(0, min(num_nodes - fp_count, int8_count))
    if int8_count > 0:
        actions[order[fp_count : fp_count + int8_count]] = 8
    return actions


def select_real_quant_policy_actions(policy_name, scores, errors, args):
    if policy_name in ("degree", "tser", "all_fp", "all_int8", "all_int4"):
        return select_real_quant_actions(policy_name, scores, errors, args)
    if policy_name == "degree_topk":
        return select_ranked_budget_actions("degree", scores, args, mode="topk")
    if policy_name == "tser_topk":
        return select_ranked_budget_actions("tser", scores, args, mode="topk")
    if policy_name == "degree_cascade":
        return select_ranked_budget_actions("degree", scores, args, mode="cascade")
    if policy_name == "tser_cascade":
        return select_ranked_budget_actions("tser", scores, args, mode="cascade")
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
