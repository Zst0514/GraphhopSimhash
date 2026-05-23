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


def _quantize_positive_to_0_15(values):
    values = values.float().clamp(min=0.0)
    vmax = values.max().clamp(min=1e-8)
    return torch.round((values / vmax).clamp(0.0, 1.0) * 15.0).to(torch.int64)


def compute_graph_impact_q(edge_index, num_nodes, device):
    row, col = edge_index
    nodes = torch.arange(num_nodes, device=device)
    target = torch.cat([row, col, nodes], dim=0)
    source = torch.cat([col, row, nodes], dim=0)

    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, target, torch.ones_like(target, dtype=torch.float32))
    degree = degree.clamp(min=1.0)

    weight = torch.rsqrt(degree[target] * degree[source])
    impact_1hop = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    impact_1hop.index_add_(0, source, weight)

    impact_2hop = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    impact_2hop.index_add_(0, source, weight * impact_1hop[target])

    impact = torch.log1p(impact_1hop + impact_2hop)
    return _quantize_positive_to_0_15(impact), impact


def compute_margin_risk_q(logits, margin_norm):
    top2 = torch.topk(logits.float(), k=min(2, logits.size(1)), dim=1).values
    if top2.size(1) == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    margin_norm = max(float(margin_norm), 1e-6)
    margin_risk = 1.0 - (margin / margin_norm).clamp(0.0, 1.0)
    return torch.round(margin_risk.clamp(0.0, 1.0) * 15.0).to(torch.int64), margin


def augment_quant_vulnerability_scores(scores, errors, data, baseline_logits, args, device):
    num_nodes = int(scores["sensitivity_q"].numel())
    graph_impact_q, graph_impact = compute_graph_impact_q(data.edge_index, num_nodes, device)
    margin_risk_q, margin = compute_margin_risk_q(baseline_logits, args.tserq_margin_norm)

    quant_sensitivity_q = (
        int(args.tserq_graph_impact_weight) * graph_impact_q
        + int(args.tserq_margin_weight) * margin_risk_q
        + int(args.tserq_graph_context_weight) * scores["graph_context_q"]
        + int(args.tserq_low_unique_weight) * scores["low_degree_unique_q"]
    ).to(torch.int64)

    int4_risk_q = quant_sensitivity_q * errors["int4_err_q"]
    int8_risk_q = quant_sensitivity_q * errors["int8_err_q"]
    protect_gain_q = (int4_risk_q - int8_risk_q).clamp(min=0)
    degree_error_gain_q = (
        scores["propagation_q"].to(torch.int64)
        * (errors["int4_err_q"] - errors["int8_err_q"]).clamp(min=0)
    )
    error_gain_q = (errors["int4_err_q"] - errors["int8_err_q"]).clamp(min=0)

    scores.update(
        {
            "graph_impact": graph_impact,
            "graph_impact_q": graph_impact_q,
            "margin": margin,
            "margin_risk_q": margin_risk_q,
            "quant_sensitivity_q": quant_sensitivity_q,
            "int4_quant_risk_q": int4_risk_q,
            "int8_quant_risk_q": int8_risk_q,
            "tserq_protect_gain_q": protect_gain_q,
            "degree_error_gain_q": degree_error_gain_q,
            "error_gain_q": error_gain_q,
        }
    )
    return scores


def augment_w4a4_safe_scores(scores, args):
    """Build a deployable graph/hash stability score for aggressive W4A4 routing."""
    propagation_q = scores["propagation_q"].to(torch.int64)
    graph_context_q = scores["graph_context_q"].to(torch.int64)
    rarity_q = scores["rarity_q"].to(torch.int64)
    low_unique_q = scores["low_degree_unique_q"].to(torch.int64)

    bucket_density_q = (15 - rarity_q).clamp(min=0, max=15)
    context_consistency_q = (15 - graph_context_q).clamp(min=0, max=15)
    low_propagation_q = (15 - propagation_q).clamp(min=0, max=15)
    non_unique_q = (15 - low_unique_q).clamp(min=0, max=15)

    similar_count = scores.get("similar_count")
    if similar_count is None:
        hash_support_q = bucket_density_q
    else:
        hash_support_q = _positive_to_coarse_bin(torch.log1p(similar_count.float()), 16)

    # A proxy for multi-head agreement in real-quant-only experiments where
    # route hit counts are not available: dense hash buckets that are also
    # locally context-consistent are considered more reliable W4A4 candidates.
    hash_agreement_proxy_q = torch.minimum(hash_support_q, context_consistency_q)

    density_w = int(getattr(args, "w4a4_safe_density_weight", 2))
    agreement_w = int(getattr(args, "w4a4_safe_agreement_weight", 1))
    context_w = int(getattr(args, "w4a4_safe_context_weight", 2))
    low_prop_w = int(getattr(args, "w4a4_safe_low_propagation_weight", 3))
    non_unique_w = int(getattr(args, "w4a4_safe_non_unique_weight", 2))

    w4a4_safe_q = (
        density_w * bucket_density_q
        + agreement_w * hash_agreement_proxy_q
        + context_w * context_consistency_q
        + low_prop_w * low_propagation_q
        + non_unique_w * non_unique_q
    ).to(torch.int64)

    scores.update(
        {
            "bucket_density_q": bucket_density_q,
            "hash_support_q": hash_support_q,
            "hash_agreement_proxy_q": hash_agreement_proxy_q,
            "context_consistency_q": context_consistency_q,
            "low_propagation_q": low_propagation_q,
            "non_unique_q": non_unique_q,
            "w4a4_safe_q": w4a4_safe_q,
        }
    )
    return scores


def _q_to_coarse_bin(values, bins):
    bins = max(1, int(bins))
    return ((values.to(torch.int64).clamp(min=0, max=15) * bins) // 16).clamp(max=bins - 1)


def _positive_to_coarse_bin(values, bins):
    bins = max(1, int(bins))
    values = values.float().clamp(min=0.0)
    vmax = values.max().clamp(min=1e-8)
    return torch.round((values / vmax).clamp(0.0, 1.0) * float(bins - 1)).to(torch.int64)


def _select_calibration_proxy_indices(scores, args, device):
    num_nodes = int(scores["sensitivity_q"].numel())
    sample_count = int(getattr(args, "calib_proxy_size", 256))
    sample_count = max(0, min(num_nodes, sample_count))
    if sample_count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    seed = int(getattr(args, "run_seed", getattr(args, "seed", 0))) + 104729
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    strategy = str(getattr(args, "calib_proxy_strategy", "score_stratified"))
    if strategy == "random":
        return torch.randperm(num_nodes, generator=generator, device="cpu")[:sample_count].to(device)

    if strategy != "score_stratified":
        raise ValueError(f"Unknown --calib_proxy_strategy={strategy}")

    score = scores.get("quant_sensitivity_q", scores["sensitivity_q"]).detach().float().cpu()
    order = torch.argsort(score)
    sample_bins = max(1, int(getattr(args, "calib_proxy_sample_bins", 8)))
    chunks = torch.chunk(order, sample_bins)
    per_bin = max(1, (sample_count + sample_bins - 1) // sample_bins)

    selected = []
    for chunk in chunks:
        if len(selected) >= sample_count or chunk.numel() == 0:
            continue
        take = min(per_bin, int(chunk.numel()), sample_count - len(selected))
        perm = torch.randperm(int(chunk.numel()), generator=generator)[:take]
        selected.append(chunk[perm])

    if selected:
        selected = torch.cat(selected, dim=0)
    else:
        selected = torch.empty(0, dtype=torch.long)

    if selected.numel() < sample_count:
        chosen = torch.zeros(num_nodes, dtype=torch.bool)
        chosen[selected] = True
        rest = torch.nonzero(~chosen, as_tuple=False).view(-1)
        fill = sample_count - int(selected.numel())
        rest_perm = rest[torch.randperm(int(rest.numel()), generator=generator)[:fill]]
        selected = torch.cat([selected, rest_perm], dim=0)

    return selected[:sample_count].to(device)


def build_calibration_error_proxy_scores(scores, errors, args, device):
    """Predict quantization error from a small calibration subset and score bins.

    This keeps full per-node errors only for evaluation/reporting. The returned
    proxy gains are computed from calibration-node averages and can be used as a
    deployable approximation to the oracle ErrorBudget/TSERQBudget rows.
    """
    num_nodes = int(scores["sensitivity_q"].numel())
    calib_idx = _select_calibration_proxy_indices(scores, args, device)
    if calib_idx.numel() == 0:
        return {
            "enabled": False,
            "calib_size": 0,
        }

    bins = max(1, int(getattr(args, "calib_proxy_bins", 4)))
    impact_bin = _q_to_coarse_bin(scores.get("graph_impact_q", scores["propagation_q"]), bins)
    margin_bin = _q_to_coarse_bin(scores.get("margin_risk_q", scores["graph_context_q"]), bins)
    context_bin = _q_to_coarse_bin(scores["graph_context_q"], bins)
    low_unique_bin = _q_to_coarse_bin(scores["low_degree_unique_q"], bins)
    bucket = (((impact_bin * bins + margin_bin) * bins + context_bin) * bins + low_unique_bin).to(torch.long)
    num_buckets = bins ** 4

    sens = scores.get("quant_sensitivity_q", scores["sensitivity_q"]).float()
    sens_bin = _positive_to_coarse_bin(sens, bins)

    def predict(error_key):
        target = errors[error_key].to(torch.float32)
        calib_target = target[calib_idx]
        global_mean = calib_target.mean()

        bucket_sum = torch.zeros(num_buckets, dtype=torch.float32, device=device)
        bucket_count = torch.zeros(num_buckets, dtype=torch.float32, device=device)
        bucket_sum.index_add_(0, bucket[calib_idx], calib_target)
        bucket_count.index_add_(0, bucket[calib_idx], torch.ones_like(calib_target))

        bucket_mean = torch.full((num_buckets,), float(global_mean.item()), dtype=torch.float32, device=device)
        nonempty_bucket = bucket_count > 0
        bucket_mean[nonempty_bucket] = bucket_sum[nonempty_bucket] / bucket_count[nonempty_bucket].clamp(min=1.0)

        sens_sum = torch.zeros(bins, dtype=torch.float32, device=device)
        sens_count = torch.zeros(bins, dtype=torch.float32, device=device)
        sens_sum.index_add_(0, sens_bin[calib_idx], calib_target)
        sens_count.index_add_(0, sens_bin[calib_idx], torch.ones_like(calib_target))
        sens_mean = torch.full((bins,), float(global_mean.item()), dtype=torch.float32, device=device)
        nonempty_sens = sens_count > 0
        sens_mean[nonempty_sens] = sens_sum[nonempty_sens] / sens_count[nonempty_sens].clamp(min=1.0)

        pred = bucket_mean[bucket]
        empty_for_node = bucket_count[bucket] <= 0
        if empty_for_node.any():
            pred[empty_for_node] = sens_mean[sens_bin[empty_for_node]]
        return torch.round(pred.clamp(0.0, 15.0)).to(torch.int64), pred

    pred_int4_q, pred_int4_float = predict("int4_err_q")
    pred_int8_q, pred_int8_float = predict("int8_err_q")
    pred_gain_q = (pred_int4_q - pred_int8_q).clamp(min=0)
    pred_tserq_gain_q = (
        scores.get("quant_sensitivity_q", scores["sensitivity_q"]).to(torch.int64) * pred_gain_q
    )
    pred_degree_gain_q = scores["propagation_q"].to(torch.int64) * pred_gain_q

    scores.update(
        {
            "calib_proxy_int4_err_q": pred_int4_q,
            "calib_proxy_int8_err_q": pred_int8_q,
            "calib_proxy_error_gain_q": pred_gain_q,
            "calib_proxy_degree_error_gain_q": pred_degree_gain_q,
            "calib_proxy_tserq_protect_gain_q": pred_tserq_gain_q,
            "calib_proxy_indices": calib_idx,
        }
    )

    true_int4 = errors["int4_err_q"].float()
    true_int8 = errors["int8_err_q"].float()
    covered = int((torch.bincount(bucket[calib_idx].detach().cpu(), minlength=num_buckets) > 0).sum().item())
    return {
        "enabled": True,
        "calib_size": int(calib_idx.numel()),
        "bins": bins,
        "num_buckets": int(num_buckets),
        "covered_buckets": covered,
        "int4_mae_q": float((pred_int4_float - true_int4).abs().mean().item()),
        "int8_mae_q": float((pred_int8_float - true_int8).abs().mean().item()),
        "int4_calib_mean_q": float(errors["int4_err_q"][calib_idx].float().mean().item()),
        "int8_calib_mean_q": float(errors["int8_err_q"][calib_idx].float().mean().item()),
    }


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


def _map_calibration_bit_to_action(bit):
    bit = int(bit)
    if bit >= 16:
        return 16
    if bit >= 8:
        return 8
    return 4


def select_random_int8_budget_actions(scores, args):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    actions = torch.full((num_nodes,), 4, dtype=torch.int64, device=device)
    int8_count = int(round(float(args.real_quant_int8_ratio) * num_nodes))
    int8_count = max(0, min(num_nodes, int8_count))
    if int8_count <= 0:
        return actions

    seed = int(getattr(args, "run_seed", getattr(args, "seed", 0)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(num_nodes, generator=generator, device="cpu").to(device)
    actions[order[:int8_count]] = 8
    return actions


def select_ranked_int8_budget_actions(policy_name, scores, args):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    actions = torch.full((num_nodes,), 4, dtype=torch.int64, device=device)
    rank_score = get_rank_score(policy_name, scores)
    order = torch.argsort(rank_score, descending=True)

    int8_count = int(round(float(args.real_quant_int8_ratio) * num_nodes))
    int8_count = max(0, min(num_nodes, int8_count))
    if int8_count > 0:
        actions[order[:int8_count]] = 8
    return actions


def select_gain_int8_budget_actions(gain_name, scores, args):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    actions = torch.full((num_nodes,), 4, dtype=torch.int64, device=device)
    gain = scores[gain_name].float()
    tie_break = scores.get("quant_sensitivity_q", scores["sensitivity_q"]).float()
    order = torch.argsort(gain + 1e-3 * tie_break, descending=True)

    int8_count = int(round(float(args.real_quant_int8_ratio) * num_nodes))
    int8_count = max(0, min(num_nodes, int8_count))
    if int8_count > 0:
        actions[order[:int8_count]] = 8
    return actions


def select_w4a4_safe_budget_actions(scores, args):
    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    actions = torch.full((num_nodes,), 8, dtype=torch.int64, device=device)
    int8_count = int(round(float(args.real_quant_int8_ratio) * num_nodes))
    int8_count = max(0, min(num_nodes, int8_count))
    int4_count = max(0, min(num_nodes, num_nodes - int8_count))
    if int4_count <= 0:
        return actions

    safe_score = scores["w4a4_safe_q"].float()
    order = torch.argsort(safe_score, descending=True)
    actions[order[:int4_count]] = 4
    return actions


def select_internal_split_actions(scores, args):
    bundle = getattr(args, "_internal_split_calibration_bundle", None)
    if bundle is None:
        raise ValueError("internal_split policy requires --internal_split_calibration")

    num_nodes = int(scores["sensitivity_q"].numel())
    device = scores["sensitivity_q"].device
    low_bit = int(getattr(args, "internal_calib_low_a_bit", 4))
    actions = torch.full(
        (num_nodes,),
        _map_calibration_bit_to_action(low_bit),
        dtype=torch.int64,
        device=device,
    )

    assignment = bundle.get("assignment", {})
    activation_bits = assignment.get("activation_bits")
    if activation_bits is not None:
        bit_tensor = torch.as_tensor(activation_bits, dtype=torch.long, device=device)
        if bit_tensor.numel() != num_nodes:
            raise ValueError(
                f"internal_split activation_bits length must match node count {num_nodes}, got {bit_tensor.numel()}"
            )
        actions.fill_(4)
        actions[bit_tensor >= 8] = 8
        actions[bit_tensor >= 16] = 16
        return actions

    for pass_spec in bundle.get("passes", []):
        node_indices = pass_spec.get("node_indices", [])
        if not node_indices:
            continue
        idx = torch.as_tensor(node_indices, dtype=torch.long, device=device)
        idx = idx[(idx >= 0) & (idx < num_nodes)]
        if idx.numel() == 0:
            continue
        bit = int(pass_spec.get("bit", low_bit))
        actions[idx] = _map_calibration_bit_to_action(bit)
    return actions


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
    if policy_name == "random_int8_budget":
        return select_random_int8_budget_actions(scores, args)
    if policy_name == "degree_int8_budget":
        return select_ranked_int8_budget_actions("degree", scores, args)
    if policy_name == "tser_int8_budget":
        return select_ranked_int8_budget_actions("tser", scores, args)
    if policy_name == "error_int8_budget":
        return select_gain_int8_budget_actions("error_gain_q", scores, args)
    if policy_name == "degree_error_int8_budget":
        return select_gain_int8_budget_actions("degree_error_gain_q", scores, args)
    if policy_name == "tserq_int8_budget":
        return select_gain_int8_budget_actions("tserq_protect_gain_q", scores, args)
    if policy_name == "w4a4_safe_budget":
        return select_w4a4_safe_budget_actions(scores, args)
    if policy_name == "calib_error_int8_budget":
        return select_gain_int8_budget_actions("calib_proxy_error_gain_q", scores, args)
    if policy_name == "calib_degree_error_int8_budget":
        return select_gain_int8_budget_actions("calib_proxy_degree_error_gain_q", scores, args)
    if policy_name == "calib_tserq_int8_budget":
        return select_gain_int8_budget_actions("calib_proxy_tserq_protect_gain_q", scores, args)
    if policy_name == "internal_split":
        return select_internal_split_actions(scores, args)
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
