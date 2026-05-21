import json
import os

import numpy as np
import torch

from .data import load_raw_texts
from .real_quant import build_real_quant_scores


def _json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    return str(obj)


def compute_total_degree(edge_index, num_nodes, device):
    num_nodes = int(num_nodes)
    if edge_index is None or num_nodes <= 0:
        return torch.zeros(max(0, num_nodes), dtype=torch.float32, device=device)

    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    if edge_index.dim() != 2 or edge_index.size(0) < 2:
        return torch.zeros(num_nodes, dtype=torch.float32, device=device)

    src = edge_index[0]
    dst = edge_index[1]
    src = src[(src >= 0) & (src < num_nodes)]
    dst = dst[(dst >= 0) & (dst < num_nodes)]

    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    if src.numel() > 0:
        degree.index_add_(0, src, torch.ones(src.numel(), dtype=torch.float32, device=device))
    if dst.numel() > 0:
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.float32, device=device))
    return degree


def _priority_values(data, verify_features, args, device):
    degree = compute_total_degree(data.edge_index, int(data.num_nodes), device)
    policy = str(args.internal_split_priority).lower()
    scores = None

    if policy == "degree":
        priority = degree
    elif policy == "bottom_degree":
        priority = -degree
    elif policy == "tser":
        scores = build_real_quant_scores(verify_features, data, args, device)
        priority = scores["sensitivity_q"].to(device=device, dtype=torch.float32)
    elif policy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(args, "run_seed", getattr(args, "seed", 0))))
        priority = torch.rand(int(data.num_nodes), generator=generator, device="cpu").to(device)
    else:
        raise ValueError(f"Unknown internal split priority: {policy}")

    return priority.to(torch.float32), degree, scores


def resolve_high_low_assignment(data, verify_features, args, device):
    num_nodes = int(data.num_nodes)
    priority, degree, scores = _priority_values(data, verify_features, args, device)
    if priority.numel() < num_nodes:
        pad = torch.zeros(num_nodes - priority.numel(), dtype=priority.dtype, device=device)
        priority = torch.cat([priority, pad], dim=0)
    priority = priority[:num_nodes]

    topk_ratio = float(args.internal_split_topk_ratio)
    mass_target = float(args.internal_split_mass_target)
    total_priority = float(priority.clamp(min=0).sum().item())
    strategy = "ratio"
    topk_count = int(np.ceil(topk_ratio * num_nodes))

    if mass_target >= 0.0:
        strategy = "mass_target"
        if num_nodes <= 0:
            topk_count = 0
        elif total_priority <= 0.0:
            strategy = "mass_target_fallback_ratio_zero_priority"
            topk_count = int(np.ceil(topk_ratio * num_nodes))
        elif mass_target <= 0.0:
            topk_count = 0
        elif mass_target >= 1.0:
            topk_count = num_nodes
        else:
            sorted_priority, _ = torch.sort(priority.clamp(min=0), descending=True)
            cumulative = torch.cumsum(sorted_priority, dim=0)
            target = cumulative.new_tensor(float(mass_target) * total_priority)
            topk_count = int(torch.searchsorted(cumulative, target, right=False).item()) + 1

    topk_count = max(0, min(num_nodes, int(topk_count)))
    high_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if topk_count > 0:
        high_idx = torch.topk(priority, k=topk_count, largest=True, sorted=False).indices
        high_mask[high_idx] = True
    low_mask = ~high_mask

    positive_priority = priority.clamp(min=0)
    priority_mass = 0.0
    if total_priority > 0.0 and topk_count > 0:
        priority_mass = float(positive_priority[high_mask].sum().item() / total_priority)

    total_degree = float(degree.sum().item())
    degree_mass = 0.0
    if total_degree > 0.0 and topk_count > 0:
        degree_mass = float(degree[high_mask].sum().item() / total_degree)

    activation_bits = torch.full((num_nodes,), int(args.internal_calib_low_a_bit), dtype=torch.int64, device=device)
    activation_bits[high_mask] = int(args.internal_calib_high_a_bit)

    report = {
        "priority_policy": str(args.internal_split_priority),
        "strategy": strategy,
        "requested_topk_ratio": float(topk_ratio),
        "requested_mass_target": float(mass_target),
        "node_count": int(num_nodes),
        "high_count": int(high_mask.sum().item()),
        "low_count": int(low_mask.sum().item()),
        "topk_ratio": 0.0 if num_nodes <= 0 else float(high_mask.float().mean().item()),
        "priority_mass": float(priority_mass),
        "degree_mass": float(degree_mass),
        "total_priority": float(total_priority),
        "total_degree": float(total_degree),
        "high_bit": int(args.internal_calib_high_a_bit),
        "low_bit": int(args.internal_calib_low_a_bit),
        "priority_mean": float(priority.float().mean().item()) if num_nodes > 0 else 0.0,
        "priority_high_mean": float(priority[high_mask].float().mean().item()) if bool(high_mask.any()) else 0.0,
        "priority_low_mean": float(priority[low_mask].float().mean().item()) if bool(low_mask.any()) else 0.0,
        "degree_mean": float(degree.float().mean().item()) if num_nodes > 0 else 0.0,
        "degree_high_mean": float(degree[high_mask].float().mean().item()) if bool(high_mask.any()) else 0.0,
        "degree_low_mean": float(degree[low_mask].float().mean().item()) if bool(low_mask.any()) else 0.0,
    }
    if scores is not None:
        report["tser_mean"] = float(scores["sensitivity_q"].float().mean().item())
        report["tser_high_mean"] = (
            float(scores["sensitivity_q"][high_mask].float().mean().item()) if bool(high_mask.any()) else 0.0
        )
        report["tser_low_mean"] = (
            float(scores["sensitivity_q"][low_mask].float().mean().item()) if bool(low_mask.any()) else 0.0
        )

    return {
        "high_indices": high_mask.nonzero(as_tuple=False).view(-1).detach().cpu().numpy().astype(np.int64),
        "low_indices": low_mask.nonzero(as_tuple=False).view(-1).detach().cpu().numpy().astype(np.int64),
        "activation_bits": activation_bits.detach().cpu().numpy().astype(np.int64),
        "degree": degree.detach().cpu().numpy().astype(np.float32),
        "priority": priority.detach().cpu().numpy().astype(np.float32),
        "report": report,
    }


def allocate_internal_split_node_budgets(high_available, low_available, node_budget, high_ratio):
    if int(node_budget) <= 0:
        return 0, 0

    high_available = max(0, int(high_available))
    low_available = max(0, int(low_available))
    node_budget = int(node_budget)
    requested_high = int(round(float(node_budget) * float(high_ratio)))
    requested_high = max(0, min(node_budget, requested_high))

    high_budget = min(high_available, requested_high)
    low_budget = min(low_available, node_budget - high_budget)
    remaining = node_budget - high_budget - low_budget

    if remaining > 0 and high_available > high_budget:
        extra = min(remaining, high_available - high_budget)
        high_budget += extra
        remaining -= extra
    if remaining > 0 and low_available > low_budget:
        extra = min(remaining, low_available - low_budget)
        low_budget += extra
    return int(high_budget), int(low_budget)


def _sample_uniform_indices(indices, target_samples, rng):
    arr = np.asarray(indices, dtype=np.int64)
    if arr.size == 0 or int(target_samples) <= 0:
        return np.zeros(0, dtype=np.int64)
    if int(target_samples) >= int(arr.size):
        arr = arr.copy()
        rng.shuffle(arr)
        return arr
    chosen = np.asarray(rng.choice(arr, size=int(target_samples), replace=False), dtype=np.int64)
    rng.shuffle(chosen)
    return chosen


def _bucket_info_report_entry(info):
    return {
        "name": str(info.get("name")),
        "size": int(info.get("size", 0)),
        "mean_degree": float(info.get("mean_degree", 0.0)),
        "min_degree": float(info.get("min_degree", 0.0)),
        "max_degree": float(info.get("max_degree", 0.0)),
    }


def _build_degree_bucket_infos(degree, mode="degree_quantile", bucket_count=4, topk_ratio=0.1):
    degree = np.asarray(degree, dtype=np.float32)
    node_count = int(degree.shape[0])
    if node_count <= 0:
        return []

    if str(mode) == "degree_topk":
        ratio = 0.1 if float(topk_ratio) < 0.0 else float(topk_ratio)
        topk_count = max(0, min(node_count, int(np.ceil(ratio * node_count))))
        high_mask = np.zeros(node_count, dtype=bool)
        if topk_count > 0:
            high_idx = np.argpartition(-degree, kth=max(topk_count - 1, 0))[:topk_count]
            high_mask[high_idx] = True
        bucket_specs = [
            ("high_degree", np.where(high_mask)[0]),
            ("low_degree", np.where(~high_mask)[0]),
        ]
    else:
        bucket_count = max(1, int(bucket_count))
        order = np.argsort(degree, kind="stable")
        bucket_specs = [
            (f"degree_bucket_{idx}", split.astype(np.int64, copy=False))
            for idx, split in enumerate(np.array_split(order, bucket_count))
        ]

    infos = []
    for name, idx in bucket_specs:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        bucket_degree = degree[idx]
        infos.append(
            {
                "name": str(name),
                "indices": idx,
                "size": int(idx.size),
                "mean_degree": float(bucket_degree.mean()),
                "min_degree": float(bucket_degree.min()),
                "max_degree": float(bucket_degree.max()),
            }
        )
    return infos


def _restrict_bucket_infos_to_indices(bucket_infos, degree, allowed_indices):
    degree = np.asarray(degree, dtype=np.float32)
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    if allowed.size == 0:
        return []

    allowed = allowed[(allowed >= 0) & (allowed < degree.shape[0])]
    allowed_mask = np.zeros(degree.shape[0], dtype=bool)
    allowed_mask[allowed] = True

    restricted = []
    for info in bucket_infos:
        idx = np.asarray(info["indices"], dtype=np.int64)
        idx = idx[allowed_mask[idx]]
        if idx.size == 0:
            continue
        bucket_degree = degree[idx]
        restricted.append(
            {
                "name": str(info["name"]),
                "indices": idx,
                "size": int(idx.size),
                "mean_degree": float(bucket_degree.mean()),
                "min_degree": float(bucket_degree.min()),
                "max_degree": float(bucket_degree.max()),
            }
        )
    return restricted


def _filter_bucket_infos_excluding_indices(bucket_infos, degree, excluded_indices):
    degree = np.asarray(degree, dtype=np.float32)
    excluded = np.asarray(excluded_indices, dtype=np.int64)
    if excluded.size == 0:
        return list(bucket_infos)

    excluded = excluded[(excluded >= 0) & (excluded < degree.shape[0])]
    excluded_mask = np.zeros(degree.shape[0], dtype=bool)
    excluded_mask[excluded] = True

    filtered = []
    for info in bucket_infos:
        idx = np.asarray(info["indices"], dtype=np.int64)
        idx = idx[~excluded_mask[idx]]
        if idx.size == 0:
            continue
        bucket_degree = degree[idx]
        filtered.append(
            {
                "name": str(info["name"]),
                "indices": idx,
                "size": int(idx.size),
                "mean_degree": float(bucket_degree.mean()),
                "min_degree": float(bucket_degree.min()),
                "max_degree": float(bucket_degree.max()),
            }
        )
    return filtered


def _sample_balanced_from_bucket_infos(bucket_infos, target_samples, rng):
    available = int(sum(int(info["size"]) for info in bucket_infos))
    if available <= 0:
        return np.zeros(0, dtype=np.int64), []

    if int(target_samples) <= 0 or int(target_samples) >= available:
        merged = np.concatenate([np.asarray(info["indices"], dtype=np.int64) for info in bucket_infos], axis=0)
        merged = merged.copy()
        rng.shuffle(merged)
        return merged, [int(info["size"]) for info in bucket_infos]

    nonempty = [info for info in bucket_infos if int(info["size"]) > 0]
    if not nonempty:
        return np.zeros(0, dtype=np.int64), [0 for _ in bucket_infos]

    per_bucket = int(target_samples) // len(nonempty)
    remainder = int(target_samples) % len(nonempty)
    selected = []
    leftovers = []
    counts = [0 for _ in bucket_infos]

    for orig_idx, info in enumerate(bucket_infos):
        arr = np.asarray(info["indices"], dtype=np.int64).copy()
        if arr.size == 0:
            continue
        rng.shuffle(arr)
        target = per_bucket + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        take = min(int(arr.size), int(target))
        counts[orig_idx] += int(take)
        if take > 0:
            selected.extend((orig_idx, int(value)) for value in arr[:take].tolist())
        if take < arr.size:
            leftovers.extend((orig_idx, int(value)) for value in arr[take:].tolist())

    if len(selected) < int(target_samples) and leftovers:
        rng.shuffle(leftovers)
        need = int(target_samples) - len(selected)
        for orig_idx, value in leftovers[:need]:
            counts[orig_idx] += 1
            selected.append((orig_idx, int(value)))

    chosen = np.asarray([value for _, value in selected[: int(target_samples)]], dtype=np.int64)
    rng.shuffle(chosen)
    return chosen, counts


def sample_internal_split_nodes(candidate_indices, target_samples, degree, args, rng, path_name):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    report = {
        "path_name": str(path_name),
        "strategy": str(args.internal_calib_strategy),
        "candidate_count": int(candidate_indices.size),
        "selected_count": 0,
        "bucket_mode": None,
        "bucket_count": 0,
        "bucket_sample_counts": [],
        "bucket_stats": [],
        "uniform_ratio": 0.0,
        "uniform_selected_count": 0,
        "bucket_selected_count": 0,
    }

    if candidate_indices.size == 0 or int(target_samples) == 0:
        return np.zeros(0, dtype=np.int64), report

    if int(target_samples) < 0 or int(target_samples) >= int(candidate_indices.size):
        selected = candidate_indices.copy()
        rng.shuffle(selected)
        report["selected_count"] = int(selected.size)
        report["strategy"] = "all_candidates"
        return selected, report

    if str(args.internal_calib_strategy) != "topology":
        selected = _sample_uniform_indices(candidate_indices, int(target_samples), rng)
        report["selected_count"] = int(selected.size)
        report["strategy"] = "uniform_subset_sampling"
        return selected, report

    bucket_infos = _build_degree_bucket_infos(
        degree=degree,
        mode=args.internal_calib_bucket_mode,
        bucket_count=args.internal_calib_bucket_count,
        topk_ratio=(
            args.internal_split_topk_ratio
            if float(args.internal_calib_bucket_topk_ratio) < 0.0
            else args.internal_calib_bucket_topk_ratio
        ),
    )
    bucket_infos = _restrict_bucket_infos_to_indices(bucket_infos, degree, candidate_indices)

    uniform_budget = int(round(float(target_samples) * float(args.internal_calib_uniform_ratio)))
    uniform_budget = max(0, min(int(target_samples), int(uniform_budget)))
    uniform_idx = _sample_uniform_indices(candidate_indices, uniform_budget, rng)
    bucket_candidate_infos = _filter_bucket_infos_excluding_indices(bucket_infos, degree, uniform_idx)
    bucket_budget = max(0, int(target_samples) - int(uniform_idx.size))
    if bucket_budget > 0:
        bucket_idx, bucket_counts = _sample_balanced_from_bucket_infos(bucket_candidate_infos, bucket_budget, rng)
    else:
        bucket_idx = np.zeros(0, dtype=np.int64)
        bucket_counts = [0 for _ in bucket_candidate_infos]

    selected = np.concatenate([uniform_idx, bucket_idx], axis=0)
    if selected.size > 1:
        selected = selected.copy()
        rng.shuffle(selected)

    report.update(
        {
            "strategy": "topology_subset_sampling",
            "bucket_mode": str(args.internal_calib_bucket_mode),
            "bucket_count": int(len(bucket_candidate_infos)),
            "bucket_sample_counts": [int(value) for value in bucket_counts],
            "bucket_stats": [_bucket_info_report_entry(info) for info in bucket_candidate_infos],
            "uniform_ratio": float(args.internal_calib_uniform_ratio),
            "uniform_selected_count": int(uniform_idx.size),
            "bucket_selected_count": int(bucket_idx.size),
            "selected_count": int(selected.size),
        }
    )
    return selected.astype(np.int64, copy=False), report


def _texts_for_indices(node_texts, indices):
    if node_texts is None:
        return []
    texts = []
    count = len(node_texts)
    for idx in np.asarray(indices, dtype=np.int64).tolist():
        if 0 <= int(idx) < count:
            texts.append(str(node_texts[int(idx)]))
    return texts


def build_internal_split_calibration(ds_key, data, verify_features, args, device):
    node_texts = None
    text_error = None
    try:
        node_texts = load_raw_texts(ds_key)
    except Exception as exc:  # keep calibration usable for graph-only reports
        text_error = str(exc)

    assignment = resolve_high_low_assignment(data, verify_features, args, device)
    node_count = int(data.num_nodes)
    target_samples = int(args.internal_calib_samples)
    if target_samples <= 0:
        node_budget = node_count
    else:
        node_budget = min(node_count, target_samples)

    high_budget, low_budget = allocate_internal_split_node_budgets(
        high_available=len(assignment["high_indices"]),
        low_available=len(assignment["low_indices"]),
        node_budget=node_budget,
        high_ratio=float(args.internal_calib_high_ratio),
    )

    seed = int(getattr(args, "run_seed", getattr(args, "seed", 0)))
    rng = np.random.RandomState(seed)
    high_selected, high_sampling = sample_internal_split_nodes(
        candidate_indices=assignment["high_indices"],
        target_samples=high_budget,
        degree=assignment["degree"],
        args=args,
        rng=rng,
        path_name="high",
    )
    low_selected, low_sampling = sample_internal_split_nodes(
        candidate_indices=assignment["low_indices"],
        target_samples=low_budget,
        degree=assignment["degree"],
        args=args,
        rng=rng,
        path_name="low",
    )

    high_texts = _texts_for_indices(node_texts, high_selected)
    low_texts = _texts_for_indices(node_texts, low_selected)
    high_bit = int(args.internal_calib_high_a_bit)
    low_bit = int(args.internal_calib_low_a_bit)

    bundle = {
        "mode": "internal_split",
        "dataset": str(ds_key),
        "node_count": int(node_count),
        "prompt_selected_count": 0,
        "assignment": {
            "high_indices": assignment["high_indices"].astype(np.int64, copy=False).tolist(),
            "low_indices": assignment["low_indices"].astype(np.int64, copy=False).tolist(),
            "activation_bits": assignment["activation_bits"].astype(np.int64, copy=False).tolist(),
            "high_bit": high_bit,
            "low_bit": low_bit,
        },
        "passes": [
            {
                "name": "high",
                "bit": high_bit,
                "node_indices": high_selected.astype(np.int64, copy=False).tolist(),
                "texts": high_texts,
                "per_text_bits": [high_bit] * len(high_texts),
            },
            {
                "name": "low",
                "bit": low_bit,
                "node_indices": low_selected.astype(np.int64, copy=False).tolist(),
                "texts": low_texts,
                "per_text_bits": [low_bit] * len(low_texts),
            },
        ],
    }

    report = {
        "enabled": True,
        "mode": "internal_split",
        "dataset": str(ds_key),
        "seed": int(seed),
        "node_count": int(node_count),
        "text_count": 0 if node_texts is None else int(len(node_texts)),
        "text_error": text_error,
        "target_samples": int(target_samples),
        "node_budget": int(node_budget),
        "prompt_selected_count": 0,
        "selected_text_count": int(len(high_texts) + len(low_texts)),
        "selected_node_count": int(len(high_selected) + len(low_selected)),
        "high_node_budget": int(high_budget),
        "low_node_budget": int(low_budget),
        "assignment": assignment["report"],
        "sampling": {
            "enabled": True,
            "strategy": "internal_degree_aware_split_calibration",
            "base_strategy": str(args.internal_calib_strategy),
            "high_sampling": high_sampling,
            "low_sampling": low_sampling,
        },
    }
    return bundle, report


def save_internal_split_calibration(bundle, report, out_dir, ds_key, seed):
    os.makedirs(out_dir, exist_ok=True)
    assignment = report.get("assignment", {})
    sampling = report.get("sampling", {})
    priority = str(assignment.get("priority_policy", "unknown")).replace("/", "_")
    strategy = str(sampling.get("base_strategy", "unknown")).replace("/", "_")
    samples = int(report.get("target_samples", 0))
    path = os.path.join(
        out_dir,
        f"{ds_key}_internal_split_calibration_{priority}_{strategy}_n{samples}_seed{int(seed)}.json",
    )
    payload = {"report": report, "bundle": bundle}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
