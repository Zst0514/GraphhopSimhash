import os

import torch
import torch.nn.functional as F

from .config import DATASET_CONFIGS
from .data import load_cheap_features
from .features import _compute_neighbor_mean, build_hash_features
from .scoring import build_node_risk_scores, build_random_matrix


def _load_processed_graph(ds_key, device):
    ds_key = ds_key.lower()
    if ds_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset for calibration: {ds_key}")

    data_dir = DATASET_CONFIGS[ds_key]["data_dir"]
    processed_path = os.path.join(
        "cache_data",
        data_dir,
        "ST",
        "processed",
        "geometric_data_processed.pt",
    )
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"{processed_path} is required for graph-aware calibration. "
            "Run the normal GraphhopSimhash pipeline once to build the ST graph cache."
        )

    loaded = torch.load(processed_path, map_location=device)
    data = loaded[0] if isinstance(loaded, tuple) else loaded
    if not hasattr(data, "edge_index"):
        raise ValueError(f"{processed_path} does not contain edge_index")
    return data.to(device)


def _total_degree(edge_index, num_nodes, device):
    row, col = edge_index
    sym_row = torch.cat([row, col], dim=0)
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, sym_row, torch.ones(sym_row.size(0), device=device))
    return degree


def _make_minimal_score_args(
    hash_bits,
    hash_mix_weights,
    controller_seed,
    rarity_bits,
    rarity_seed,
    propagation_weight,
    graph_context_weight,
    low_unique_weight,
):
    return {
        "hash_bits": int(hash_bits),
        "hash_mix_weights": tuple(float(v) for v in hash_mix_weights),
        "controller_seed": int(controller_seed),
        "rarity_bits": int(rarity_bits),
        "rarity_seed": int(rarity_seed),
        "propagation_weight": int(propagation_weight),
        "graph_context_weight": int(graph_context_weight),
        "low_unique_weight": int(low_unique_weight),
    }


def build_calibration_scores(
    ds_key,
    device,
    hash_bits=14,
    hash_mix_weights=(0.3, 0.7, 0.0),
    controller_seed=42,
    rarity_bits=16,
    rarity_seed=98765,
    propagation_weight=3,
    graph_context_weight=2,
    low_unique_weight=2,
):
    score_args = _make_minimal_score_args(
        hash_bits=hash_bits,
        hash_mix_weights=hash_mix_weights,
        controller_seed=controller_seed,
        rarity_bits=rarity_bits,
        rarity_seed=rarity_seed,
        propagation_weight=propagation_weight,
        graph_context_weight=graph_context_weight,
        low_unique_weight=low_unique_weight,
    )

    data = _load_processed_graph(ds_key, device)
    verify_features = load_cheap_features(ds_key.lower(), data, device)
    hash_features, _weights = build_hash_features(
        verify_features,
        data.edge_index,
        "self",
        score_args["hash_mix_weights"],
    )

    neighbor_mean = _compute_neighbor_mean(verify_features, data.edge_index)
    context_signature = F.normalize(0.5 * verify_features + 0.5 * neighbor_mean, p=2, dim=1)
    total_degree = _total_degree(data.edge_index, verify_features.size(0), device)
    hash_matrix = build_random_matrix(
        hash_features.size(1),
        score_args["hash_bits"],
        device,
        score_args["controller_seed"],
    )

    scores = build_node_risk_scores(
        verify_features=verify_features,
        hash_features=hash_features,
        edge_index=data.edge_index,
        total_degree=total_degree,
        context_signature=context_signature,
        hash_matrix=hash_matrix,
        rarity_bits=score_args["rarity_bits"],
        rarity_seed=score_args["rarity_seed"],
        propagation_weight=score_args["propagation_weight"],
        graph_context_weight=score_args["graph_context_weight"],
        low_unique_weight=score_args["low_unique_weight"],
    )
    scores["degree"] = total_degree
    return scores


def _topk_indices(values, k, largest=True):
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    k = min(int(k), int(values.numel()))
    return torch.topk(values.float(), k=k, largest=largest).indices


def _rank_slice_indices(values, start_frac, end_frac):
    total = int(values.numel())
    start = max(0, min(total, int(round(total * float(start_frac)))))
    end = max(start, min(total, int(round(total * float(end_frac)))))
    order = torch.argsort(values.float(), descending=True)
    return order[start:end]


def _dedup_preserve_order(chunks, limit):
    selected = []
    seen = set()
    for chunk in chunks:
        for idx in chunk.detach().cpu().tolist():
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
            if len(selected) >= limit:
                return selected
    return selected


def _random_indices(num_nodes, size, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(num_nodes, generator=generator, device="cpu")[:size].to(device)


def select_calibration_indices(
    ds_key,
    strategy,
    size,
    seed,
    device,
    hash_bits=14,
    hash_mix_weights=(0.3, 0.7, 0.0),
    rarity_bits=16,
    rarity_seed=98765,
    propagation_weight=3,
    graph_context_weight=2,
    low_unique_weight=2,
):
    strategy = strategy.lower()
    if size <= 0:
        raise ValueError("calibration size must be positive")

    if strategy == "prefix":
        return list(range(int(size))), None

    scores = build_calibration_scores(
        ds_key=ds_key,
        device=device,
        hash_bits=hash_bits,
        hash_mix_weights=hash_mix_weights,
        controller_seed=seed,
        rarity_bits=rarity_bits,
        rarity_seed=rarity_seed,
        propagation_weight=propagation_weight,
        graph_context_weight=graph_context_weight,
        low_unique_weight=low_unique_weight,
    )
    num_nodes = int(scores["sensitivity_q"].numel())
    size = min(int(size), num_nodes)
    random_fill = _random_indices(num_nodes, num_nodes, seed, scores["sensitivity_q"].device)

    if strategy == "random":
        selected = random_fill[:size].detach().cpu().tolist()
    elif strategy == "degree":
        selected = _topk_indices(scores["degree"], size, largest=True).detach().cpu().tolist()
    elif strategy == "tser":
        selected = _topk_indices(scores["sensitivity_q"], size, largest=True).detach().cpu().tolist()
    elif strategy == "tser_stratified":
        n_high = max(1, int(round(size * 0.50)))
        n_context = max(0, int(round(size * 0.20)))
        n_unique = max(0, int(round(size * 0.20)))
        n_mid = max(0, int(round(size * 0.10)))
        chunks = [
            _topk_indices(scores["sensitivity_q"], n_high, largest=True),
            _topk_indices(scores["graph_context_q"], n_context, largest=True),
            _topk_indices(scores["low_degree_unique_q"], n_unique, largest=True),
            _rank_slice_indices(scores["sensitivity_q"], 0.45, 0.65)[:n_mid],
            random_fill,
        ]
        selected = _dedup_preserve_order(chunks, size)
    else:
        raise ValueError(f"Unknown calibration strategy: {strategy}")

    if len(selected) < size:
        selected = _dedup_preserve_order(
            [torch.tensor(selected, device=scores["sensitivity_q"].device), random_fill],
            size,
        )

    selected_tensor = torch.tensor(selected, dtype=torch.long, device=scores["sensitivity_q"].device)
    summary = {
        "num_nodes": num_nodes,
        "sensitivity_all": float(scores["sensitivity_q"].float().mean().item()),
        "sensitivity_selected": float(scores["sensitivity_q"][selected_tensor].float().mean().item()),
        "degree_all": float(scores["degree"].float().mean().item()),
        "degree_selected": float(scores["degree"][selected_tensor].float().mean().item()),
        "graph_context_all": float(scores["graph_context_q"].float().mean().item()),
        "graph_context_selected": float(scores["graph_context_q"][selected_tensor].float().mean().item()),
        "low_unique_all": float(scores["low_degree_unique_q"].float().mean().item()),
        "low_unique_selected": float(scores["low_degree_unique_q"][selected_tensor].float().mean().item()),
    }
    return selected, summary
