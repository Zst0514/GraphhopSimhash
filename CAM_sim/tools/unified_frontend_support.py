from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_tser_scores(
    *,
    build_node_risk_scores,
    compute_neighbor_mean,
    verify_features: torch.Tensor,
    hash_features: torch.Tensor,
    hash_matrix: torch.Tensor,
    edge_index: torch.Tensor,
    rarity_bits: int,
    rarity_seed: int,
    propagation_weight: int,
    graph_context_weight: int,
    low_unique_weight: int,
) -> dict:
    device = verify_features.device
    row, col = edge_index
    sym_row = torch.cat([row, col], dim=0)
    total_degree = torch.zeros(verify_features.size(0), dtype=torch.float32, device=device)
    total_degree.index_add_(0, sym_row, torch.ones(sym_row.numel(), dtype=torch.float32, device=device))

    neighbor_mean = compute_neighbor_mean(verify_features, edge_index)
    context_signature = F.normalize(0.5 * verify_features + 0.5 * neighbor_mean, p=2, dim=1)

    return build_node_risk_scores(
        verify_features=verify_features,
        hash_features=hash_features,
        edge_index=edge_index,
        total_degree=total_degree,
        context_signature=context_signature,
        hash_matrix=hash_matrix,
        rarity_bits=rarity_bits,
        rarity_seed=rarity_seed,
        propagation_weight=propagation_weight,
        graph_context_weight=graph_context_weight,
        low_unique_weight=low_unique_weight,
    )
