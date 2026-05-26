import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import _compute_neighbor_mean


class LowRankResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, rank=32):
        super().__init__()
        self.down = nn.Linear(int(input_dim), int(rank))
        self.up = nn.Linear(int(rank), int(output_dim))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


def compute_total_degree(edge_index, num_nodes, device):
    row, col = edge_index
    sym_nodes = torch.cat([row, col], dim=0)
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, sym_nodes, torch.ones(sym_nodes.numel(), dtype=torch.float32, device=device))
    return degree


def build_context_signature(verify_features, edge_index):
    neighbor_mean = _compute_neighbor_mean(verify_features, edge_index)
    return F.normalize(0.5 * verify_features + 0.5 * neighbor_mean, p=2, dim=1)


def build_residual_pair_inputs(verify_features, edge_index, trace, node_indices, risk_scores=None):
    device = verify_features.device
    node_indices = node_indices.to(device=device, dtype=torch.long)
    source_ids = trace["source_ids"][node_indices].to(device=device, dtype=torch.long)

    context = build_context_signature(verify_features, edge_index)
    degree = compute_total_degree(edge_index, verify_features.size(0), device)

    cheap_delta = verify_features[node_indices] - verify_features[source_ids]
    context_delta = context[node_indices] - context[source_ids]

    dist = trace["best_dists"][node_indices].to(device=device, dtype=torch.float32).clamp(min=0.0)
    route_hits = trace["route_hit_counts"][node_indices].to(device=device, dtype=torch.float32).clamp(min=0.0)
    base_hits = trace["base_route_hit_counts"][node_indices].to(device=device, dtype=torch.float32).clamp(min=0.0)
    base_table_hits = trace["winning_base_table_hit_counts"][node_indices].to(device=device, dtype=torch.float32).clamp(min=0.0)
    best_cos = trace["best_cosines"][node_indices].to(device=device, dtype=torch.float32)

    deg_v = degree[node_indices]
    deg_u = degree[source_ids]
    log_degree_ratio = torch.log1p(deg_v) - torch.log1p(deg_u)
    cheap_cos = F.cosine_similarity(verify_features[node_indices], verify_features[source_ids], dim=1)
    context_cos = F.cosine_similarity(context[node_indices], context[source_ids], dim=1)

    if risk_scores is not None and "sensitivity_q" in risk_scores:
        sensitivity = risk_scores["sensitivity_q"][node_indices].to(device=device, dtype=torch.float32) / 64.0
    else:
        sensitivity = torch.zeros_like(dist)

    scalars = torch.stack(
        [
            dist / 16.0,
            route_hits / 8.0,
            base_hits / 4.0,
            base_table_hits / 8.0,
            best_cos,
            cheap_cos,
            context_cos,
            log_degree_ratio,
            sensitivity,
        ],
        dim=1,
    )
    return torch.cat([cheap_delta, context_delta, scalars], dim=1)


def select_residual_train_nodes(
    trace,
    data,
    split="train_val",
    max_pairs=4096,
    correction_mask=None,
    min_dist=0.0,
):
    hit_mask = trace["hit_mask"]
    source_ok = trace["source_ids"] >= 0
    mask = hit_mask & source_ok
    if correction_mask is not None:
        mask = mask & correction_mask.to(device=mask.device, dtype=torch.bool)
    if float(min_dist) > 0.0:
        mask = mask & (trace["best_dists"].to(device=mask.device, dtype=torch.float32) >= float(min_dist))
    if split == "train":
        mask = mask & data.train_mask
    elif split == "train_val":
        mask = mask & (data.train_mask | data.val_mask)
    elif split == "all_hits":
        pass
    else:
        raise ValueError(f"Unknown residual train split: {split}")

    indices = mask.nonzero(as_tuple=False).view(-1)
    if int(max_pairs) > 0 and indices.numel() > int(max_pairs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        perm = torch.randperm(indices.numel(), generator=generator, device="cpu")[: int(max_pairs)]
        indices = indices[perm.to(indices.device)]
    return indices


def train_residual_adapter(
    target_embeddings,
    verify_features,
    edge_index,
    trace,
    data,
    risk_scores=None,
    rank=32,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    residual_l2=1e-4,
    train_split="train_val",
    max_pairs=4096,
    correction_mask=None,
    min_dist=0.0,
):
    device = target_embeddings.device
    train_nodes = select_residual_train_nodes(
        trace,
        data,
        split=train_split,
        max_pairs=max_pairs,
        correction_mask=correction_mask,
        min_dist=min_dist,
    )
    if train_nodes.numel() == 0:
        return None, {"train_pairs": 0, "loss": 0.0}

    x_train = build_residual_pair_inputs(verify_features, edge_index, trace, train_nodes, risk_scores=risk_scores)
    source_ids = trace["source_ids"][train_nodes].to(device=device, dtype=torch.long)
    anchors = target_embeddings[source_ids]
    targets = target_embeddings[train_nodes]

    adapter = LowRankResidualAdapter(x_train.size(1), target_embeddings.size(1), rank=rank).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    last_loss = 0.0
    for _epoch in range(int(epochs)):
        adapter.train()
        optimizer.zero_grad()
        residual = adapter(x_train)
        pred = F.normalize(anchors + residual, p=2, dim=1)
        target_norm = F.normalize(targets, p=2, dim=1)
        loss = (1.0 - F.cosine_similarity(pred, target_norm, dim=1)).mean()
        if float(residual_l2) > 0.0:
            loss = loss + float(residual_l2) * residual.pow(2).mean()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())

    return adapter, {"train_pairs": int(train_nodes.numel()), "loss": last_loss}


def apply_residual_adapter(
    direct_embeddings,
    target_embeddings,
    verify_features,
    edge_index,
    trace,
    adapter,
    risk_scores=None,
    alpha=1.0,
    min_dist=1.0,
    correction_mask=None,
):
    if adapter is None:
        return direct_embeddings, {"corrected": 0, "alpha": 0.0}
    device = direct_embeddings.device
    hit_nodes = (trace["hit_mask"] & (trace["source_ids"] >= 0)).nonzero(as_tuple=False).view(-1)
    if hit_nodes.numel() == 0:
        return direct_embeddings, {"corrected": 0, "alpha": 0.0}

    if float(min_dist) > 0.0:
        dist = trace["best_dists"][hit_nodes].to(device=device, dtype=torch.float32)
        hit_nodes = hit_nodes[dist >= float(min_dist)]
        if hit_nodes.numel() == 0:
            return direct_embeddings, {"corrected": 0, "alpha": float(alpha)}
    if correction_mask is not None:
        active = correction_mask.to(device=device, dtype=torch.bool)
        hit_nodes = hit_nodes[active[hit_nodes]]
        if hit_nodes.numel() == 0:
            return direct_embeddings, {"corrected": 0, "alpha": float(alpha)}

    adapter.eval()
    corrected = direct_embeddings.clone()
    with torch.no_grad():
        x = build_residual_pair_inputs(verify_features, edge_index, trace, hit_nodes, risk_scores=risk_scores)
        source_ids = trace["source_ids"][hit_nodes].to(device=device, dtype=torch.long)
        anchors = target_embeddings[source_ids]
        residual = adapter(x)
        corrected[hit_nodes] = F.normalize(anchors + float(alpha) * residual, p=2, dim=1)
    return corrected, {"corrected": int(hit_nodes.numel()), "alpha": float(alpha)}


def embedding_error(reference_embeddings, approx_embeddings):
    return (1.0 - F.cosine_similarity(reference_embeddings, approx_embeddings, dim=1)).clamp(min=0.0)
