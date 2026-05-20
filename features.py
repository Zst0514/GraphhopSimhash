import torch
import torch.nn.functional as F

# Inlined hash reuse implementation

HASH_VIEW_PRESETS = {
    "self": (1.0, 0.0, 0.0),
    "self_1hop_2hop": (0.2, 0.6, 0.2),
}


def _compute_neighbor_mean(features, edge_index):
    num_nodes = features.size(0)
    row, col = edge_index
    sym_row = torch.cat([row, col], dim=0)
    sym_col = torch.cat([col, row], dim=0)

    neighbor_sum = torch.zeros_like(features)
    neighbor_sum.index_add_(0, sym_row, features[sym_col])

    total_degree = torch.zeros(num_nodes, device=features.device)
    total_degree.index_add_(0, sym_row, torch.ones(sym_row.size(0), device=features.device))

    neighbor_mean = neighbor_sum / total_degree.clamp(min=1).unsqueeze(1)
    isolated_mask = total_degree == 0
    if isolated_mask.any():
        neighbor_mean[isolated_mask] = features[isolated_mask]
    return neighbor_mean


def build_hash_features(verify_features, edge_index, hash_view, hash_mix_weights=None):
    if hash_mix_weights is None:
        weights = HASH_VIEW_PRESETS[hash_view]
    else:
        weights = tuple(float(weight) for weight in hash_mix_weights)
        if len(weights) != 3:
            raise ValueError("hash_mix_weights must contain exactly 3 values")

    weights_sum = sum(weights)
    if weights_sum <= 0:
        raise ValueError("hash view weights must sum to a positive value")
    weights = tuple(weight / weights_sum for weight in weights)

    self_feat = verify_features
    hop1_feat = None
    hop2_feat = None

    if weights[1] > 0.0 or weights[2] > 0.0:
        hop1_feat = _compute_neighbor_mean(self_feat, edge_index)
    if weights[2] > 0.0:
        hop2_feat = _compute_neighbor_mean(F.normalize(hop1_feat, p=2, dim=1), edge_index)

    mixed_feat = weights[0] * self_feat
    if hop1_feat is not None:
        mixed_feat = mixed_feat + weights[1] * hop1_feat
    if hop2_feat is not None:
        mixed_feat = mixed_feat + weights[2] * hop2_feat

    return F.normalize(mixed_feat, p=2, dim=1), weights


def build_topology_hash_features(verify_features, edge_index):
    neighbor_mean = _compute_neighbor_mean(verify_features, edge_index)
    topology_feat = 0.5 * verify_features + 0.5 * neighbor_mean
    return F.normalize(topology_feat, p=2, dim=1)
def make_hash_view_tag(hash_view, hash_mix_weights):
    if hash_mix_weights is None:
        return hash_view
    rounded = [round(float(weight), 2) for weight in hash_mix_weights]
    return "mix_" + "_".join(f"{value:.2f}".replace(".", "p") for value in rounded)


def build_hash_feature_routes(
    verify_features,
    edge_index,
    hash_view,
    hash_mix_weights=None,
    union_hash_views=None,
):
    route_specs = []
    seen_tags = set()

    def add_route(route_view, route_mix_weights=None):
        route_tag = make_hash_view_tag(route_view, route_mix_weights)
        if route_tag in seen_tags:
            return
        base_route_idx = len(route_specs)
        route_features, route_weights = build_hash_features(
            verify_features,
            edge_index,
            route_view,
            route_mix_weights,
        )
        route_specs.append(
            {
                "name": route_tag,
                "view": route_view,
                "features": route_features,
                "weights": route_weights,
                "base_route_idx": base_route_idx,
                "base_name": route_tag,
                "table_idx": 0,
                "table_count": 1,
            }
        )
        seen_tags.add(route_tag)

    add_route(hash_view, hash_mix_weights)
    for extra_view in union_hash_views or []:
        add_route(extra_view, None)

    return route_specs


def _format_hash_view(weights):
    return f"self={weights[0]:.2f},1hop={weights[1]:.2f},2hop={weights[2]:.2f}"


def format_hash_route_specs(route_specs, route_score_weights=None):
    parts = []
    for idx, spec in enumerate(route_specs):
        route_role = str(spec.get("route_role", "semantic"))
        base_route_idx = int(spec.get("base_route_idx", idx))
        role = "main" if base_route_idx == 0 else f"union{base_route_idx}"
        if route_role == "topology":
            role = "topology"
        table_idx = int(spec.get("table_idx", 0))
        if route_role == "topology":
            part = f"{role}:{spec['name']}(context)"
        else:
            part = f"{role}:{spec['name']}({_format_hash_view(spec['weights'])})"
        if table_idx > 0:
            part += f"[h{table_idx}]"
        if route_score_weights is not None and idx < len(route_score_weights):
            part += f"[w={route_score_weights[idx]:.2f}]"
        parts.append(part)
    return " | ".join(parts)
