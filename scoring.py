from dataclasses import dataclass

import torch
import torch.nn.functional as F


def quantize_0_15(values):
    return torch.round(values.clamp(0.0, 1.0) * 15.0).to(torch.int64)


def build_random_matrix(input_dim, bits, device, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(int(input_dim), int(bits), generator=generator)
    matrix = F.normalize(matrix, p=2, dim=0)
    return matrix.to(device)


def hash_bits(features, matrix):
    return torch.matmul(features, matrix) > 0


def hash_bits_to_ints(bits):
    vals = []
    arr = bits.detach().cpu().numpy().astype("uint8")
    for row in arr:
        val = 0
        for bit in row:
            val = (val << 1) | int(bit)
        vals.append(val)
    return vals


def bucket_rarity_q(features, bits, seed):
    """Estimate global self-feature rarity using exact SimHash bucket density."""
    matrix = build_random_matrix(features.size(1), bits, features.device, seed)
    hashes = hash_bits_to_ints(hash_bits(features, matrix))
    counts = {}
    for h in hashes:
        counts[h] = counts.get(h, 0) + 1

    similar_count = torch.tensor(
        [max(0, counts[h] - 1) for h in hashes],
        dtype=torch.int64,
        device=features.device,
    )
    rarity_q = torch.full_like(similar_count, 15)
    rarity_q[similar_count >= 1] = 12
    rarity_q[similar_count >= 2] = 8
    rarity_q[similar_count >= 4] = 4
    rarity_q[similar_count >= 8] = 0
    return rarity_q, similar_count


def build_node_risk_scores(
    verify_features,
    hash_features,
    edge_index,
    total_degree,
    context_signature,
    hash_matrix,
    rarity_bits=16,
    rarity_seed=98765,
    propagation_weight=3,
    graph_context_weight=2,
    low_unique_weight=2,
):
    """Build degree-first risk scores with a low-degree rare-leaf correction."""
    device = verify_features.device
    num_nodes = verify_features.size(0)

    propagation_risk = torch.log1p(total_degree.float()) / torch.log1p(total_degree.max().clamp(min=1.0))
    propagation_q = quantize_0_15(propagation_risk)

    context_bits = hash_bits(hash_features, hash_matrix)
    sketch_bits = max(1, int(context_bits.size(1)))
    row, col = edge_index
    sym_row = torch.cat([row, col], dim=0)
    sym_col = torch.cat([col, row], dim=0)
    edge_dist = (context_bits[sym_row] != context_bits[sym_col]).sum(dim=1).float()
    sum_dist = torch.zeros(num_nodes, dtype=torch.float, device=device)
    count = torch.zeros(num_nodes, dtype=torch.float, device=device)
    sum_dist.index_add_(0, sym_row, edge_dist)
    count.index_add_(0, sym_row, torch.ones_like(edge_dist))
    neutral_dist = torch.full_like(sum_dist, sketch_bits * 0.5)
    avg_dist = torch.where(count > 0, sum_dist / count.clamp(min=1.0), neutral_dist)
    boundary_risk = avg_dist / float(sketch_bits)

    self_norm = F.normalize(verify_features, p=2, dim=1)
    context_norm = F.normalize(context_signature, p=2, dim=1)
    context_shift = (1.0 - (self_norm * context_norm).sum(dim=1)).clamp(0.0, 2.0) * 0.5
    graph_context_risk = torch.maximum(boundary_risk, context_shift)
    graph_context_q = quantize_0_15(graph_context_risk)

    rarity_q, similar_count = bucket_rarity_q(verify_features, rarity_bits, rarity_seed)
    low_degree_factor_q = 15 - propagation_q
    low_degree_unique_q = torch.round(
        low_degree_factor_q.float() * rarity_q.float() / 15.0
    ).to(torch.int64)

    sensitivity_q = (
        propagation_weight * propagation_q
        + graph_context_weight * graph_context_q
        + low_unique_weight * low_degree_unique_q
    )

    return {
        "propagation_risk": propagation_risk,
        "boundary_risk": boundary_risk,
        "context_shift": context_shift,
        "graph_context_risk": graph_context_risk,
        "propagation_q": propagation_q,
        "graph_context_q": graph_context_q,
        "rarity_q": rarity_q,
        "similar_count": similar_count,
        "low_degree_unique_q": low_degree_unique_q,
        "sensitivity_q": sensitivity_q,
    }


def reuse_error_q(hamming_dist, route_hit_count=1, support_discount=True):
    hamming_dist = int(hamming_dist)
    if hamming_dist <= 0:
        return 1
    if hamming_dist == 1:
        error = 2
    elif hamming_dist == 2:
        error = 4
    else:
        error = max(4, 2 * hamming_dist)

    if not support_discount:
        return error

    support = max(1, int(route_hit_count))
    if support >= 4:
        return max(1, error - 2)
    if support >= 2:
        return max(1, error - 1)
    return error


@dataclass
class RiskGateConfig:
    enabled: bool = True
    reuse_threshold: int = 45
    hub_threshold: int = 12
    rare_threshold: int = 10
    protect_hub_exact: bool = False
    protect_hub_fuzzy: bool = True
    forbid_rare_fuzzy: bool = True
    support_discount: bool = True
    rare_gate_mode: str = "support"
    rare_min_dist: int = 2
    rare_min_route_hits: int = 2
    rare_min_base_hits: int = 2
    confidence_discount: int = 1
    confidence_max_dist: int = 1
    confidence_min_route_hits: int = 2
    confidence_min_base_hits: int = 2
    confidence_min_cos_margin: float = 0.02


@dataclass
class QuantPolicyConfig:
    enabled: bool = False
    int4_threshold: int = 90
    int8_threshold: int = 45
    int4_error: int = 3
    int8_error: int = 1


class ReuseRiskGate:
    def __init__(self, scores, config=None):
        self.scores = scores
        self.config = config or RiskGateConfig()

    def evaluate(
        self,
        node_idx,
        hamming_dist,
        route_hit_count=1,
        base_route_hit_count=1,
        cos_margin=None,
    ):
        dist = int(hamming_dist)
        sensitivity = int(self.scores["sensitivity_q"][node_idx])
        propagation = int(self.scores["propagation_q"][node_idx])
        graph_context = int(self.scores["graph_context_q"][node_idx])
        low_unique = int(self.scores["low_degree_unique_q"][node_idx])
        rarity = int(self.scores["rarity_q"][node_idx])
        error = reuse_error_q(
            dist,
            route_hit_count=route_hit_count,
            support_discount=self.config.support_discount,
        )
        confidence_discount = self._confidence_discount(
            dist=dist,
            route_hit_count=route_hit_count,
            base_route_hit_count=base_route_hit_count,
            cos_margin=cos_margin,
        )
        if confidence_discount > 0:
            error = max(1, error - confidence_discount)
        risk = sensitivity * error

        result = {
            "allow": True,
            "reason": "allow",
            "risk": risk,
            "sensitivity": sensitivity,
            "approx_error": error,
            "propagation": propagation,
            "graph_context": graph_context,
            "low_unique": low_unique,
            "rarity": rarity,
            "route_hit_count": int(route_hit_count),
            "base_route_hit_count": int(base_route_hit_count),
            "confidence_discount": int(confidence_discount),
        }
        if not self.config.enabled:
            return result

        if (
            propagation >= self.config.hub_threshold
            and (
                (dist > 0 and self.config.protect_hub_fuzzy)
                or (dist == 0 and self.config.protect_hub_exact)
            )
        ):
            result["allow"] = False
            result["reason"] = "hub_protect"
            return result

        if self._rare_gate_rejects(
            low_unique=low_unique,
            dist=dist,
            route_hit_count=route_hit_count,
            base_route_hit_count=base_route_hit_count,
        ):
            result["allow"] = False
            result["reason"] = "rare_leaf"
            return result

        if risk > self.config.reuse_threshold:
            result["allow"] = False
            result["reason"] = "risk"
            return result

        return result

    def _confidence_discount(self, dist, route_hit_count, base_route_hit_count, cos_margin):
        discount = max(0, int(self.config.confidence_discount))
        if discount <= 0:
            return 0
        if int(dist) > int(self.config.confidence_max_dist):
            return 0

        route_supported = int(route_hit_count) >= int(self.config.confidence_min_route_hits)
        base_supported = int(base_route_hit_count) >= int(self.config.confidence_min_base_hits)
        cos_supported = (
            cos_margin is not None
            and float(cos_margin) >= float(self.config.confidence_min_cos_margin)
        )
        if route_supported or base_supported or cos_supported:
            return discount
        return 0

    def _rare_gate_rejects(self, low_unique, dist, route_hit_count, base_route_hit_count):
        if not self.config.forbid_rare_fuzzy:
            return False
        if int(low_unique) < int(self.config.rare_threshold) or int(dist) <= 0:
            return False

        mode = str(self.config.rare_gate_mode).lower()
        if mode == "risk":
            return False
        if mode == "hard":
            return True
        if mode != "support":
            raise ValueError(f"Unknown rare_gate_mode={self.config.rare_gate_mode}")

        if int(dist) < int(self.config.rare_min_dist):
            return False
        route_supported = int(route_hit_count) >= int(self.config.rare_min_route_hits)
        base_supported = int(base_route_hit_count) >= int(self.config.rare_min_base_hits)
        return not (route_supported or base_supported)


class QuantExecutionPolicy:
    def __init__(self, scores, config=None):
        self.scores = scores
        self.config = config or QuantPolicyConfig()

    def decide(self, node_idx):
        sensitivity = int(self.scores["sensitivity_q"][node_idx])
        propagation = int(self.scores["propagation_q"][node_idx])
        graph_context = int(self.scores["graph_context_q"][node_idx])
        low_unique = int(self.scores["low_degree_unique_q"][node_idx])
        rarity = int(self.scores["rarity_q"][node_idx])

        int4_error = max(1, int(self.config.int4_error))
        int8_error = max(1, int(self.config.int8_error))
        int4_risk = sensitivity * int4_error
        int8_risk = sensitivity * int8_error

        if not self.config.enabled:
            action = "full_precision"
            reason = "quant_disabled"
        elif int4_risk <= int(self.config.int4_threshold):
            action = "int4"
            reason = "int4_safe"
        elif int8_risk <= int(self.config.int8_threshold):
            action = "int8"
            reason = "int8_safe"
        else:
            action = "protected"
            reason = "risk_protect"

        return {
            "action": action,
            "reason": reason,
            "sensitivity": sensitivity,
            "propagation": propagation,
            "graph_context": graph_context,
            "low_unique": low_unique,
            "rarity": rarity,
            "int4_error": int4_error,
            "int8_error": int8_error,
            "int4_risk": int4_risk,
            "int8_risk": int8_risk,
        }


def summarize_scores(scores):
    def stat(t):
        return float(t.float().min().item()), float(t.float().mean().item()), float(t.float().max().item())

    return {
        "propagation_q": stat(scores["propagation_q"]),
        "graph_context_q": stat(scores["graph_context_q"]),
        "rarity_q": stat(scores["rarity_q"]),
        "low_degree_unique_q": stat(scores["low_degree_unique_q"]),
        "sensitivity_q": stat(scores["sensitivity_q"]),
    }
