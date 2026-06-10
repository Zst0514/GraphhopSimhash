#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace ghhw {

struct UnifiedFrontendConfig {
    bool score_gate_enabled = false;
    int score_reuse_threshold = 45;
    int score_hub_threshold = 12;
    int score_rare_threshold = 10;
    bool score_protect_hub_exact = false;
    bool score_protect_hub_fuzzy = true;
    bool score_forbid_rare_fuzzy = true;
    bool score_support_discount = true;
    int score_rare_min_dist = 2;
    int score_rare_min_route_hits = 2;
    int score_rare_min_base_hits = 2;
    int score_pair_confidence_discount = 1;
    int score_pair_confidence_max_dist = 1;
    int score_pair_confidence_min_route_hits = 2;
    int score_pair_confidence_min_base_hits = 2;
    int direct_support_threshold = 5;
};

struct UnifiedFrontendDecision {
    bool candidate_found = false;
    bool accepted = false;
    std::string route = "compute";
    std::string score_reason = "no_candidate";
    int score_error_q = 0;
    int score_risk = 0;
};

inline int reuse_error_q(int hamming_dist, int route_hit_count, bool support_discount) {
    const int dist = std::max(0, hamming_dist);
    if (dist <= 0) {
        return 1;
    }

    int error = 0;
    if (dist == 1) {
        error = 2;
    } else if (dist == 2) {
        error = 4;
    } else {
        error = std::max(4, 2 * dist);
    }

    if (!support_discount) {
        return error;
    }

    const int support = std::max(1, route_hit_count);
    if (support >= 4) {
        return std::max(1, error - 2);
    }
    if (support >= 2) {
        return std::max(1, error - 1);
    }
    return error;
}

inline int confidence_discount_q(
    int hamming_dist,
    int route_hit_count,
    int base_route_hit_count,
    const UnifiedFrontendConfig& config
) {
    if (config.score_pair_confidence_discount <= 0) {
        return 0;
    }
    if (hamming_dist > config.score_pair_confidence_max_dist) {
        return 0;
    }
    const bool route_supported = route_hit_count >= config.score_pair_confidence_min_route_hits;
    const bool base_supported = base_route_hit_count >= config.score_pair_confidence_min_base_hits;
    if (route_supported || base_supported) {
        return std::max(0, config.score_pair_confidence_discount);
    }
    return 0;
}

inline bool rare_gate_rejects(
    uint8_t low_unique_q,
    int hamming_dist,
    int route_hit_count,
    int base_route_hit_count,
    const UnifiedFrontendConfig& config
) {
    if (!config.score_forbid_rare_fuzzy) {
        return false;
    }
    if (static_cast<int>(low_unique_q) < config.score_rare_threshold || hamming_dist <= 0) {
        return false;
    }
    if (hamming_dist < config.score_rare_min_dist) {
        return false;
    }
    const bool route_supported = route_hit_count >= config.score_rare_min_route_hits;
    const bool base_supported = base_route_hit_count >= config.score_rare_min_base_hits;
    return !(route_supported || base_supported);
}

inline UnifiedFrontendDecision apply_unified_frontend_policy(
    uint16_t sensitivity_q,
    uint8_t propagation_q,
    uint8_t low_unique_q,
    int route_hit_count,
    int base_route_hit_count,
    int min_dist,
    const UnifiedFrontendConfig& config
) {
    UnifiedFrontendDecision out;
    if (route_hit_count <= 0 || min_dist < 0) {
        return out;
    }

    out.candidate_found = true;
    out.score_error_q = reuse_error_q(min_dist, route_hit_count, config.score_support_discount);
    const int confidence_discount =
        confidence_discount_q(min_dist, route_hit_count, base_route_hit_count, config);
    if (confidence_discount > 0) {
        out.score_error_q = std::max(1, out.score_error_q - confidence_discount);
    }
    out.score_risk = static_cast<int>(sensitivity_q) * out.score_error_q;

    if (config.score_gate_enabled) {
        const bool fuzzy = min_dist > 0;
        const bool exact = min_dist == 0;
        if (static_cast<int>(propagation_q) >= config.score_hub_threshold
            && ((fuzzy && config.score_protect_hub_fuzzy)
                || (exact && config.score_protect_hub_exact))) {
            out.accepted = false;
            out.route = "compute";
            out.score_reason = "hub_protect";
            return out;
        }
        if (rare_gate_rejects(low_unique_q, min_dist, route_hit_count, base_route_hit_count, config)) {
            out.accepted = false;
            out.route = "compute";
            out.score_reason = "rare_leaf";
            return out;
        }
        if (out.score_risk > config.score_reuse_threshold) {
            out.accepted = false;
            out.route = "compute";
            out.score_reason = "risk";
            return out;
        }
    }

    out.accepted = true;
    out.route = route_hit_count >= config.direct_support_threshold ? "direct" : "residual";
    out.score_reason = config.score_gate_enabled ? "allow" : "disabled";
    return out;
}

}  // namespace ghhw
