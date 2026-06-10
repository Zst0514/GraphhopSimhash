#include "digital_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../common/hash_utils.h"

namespace ghhw {

DigitalConfig digital_config_from_file(const std::string& path) {
    ConfigText cfg = load_config_text(path);
    DigitalConfig out;
    out.clock_mhz = config_number(cfg, "clock_mhz", out.clock_mhz);
    out.radius = config_int(cfg, "radius", out.radius);
    out.support_threshold = config_int(cfg, "support_threshold", out.support_threshold);
    out.memo_k = config_int(cfg, "memo_k", out.memo_k);
    out.candidate_cam_entries = config_int(cfg, "candidate_cam_entries", out.candidate_cam_entries);
    out.subarray_rows = config_int(cfg, "subarray_rows", out.subarray_rows);
    out.parallel_subarrays = config_int(cfg, "parallel_subarrays", out.parallel_subarrays);
    out.cam_chunk_bits = config_int(cfg, "cam_chunk_bits", out.cam_chunk_bits);
    out.cam_search_cycles = config_int(cfg, "cam_search_cycles", out.cam_search_cycles);
    out.shared_verifier_lanes = config_int(cfg, "shared_verifier_lanes", out.shared_verifier_lanes);
    out.verify_lanes = config_int(cfg, "verify_lanes", out.verify_lanes);
    out.verify_cycles = config_int(cfg, "verify_cycles", out.verify_cycles);
    out.candidate_select_cycles = config_int(cfg, "candidate_select_cycles", out.candidate_select_cycles);
    out.cache_write_cycles = config_int(cfg, "cache_write_cycles", out.cache_write_cycles);
    out.direct_support_threshold = config_int(cfg, "direct_support_threshold", out.direct_support_threshold);
    out.score_gate_enabled = config_int(cfg, "score_gate_enabled", out.score_gate_enabled);
    out.score_reuse_threshold = config_int(cfg, "score_reuse_threshold", out.score_reuse_threshold);
    out.score_hub_threshold = config_int(cfg, "score_hub_threshold", out.score_hub_threshold);
    out.score_rare_threshold = config_int(cfg, "score_rare_threshold", out.score_rare_threshold);
    out.score_protect_hub_exact = config_int(cfg, "score_protect_hub_exact", out.score_protect_hub_exact);
    out.score_protect_hub_fuzzy = config_int(cfg, "score_protect_hub_fuzzy", out.score_protect_hub_fuzzy);
    out.score_forbid_rare_fuzzy = config_int(cfg, "score_forbid_rare_fuzzy", out.score_forbid_rare_fuzzy);
    out.score_support_discount = config_int(cfg, "score_support_discount", out.score_support_discount);
    out.score_rare_min_dist = config_int(cfg, "score_rare_min_dist", out.score_rare_min_dist);
    out.score_rare_min_route_hits = config_int(cfg, "score_rare_min_route_hits", out.score_rare_min_route_hits);
    out.score_rare_min_base_hits = config_int(cfg, "score_rare_min_base_hits", out.score_rare_min_base_hits);
    out.score_pair_confidence_discount =
        config_int(cfg, "score_pair_confidence_discount", out.score_pair_confidence_discount);
    out.score_pair_confidence_max_dist =
        config_int(cfg, "score_pair_confidence_max_dist", out.score_pair_confidence_max_dist);
    out.score_pair_confidence_min_route_hits =
        config_int(cfg, "score_pair_confidence_min_route_hits", out.score_pair_confidence_min_route_hits);
    out.score_pair_confidence_min_base_hits =
        config_int(cfg, "score_pair_confidence_min_base_hits", out.score_pair_confidence_min_base_hits);
    out.cam_compare_energy_fj_per_bit =
        config_number(cfg, "cam_compare_energy_fj_per_bit", out.cam_compare_energy_fj_per_bit);
    out.xor_popcount_energy_fj_per_bit =
        config_number(cfg, "xor_popcount_energy_fj_per_bit", out.xor_popcount_energy_fj_per_bit);
    out.candidate_cam_probe_energy_pj =
        config_number(cfg, "candidate_cam_probe_energy_pj", out.candidate_cam_probe_energy_pj);
    out.cam_write_energy_pj = config_number(cfg, "cam_write_energy_pj", out.cam_write_energy_pj);
    out.cam_cell_area_um2 = config_number(cfg, "cam_cell_area_um2", out.cam_cell_area_um2);
    out.xor_popcount_lane_area_um2 =
        config_number(cfg, "xor_popcount_lane_area_um2", out.xor_popcount_lane_area_um2);
    out.replacement_policy = config_string(cfg, "replacement_policy", out.replacement_policy);
    out.total_cam_bytes = config_int(cfg, "total_cam_bytes", out.total_cam_bytes);
    out.node_entry_bytes = config_int(cfg, "node_entry_bytes", out.node_entry_bytes);
    return out;
}

DigitalHashReuseEngine::DigitalHashReuseEngine(DigitalConfig config) : config_(config) {
    if (config_.radius < 0 || config_.radius > 2) {
        throw std::invalid_argument("digital engine supports radius 0..2");
    }
    if (config_.support_threshold <= 0 || config_.support_threshold > static_cast<int>(kDefaultHeads)) {
        throw std::invalid_argument("support_threshold must be in 1..8");
    }
    if (config_.memo_k <= 0 || config_.candidate_cam_entries <= 0 || config_.subarray_rows <= 0) {
        throw std::invalid_argument("memo_k, candidate_cam_entries and subarray_rows must be positive");
    }
    if (config_.cam_chunk_bits <= 0 || config_.cam_chunk_bits > static_cast<int>(kDefaultHashBits)) {
        throw std::invalid_argument("cam_chunk_bits must be in 1..16");
    }
    if (config_.verify_lanes <= 0 || config_.verify_cycles <= 0 || config_.cam_search_cycles <= 0) {
        throw std::invalid_argument("verify_lanes, verify_cycles and cam_search_cycles must be positive");
    }
    if (config_.direct_support_threshold < config_.support_threshold
        || config_.direct_support_threshold > static_cast<int>(kDefaultHeads)) {
        throw std::invalid_argument("direct_support_threshold must be in support_threshold..8");
    }
    if (config_.score_gate_enabled != 0 && config_.score_gate_enabled != 1) {
        throw std::invalid_argument("score_gate_enabled must be 0 or 1");
    }
    if ((config_.score_protect_hub_exact != 0 && config_.score_protect_hub_exact != 1)
        || (config_.score_protect_hub_fuzzy != 0 && config_.score_protect_hub_fuzzy != 1)
        || (config_.score_forbid_rare_fuzzy != 0 && config_.score_forbid_rare_fuzzy != 1)
        || (config_.score_support_discount != 0 && config_.score_support_discount != 1)) {
        throw std::invalid_argument("score-gate boolean flags must be 0 or 1");
    }
    if (config_.score_reuse_threshold < 0
        || config_.score_hub_threshold < 0
        || config_.score_rare_threshold < 0
        || config_.score_rare_min_dist < 1
        || config_.score_rare_min_route_hits < 1
        || config_.score_rare_min_base_hits < 1
        || config_.score_pair_confidence_discount < 0
        || config_.score_pair_confidence_max_dist < 0
        || config_.score_pair_confidence_min_route_hits <= 0
        || config_.score_pair_confidence_min_base_hits <= 0) {
        throw std::invalid_argument("score-gate thresholds must be non-negative and support floors positive");
    }
    if (config_.shared_verifier_lanes != 0 && config_.shared_verifier_lanes != 1) {
        throw std::invalid_argument("shared_verifier_lanes must be 0 or 1");
    }
    if (config_.clock_mhz <= 0.0) {
        throw std::invalid_argument("clock_mhz must be positive");
    }
    if (config_.cam_compare_energy_fj_per_bit < 0.0
        || config_.xor_popcount_energy_fj_per_bit < 0.0
        || config_.candidate_cam_probe_energy_pj < 0.0
        || config_.cam_write_energy_pj < 0.0
        || config_.cam_cell_area_um2 < 0.0
        || config_.xor_popcount_lane_area_um2 < 0.0) {
        throw std::invalid_argument("energy and area parameters must be non-negative");
    }
    if (config_.replacement_policy == "per_hash_fifo") {
        use_global_lru_ = false;
    } else if (config_.replacement_policy == "global_lru") {
        use_global_lru_ = true;
    } else if (config_.replacement_policy == "global_unbounded") {
        use_global_lru_ = true;
    } else {
        throw std::invalid_argument("replacement_policy must be per_hash_fifo, global_lru or global_unbounded");
    }
    if (config_.replacement_policy == "global_lru") {
        if (config_.total_cam_bytes <= 0 || config_.node_entry_bytes <= 0) {
            throw std::invalid_argument("global_lru requires positive total_cam_bytes and node_entry_bytes");
        }
        node_capacity_entries_ = static_cast<size_t>(config_.total_cam_bytes / config_.node_entry_bytes);
        if (node_capacity_entries_ == 0) {
            throw std::invalid_argument("global_lru capacity must fit at least one node entry");
        }
    } else if (config_.replacement_policy == "global_unbounded") {
        node_capacity_entries_ = std::numeric_limits<size_t>::max();
    }
    for (auto& head_buckets : buckets_) {
        head_buckets.resize(65536);
    }
}

UnifiedFrontendConfig DigitalHashReuseEngine::frontend_policy_config() const {
    UnifiedFrontendConfig cfg;
    cfg.score_gate_enabled = config_.score_gate_enabled != 0;
    cfg.score_reuse_threshold = config_.score_reuse_threshold;
    cfg.score_hub_threshold = config_.score_hub_threshold;
    cfg.score_rare_threshold = config_.score_rare_threshold;
    cfg.score_protect_hub_exact = config_.score_protect_hub_exact != 0;
    cfg.score_protect_hub_fuzzy = config_.score_protect_hub_fuzzy != 0;
    cfg.score_forbid_rare_fuzzy = config_.score_forbid_rare_fuzzy != 0;
    cfg.score_support_discount = config_.score_support_discount != 0;
    cfg.score_rare_min_dist = config_.score_rare_min_dist;
    cfg.score_rare_min_route_hits = config_.score_rare_min_route_hits;
    cfg.score_rare_min_base_hits = config_.score_rare_min_base_hits;
    cfg.score_pair_confidence_discount = config_.score_pair_confidence_discount;
    cfg.score_pair_confidence_max_dist = config_.score_pair_confidence_max_dist;
    cfg.score_pair_confidence_min_route_hits = config_.score_pair_confidence_min_route_hits;
    cfg.score_pair_confidence_min_base_hits = config_.score_pair_confidence_min_base_hits;
    cfg.direct_support_threshold = config_.direct_support_threshold;
    return cfg;
}

uint64_t DigitalHashReuseEngine::active_rows_for_head(uint32_t head) const {
    if (use_global_lru_) {
        (void)head;
        return static_cast<uint64_t>(active_nodes_.size());
    }
    uint64_t active = 0;
    for (const CamEntry& entry : cam_rows_[head]) {
        if (entry.active) {
            active += 1;
        }
    }
    return active;
}

uint64_t DigitalHashReuseEngine::active_rows_all_heads() const {
    if (use_global_lru_) {
        return static_cast<uint64_t>(active_nodes_.size()) * static_cast<uint64_t>(kDefaultHeads);
    }
    uint64_t total = 0;
    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        total += active_rows_for_head(head);
    }
    return total;
}

uint64_t DigitalHashReuseEngine::active_node_count() const {
    return static_cast<uint64_t>(active_nodes_.size());
}

uint64_t DigitalHashReuseEngine::search_cycles_for_active_rows(uint64_t max_active_rows) const {
    if (config_.parallel_subarrays != 0) {
        return static_cast<uint64_t>(config_.cam_search_cycles);
    }
    const uint64_t subarrays = std::max<uint64_t>(
        1,
        static_cast<uint64_t>(
            std::ceil(static_cast<double>(max_active_rows) / static_cast<double>(config_.subarray_rows))
        )
    );
    return subarrays * static_cast<uint64_t>(config_.cam_search_cycles);
}

uint64_t DigitalHashReuseEngine::verify_cycles_for_survivors(
    const std::array<uint64_t, kDefaultHeads>& verified_rows_per_head
) const {
    if (config_.shared_verifier_lanes != 0) {
        uint64_t total_verified_rows = 0;
        for (uint64_t count : verified_rows_per_head) {
            total_verified_rows += count;
        }
        if (total_verified_rows == 0) {
            return 0;
        }
        return static_cast<uint64_t>(
            std::ceil(static_cast<double>(total_verified_rows) / static_cast<double>(config_.verify_lanes))
        ) * static_cast<uint64_t>(config_.verify_cycles);
    }

    uint64_t worst_case_cycles = 0;
    for (uint64_t count : verified_rows_per_head) {
        if (count == 0) {
            continue;
        }
        const uint64_t head_cycles = static_cast<uint64_t>(
            std::ceil(static_cast<double>(count) / static_cast<double>(config_.verify_lanes))
        ) * static_cast<uint64_t>(config_.verify_cycles);
        worst_case_cycles = std::max(worst_case_cycles, head_cycles);
    }
    return worst_case_cycles;
}

int DigitalHashReuseEngine::chunk_count_for_word_bits(uint32_t word_bits) const {
    return static_cast<int>(
        std::ceil(static_cast<double>(word_bits) / static_cast<double>(config_.cam_chunk_bits))
    );
}

int DigitalHashReuseEngine::matching_chunks(uint16_t lhs, uint16_t rhs, uint32_t word_bits) const {
    int matches = 0;
    uint32_t bit_offset = 0;
    while (bit_offset < word_bits) {
        const uint32_t chunk_bits = std::min<uint32_t>(
            static_cast<uint32_t>(config_.cam_chunk_bits),
            word_bits - bit_offset
        );
        const uint16_t mask = static_cast<uint16_t>(((1u << chunk_bits) - 1u) << bit_offset);
        if ((lhs & mask) == (rhs & mask)) {
            matches += 1;
        }
        bit_offset += chunk_bits;
    }
    return matches;
}

bool DigitalHashReuseEngine::coarse_filter_hit(
    uint16_t row_hash,
    uint16_t query_hash,
    uint32_t word_bits
) const {
    const int chunk_count = chunk_count_for_word_bits(word_bits);
    const int required_exact_chunks = std::max(0, chunk_count - config_.radius);
    return matching_chunks(row_hash, query_hash, word_bits) >= required_exact_chunks;
}

const std::vector<uint16_t>& DigitalHashReuseEngine::coarse_filter_hashes(
    uint16_t query_hash,
    uint32_t word_bits
) const {
    const uint32_t cache_key =
        (word_bits << 24)
        | (static_cast<uint32_t>(config_.cam_chunk_bits) << 20)
        | (static_cast<uint32_t>(config_.radius) << 16)
        ^ static_cast<uint32_t>(query_hash);
    auto cached = coarse_hash_cache_.find(cache_key);
    if (cached != coarse_hash_cache_.end()) {
        return cached->second;
    }

    const int chunk_count = chunk_count_for_word_bits(word_bits);
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> widths;
    std::vector<uint16_t> masks;
    std::vector<uint16_t> base_values;
    offsets.reserve(static_cast<size_t>(chunk_count));
    widths.reserve(static_cast<size_t>(chunk_count));
    masks.reserve(static_cast<size_t>(chunk_count));
    base_values.reserve(static_cast<size_t>(chunk_count));

    uint32_t bit_offset = 0;
    while (bit_offset < word_bits) {
        const uint32_t chunk_bits = std::min<uint32_t>(
            static_cast<uint32_t>(config_.cam_chunk_bits),
            word_bits - bit_offset
        );
        const uint16_t raw_mask = static_cast<uint16_t>((1u << chunk_bits) - 1u);
        const uint16_t mask = static_cast<uint16_t>(raw_mask << bit_offset);
        offsets.push_back(bit_offset);
        widths.push_back(chunk_bits);
        masks.push_back(mask);
        base_values.push_back(static_cast<uint16_t>((query_hash & mask) >> bit_offset));
        bit_offset += chunk_bits;
    }

    auto set_chunk = [&](uint16_t hash, int chunk_idx, uint16_t value) {
        hash = static_cast<uint16_t>(hash & ~masks[static_cast<size_t>(chunk_idx)]);
        hash = static_cast<uint16_t>(
            hash | static_cast<uint16_t>(value << offsets[static_cast<size_t>(chunk_idx)])
        );
        return hash;
    };

    std::vector<uint16_t> values;
    values.reserve(1500);
    values.push_back(query_hash);
    if (config_.radius >= 1) {
        for (int i = 0; i < chunk_count; ++i) {
            const uint16_t limit = static_cast<uint16_t>(1u << widths[static_cast<size_t>(i)]);
            for (uint16_t vi = 0; vi < limit; ++vi) {
                if (vi == base_values[static_cast<size_t>(i)]) {
                    continue;
                }
                values.push_back(set_chunk(query_hash, i, vi));
            }
        }
    }
    if (config_.radius >= 2) {
        for (int i = 0; i < chunk_count; ++i) {
            const uint16_t limit_i = static_cast<uint16_t>(1u << widths[static_cast<size_t>(i)]);
            for (int j = i + 1; j < chunk_count; ++j) {
                const uint16_t limit_j = static_cast<uint16_t>(1u << widths[static_cast<size_t>(j)]);
                for (uint16_t vi = 0; vi < limit_i; ++vi) {
                    if (vi == base_values[static_cast<size_t>(i)]) {
                        continue;
                    }
                    const uint16_t partial = set_chunk(query_hash, i, vi);
                    for (uint16_t vj = 0; vj < limit_j; ++vj) {
                        if (vj == base_values[static_cast<size_t>(j)]) {
                            continue;
                        }
                        values.push_back(set_chunk(partial, j, vj));
                    }
                }
            }
        }
    }

    auto inserted = coarse_hash_cache_.emplace(cache_key, std::move(values));
    return inserted.first->second;
}

void DigitalHashReuseEngine::deactivate_active_node(
    uint32_t node_id,
    SimulationStats& stats,
    bool count_eviction
) {
    auto node_it = active_nodes_.find(node_id);
    if (node_it == active_nodes_.end()) {
        return;
    }
    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        const size_t cam_index = node_it->second.cam_indices[head];
        if (cam_index < cam_rows_[head].size()) {
            cam_rows_[head][cam_index].active = false;
        }
    }
    active_nodes_.erase(node_it);

    auto lru_it = lru_iters_.find(node_id);
    if (lru_it != lru_iters_.end()) {
        lru_order_.erase(lru_it->second);
        lru_iters_.erase(lru_it);
    }
    if (count_eviction) {
        stats.cam_evictions += 1;
    }
}

void DigitalHashReuseEngine::evict_lru_node(SimulationStats& stats) {
    if (!use_global_lru_ || lru_order_.empty()) {
        return;
    }
    const uint32_t victim = lru_order_.back();
    deactivate_active_node(victim, stats, true);
}

void DigitalHashReuseEngine::touch_node(uint32_t node_id) {
    if (!use_global_lru_) {
        return;
    }
    auto it = lru_iters_.find(node_id);
    if (it == lru_iters_.end()) {
        return;
    }
    lru_order_.splice(lru_order_.begin(), lru_order_, it->second);
    it->second = lru_order_.begin();
}

void DigitalHashReuseEngine::add_candidate(
    std::unordered_map<uint32_t, Candidate>& candidates,
    const CamEntry& entry,
    int dist,
    SimulationStats& stats
) const {
    auto it = candidates.find(entry.node_id);
    if (it == candidates.end()) {
        if (static_cast<int>(candidates.size()) >= config_.candidate_cam_entries) {
            stats.candidate_overflows += 1;
            return;
        }
        Candidate candidate;
        candidate.node_id = entry.node_id;
        candidate.timestamp = entry.timestamp;
        candidate.support = 1;
        candidate.min_dist = dist;
        candidates.emplace(entry.node_id, candidate);
        stats.candidate_inserts += 1;
    } else {
        Candidate& candidate = it->second;
        candidate.support += 1;
        candidate.min_dist = std::min(candidate.min_dist, dist);
        candidate.timestamp = std::max(candidate.timestamp, entry.timestamp);
    }
}

Decision DigitalHashReuseEngine::select_candidate(
    const std::unordered_map<uint32_t, Candidate>& candidates,
    uint32_t query_node_id,
    const std::string& kind
) const {
    Decision decision;
    decision.node_id = query_node_id;
    decision.source_id = std::numeric_limits<uint32_t>::max();
    decision.kind = "miss";

    const Candidate* best = nullptr;
    for (const auto& kv : candidates) {
        const Candidate& candidate = kv.second;
        if (candidate.node_id == query_node_id) {
            continue;
        }
        if (candidate.support < config_.support_threshold) {
            continue;
        }
        if (best == nullptr
            || candidate.support > best->support
            || (candidate.support == best->support && candidate.min_dist < best->min_dist)
            || (candidate.support == best->support && candidate.min_dist == best->min_dist
                && candidate.timestamp > best->timestamp)
            || (candidate.support == best->support && candidate.min_dist == best->min_dist
                && candidate.timestamp == best->timestamp && candidate.node_id < best->node_id)) {
            best = &candidate;
        }
    }

    if (best != nullptr) {
        decision.hit = true;
        decision.source_id = best->node_id;
        decision.support = best->support;
        decision.route_hit_count = best->support;
        decision.base_route_hit_count = best->support > 0 ? 1 : 0;
        decision.winning_base_table_hit_count = best->support;
        decision.min_dist = best->min_dist;
        decision.kind = kind;
    }
    return decision;
}

void DigitalHashReuseEngine::insert_record(const TraceRecord& rec, SimulationStats& stats) {
    const uint64_t ts = timestamp_++;
    if (use_global_lru_) {
        deactivate_active_node(rec.node_id, stats, false);
        if (active_nodes_.size() >= node_capacity_entries_) {
            evict_lru_node(stats);
        }

        ActiveNode node;
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            CamEntry row;
            row.node_id = rec.node_id;
            row.hash = rec.head_hashes[head];
            row.timestamp = ts;
            row.active = true;
            cam_rows_[head].push_back(row);
            node.cam_indices[head] = cam_rows_[head].size() - 1;
            buckets_[head][row.hash].push_front(BucketEntry{node.cam_indices[head]});
            stats.bucket_writes += 1;
        }
        active_nodes_[rec.node_id] = node;
        lru_order_.push_front(rec.node_id);
        lru_iters_[rec.node_id] = lru_order_.begin();
        return;
    }

    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        CamEntry row;
        row.node_id = rec.node_id;
        row.hash = rec.head_hashes[head];
        row.timestamp = ts;
        row.active = true;
        cam_rows_[head].push_back(row);
        const size_t cam_index = cam_rows_[head].size() - 1;

        Bucket& bucket = buckets_[head][rec.head_hashes[head]];
        bucket.push_front(BucketEntry{cam_index});
        while (static_cast<int>(bucket.size()) > config_.memo_k) {
            const BucketEntry evicted = bucket.back();
            bucket.pop_back();
            if (evicted.cam_index < cam_rows_[head].size()) {
                cam_rows_[head][evicted.cam_index].active = false;
            }
        }
        stats.bucket_writes += 1;
    }
}

DigitalResult DigitalHashReuseEngine::run(const TraceData& trace, ProgressBar* progress) {
    DigitalResult result;
    SimulationStats& stats = result.stats;
    stats.implementation =
        config_.shared_verifier_lanes != 0
            ? "digital_cam_xor_popcount_shared_verify"
            : "digital_cam_xor_popcount_per_head_verify";
    stats.calibration = "proxy";
    stats.replacement_policy = config_.replacement_policy;
    stats.total_cam_bytes = config_.replacement_policy == "global_lru" ? static_cast<uint64_t>(config_.total_cam_bytes) : 0;
    stats.node_entry_bytes = config_.replacement_policy == "global_lru" ? static_cast<uint64_t>(config_.node_entry_bytes) : 0;
    stats.capacity_limit_nodes = config_.replacement_policy == "global_lru" ? static_cast<uint64_t>(node_capacity_entries_) : 0;
    stats.clock_mhz = config_.clock_mhz;
    stats.total_queries = trace.records.size();
    result.decisions.reserve(trace.records.size());
    if (progress != nullptr) {
        progress->update(ProgressSnapshot{0, stats.total_queries, 0, 0, 0});
    }

    uint64_t max_active_rows_seen = 0;
    uint64_t processed = 0;
    const UnifiedFrontendConfig policy_cfg = frontend_policy_config();
    for (const TraceRecord& rec : trace.records) {
        const uint64_t active_rows = active_rows_all_heads();
        max_active_rows_seen = std::max(max_active_rows_seen, active_rows);
        stats.max_active_rows = std::max(stats.max_active_rows, active_rows);
        if (use_global_lru_) {
            stats.max_active_nodes = std::max(stats.max_active_nodes, active_node_count());
        }
        uint64_t max_head_rows = 0;
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            max_head_rows = std::max(max_head_rows, active_rows_for_head(head));
        }
        const uint64_t search_cycles = search_cycles_for_active_rows(max_head_rows);

        std::unordered_map<uint32_t, Candidate> exact_candidates;
        std::unordered_map<uint32_t, Candidate> fuzzy_candidates;
        exact_candidates.reserve(128);
        fuzzy_candidates.reserve(256);
        uint64_t verified_rows = 0;
        std::array<uint64_t, kDefaultHeads> verified_rows_per_head{};
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            if (use_global_lru_) {
                const auto& hash_values = coarse_filter_hashes(rec.head_hashes[head], trace.header.hash_bits);
                for (const uint16_t hash_value : hash_values) {
                    const Bucket& bucket = buckets_[head][hash_value];
                    for (const BucketEntry& bucket_entry : bucket) {
                        if (bucket_entry.cam_index >= cam_rows_[head].size()) {
                            continue;
                        }
                        const CamEntry& row = cam_rows_[head][bucket_entry.cam_index];
                        if (!row.active) {
                            continue;
                        }
                        verified_rows += 1;
                        verified_rows_per_head[head] += 1;
                        const int dist = hamming_distance16(rec.head_hashes[head], row.hash);
                        if (dist <= config_.radius) {
                            if (dist == 0) {
                                add_candidate(exact_candidates, row, dist, stats);
                            } else {
                                add_candidate(fuzzy_candidates, row, dist, stats);
                            }
                        }
                    }
                }
            } else {
                for (const CamEntry& row : cam_rows_[head]) {
                    if (!row.active) {
                        continue;
                    }
                    if (!coarse_filter_hit(row.hash, rec.head_hashes[head], trace.header.hash_bits)) {
                        continue;
                    }
                    verified_rows += 1;
                    verified_rows_per_head[head] += 1;
                    const int dist = hamming_distance16(rec.head_hashes[head], row.hash);
                    if (dist <= config_.radius) {
                        if (dist == 0) {
                            add_candidate(exact_candidates, row, dist, stats);
                        } else {
                            add_candidate(fuzzy_candidates, row, dist, stats);
                        }
                    }
                }
            }
        }

        stats.cam_searches += kDefaultHeads;
        stats.cam_compared_rows += active_rows;
        stats.cycles += search_cycles;
        stats.frontend_search_cycles += search_cycles;
        const uint64_t verify_cycles = verify_cycles_for_survivors(verified_rows_per_head);
        if (verify_cycles > 0) {
            stats.cycles += verify_cycles;
            stats.frontend_verify_cycles += verify_cycles;
        }
        stats.frontend_verified_rows += verified_rows;
        stats.energy_pj +=
            (static_cast<double>(active_rows) * static_cast<double>(trace.header.hash_bits)
             * config_.cam_compare_energy_fj_per_bit)
            / 1000.0;
        stats.energy_pj +=
            (static_cast<double>(verified_rows) * static_cast<double>(trace.header.hash_bits)
             * config_.xor_popcount_energy_fj_per_bit)
            / 1000.0;

        uint64_t score_checks_for_query = 0;
        uint64_t score_rejects_for_query = 0;
        uint64_t score_risk_rejects_for_query = 0;
        uint64_t score_hub_rejects_for_query = 0;
        uint64_t score_rare_rejects_for_query = 0;
        auto apply_route_policy = [&](Decision& candidate) {
            candidate.candidate_found = candidate.hit;
            candidate.sensitivity_q = rec.sensitivity_q;
            candidate.propagation_q = rec.propagation_q;
            candidate.graph_context_q = rec.graph_context_q;
            candidate.low_unique_q = rec.low_unique_q;
            candidate.rarity_q = rec.rarity_q;
            const UnifiedFrontendDecision route = apply_unified_frontend_policy(
                rec.sensitivity_q,
                rec.propagation_q,
                rec.low_unique_q,
                candidate.route_hit_count,
                candidate.base_route_hit_count,
                candidate.min_dist,
                policy_cfg
            );
            candidate.score_gate_checked = candidate.candidate_found && (config_.score_gate_enabled != 0);
            candidate.score_gate_allow = candidate.candidate_found && route.accepted;
            candidate.score_error_q = route.score_error_q;
            candidate.score_risk = route.score_risk;
            candidate.score_reason = route.score_reason;
            candidate.route = route.route;
            if (candidate.score_gate_checked) {
                score_checks_for_query += 1;
                if (!route.accepted) {
                    score_rejects_for_query += 1;
                    if (route.score_reason == "risk") {
                        score_risk_rejects_for_query += 1;
                    } else if (route.score_reason == "hub_protect") {
                        score_hub_rejects_for_query += 1;
                    } else if (route.score_reason == "rare_leaf") {
                        score_rare_rejects_for_query += 1;
                    }
                }
            }
            return route;
        };

        Decision decision = select_candidate(exact_candidates, rec.node_id, "exact");
        UnifiedFrontendDecision route_decision = apply_route_policy(decision);
        if (!decision.candidate_found || !route_decision.accepted) {
            Decision fuzzy_decision = select_candidate(fuzzy_candidates, rec.node_id, "fuzzy");
            if (fuzzy_decision.hit) {
                UnifiedFrontendDecision fuzzy_route_decision = apply_route_policy(fuzzy_decision);
                decision = fuzzy_decision;
                route_decision = fuzzy_route_decision;
            }
        }
        decision.hit = route_decision.accepted;
        if (decision.hit) {
            touch_node(decision.source_id);
        }
        decision.active_rows = active_rows;
        decision.search_cycles = search_cycles;
        decision.verify_cycles = verify_cycles;
        decision.verified_rows = verified_rows;

        stats.cycles += static_cast<uint64_t>(config_.candidate_select_cycles);
        stats.score_checked += score_checks_for_query;
        stats.score_reject += score_rejects_for_query;
        stats.score_reject_risk += score_risk_rejects_for_query;
        stats.score_reject_hub_protect += score_hub_rejects_for_query;
        stats.score_reject_rare_leaf += score_rare_rejects_for_query;
        if (decision.hit) {
            stats.reuse += 1;
            if (decision.route == "direct") {
                stats.direct_reuse += 1;
            } else if (decision.route == "residual") {
                stats.residual_route += 1;
            }
            if (decision.kind == "exact") {
                stats.exact_reuse += 1;
            } else {
                stats.fuzzy_reuse += 1;
            }
        } else {
            stats.computed += 1;
            stats.cycles += static_cast<uint64_t>(config_.cache_write_cycles);
            insert_record(rec, stats);
            stats.energy_pj += static_cast<double>(kDefaultHeads) * config_.cam_write_energy_pj;
        }
        result.decisions.push_back(decision);
        processed += 1;
        if (progress != nullptr) {
            progress->update(ProgressSnapshot{
                processed,
                stats.total_queries,
                stats.reuse,
                stats.computed,
                stats.cam_evictions,
            });
        }
    }

    stats.energy_pj += static_cast<double>(stats.candidate_inserts) * config_.candidate_cam_probe_energy_pj;
    stats.area_proxy_um2 =
        static_cast<double>(max_active_rows_seen) * static_cast<double>(trace.header.hash_bits) * config_.cam_cell_area_um2
        + static_cast<double>(config_.candidate_cam_entries) * 64.0 * 0.08
        + static_cast<double>(config_.verify_lanes)
            * config_.xor_popcount_lane_area_um2
            * static_cast<double>(config_.shared_verifier_lanes != 0 ? 1 : kDefaultHeads);
    if (progress != nullptr) {
        progress->finish(ProgressSnapshot{
            stats.total_queries,
            stats.total_queries,
            stats.reuse,
            stats.computed,
            stats.cam_evictions,
        });
    }
    return result;
}

}  // namespace ghhw
