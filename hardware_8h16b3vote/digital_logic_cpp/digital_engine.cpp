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
    for (auto& head_buckets : buckets_) {
        head_buckets.resize(65536);
    }
}

uint64_t DigitalHashReuseEngine::active_rows_for_head(uint32_t head) const {
    uint64_t active = 0;
    for (const CamEntry& entry : cam_rows_[head]) {
        if (entry.active) {
            active += 1;
        }
    }
    return active;
}

uint64_t DigitalHashReuseEngine::active_rows_all_heads() const {
    uint64_t total = 0;
    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        total += active_rows_for_head(head);
    }
    return total;
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
        decision.min_dist = best->min_dist;
        decision.kind = kind;
    }
    return decision;
}

void DigitalHashReuseEngine::insert_record(const TraceRecord& rec, SimulationStats& stats) {
    const uint64_t ts = timestamp_++;
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

DigitalResult DigitalHashReuseEngine::run(const TraceData& trace) {
    DigitalResult result;
    SimulationStats& stats = result.stats;
    stats.implementation =
        config_.shared_verifier_lanes != 0
            ? "digital_cam_xor_popcount_shared_verify"
            : "digital_cam_xor_popcount_per_head_verify";
    stats.calibration = "proxy";
    stats.clock_mhz = config_.clock_mhz;
    stats.total_queries = trace.records.size();
    result.decisions.reserve(trace.records.size());

    uint64_t max_active_rows_seen = 0;
    for (const TraceRecord& rec : trace.records) {
        const uint64_t active_rows = active_rows_all_heads();
        max_active_rows_seen = std::max(max_active_rows_seen, active_rows);
        uint64_t max_head_rows = 0;
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            max_head_rows = std::max(max_head_rows, active_rows_for_head(head));
        }
        const uint64_t search_cycles = search_cycles_for_active_rows(max_head_rows);

        std::unordered_map<uint32_t, Candidate> candidates;
        candidates.reserve(256);
        uint64_t verified_rows = 0;
        std::array<uint64_t, kDefaultHeads> verified_rows_per_head{};
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
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
                    add_candidate(candidates, row, dist, stats);
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

        Decision decision = select_candidate(candidates, rec.node_id, "fuzzy");
        if (decision.hit && decision.min_dist == 0) {
            decision.kind = "exact";
        }
        decision.active_rows = active_rows;
        decision.search_cycles = search_cycles;
        decision.verify_cycles = verify_cycles;
        decision.verified_rows = verified_rows;

        stats.cycles += static_cast<uint64_t>(config_.candidate_select_cycles);
        if (decision.hit) {
            stats.reuse += 1;
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
    }

    stats.energy_pj += static_cast<double>(stats.candidate_inserts) * config_.candidate_cam_probe_energy_pj;
    stats.area_proxy_um2 =
        static_cast<double>(max_active_rows_seen) * static_cast<double>(trace.header.hash_bits) * config_.cam_cell_area_um2
        + static_cast<double>(config_.candidate_cam_entries) * 64.0 * 0.08
        + static_cast<double>(config_.verify_lanes)
            * config_.xor_popcount_lane_area_um2
            * static_cast<double>(config_.shared_verifier_lanes != 0 ? 1 : kDefaultHeads);
    return result;
}

}  // namespace ghhw
