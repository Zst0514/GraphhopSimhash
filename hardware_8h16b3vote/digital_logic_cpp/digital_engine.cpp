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
    out.neighbor_lookup_lanes = config_int(cfg, "neighbor_lookup_lanes", out.neighbor_lookup_lanes);
    out.candidate_cam_entries = config_int(cfg, "candidate_cam_entries", out.candidate_cam_entries);
    out.exact_lookup_cycles = config_int(cfg, "exact_lookup_cycles", out.exact_lookup_cycles);
    out.candidate_select_cycles = config_int(cfg, "candidate_select_cycles", out.candidate_select_cycles);
    out.cache_write_cycles = config_int(cfg, "cache_write_cycles", out.cache_write_cycles);
    out.sram_probe_energy_pj = config_number(cfg, "sram_probe_energy_pj", out.sram_probe_energy_pj);
    out.candidate_cam_probe_energy_pj = config_number(cfg, "candidate_cam_probe_energy_pj", out.candidate_cam_probe_energy_pj);
    out.bucket_write_energy_pj = config_number(cfg, "bucket_write_energy_pj", out.bucket_write_energy_pj);
    out.sram_bitcell_area_um2 = config_number(cfg, "sram_bitcell_area_um2", out.sram_bitcell_area_um2);
    out.bucket_pointer_bits = config_int(cfg, "bucket_pointer_bits", out.bucket_pointer_bits);
    return out;
}

DigitalHashReuseEngine::DigitalHashReuseEngine(DigitalConfig config) : config_(config) {
    if (config_.radius < 0 || config_.radius > 2) {
        throw std::invalid_argument("digital engine supports radius 0..2");
    }
    if (config_.support_threshold <= 0 || config_.support_threshold > static_cast<int>(kDefaultHeads)) {
        throw std::invalid_argument("support_threshold must be in 1..8");
    }
    if (config_.memo_k <= 0) {
        throw std::invalid_argument("memo_k must be positive");
    }
    if (config_.neighbor_lookup_lanes <= 0 || config_.candidate_cam_entries <= 0) {
        throw std::invalid_argument("neighbor_lookup_lanes and candidate_cam_entries must be positive");
    }
    for (auto& head_table : tables_) {
        head_table.resize(65536);
    }
}

void DigitalHashReuseEngine::add_bucket_candidates(
    std::unordered_map<uint32_t, Candidate>& candidates,
    const Bucket& bucket,
    int dist,
    SimulationStats& stats
) const {
    for (const BucketEntry& entry : bucket) {
        auto it = candidates.find(entry.node_id);
        if (it == candidates.end()) {
            if (static_cast<int>(candidates.size()) >= config_.candidate_cam_entries) {
                stats.candidate_overflows += 1;
                continue;
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
    BucketEntry entry;
    entry.node_id = rec.node_id;
    entry.timestamp = timestamp_++;
    entry.head_hashes = rec.head_hashes;

    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        Bucket& bucket = tables_[head][rec.head_hashes[head]];
        bucket.push_front(entry);
        while (static_cast<int>(bucket.size()) > config_.memo_k) {
            bucket.pop_back();
        }
        stats.bucket_writes += 1;
    }
}

DigitalResult DigitalHashReuseEngine::run(const TraceData& trace) {
    DigitalResult result;
    SimulationStats& stats = result.stats;
    stats.implementation = "digital_logic_sram";
    stats.calibration = "proxy";
    stats.clock_mhz = config_.clock_mhz;
    stats.total_queries = trace.records.size();
    result.decisions.reserve(trace.records.size());

    const int neighbor_count = hamming_ball_size(static_cast<int>(trace.header.hash_bits), config_.radius);
    const int fuzzy_cycles = static_cast<int>(
        std::ceil(static_cast<double>(neighbor_count) / static_cast<double>(config_.neighbor_lookup_lanes))
    );

    for (const TraceRecord& rec : trace.records) {
        stats.cycles += static_cast<uint64_t>(config_.exact_lookup_cycles);
        stats.sram_probes += kDefaultHeads;

        std::unordered_map<uint32_t, Candidate> exact_candidates;
        exact_candidates.reserve(64);
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            const Bucket& bucket = tables_[head][rec.head_hashes[head]];
            add_bucket_candidates(exact_candidates, bucket, 0, stats);
        }

        Decision decision = select_candidate(exact_candidates, rec.node_id, "exact");
        if (!decision.hit) {
            std::unordered_map<uint32_t, Candidate> fuzzy_candidates;
            fuzzy_candidates.reserve(256);
            for (uint32_t head = 0; head < kDefaultHeads; ++head) {
                const auto neighbors = generate_hamming_neighbors16(rec.head_hashes[head], config_.radius, 16);
                for (uint16_t key : neighbors) {
                    const Bucket& bucket = tables_[head][key];
                    int dist = hamming_distance16(rec.head_hashes[head], key);
                    add_bucket_candidates(fuzzy_candidates, bucket, dist, stats);
                }
            }
            stats.cycles += static_cast<uint64_t>(fuzzy_cycles);
            stats.sram_probes += static_cast<uint64_t>(neighbor_count) * kDefaultHeads;
            decision = select_candidate(fuzzy_candidates, rec.node_id, "fuzzy");
        }

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
        }
        result.decisions.push_back(decision);
    }

    stats.energy_pj =
        static_cast<double>(stats.sram_probes) * config_.sram_probe_energy_pj
        + static_cast<double>(stats.candidate_inserts) * config_.candidate_cam_probe_energy_pj
        + static_cast<double>(stats.bucket_writes) * config_.bucket_write_energy_pj;
    stats.area_proxy_um2 =
        static_cast<double>(kDefaultHeads) * 65536.0 * static_cast<double>(config_.bucket_pointer_bits)
            * config_.sram_bitcell_area_um2
        + static_cast<double>(config_.candidate_cam_entries) * 64.0 * config_.sram_bitcell_area_um2;
    return result;
}

}  // namespace ghhw
