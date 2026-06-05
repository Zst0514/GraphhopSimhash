#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>
#include <unordered_map>

#include "../common/metrics.h"
#include "../common/trace_format.h"

namespace ghhw {

struct DigitalConfig {
    double clock_mhz = 1000.0;
    int radius = 2;
    int support_threshold = 3;
    int memo_k = 3;
    int candidate_cam_entries = 512;
    int subarray_rows = 512;
    int parallel_subarrays = 1;
    int cam_chunk_bits = 4;
    int cam_search_cycles = 1;
    int shared_verifier_lanes = 1;
    int verify_lanes = 32;
    int verify_cycles = 1;
    int candidate_select_cycles = 1;
    int cache_write_cycles = 1;
    double cam_compare_energy_fj_per_bit = 0.35;
    double xor_popcount_energy_fj_per_bit = 1.20;
    double candidate_cam_probe_energy_pj = 0.20;
    double cam_write_energy_pj = 0.30;
    double cam_cell_area_um2 = 0.18;
    double xor_popcount_lane_area_um2 = 20.0;
};

struct DigitalResult {
    SimulationStats stats;
    std::vector<Decision> decisions;
};

class DigitalHashReuseEngine {
public:
    explicit DigitalHashReuseEngine(DigitalConfig config);
    DigitalResult run(const TraceData& trace);

private:
    struct CamEntry {
        uint32_t node_id = 0;
        uint16_t hash = 0;
        uint64_t timestamp = 0;
        bool active = true;
    };

    struct BucketEntry {
        size_t cam_index = 0;
    };

    struct Candidate {
        uint32_t node_id = 0;
        uint64_t timestamp = 0;
        int support = 0;
        int min_dist = 999;
    };

    using Bucket = std::deque<BucketEntry>;
    DigitalConfig config_;
    std::array<std::vector<Bucket>, kDefaultHeads> buckets_;
    std::array<std::vector<CamEntry>, kDefaultHeads> cam_rows_;
    uint64_t timestamp_ = 0;

    void insert_record(const TraceRecord& rec, SimulationStats& stats);
    void add_candidate(
        std::unordered_map<uint32_t, Candidate>& candidates,
        const CamEntry& entry,
        int dist,
        SimulationStats& stats
    ) const;
    Decision select_candidate(
        const std::unordered_map<uint32_t, Candidate>& candidates,
        uint32_t query_node_id,
        const std::string& kind
    ) const;
    uint64_t active_rows_for_head(uint32_t head) const;
    uint64_t active_rows_all_heads() const;
    uint64_t search_cycles_for_active_rows(uint64_t max_active_rows) const;
    uint64_t verify_cycles_for_survivors(const std::array<uint64_t, kDefaultHeads>& verified_rows_per_head) const;
    int chunk_count_for_word_bits(uint32_t word_bits) const;
    int matching_chunks(uint16_t lhs, uint16_t rhs, uint32_t word_bits) const;
    bool coarse_filter_hit(uint16_t row_hash, uint16_t query_hash, uint32_t word_bits) const;
};

DigitalConfig digital_config_from_file(const std::string& path);

}  // namespace ghhw
