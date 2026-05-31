#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/metrics.h"
#include "../common/trace_format.h"

namespace ghhw {

struct DigitalConfig {
    double clock_mhz = 1000.0;
    int radius = 2;
    int support_threshold = 3;
    int memo_k = 3;
    int neighbor_lookup_lanes = 16;
    int candidate_cam_entries = 512;
    int exact_lookup_cycles = 1;
    int candidate_select_cycles = 1;
    int cache_write_cycles = 1;
    double sram_probe_energy_pj = 0.025;
    double candidate_cam_probe_energy_pj = 0.20;
    double bucket_write_energy_pj = 0.08;
    double sram_bitcell_area_um2 = 0.08;
    int bucket_pointer_bits = 32;
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
    struct BucketEntry {
        uint32_t node_id = 0;
        uint64_t timestamp = 0;
        std::array<uint16_t, kDefaultHeads> head_hashes{};
    };

    struct Candidate {
        uint32_t node_id = 0;
        uint64_t timestamp = 0;
        int support = 0;
        int min_dist = 999;
    };

    using Bucket = std::deque<BucketEntry>;
    DigitalConfig config_;
    std::array<std::vector<Bucket>, kDefaultHeads> tables_;
    uint64_t timestamp_ = 0;

    void insert_record(const TraceRecord& rec, SimulationStats& stats);
    void add_bucket_candidates(
        std::unordered_map<uint32_t, Candidate>& candidates,
        const Bucket& bucket,
        int dist,
        SimulationStats& stats
    ) const;
    Decision select_candidate(
        const std::unordered_map<uint32_t, Candidate>& candidates,
        uint32_t query_node_id,
        const std::string& kind
    ) const;
};

DigitalConfig digital_config_from_file(const std::string& path);

}  // namespace ghhw
