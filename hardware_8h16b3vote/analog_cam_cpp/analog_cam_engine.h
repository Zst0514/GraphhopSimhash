#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/metrics.h"
#include "../common/trace_format.h"

namespace ghhw {

struct AnalogCamConfig {
    double clock_mhz = 500.0;
    int radius = 2;
    int support_threshold = 3;
    int memo_k = 3;
    int candidate_cam_entries = 512;
    int subarray_rows = 512;
    int parallel_subarrays = 1;
    int cam_search_cycles = 3;
    int candidate_select_cycles = 1;
    int cache_write_cycles = 1;
    double cam_compare_energy_fj_per_bit = 0.35;
    double candidate_cam_probe_energy_pj = 0.20;
    double cam_write_energy_pj = 0.30;
    double cam_cell_area_um2 = 0.18;
    double vdd = 1.0;
    double matchline_base_cap_f = 2.0e-15;
    double matchline_cap_per_bit_f = 0.5e-15;
    double mismatch_conductance_s = 4.0e-5;
    double match_leak_conductance_s = 2.0e-7;
    double precharge_time_ps = 40.0;
    double eval_time_ps = 100.0;
    double sense_time_ps = 20.0;
    double comparator_vref = -1.0;
    double device_sigma_rel = 0.0;
    double sense_noise_sigma_v = 0.0;
    double comparator_noise_sigma_v = 0.0;
    int rng_seed = 12345;
    std::string calibration = "proxy";
};

struct AnalogCamResult {
    SimulationStats stats;
    std::vector<Decision> decisions;
};

class AnalogCamHashReuseEngine {
public:
    explicit AnalogCamHashReuseEngine(AnalogCamConfig config);
    AnalogCamResult run(const TraceData& trace);

private:
    struct CamEntry {
        uint32_t node_id = 0;
        uint16_t hash = 0;
        uint64_t timestamp = 0;
        bool active = true;
        double mismatch_scale = 1.0;
        double leak_scale = 1.0;
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

    AnalogCamConfig config_;
    std::array<std::vector<Bucket>, kDefaultHeads> buckets_;
    std::array<std::vector<CamEntry>, kDefaultHeads> cam_rows_;
    uint64_t timestamp_ = 0;
    std::mt19937_64 rng_;

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
    double sample_zero_mean_gaussian(double sigma);
    double matchline_cap_f(uint32_t word_bits) const;
    double nominal_matchline_voltage(int dist, uint32_t word_bits) const;
    double row_matchline_voltage(const CamEntry& row, int dist, uint32_t word_bits) const;
    double comparator_vref_for_word_bits(uint32_t word_bits) const;
    bool row_threshold_hit(const CamEntry& row, uint16_t query_hash, uint32_t word_bits, int* dist_out);
};

AnalogCamConfig analog_cam_config_from_file(const std::string& path);

}  // namespace ghhw
