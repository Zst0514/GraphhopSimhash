#include "analog_cam_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "../common/hash_utils.h"

namespace ghhw {

AnalogCamConfig analog_cam_config_from_file(const std::string& path) {
    ConfigText cfg = load_config_text(path);
    AnalogCamConfig out;
    out.clock_mhz = config_number(cfg, "clock_mhz", out.clock_mhz);
    out.radius = config_int(cfg, "radius", out.radius);
    out.support_threshold = config_int(cfg, "support_threshold", out.support_threshold);
    out.memo_k = config_int(cfg, "memo_k", out.memo_k);
    out.candidate_cam_entries = config_int(cfg, "candidate_cam_entries", out.candidate_cam_entries);
    out.subarray_rows = config_int(cfg, "subarray_rows", out.subarray_rows);
    out.parallel_subarrays = config_int(cfg, "parallel_subarrays", out.parallel_subarrays);
    out.cam_search_cycles = config_int(cfg, "cam_search_cycles", out.cam_search_cycles);
    out.candidate_select_cycles = config_int(cfg, "candidate_select_cycles", out.candidate_select_cycles);
    out.cache_write_cycles = config_int(cfg, "cache_write_cycles", out.cache_write_cycles);
    out.cam_compare_energy_fj_per_bit = config_number(cfg, "cam_compare_energy_fj_per_bit", out.cam_compare_energy_fj_per_bit);
    out.candidate_cam_probe_energy_pj = config_number(cfg, "candidate_cam_probe_energy_pj", out.candidate_cam_probe_energy_pj);
    out.cam_write_energy_pj = config_number(cfg, "cam_write_energy_pj", out.cam_write_energy_pj);
    out.cam_cell_area_um2 = config_number(cfg, "cam_cell_area_um2", out.cam_cell_area_um2);
    out.vdd = config_number(cfg, "vdd", out.vdd);
    out.matchline_base_cap_f = config_number(cfg, "matchline_base_cap_f", out.matchline_base_cap_f);
    out.matchline_cap_per_bit_f = config_number(cfg, "matchline_cap_per_bit_f", out.matchline_cap_per_bit_f);
    out.mismatch_conductance_s = config_number(cfg, "mismatch_conductance_s", out.mismatch_conductance_s);
    out.match_leak_conductance_s = config_number(cfg, "match_leak_conductance_s", out.match_leak_conductance_s);
    out.precharge_time_ps = config_number(cfg, "precharge_time_ps", out.precharge_time_ps);
    out.eval_time_ps = config_number(cfg, "eval_time_ps", out.eval_time_ps);
    out.sense_time_ps = config_number(cfg, "sense_time_ps", out.sense_time_ps);
    out.comparator_vref = config_number(cfg, "comparator_vref", out.comparator_vref);
    out.device_sigma_rel = config_number(cfg, "device_sigma_rel", out.device_sigma_rel);
    out.sense_noise_sigma_v = config_number(cfg, "sense_noise_sigma_v", out.sense_noise_sigma_v);
    out.comparator_noise_sigma_v = config_number(cfg, "comparator_noise_sigma_v", out.comparator_noise_sigma_v);
    out.rng_seed = config_int(cfg, "rng_seed", out.rng_seed);
    out.calibration = config_string(cfg, "calibration", out.calibration);
    return out;
}

AnalogCamHashReuseEngine::AnalogCamHashReuseEngine(AnalogCamConfig config) : config_(config), rng_(static_cast<uint64_t>(config.rng_seed)) {
    if (config_.radius < 0 || config_.radius > 2) {
        throw std::invalid_argument("analog CAM engine supports radius 0..2");
    }
    if (config_.support_threshold <= 0 || config_.support_threshold > static_cast<int>(kDefaultHeads)) {
        throw std::invalid_argument("support_threshold must be in 1..8");
    }
    if (config_.memo_k <= 0 || config_.candidate_cam_entries <= 0 || config_.subarray_rows <= 0) {
        throw std::invalid_argument("memo_k, candidate_cam_entries and subarray_rows must be positive");
    }
    if (config_.clock_mhz <= 0.0 || config_.vdd <= 0.0) {
        throw std::invalid_argument("clock_mhz and vdd must be positive");
    }
    if (config_.matchline_base_cap_f <= 0.0 || config_.matchline_cap_per_bit_f < 0.0) {
        throw std::invalid_argument("matchline capacitance parameters must be valid");
    }
    if (config_.mismatch_conductance_s <= 0.0 || config_.match_leak_conductance_s < 0.0) {
        throw std::invalid_argument("conductance parameters must be valid");
    }
    if (config_.precharge_time_ps < 0.0 || config_.eval_time_ps <= 0.0 || config_.sense_time_ps < 0.0) {
        throw std::invalid_argument("RC timing parameters must be valid");
    }
    if (config_.device_sigma_rel < 0.0 || config_.sense_noise_sigma_v < 0.0 || config_.comparator_noise_sigma_v < 0.0) {
        throw std::invalid_argument("noise sigma values must be non-negative");
    }
    for (auto& head_buckets : buckets_) {
        head_buckets.resize(65536);
    }
}

uint64_t AnalogCamHashReuseEngine::active_rows_for_head(uint32_t head) const {
    uint64_t active = 0;
    for (const CamEntry& entry : cam_rows_[head]) {
        if (entry.active) {
            active += 1;
        }
    }
    return active;
}

uint64_t AnalogCamHashReuseEngine::active_rows_all_heads() const {
    uint64_t total = 0;
    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        total += active_rows_for_head(head);
    }
    return total;
}

uint64_t AnalogCamHashReuseEngine::search_cycles_for_active_rows(uint64_t max_active_rows) const {
    const double clock_period_ps = 1000000.0 / config_.clock_mhz;
    const double search_time_ps = config_.precharge_time_ps + config_.eval_time_ps + config_.sense_time_ps;
    uint64_t per_subarray_cycles = std::max<uint64_t>(
        static_cast<uint64_t>(config_.cam_search_cycles),
        static_cast<uint64_t>(std::ceil(search_time_ps / clock_period_ps))
    );
    if (config_.parallel_subarrays != 0) {
        return per_subarray_cycles;
    }
    const uint64_t subarrays = std::max<uint64_t>(
        1,
        static_cast<uint64_t>(
            std::ceil(static_cast<double>(max_active_rows) / static_cast<double>(config_.subarray_rows))
        )
    );
    return subarrays * per_subarray_cycles;
}

double AnalogCamHashReuseEngine::sample_zero_mean_gaussian(double sigma) {
    if (sigma <= 0.0) {
        return 0.0;
    }
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng_);
}

double AnalogCamHashReuseEngine::matchline_cap_f(uint32_t word_bits) const {
    return config_.matchline_base_cap_f + static_cast<double>(word_bits) * config_.matchline_cap_per_bit_f;
}

double AnalogCamHashReuseEngine::nominal_matchline_voltage(int dist, uint32_t word_bits) const {
    const int clamped_dist = std::max(0, std::min(dist, static_cast<int>(word_bits)));
    const double total_g =
        static_cast<double>(clamped_dist) * config_.mismatch_conductance_s
        + static_cast<double>(static_cast<int>(word_bits) - clamped_dist) * config_.match_leak_conductance_s;
    const double exponent = -total_g * (config_.eval_time_ps * 1.0e-12) / matchline_cap_f(word_bits);
    return config_.vdd * std::exp(exponent);
}

double AnalogCamHashReuseEngine::row_matchline_voltage(const CamEntry& row, int dist, uint32_t word_bits) const {
    const int clamped_dist = std::max(0, std::min(dist, static_cast<int>(word_bits)));
    const double mismatch_g = config_.mismatch_conductance_s * std::max(0.01, row.mismatch_scale);
    const double leak_g = config_.match_leak_conductance_s * std::max(0.01, row.leak_scale);
    const double total_g =
        static_cast<double>(clamped_dist) * mismatch_g
        + static_cast<double>(static_cast<int>(word_bits) - clamped_dist) * leak_g;
    const double exponent = -total_g * (config_.eval_time_ps * 1.0e-12) / matchline_cap_f(word_bits);
    return config_.vdd * std::exp(exponent);
}

double AnalogCamHashReuseEngine::comparator_vref_for_word_bits(uint32_t word_bits) const {
    if (config_.comparator_vref >= 0.0) {
        return config_.comparator_vref;
    }
    const int positive_edge = std::min(2, static_cast<int>(word_bits));
    const int negative_edge = std::min(3, static_cast<int>(word_bits));
    const double v_pos = nominal_matchline_voltage(positive_edge, word_bits);
    const double v_neg = nominal_matchline_voltage(negative_edge, word_bits);
    return 0.5 * (v_pos + v_neg);
}

bool AnalogCamHashReuseEngine::row_threshold_hit(
    const CamEntry& row,
    uint16_t query_hash,
    uint32_t word_bits,
    int* dist_out
) {
    const int dist = hamming_distance16(query_hash, row.hash);
    if (dist_out != nullptr) {
        *dist_out = dist;
    }
    const double v_ml = row_matchline_voltage(row, dist, word_bits);
    const double effective_v_ml = std::clamp(v_ml + sample_zero_mean_gaussian(config_.sense_noise_sigma_v), 0.0, config_.vdd);
    const double effective_v_ref =
        std::clamp(comparator_vref_for_word_bits(word_bits) + sample_zero_mean_gaussian(config_.comparator_noise_sigma_v), 0.0, config_.vdd);
    return effective_v_ml >= effective_v_ref;
}

void AnalogCamHashReuseEngine::add_candidate(
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

Decision AnalogCamHashReuseEngine::select_candidate(
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

void AnalogCamHashReuseEngine::insert_record(const TraceRecord& rec, SimulationStats& stats) {
    const uint64_t ts = timestamp_++;
    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        CamEntry row;
        row.node_id = rec.node_id;
        row.hash = rec.head_hashes[head];
        row.timestamp = ts;
        row.active = true;
        row.mismatch_scale = std::max(0.05, 1.0 + sample_zero_mean_gaussian(config_.device_sigma_rel));
        row.leak_scale = std::max(0.05, 1.0 + sample_zero_mean_gaussian(config_.device_sigma_rel));
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

AnalogCamResult AnalogCamHashReuseEngine::run(const TraceData& trace) {
    AnalogCamResult result;
    SimulationStats& stats = result.stats;
    stats.implementation = "analog_cam_rc_threshold";
    stats.calibration = config_.calibration;
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

        std::unordered_map<uint32_t, Candidate> candidates;
        candidates.reserve(256);
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            for (const CamEntry& row : cam_rows_[head]) {
                if (!row.active) {
                    continue;
                }
                int dist = 0;
                if (row_threshold_hit(row, rec.head_hashes[head], trace.header.hash_bits, &dist)) {
                    add_candidate(candidates, row, dist, stats);
                }
            }
        }
        stats.cam_searches += kDefaultHeads;
        stats.cam_compared_rows += active_rows;
        stats.cycles += search_cycles_for_active_rows(max_head_rows);
        stats.energy_pj +=
            (static_cast<double>(active_rows) * static_cast<double>(trace.header.hash_bits)
             * config_.cam_compare_energy_fj_per_bit)
            / 1000.0;

        Decision decision = select_candidate(candidates, rec.node_id, "fuzzy");
        if (decision.hit && decision.min_dist == 0) {
            decision.kind = "exact";
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
            stats.energy_pj += static_cast<double>(kDefaultHeads) * config_.cam_write_energy_pj;
        }
        result.decisions.push_back(decision);
    }

    stats.energy_pj += static_cast<double>(stats.candidate_inserts) * config_.candidate_cam_probe_energy_pj;
    stats.area_proxy_um2 =
        static_cast<double>(max_active_rows_seen) * static_cast<double>(trace.header.hash_bits) * config_.cam_cell_area_um2
        + static_cast<double>(config_.candidate_cam_entries) * 64.0 * 0.08;
    return result;
}

}  // namespace ghhw
