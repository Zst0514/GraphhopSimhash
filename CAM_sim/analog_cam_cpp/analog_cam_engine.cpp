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
    out.veval = config_number(cfg, "veval", out.veval);
    out.meval_threshold_v = config_number(cfg, "meval_threshold_v", out.meval_threshold_v);
    out.matchline_base_cap_f = config_number(cfg, "matchline_base_cap_f", out.matchline_base_cap_f);
    out.matchline_cap_per_bit_f = config_number(cfg, "matchline_cap_per_bit_f", out.matchline_cap_per_bit_f);
    out.mismatch_conductance_s = config_number(cfg, "mismatch_conductance_s", out.mismatch_conductance_s);
    out.exact_mismatch_conductance_s = config_number(cfg, "exact_mismatch_conductance_s", out.exact_mismatch_conductance_s);
    out.match_leak_conductance_s = config_number(cfg, "match_leak_conductance_s", out.match_leak_conductance_s);
    out.precharge_time_ps = config_number(cfg, "precharge_time_ps", out.precharge_time_ps);
    out.eval_time_ps = config_number(cfg, "eval_time_ps", out.eval_time_ps);
    out.sense_time_ps = config_number(cfg, "sense_time_ps", out.sense_time_ps);
    out.fixed_vref = config_number(cfg, "fixed_vref", out.fixed_vref);
    out.comparator_vref = config_number(cfg, "comparator_vref", out.comparator_vref);
    out.device_sigma_rel = config_number(cfg, "device_sigma_rel", out.device_sigma_rel);
    out.sense_noise_sigma_v = config_number(cfg, "sense_noise_sigma_v", out.sense_noise_sigma_v);
    out.comparator_noise_sigma_v = config_number(cfg, "comparator_noise_sigma_v", out.comparator_noise_sigma_v);
    out.rng_seed = config_int(cfg, "rng_seed", out.rng_seed);
    out.calibration = config_string(cfg, "calibration", out.calibration);
    out.replacement_policy = config_string(cfg, "replacement_policy", out.replacement_policy);
    out.total_cam_bytes = config_int(cfg, "total_cam_bytes", out.total_cam_bytes);
    out.node_entry_bytes = config_int(cfg, "node_entry_bytes", out.node_entry_bytes);
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
    if (config_.veval >= 0.0 && config_.veval > config_.vdd) {
        throw std::invalid_argument("veval must not exceed vdd");
    }
    if (config_.meval_threshold_v < 0.0 || config_.meval_threshold_v >= config_.vdd) {
        throw std::invalid_argument("meval_threshold_v must be in [0, vdd)");
    }
    if (config_.matchline_base_cap_f <= 0.0 || config_.matchline_cap_per_bit_f < 0.0) {
        throw std::invalid_argument("matchline capacitance parameters must be valid");
    }
    if (config_.mismatch_conductance_s <= 0.0 || config_.match_leak_conductance_s < 0.0) {
        throw std::invalid_argument("conductance parameters must be valid");
    }
    if (config_.exact_mismatch_conductance_s != -1.0 && config_.exact_mismatch_conductance_s <= 0.0) {
        throw std::invalid_argument("exact_mismatch_conductance_s must be positive or -1");
    }
    if (config_.precharge_time_ps < 0.0 || config_.eval_time_ps <= 0.0 || config_.sense_time_ps < 0.0) {
        throw std::invalid_argument("RC timing parameters must be valid");
    }
    if (config_.device_sigma_rel < 0.0 || config_.sense_noise_sigma_v < 0.0 || config_.comparator_noise_sigma_v < 0.0) {
        throw std::invalid_argument("noise sigma values must be non-negative");
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

uint64_t AnalogCamHashReuseEngine::active_rows_for_head(uint32_t head) const {
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

uint64_t AnalogCamHashReuseEngine::active_rows_all_heads() const {
    if (use_global_lru_) {
        return static_cast<uint64_t>(active_nodes_.size()) * static_cast<uint64_t>(kDefaultHeads);
    }
    uint64_t total = 0;
    for (uint32_t head = 0; head < kDefaultHeads; ++head) {
        total += active_rows_for_head(head);
    }
    return total;
}

uint64_t AnalogCamHashReuseEngine::active_node_count() const {
    return static_cast<uint64_t>(active_nodes_.size());
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

double AnalogCamHashReuseEngine::effective_veval_scale() const {
    if (config_.veval < 0.0) {
        return 1.0;
    }
    const double denom = config_.vdd - config_.meval_threshold_v;
    if (denom <= 0.0) {
        return 1.0;
    }
    return std::clamp((config_.veval - config_.meval_threshold_v) / denom, 0.0, 1.0);
}

double AnalogCamHashReuseEngine::nominal_matchline_voltage(int dist, uint32_t word_bits) const {
    const int clamped_dist = std::max(0, std::min(dist, static_cast<int>(word_bits)));
    const double mismatch_g = config_.mismatch_conductance_s * effective_veval_scale();
    const double total_g =
        static_cast<double>(clamped_dist) * mismatch_g
        + static_cast<double>(static_cast<int>(word_bits) - clamped_dist) * config_.match_leak_conductance_s;
    const double exponent = -total_g * (config_.eval_time_ps * 1.0e-12) / matchline_cap_f(word_bits);
    return config_.vdd * std::exp(exponent);
}

double AnalogCamHashReuseEngine::row_matchline_voltage(const CamEntry& row, int dist, uint32_t word_bits) const {
    const int clamped_dist = std::max(0, std::min(dist, static_cast<int>(word_bits)));
    const double mismatch_g =
        config_.mismatch_conductance_s * effective_veval_scale() * std::max(0.01, row.mismatch_scale);
    const double leak_g = config_.match_leak_conductance_s * std::max(0.01, row.leak_scale);
    const double total_g =
        static_cast<double>(clamped_dist) * mismatch_g
        + static_cast<double>(static_cast<int>(word_bits) - clamped_dist) * leak_g;
    const double exponent = -total_g * (config_.eval_time_ps * 1.0e-12) / matchline_cap_f(word_bits);
    return config_.vdd * std::exp(exponent);
}

double AnalogCamHashReuseEngine::comparator_vref_for_word_bits(uint32_t word_bits) const {
    if (config_.fixed_vref >= 0.0) {
        return config_.fixed_vref;
    }
    if (config_.comparator_vref >= 0.0) {
        return config_.comparator_vref;
    }
    const int positive_edge = std::min(2, static_cast<int>(word_bits));
    const int negative_edge = std::min(3, static_cast<int>(word_bits));
    const double v_pos = nominal_matchline_voltage(positive_edge, word_bits);
    const double v_neg = nominal_matchline_voltage(negative_edge, word_bits);
    return 0.5 * (v_pos + v_neg);
}

namespace {

double normalized_veval_scale(const AnalogCamConfig& config) {
    if (config.veval < 0.0) {
        return 1.0;
    }
    const double denom = config.vdd - config.meval_threshold_v;
    if (denom <= 0.0) {
        return 1.0;
    }
    return std::clamp((config.veval - config.meval_threshold_v) / denom, 0.0, 1.0);
}

double matchline_cap_f(const AnalogCamConfig& config, uint32_t word_bits) {
    return config.matchline_base_cap_f + static_cast<double>(word_bits) * config.matchline_cap_per_bit_f;
}

double nominal_voltage_for_conductance(
    const AnalogCamConfig& config,
    double mismatch_g,
    int dist,
    uint32_t word_bits,
    double eval_time_ps
) {
    const int clamped_dist = std::max(0, std::min(dist, static_cast<int>(word_bits)));
    const double total_g =
        static_cast<double>(clamped_dist) * mismatch_g
        + static_cast<double>(static_cast<int>(word_bits) - clamped_dist) * config.match_leak_conductance_s;
    const double exponent = -total_g * (eval_time_ps * 1.0e-12) / matchline_cap_f(config, word_bits);
    return config.vdd * std::exp(exponent);
}

double crossing_time_ps_for_conductance(
    const AnalogCamConfig& config,
    double mismatch_g,
    int dist,
    uint32_t word_bits,
    double vref
) {
    if (vref <= 0.0 || vref >= config.vdd) {
        return std::numeric_limits<double>::infinity();
    }
    const int clamped_dist = std::max(0, std::min(dist, static_cast<int>(word_bits)));
    const double total_g =
        static_cast<double>(clamped_dist) * mismatch_g
        + static_cast<double>(static_cast<int>(word_bits) - clamped_dist) * config.match_leak_conductance_s;
    if (total_g <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double c_ml = matchline_cap_f(config, word_bits);
    return -(c_ml / total_g) * std::log(vref / config.vdd) * 1.0e12;
}

double frontend_vref(const AnalogCamConfig& config, uint32_t word_bits) {
    if (config.fixed_vref >= 0.0) {
        return config.fixed_vref;
    }
    if (config.comparator_vref >= 0.0) {
        return config.comparator_vref;
    }
    const double hdcam_mismatch_g = config.mismatch_conductance_s * normalized_veval_scale(config);
    const int positive_edge = std::min(2, static_cast<int>(word_bits));
    const int negative_edge = std::min(3, static_cast<int>(word_bits));
    const double v_pos = nominal_voltage_for_conductance(
        config, hdcam_mismatch_g, positive_edge, word_bits, config.eval_time_ps
    );
    const double v_neg = nominal_voltage_for_conductance(
        config, hdcam_mismatch_g, negative_edge, word_bits, config.eval_time_ps
    );
    return 0.5 * (v_pos + v_neg);
}

}  // namespace

FrontendSpeedAnalysis analyze_cam_frontends(const AnalogCamConfig& config, uint32_t word_bits, int max_dist) {
    FrontendSpeedAnalysis out;
    out.word_bits = word_bits;
    out.max_dist = std::max(0, max_dist);
    out.hd_boundary = config.radius;
    out.vdd = config.vdd;
    out.veval = config.veval;
    out.vref = frontend_vref(config, word_bits);
    out.matchline_cap_f = matchline_cap_f(config, word_bits);

    const double hdcam_mismatch_g = config.mismatch_conductance_s * normalized_veval_scale(config);
    const double exact_mismatch_g =
        config.exact_mismatch_conductance_s > 0.0 ? config.exact_mismatch_conductance_s : config.mismatch_conductance_s;

    out.points.reserve(static_cast<size_t>(out.max_dist + 1));
    for (int dist = 0; dist <= out.max_dist; ++dist) {
        FrontendVoltagePoint point;
        point.dist = dist;
        point.hdcam_v_ml = nominal_voltage_for_conductance(config, hdcam_mismatch_g, dist, word_bits, config.eval_time_ps);
        point.hdcam_t_cross_ps = crossing_time_ps_for_conductance(config, hdcam_mismatch_g, dist, word_bits, out.vref);
        point.exact_cam_v_ml = nominal_voltage_for_conductance(config, exact_mismatch_g, dist, word_bits, config.eval_time_ps);
        point.exact_cam_t_cross_ps = crossing_time_ps_for_conductance(config, exact_mismatch_g, dist, word_bits, out.vref);
        out.points.push_back(point);
    }

    const int hd_negative_edge = std::min(static_cast<int>(word_bits), config.radius + 1);
    out.hdcam_eval_time_ps = crossing_time_ps_for_conductance(config, hdcam_mismatch_g, hd_negative_edge, word_bits, out.vref);
    out.exact_cam_eval_time_ps = crossing_time_ps_for_conductance(config, exact_mismatch_g, 1, word_bits, out.vref);
    out.hdcam_search_time_ps = config.precharge_time_ps + out.hdcam_eval_time_ps + config.sense_time_ps;
    out.exact_cam_search_time_ps = config.precharge_time_ps + out.exact_cam_eval_time_ps + config.sense_time_ps;
    out.search_time_ratio =
        out.exact_cam_search_time_ps > 0.0 ? out.hdcam_search_time_ps / out.exact_cam_search_time_ps : 0.0;
    return out;
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

void AnalogCamHashReuseEngine::deactivate_active_node(
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

void AnalogCamHashReuseEngine::evict_lru_node(SimulationStats& stats) {
    if (!use_global_lru_ || lru_order_.empty()) {
        return;
    }
    const uint32_t victim = lru_order_.back();
    deactivate_active_node(victim, stats, true);
}

void AnalogCamHashReuseEngine::touch_node(uint32_t node_id) {
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
            row.mismatch_scale = std::max(0.05, 1.0 + sample_zero_mean_gaussian(config_.device_sigma_rel));
            row.leak_scale = std::max(0.05, 1.0 + sample_zero_mean_gaussian(config_.device_sigma_rel));
            cam_rows_[head].push_back(row);
            node.cam_indices[head] = cam_rows_[head].size() - 1;
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

AnalogCamResult AnalogCamHashReuseEngine::run(const TraceData& trace, ProgressBar* progress) {
    AnalogCamResult result;
    SimulationStats& stats = result.stats;
    stats.implementation = "analog_cam_rc_threshold";
    stats.calibration = config_.calibration;
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

        std::unordered_map<uint32_t, Candidate> candidates;
        candidates.reserve(256);
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            if (use_global_lru_) {
                for (const auto& kv : active_nodes_) {
                    const CamEntry& row = cam_rows_[head][kv.second.cam_indices[head]];
                    int dist = 0;
                    if (row_threshold_hit(row, rec.head_hashes[head], trace.header.hash_bits, &dist)) {
                        add_candidate(candidates, row, dist, stats);
                    }
                }
            } else {
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
        }
        stats.cam_searches += kDefaultHeads;
        stats.cam_compared_rows += active_rows;
        stats.cycles += search_cycles;
        stats.frontend_search_cycles += search_cycles;
        stats.energy_pj +=
            (static_cast<double>(active_rows) * static_cast<double>(trace.header.hash_bits)
             * config_.cam_compare_energy_fj_per_bit)
            / 1000.0;

        Decision decision = select_candidate(candidates, rec.node_id, "fuzzy");
        if (decision.hit && decision.min_dist == 0) {
            decision.kind = "exact";
        }
        if (decision.hit) {
            touch_node(decision.source_id);
        }
        decision.active_rows = active_rows;
        decision.search_cycles = search_cycles;
        decision.verify_cycles = 0;
        decision.verified_rows = 0;

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
        + static_cast<double>(config_.candidate_cam_entries) * 64.0 * 0.08;
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
