#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ghhw {

struct ConfigText {
    std::string path;
    std::string text;
};

struct Decision {
    uint32_t node_id = 0;
    bool hit = false;
    uint32_t source_id = 0;
    int support = 0;
    int min_dist = -1;
    std::string kind = "miss";
    uint64_t active_rows = 0;
    uint64_t search_cycles = 0;
    uint64_t verify_cycles = 0;
    uint64_t verified_rows = 0;
};

struct SimulationStats {
    std::string implementation;
    std::string calibration = "proxy";
    std::string replacement_policy = "per_hash_fifo";
    uint64_t total_cam_bytes = 0;
    uint64_t node_entry_bytes = 0;
    uint64_t capacity_limit_nodes = 0;
    uint64_t total_queries = 0;
    uint64_t reuse = 0;
    uint64_t exact_reuse = 0;
    uint64_t fuzzy_reuse = 0;
    uint64_t computed = 0;
    uint64_t candidate_inserts = 0;
    uint64_t candidate_overflows = 0;
    uint64_t sram_probes = 0;
    uint64_t cam_searches = 0;
    uint64_t cam_compared_rows = 0;
    uint64_t bucket_writes = 0;
    uint64_t cam_evictions = 0;
    uint64_t max_active_nodes = 0;
    uint64_t max_active_rows = 0;
    uint64_t frontend_search_cycles = 0;
    uint64_t frontend_verify_cycles = 0;
    uint64_t frontend_verified_rows = 0;
    uint64_t cycles = 0;
    double clock_mhz = 1000.0;
    double energy_pj = 0.0;
    double area_proxy_um2 = 0.0;

    double reuse_rate() const {
        return total_queries == 0 ? 0.0 : static_cast<double>(reuse) / static_cast<double>(total_queries);
    }

    double cycles_per_query() const {
        return total_queries == 0 ? 0.0 : static_cast<double>(cycles) / static_cast<double>(total_queries);
    }

    double energy_per_query_pj() const {
        return total_queries == 0 ? 0.0 : energy_pj / static_cast<double>(total_queries);
    }

    double search_cycles_per_query() const {
        return total_queries == 0 ? 0.0 : static_cast<double>(frontend_search_cycles) / static_cast<double>(total_queries);
    }

    double verify_cycles_per_query() const {
        return total_queries == 0 ? 0.0 : static_cast<double>(frontend_verify_cycles) / static_cast<double>(total_queries);
    }

    double verified_rows_per_query() const {
        return total_queries == 0 ? 0.0 : static_cast<double>(frontend_verified_rows) / static_cast<double>(total_queries);
    }

    double throughput_qps() const {
        if (cycles == 0 || clock_mhz <= 0.0) {
            return 0.0;
        }
        const double seconds = static_cast<double>(cycles) / (clock_mhz * 1000000.0);
        return static_cast<double>(total_queries) / seconds;
    }

    double edp_pj_cycle_per_query() const {
        return energy_per_query_pj() * cycles_per_query();
    }
};

ConfigText load_config_text(const std::string& path);
double config_number(const ConfigText& cfg, const std::string& key, double default_value);
int config_int(const ConfigText& cfg, const std::string& key, int default_value);
std::string config_string(const ConfigText& cfg, const std::string& key, const std::string& default_value);

void write_report_json(
    const std::string& path,
    const SimulationStats& stats,
    const std::string& trace_path,
    const std::string& config_path,
    const std::string& decision_path
);

void write_decisions_csv(const std::string& path, const std::vector<Decision>& decisions);

}  // namespace ghhw
