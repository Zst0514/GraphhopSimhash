#include "metrics.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace ghhw {

namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

std::string decision_csv_path_for_report(const std::string& report_path) {
    return report_path + ".decisions.csv";
}

}  // namespace

ConfigText load_config_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open config file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return ConfigText{path, buffer.str()};
}

double config_number(const ConfigText& cfg, const std::string& key, double default_value) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(cfg.text, match, pattern)) {
        return std::stod(match[1].str());
    }
    return default_value;
}

int config_int(const ConfigText& cfg, const std::string& key, int default_value) {
    return static_cast<int>(config_number(cfg, key, static_cast<double>(default_value)));
}

std::string config_string(const ConfigText& cfg, const std::string& key, const std::string& default_value) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(cfg.text, match, pattern)) {
        return match[1].str();
    }
    return default_value;
}

void write_decisions_csv(const std::string& path, const std::vector<Decision>& decisions) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write decisions file: " + path);
    }
    out << "node_id,hit,source_id,support,min_dist,kind\n";
    for (const Decision& decision : decisions) {
        out << decision.node_id << ','
            << (decision.hit ? 1 : 0) << ','
            << decision.source_id << ','
            << decision.support << ','
            << decision.min_dist << ','
            << decision.kind << '\n';
    }
}

void write_report_json(
    const std::string& path,
    const SimulationStats& stats,
    const std::string& trace_path,
    const std::string& config_path,
    const std::string& decision_path
) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write report file: " + path);
    }

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"implementation\": \"" << json_escape(stats.implementation) << "\",\n";
    out << "  \"calibration\": \"" << json_escape(stats.calibration) << "\",\n";
    out << "  \"trace_path\": \"" << json_escape(trace_path) << "\",\n";
    out << "  \"config_path\": \"" << json_escape(config_path) << "\",\n";
    out << "  \"decision_path\": \"" << json_escape(decision_path.empty() ? decision_csv_path_for_report(path) : decision_path) << "\",\n";
    out << "  \"total_queries\": " << stats.total_queries << ",\n";
    out << "  \"reuse\": " << stats.reuse << ",\n";
    out << "  \"exact_reuse\": " << stats.exact_reuse << ",\n";
    out << "  \"fuzzy_reuse\": " << stats.fuzzy_reuse << ",\n";
    out << "  \"computed\": " << stats.computed << ",\n";
    out << "  \"reuse_rate\": " << stats.reuse_rate() << ",\n";
    out << "  \"candidate_inserts\": " << stats.candidate_inserts << ",\n";
    out << "  \"candidate_overflows\": " << stats.candidate_overflows << ",\n";
    out << "  \"sram_probes\": " << stats.sram_probes << ",\n";
    out << "  \"cam_searches\": " << stats.cam_searches << ",\n";
    out << "  \"cam_compared_rows\": " << stats.cam_compared_rows << ",\n";
    out << "  \"bucket_writes\": " << stats.bucket_writes << ",\n";
    out << "  \"cycles\": " << stats.cycles << ",\n";
    out << "  \"clock_mhz\": " << stats.clock_mhz << ",\n";
    out << "  \"cycles_per_query\": " << stats.cycles_per_query() << ",\n";
    out << "  \"throughput_qps\": " << stats.throughput_qps() << ",\n";
    out << "  \"energy_pj\": " << stats.energy_pj << ",\n";
    out << "  \"energy_per_query_pj\": " << stats.energy_per_query_pj() << ",\n";
    out << "  \"area_proxy_um2\": " << stats.area_proxy_um2 << ",\n";
    out << "  \"edp_pj_cycle_per_query\": " << stats.edp_pj_cycle_per_query() << "\n";
    out << "}\n";
}

}  // namespace ghhw
