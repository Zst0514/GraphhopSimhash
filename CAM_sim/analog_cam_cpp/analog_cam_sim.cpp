#include "analog_cam_engine.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "../common/metrics.h"
#include "../common/trace_format.h"

namespace {

std::string require_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    throw std::runtime_error("missing required argument: " + name);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string trace_path = require_arg(argc, argv, "--trace");
        const std::string config_path = require_arg(argc, argv, "--config");
        const std::string out_path = require_arg(argc, argv, "--out");
        const std::string decision_path = out_path + ".decisions.csv";

        ghhw::TraceData trace = ghhw::load_trace_file(trace_path);
        ghhw::AnalogCamConfig config = ghhw::analog_cam_config_from_file(config_path);
        ghhw::AnalogCamHashReuseEngine engine(config);
        ghhw::AnalogCamResult result = engine.run(trace);
        ghhw::write_decisions_csv(decision_path, result.decisions);
        ghhw::write_report_json(out_path, result.stats, trace_path, config_path, decision_path);

        std::cout << "[analog-cam] wrote " << out_path
                  << " reuse=" << result.stats.reuse
                  << "/" << result.stats.total_queries
                  << " cycles/query=" << result.stats.cycles_per_query()
                  << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "[analog-cam] error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
