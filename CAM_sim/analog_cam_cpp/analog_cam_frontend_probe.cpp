#include "analog_cam_engine.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string require_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    throw std::runtime_error("missing required argument: " + name);
}

std::string optional_arg(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::string format_double(double value, int precision) {
    if (!std::isfinite(value)) {
        return "inf";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string config_path = require_arg(argc, argv, "--config");
        const std::string word_bits_arg = optional_arg(argc, argv, "--word_bits", "16");
        const std::string max_dist_arg = optional_arg(argc, argv, "--max_dist", "5");
        const std::string out_path = optional_arg(argc, argv, "--out", "");

        const uint32_t word_bits = static_cast<uint32_t>(std::stoul(word_bits_arg));
        const int max_dist = std::stoi(max_dist_arg);

        ghhw::AnalogCamConfig config = ghhw::analog_cam_config_from_file(config_path);
        const ghhw::FrontendSpeedAnalysis analysis = ghhw::analyze_cam_frontends(config, word_bits, max_dist);

        std::ostringstream out;
        out << "# CAM Frontend Probe\n\n";
        out << "- config: `" << config_path << "`\n";
        out << "- word_bits: `" << analysis.word_bits << "`\n";
        out << "- max_dist: `" << analysis.max_dist << "`\n";
        out << "- vdd: `" << format_double(analysis.vdd, 3) << " V`\n";
        out << "- veval: `" << format_double(analysis.veval, 3) << " V`\n";
        out << "- vref: `" << format_double(analysis.vref, 3) << " V`\n";
        out << "- matchline_cap: `" << format_double(analysis.matchline_cap_f * 1.0e15, 3) << " fF`\n";
        out << "- HD boundary: `HD <= " << analysis.hd_boundary << "`\n\n";

        out << "## Timing Summary\n\n";
        out << "| Frontend | Eval boundary | Eval time (ps) | Search time (ps) |\n";
        out << "|---|---:|---:|---:|\n";
        out << "| Exact CAM | `d=0/1` | " << format_double(analysis.exact_cam_eval_time_ps, 3)
            << " | " << format_double(analysis.exact_cam_search_time_ps, 3) << " |\n";
        out << "| HD-CAM | `d=" << analysis.hd_boundary << "/" << (analysis.hd_boundary + 1) << "` | "
            << format_double(analysis.hdcam_eval_time_ps, 3)
            << " | " << format_double(analysis.hdcam_search_time_ps, 3) << " |\n\n";

        const double eval_ratio =
            analysis.exact_cam_eval_time_ps > 0.0 ? analysis.hdcam_eval_time_ps / analysis.exact_cam_eval_time_ps : 0.0;
        out << "- eval time ratio (`HD-CAM / Exact-CAM`): `"
            << format_double(eval_ratio, 4) << "x`\n";
        out << "- search time ratio (`HD-CAM / Exact-CAM`): `"
            << format_double(analysis.search_time_ratio, 4) << "x`\n\n";

        out << "## V_ML and Crossing Time\n\n";
        out << "| d | HD-CAM `V_ML` @ eval_time (V) | HD-CAM crossing to `Vref` (ps) | Exact CAM `V_ML` @ eval_time (V) | Exact CAM crossing to `Vref` (ps) |\n";
        out << "|---:|---:|---:|---:|---:|\n";
        for (const auto& point : analysis.points) {
            out << "| " << point.dist
                << " | " << format_double(point.hdcam_v_ml, 6)
                << " | " << format_double(point.hdcam_t_cross_ps, 3)
                << " | " << format_double(point.exact_cam_v_ml, 6)
                << " | " << format_double(point.exact_cam_t_cross_ps, 3)
                << " |\n";
        }

        if (!out_path.empty()) {
            std::ofstream fout(out_path);
            if (!fout) {
                throw std::runtime_error("failed to open output file: " + out_path);
            }
            fout << out.str();
        } else {
            std::cout << out.str();
        }
    } catch (const std::exception& ex) {
        std::cerr << "[analog-cam-probe] error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
