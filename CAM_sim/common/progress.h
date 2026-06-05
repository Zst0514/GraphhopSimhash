#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

namespace ghhw {

struct ProgressSnapshot {
    uint64_t processed = 0;
    uint64_t total = 0;
    uint64_t reuse = 0;
    uint64_t computed = 0;
    uint64_t cam_evictions = 0;
};

class ProgressBar {
public:
    ProgressBar(
        std::string label,
        bool enabled,
        uint64_t total,
        std::ostream& stream = std::cerr,
        std::chrono::milliseconds update_interval = std::chrono::milliseconds(250)
    );

    void update(const ProgressSnapshot& snapshot);
    void finish(const ProgressSnapshot& snapshot);

private:
    bool should_render(const ProgressSnapshot& snapshot) const;
    void render(const ProgressSnapshot& snapshot, bool final);

    std::string label_;
    bool enabled_ = false;
    uint64_t total_ = 0;
    std::ostream* stream_ = nullptr;
    std::chrono::milliseconds update_interval_{250};
    std::chrono::steady_clock::time_point start_time_{};
    std::chrono::steady_clock::time_point last_render_time_{};
    uint64_t last_rendered_processed_ = 0;
};

bool stderr_is_tty();

}  // namespace ghhw
