#include "progress.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include <unistd.h>

namespace ghhw {

namespace {

std::string format_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return "--:--";
    }
    const auto total_seconds = static_cast<uint64_t>(std::llround(seconds));
    const uint64_t hours = total_seconds / 3600;
    const uint64_t minutes = (total_seconds % 3600) / 60;
    const uint64_t secs = total_seconds % 60;

    std::ostringstream out;
    out << std::setfill('0');
    if (hours > 0) {
        out << std::setw(2) << hours << ":";
    }
    out << std::setw(2) << minutes << ":" << std::setw(2) << secs;
    return out.str();
}

}  // namespace

ProgressBar::ProgressBar(
    std::string label,
    bool enabled,
    uint64_t total,
    std::ostream& stream,
    std::chrono::milliseconds update_interval
)
    : label_(std::move(label)),
      enabled_(enabled && total > 0),
      total_(total),
      stream_(&stream),
      update_interval_(update_interval),
      start_time_(std::chrono::steady_clock::now()),
      last_render_time_(start_time_) {}

bool ProgressBar::should_render(const ProgressSnapshot& snapshot) const {
    if (!enabled_) {
        return false;
    }
    if (snapshot.processed == 0 || snapshot.processed >= total_) {
        return true;
    }
    if (snapshot.processed == last_rendered_processed_) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    return now - last_render_time_ >= update_interval_;
}

void ProgressBar::update(const ProgressSnapshot& snapshot) {
    if (!enabled_ || snapshot.processed >= total_) {
        return;
    }
    if (!should_render(snapshot)) {
        return;
    }
    render(snapshot, false);
}

void ProgressBar::finish(const ProgressSnapshot& snapshot) {
    if (!enabled_) {
        return;
    }
    render(snapshot, true);
}

void ProgressBar::render(const ProgressSnapshot& snapshot, bool final) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
    const double progress =
        total_ == 0 ? 1.0 : std::clamp(static_cast<double>(snapshot.processed) / static_cast<double>(total_), 0.0, 1.0);
    const double qps = elapsed_seconds > 0.0 ? static_cast<double>(snapshot.processed) / elapsed_seconds : 0.0;
    const double remaining_queries = static_cast<double>(total_ > snapshot.processed ? total_ - snapshot.processed : 0);
    const double eta_seconds = qps > 0.0 ? remaining_queries / qps : 0.0;
    const int bar_width = 24;
    const int filled = static_cast<int>(std::round(progress * static_cast<double>(bar_width)));

    std::ostringstream bar;
    bar << "[";
    for (int idx = 0; idx < bar_width; ++idx) {
        bar << (idx < filled ? '=' : ' ');
    }
    bar << "]";

    (*stream_) << "\r" << label_ << " " << bar.str() << " "
               << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
               << snapshot.processed << "/" << total_
               << " qps=" << std::setprecision(0) << qps
               << " reuse=" << snapshot.reuse
               << " miss=" << snapshot.computed;
    if (snapshot.cam_evictions > 0) {
        (*stream_) << " evict=" << snapshot.cam_evictions;
    }
    (*stream_) << " elapsed=" << format_duration(elapsed_seconds);
    if (!final && snapshot.processed < total_) {
        (*stream_) << " eta=" << format_duration(eta_seconds);
    }
    if (final) {
        (*stream_) << "\n";
    }
    stream_->flush();
    last_render_time_ = now;
    last_rendered_processed_ = snapshot.processed;
}

bool stderr_is_tty() {
    return ::isatty(fileno(stderr)) != 0;
}

}  // namespace ghhw
