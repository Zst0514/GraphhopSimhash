#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ghhw {

constexpr const char kTraceMagicText[] = "GHSIMTRACE";
constexpr uint32_t kTraceVersion = 1;
constexpr uint32_t kDefaultHeads = 8;
constexpr uint32_t kDefaultHashBits = 16;
constexpr uint32_t kDefaultRadius = 2;
constexpr uint32_t kDefaultSupportThreshold = 3;

struct TraceHeader {
    std::string magic;
    uint32_t version = kTraceVersion;
    uint32_t num_nodes = 0;
    uint32_t num_heads = kDefaultHeads;
    uint32_t hash_bits = kDefaultHashBits;
    uint32_t default_radius = kDefaultRadius;
    uint32_t support_threshold = kDefaultSupportThreshold;
};

struct TraceRecord {
    uint32_t node_id = 0;
    std::array<uint16_t, kDefaultHeads> head_hashes{};
    uint16_t sensitivity_q = 0;
    uint8_t degree_bucket = 0;
    uint8_t reserved = 0;
};

struct TraceData {
    TraceHeader header;
    std::vector<TraceRecord> records;
};

TraceData load_trace_file(const std::string& path);

}  // namespace ghhw
