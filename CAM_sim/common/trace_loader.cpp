#include "trace_format.h"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ghhw {

namespace {

uint32_t read_u32(std::istream& in) {
    uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading uint32");
    }
    return value;
}

uint16_t read_u16(std::istream& in) {
    uint16_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading uint16");
    }
    return value;
}

uint8_t read_u8(std::istream& in) {
    uint8_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading uint8");
    }
    return value;
}

}  // namespace

TraceData load_trace_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open trace file: " + path);
    }

    std::array<char, 16> magic_bytes{};
    in.read(magic_bytes.data(), static_cast<std::streamsize>(magic_bytes.size()));
    if (!in) {
        throw std::runtime_error("failed to read trace magic: " + path);
    }
    std::string magic(magic_bytes.data(), strnlen(magic_bytes.data(), magic_bytes.size()));
    if (magic != kTraceMagicText) {
        throw std::runtime_error("invalid trace magic: " + magic);
    }

    TraceData data;
    data.header.magic = magic;
    data.header.version = read_u32(in);
    data.header.num_nodes = read_u32(in);
    data.header.num_heads = read_u32(in);
    data.header.hash_bits = read_u32(in);
    data.header.default_radius = read_u32(in);
    data.header.support_threshold = read_u32(in);

    if (data.header.version != kTraceVersion) {
        throw std::runtime_error("unsupported trace version: " + std::to_string(data.header.version));
    }
    if (data.header.num_heads != kDefaultHeads) {
        throw std::runtime_error("first hardware model requires exactly 8 heads");
    }
    if (data.header.hash_bits != kDefaultHashBits) {
        throw std::runtime_error("first hardware model requires 16-bit hashes");
    }

    data.records.reserve(data.header.num_nodes);
    for (uint32_t i = 0; i < data.header.num_nodes; ++i) {
        TraceRecord rec;
        rec.node_id = read_u32(in);
        for (uint32_t head = 0; head < kDefaultHeads; ++head) {
            rec.head_hashes[head] = read_u16(in);
        }
        rec.sensitivity_q = read_u16(in);
        rec.degree_bucket = read_u8(in);
        rec.reserved = read_u8(in);
        data.records.push_back(rec);
    }

    return data;
}

}  // namespace ghhw
