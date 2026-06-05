#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ghhw {

inline int hamming_distance16(uint16_t a, uint16_t b) {
    return __builtin_popcount(static_cast<unsigned>(a ^ b));
}

inline uint16_t flip_bit16(uint16_t value, int bit_id) {
    return static_cast<uint16_t>(value ^ static_cast<uint16_t>(1u << bit_id));
}

inline std::vector<uint16_t> generate_hamming_neighbors16(uint16_t value, int radius, int bits = 16) {
    if (bits <= 0 || bits > 16) {
        throw std::invalid_argument("generate_hamming_neighbors16 supports 1..16 bits");
    }
    if (radius < 0) {
        throw std::invalid_argument("radius must be non-negative");
    }
    radius = std::min(radius, bits);
    std::vector<uint16_t> result;
    result.push_back(value);

    if (radius >= 1) {
        for (int i = 0; i < bits; ++i) {
            result.push_back(flip_bit16(value, i));
        }
    }
    if (radius >= 2) {
        for (int i = 0; i < bits; ++i) {
            for (int j = i + 1; j < bits; ++j) {
                result.push_back(static_cast<uint16_t>(value ^ (1u << i) ^ (1u << j)));
            }
        }
    }
    if (radius > 2) {
        // First version is intentionally optimized for the GraphhopSimhash R<=2 workload.
        // Higher radii are not used by the hardware comparison configs.
        throw std::invalid_argument("radius > 2 is not supported in the first hardware model");
    }
    return result;
}

inline int hamming_ball_size(int bits, int radius) {
    if (bits <= 0 || bits > 16 || radius < 0) {
        throw std::invalid_argument("invalid hamming ball parameters");
    }
    radius = std::min(radius, bits);
    int total = 1;
    if (radius >= 1) {
        total += bits;
    }
    if (radius >= 2) {
        total += bits * (bits - 1) / 2;
    }
    if (radius > 2) {
        throw std::invalid_argument("radius > 2 is not supported in the first hardware model");
    }
    return total;
}

}  // namespace ghhw
