#include <cassert>
#include <iostream>

#include "../analog_cam_cpp/analog_cam_engine.h"
#include "../common/hash_utils.h"
#include "../common/trace_format.h"
#include "../digital_logic_cpp/digital_engine.h"

namespace {

ghhw::TraceRecord make_record(uint32_t node_id, std::initializer_list<uint16_t> hashes) {
    ghhw::TraceRecord rec;
    rec.node_id = node_id;
    size_t idx = 0;
    for (uint16_t value : hashes) {
        if (idx < ghhw::kDefaultHeads) {
            rec.head_hashes[idx++] = value;
        }
    }
    while (idx < ghhw::kDefaultHeads) {
        rec.head_hashes[idx] = static_cast<uint16_t>(0x1000 + node_id * 17 + idx);
        ++idx;
    }
    return rec;
}

ghhw::TraceData make_synthetic_trace() {
    ghhw::TraceData trace;
    trace.header.magic = ghhw::kTraceMagicText;
    trace.header.version = ghhw::kTraceVersion;
    trace.header.num_heads = ghhw::kDefaultHeads;
    trace.header.hash_bits = ghhw::kDefaultHashBits;
    trace.header.default_radius = 2;
    trace.header.support_threshold = 3;

    trace.records.push_back(make_record(0, {0, 0, 0, 0, 0, 0, 0, 0}));
    trace.records.push_back(make_record(1, {0, 0, 0, 0x00ff, 0x0f0f, 0x3333, 0x5555, 0x7777}));
    trace.records.push_back(make_record(2, {0, 0, 0x00ff, 0x0f0f, 0x3333, 0x5555, 0x7777, 0x7fff}));
    trace.records.push_back(make_record(3, {1, 2, 4, 0x00ff, 0x0f0f, 0x3333, 0x5555, 0x7777}));
    trace.header.num_nodes = static_cast<uint32_t>(trace.records.size());
    return trace;
}

void check_hash_utils() {
    assert(ghhw::hamming_distance16(0x0000, 0x0000) == 0);
    assert(ghhw::hamming_distance16(0x0000, 0x0003) == 2);
    assert(ghhw::hamming_ball_size(16, 2) == 137);
    const auto neighbors = ghhw::generate_hamming_neighbors16(0, 2, 16);
    assert(neighbors.size() == 137);
}

void check_engines_align() {
    const ghhw::TraceData trace = make_synthetic_trace();

    ghhw::DigitalConfig digital_cfg;
    digital_cfg.support_threshold = 3;
    digital_cfg.radius = 2;
    ghhw::DigitalHashReuseEngine digital(digital_cfg);
    ghhw::DigitalResult digital_result = digital.run(trace);

    ghhw::AnalogCamConfig analog_cfg;
    analog_cfg.support_threshold = 3;
    analog_cfg.radius = 2;
    ghhw::AnalogCamHashReuseEngine analog(analog_cfg);
    ghhw::AnalogCamResult analog_result = analog.run(trace);

    assert(digital_result.decisions.size() == analog_result.decisions.size());
    for (size_t idx = 0; idx < digital_result.decisions.size(); ++idx) {
        const auto& d = digital_result.decisions[idx];
        const auto& a = analog_result.decisions[idx];
        assert(d.hit == a.hit);
        assert(d.support == a.support);
        assert(d.min_dist == a.min_dist);
        assert(d.kind == a.kind);
        if (d.hit) {
            assert(d.source_id == a.source_id);
        }
    }

    assert(!digital_result.decisions[0].hit);
    assert(digital_result.decisions[1].hit);
    assert(digital_result.decisions[1].kind == "exact");
    assert(!digital_result.decisions[2].hit);
    assert(digital_result.decisions[3].hit);
    assert(digital_result.decisions[3].kind == "fuzzy");
    assert(digital_result.stats.reuse == 2);
    assert(digital_result.stats.computed == 2);
}

}  // namespace

int main() {
    check_hash_utils();
    check_engines_align();
    std::cout << "hardware_model_tests passed\n";
    return 0;
}
