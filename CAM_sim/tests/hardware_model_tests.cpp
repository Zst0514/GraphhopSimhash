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

ghhw::TraceData make_global_lru_trace() {
    ghhw::TraceData trace;
    trace.header.magic = ghhw::kTraceMagicText;
    trace.header.version = ghhw::kTraceVersion;
    trace.header.num_heads = ghhw::kDefaultHeads;
    trace.header.hash_bits = ghhw::kDefaultHashBits;
    trace.header.default_radius = 2;
    trace.header.support_threshold = 3;

    trace.records.push_back(make_record(10, {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}));
    trace.records.push_back(make_record(20, {0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff}));
    trace.records.push_back(make_record(30, {0x0000, 0x0000, 0x0000, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa, 0xaaaa}));
    trace.records.push_back(make_record(40, {0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff, 0x00ff}));
    trace.records.push_back(make_record(50, {0xffff, 0xffff, 0xffff, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555}));
    trace.header.num_nodes = static_cast<uint32_t>(trace.records.size());
    return trace;
}

ghhw::TraceData make_global_unbounded_trace() {
    ghhw::TraceData trace;
    trace.header.magic = ghhw::kTraceMagicText;
    trace.header.version = ghhw::kTraceVersion;
    trace.header.num_heads = ghhw::kDefaultHeads;
    trace.header.hash_bits = ghhw::kDefaultHashBits;
    trace.header.default_radius = 2;
    trace.header.support_threshold = 3;

    trace.records.push_back(make_record(10, {0x1111, 0x2222, 0x3001, 0x4001, 0x5001, 0x6001, 0x7001, 0x8001}));
    trace.records.push_back(make_record(20, {0x1111, 0x2222, 0x3002, 0x4002, 0x5002, 0x6002, 0x7002, 0x8002}));
    trace.records.push_back(make_record(30, {0x1111, 0x2222, 0x3003, 0x4003, 0x5003, 0x6003, 0x7003, 0x8003}));
    trace.records.push_back(make_record(40, {0x1111, 0x2222, 0x3004, 0x4004, 0x5004, 0x6004, 0x7004, 0x8004}));
    trace.records.push_back(make_record(50, {0xaaaa, 0xbbbb, 0xcccc, 0xdddd, 0xeeee, 0xf111, 0xf222, 0xf333}));
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

void check_global_lru_capacity() {
    const ghhw::TraceData trace = make_global_lru_trace();

    ghhw::DigitalConfig digital_cfg;
    digital_cfg.support_threshold = 3;
    digital_cfg.radius = 2;
    digital_cfg.replacement_policy = "global_lru";
    digital_cfg.total_cam_bytes = 64;
    digital_cfg.node_entry_bytes = 32;
    ghhw::DigitalHashReuseEngine digital(digital_cfg);
    ghhw::DigitalResult digital_result = digital.run(trace);

    ghhw::AnalogCamConfig analog_cfg;
    analog_cfg.support_threshold = 3;
    analog_cfg.radius = 2;
    analog_cfg.replacement_policy = "global_lru";
    analog_cfg.total_cam_bytes = 64;
    analog_cfg.node_entry_bytes = 32;
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
    assert(!digital_result.decisions[1].hit);
    assert(digital_result.decisions[2].hit);
    assert(digital_result.decisions[2].source_id == 10);
    assert(!digital_result.decisions[3].hit);
    assert(!digital_result.decisions[4].hit);

    assert(digital_result.stats.cam_evictions == 2);
    assert(analog_result.stats.cam_evictions == 2);
    assert(digital_result.stats.capacity_limit_nodes == 2);
    assert(analog_result.stats.capacity_limit_nodes == 2);
    assert(digital_result.stats.max_active_nodes == 2);
    assert(analog_result.stats.max_active_nodes == 2);
}

void check_global_unbounded_removes_bucket_limit() {
    const ghhw::TraceData trace = make_global_unbounded_trace();

    ghhw::DigitalConfig digital_fifo_cfg;
    digital_fifo_cfg.support_threshold = 3;
    digital_fifo_cfg.radius = 2;
    digital_fifo_cfg.replacement_policy = "per_hash_fifo";
    ghhw::DigitalHashReuseEngine digital_fifo(digital_fifo_cfg);
    ghhw::DigitalResult digital_fifo_result = digital_fifo.run(trace);

    ghhw::DigitalConfig digital_unbounded_cfg;
    digital_unbounded_cfg.support_threshold = 3;
    digital_unbounded_cfg.radius = 2;
    digital_unbounded_cfg.replacement_policy = "global_unbounded";
    ghhw::DigitalHashReuseEngine digital_unbounded(digital_unbounded_cfg);
    ghhw::DigitalResult digital_unbounded_result = digital_unbounded.run(trace);

    ghhw::AnalogCamConfig analog_fifo_cfg;
    analog_fifo_cfg.support_threshold = 3;
    analog_fifo_cfg.radius = 2;
    analog_fifo_cfg.replacement_policy = "per_hash_fifo";
    ghhw::AnalogCamHashReuseEngine analog_fifo(analog_fifo_cfg);
    ghhw::AnalogCamResult analog_fifo_result = analog_fifo.run(trace);

    ghhw::AnalogCamConfig analog_unbounded_cfg;
    analog_unbounded_cfg.support_threshold = 3;
    analog_unbounded_cfg.radius = 2;
    analog_unbounded_cfg.replacement_policy = "global_unbounded";
    ghhw::AnalogCamHashReuseEngine analog_unbounded(analog_unbounded_cfg);
    ghhw::AnalogCamResult analog_unbounded_result = analog_unbounded.run(trace);

    assert(digital_fifo_result.stats.max_active_rows == 30);
    assert(analog_fifo_result.stats.max_active_rows == 30);
    assert(digital_unbounded_result.stats.max_active_rows == 32);
    assert(analog_unbounded_result.stats.max_active_rows == 32);
    assert(digital_unbounded_result.stats.cam_evictions == 0);
    assert(analog_unbounded_result.stats.cam_evictions == 0);
    assert(digital_unbounded_result.stats.capacity_limit_nodes == 0);
    assert(analog_unbounded_result.stats.capacity_limit_nodes == 0);
}

}  // namespace

int main() {
    check_hash_utils();
    check_engines_align();
    check_global_lru_capacity();
    check_global_unbounded_removes_bucket_limit();
    std::cout << "hardware_model_tests passed\n";
    return 0;
}
