# 8-head 16-bit 3-vote Hash Reuse Hardware Models

This subproject compares two non-NPU hardware frontends for the current
GraphhopSimhash reuse path:

- `digital_logic_cpp`: per-head 16-bit SRAM bucket lookup, fuzzy neighbor
  enumeration, and a small candidate CAM for support aggregation.
- `analog_cam_cpp`: per-head RC/discharge threshold CAM search, followed by
  the same digital candidate aggregation and `support >= 3` reuse rule.

Both simulators consume the same binary trace:

```text
8 heads x 16-bit hash
radius = 2
support_threshold = 3
```

The first version models only hash lookup, support voting, and cache update.
It intentionally excludes score gate, residual correction, and quantization so
that the digital and CAM lookup frontends are compared on the same surface.

## Build

```bash
cd /home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash

cmake -S hardware_8h16b3vote -B hardware_8h16b3vote/build
cmake --build hardware_8h16b3vote/build -j
ctest --test-dir hardware_8h16b3vote/build
```

## Export A Real Trace

Run from `/home/qiumingzhi/Simhash-S/OneForAll` so `python -m GraphhopSimhash`
uses the local package:

```bash
python -m GraphhopSimhash.hardware_8h16b3vote.tools.export_graphhop_trace \
  --datasets pubmed \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --route_min_support_hits 3 \
  --radius 2 \
  --output GraphhopSimhash/hardware_8h16b3vote/traces/pubmed_8h16b_r2.trace
```

## Run Both Simulators

```bash
cd /home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/hardware_8h16b3vote

./build/digital_logic_cpp/digital_hash_reuse \
  --trace traces/pubmed_8h16b_r2.trace \
  --config digital_logic_cpp/configs/digital_default.json \
  --out reports/pubmed_digital.json

./build/analog_cam_cpp/analog_cam_reuse \
  --trace traces/pubmed_8h16b_r2.trace \
  --config analog_cam_cpp/configs/analog_cam_default.json \
  --out reports/pubmed_analog_cam.json

python tools/compare_reports.py \
  reports/pubmed_digital.json \
  reports/pubmed_analog_cam.json \
  --out reports/pubmed_compare.md
```

One-command smoke path:

```bash
bash hardware_8h16b3vote/tools/run_pubmed_8h16b3vote.sh
```

## Trace Format

Binary little-endian layout:

```text
Header:
  magic[16] = "GHSIMTRACE"
  version: uint32
  num_nodes: uint32
  num_heads: uint32 = 8
  hash_bits: uint32 = 16
  default_radius: uint32 = 2
  support_threshold: uint32 = 3

Record:
  node_id: uint32
  head_hashes[8]: uint16
  sensitivity_q: uint16
  degree_bucket: uint8
  reserved: uint8
```

`sensitivity_q` is reserved for future score-gate hardware experiments. The
current lookup-only models do not use it.

## Report Fields

Each simulator writes:

- `*.json`: aggregate metrics.
- `*.json.decisions.csv`: per-node hit/miss decision.

Important metrics:

- `reuse_rate`: accepted reuse hits / total queries.
- `cycles_per_query`: cycle-model latency normalized per query.
- `throughput_qps`: estimated queries per second from the configured clock.
- `energy_per_query_pj`: analytical proxy energy per query.
- `edp_pj_cycle_per_query`: energy-delay proxy.
- `area_proxy_um2`: simple area proxy, not a physical layout result.

The analog CAM defaults now use an RC/discharge threshold model:

- every active row starts from a precharged match line,
- mismatch bits contribute discharge conductance,
- the simulator evaluates `V_ML = VDD * exp(-G*t/C)`,
- a comparator threshold decides whether the row is accepted.

This is still a compact behavioral model, not a full SPICE signoff flow. Use
`configs/camasim_cost_stub.json` as the handoff point for CAMASim/EvaCAM-
derived numbers.
