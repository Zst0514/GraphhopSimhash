# Hardware Model

## Common Algorithm

For every query node, both engines follow the same high-level policy:

```text
1. Query 8 independent 16-bit hash heads.
2. Retrieve reuse candidates in the frontend.
3. Aggregate candidates by node_id.
4. Reuse if one candidate has support_count >= 3.
5. Otherwise compute the node and insert it into the cache.
```

The cache update mirrors the software controller: reuse hits do not write a new
entry, while misses insert the newly computed node into every head table.

Tie-break:

```text
higher support_count
lower min_hamming_distance
newer timestamp
smaller node_id
```

## Digital Logic Model

The digital frontend uses direct-indexed SRAM tables:

```text
head_id + 16-bit hash -> bucket
```

Default parameters:

```text
clock_mhz = 1000
radius = 2
support_threshold = 3
memo_k = 3
neighbor_lookup_lanes = 16
candidate_cam_entries = 512
```

Exact lookup is one parallel cycle across all 8 heads. Fuzzy lookup enumerates
the 16-bit Hamming ball:

```text
C(16,0) + C(16,1) + C(16,2) = 137 keys/head
ceil(137 / 16 lanes) = 9 fuzzy cycles
```

Energy is a first-order proxy:

```text
energy =
  sram_probes * sram_probe_energy_pj
  + candidate_inserts * candidate_cam_probe_energy_pj
  + bucket_writes * bucket_write_energy_pj
```

This model is expected to look good when the radius is small and SRAM neighbor
enumeration is still cheaper than CAM matchline activation.

## Analog CAM Model

The analog CAM frontend uses an RC/discharge threshold model instead of an
ideal `dist <= radius` oracle. For every active row:

```text
1. Precharge the match line to VDD.
2. Mismatch bits contribute discharge conductance.
3. Evaluate after t_eval:
   V_ML = VDD * exp(-G_total * t_eval / C_ML)
4. Compare V_ML against V_ref.
5. If V_ML >= V_ref, the row is accepted as a threshold hit.
```

With the default auto-threshold, `V_ref` is placed between the nominal
`d = 2` and `d = 3` match-line voltages, so the default zero-noise model
implements a deterministic approximation of `dist <= 2`.

Default parameters:

```text
clock_mhz = 500
radius = 2
support_threshold = 3
memo_k = 3
subarray_rows = 512
parallel_subarrays = 1
cam_search_cycles = 3
candidate_cam_entries = 512
vdd = 1.0
matchline_base_cap_f = 2.0e-15
matchline_cap_per_bit_f = 5.0e-16
mismatch_conductance_s = 4.0e-5
match_leak_conductance_s = 2.0e-7
precharge_time_ps = 40
eval_time_ps = 100
sense_time_ps = 20
comparator_vref = -1.0 (auto midpoint between d=2 and d=3)
device_sigma_rel = 0
sense_noise_sigma_v = 0
comparator_noise_sigma_v = 0
```

The model separates two concepts:

- CAM lookup: compare query hash against active rows through RC discharge.
- Digital candidate aggregation: merge node IDs and count support.

Energy is a proxy:

```text
energy += active_rows * hash_bits * cam_compare_energy_fj_per_bit / 1000
energy += writes * cam_write_energy_pj
energy += candidate_inserts * candidate_cam_probe_energy_pj
```

Search latency is derived from:

```text
search_time_ps = precharge_time_ps + eval_time_ps + sense_time_ps
search_cycles = ceil(search_time_ps / clock_period_ps)
```

`cam_search_cycles` remains as a configurable cycle floor. If
`parallel_subarrays=1`, subarrays are assumed to evaluate in parallel, so
latency is bounded by one subarray evaluation while energy scales with active
rows. If set to 0, subarrays are serialized in the cycle model.

## Fairness Between The Two Models

Both engines:

- read the same trace,
- use the same `support_threshold`,
- use the same `memo_k` bucket retention,
- use the same candidate capacity,
- produce the same per-node decision format.

The difference is only the candidate retrieval frontend:

- Digital: enumerate Hamming neighbors and probe SRAM buckets.
- Analog CAM: scan active CAM rows with a Hamming threshold.

This is still a compact behavioral model, not a SPICE signoff model. It now
captures:

- match-line precharge,
- RC discharge versus mismatch count,
- threshold sensing,
- optional fixed device variation and dynamic sense noise.

For a physical conclusion, replace proxy parameters with CAMASim/EvaCAM-derived
numbers and record the calibration source in the config.
