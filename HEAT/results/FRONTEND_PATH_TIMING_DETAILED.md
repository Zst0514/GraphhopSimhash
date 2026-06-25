# Detailed Frontend Path Timing vs HEAT-Style Bit-Serial

## What Is Simulated

- GRACE uses the existing local BFP array trace for the fixed `W4BFPA4` encoder path.
- HEAT-style uses a separate bit-serial PE model: `32` PEs by default, each with a `32x32` 1-bit systolic array.
- HEAT key vertices execute `W8A10`; non-key vertices execute `W4A2`.
- Weight loading, activation loading, BFP exponent loading, intermediate output writes, final embedding IO, and CAM query cycles are all accounted for.

## Configuration

- Clock: `500.0 MHz`.
- Weight bandwidth: `25.6 GB/s`; activation/output bandwidth: `1024.0 GB/s`; embedding bandwidth: `25.6 GB/s`.
- HEAT weight-load mode: `dual`.
- HEAT compute mode: `throughput`.
- HEAT PE: `32` PEs, `32x32` cells/PE, utilization `0.85`.
- GRACE CAM query: search `1.0` + select `1.0` + miss update `1.0` cycles.

## Aggregate Speedup

| Scope | Policy | Reuse | Drop | HEAT Proxy Drop | Compute Norm | Weight Load Norm | Activation Load Norm | Output Norm | Total Norm, No Overlap | Speedup, No Overlap | Total Norm, Overlap | Speedup, Overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AVG_HEAT5 | HEAT-style W8A10/W4A2 | 0.00% | - | 0.83% | 2.0118x | 1.1008x | 0.6946x | 1.0000x | 1.9889x | 0.50x | 2.0118x | 0.50x |
| AVG_HEAT5 | TSER40+W4BFPA4 | 40.26% | 1.44% | - | 0.5974x | 0.5975x | 0.5974x | 0.5974x | 0.5974x | 1.67x | 0.5974x | 1.67x |
| AVG6 | HEAT-style W8A10/W4A2 | 0.00% | - | 0.85% | 2.0118x | 1.1007x | 0.6946x | 1.0000x | 1.9898x | 0.50x | 2.0118x | 0.50x |
| AVG6 | TSER40+W4BFPA4 | 40.03% | 1.39% | - | 0.5996x | 0.5997x | 0.5997x | 0.5997x | 0.5997x | 1.67x | 0.5997x | 1.67x |

## Per-Task Frontend Path

| Task | Policy | Reuse | Compute Norm | Weight Norm | Act Norm | Output Norm | Total Norm, Overlap | Speedup, Overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | HEAT-style W8A10/W4A2 | 0.00% | 2.0118x | 1.1019x | 0.6946x | 1.0000x | 2.0118x | 0.50x |
| CN | TSER40+W4BFPA4 | 39.90% | 0.6010x | 0.6012x | 0.6010x | 0.6010x | 0.6010x | 1.66x |
| CL | HEAT-style W8A10/W4A2 | 0.00% | 2.0118x | 1.1019x | 0.6946x | 1.0000x | 2.0118x | 0.50x |
| CL | TSER40+W4BFPA4 | 39.46% | 0.6054x | 0.6056x | 0.6054x | 0.6054x | 0.6054x | 1.65x |
| PN | HEAT-style W8A10/W4A2 | 0.00% | 2.0118x | 1.1000x | 0.6946x | 1.0000x | 2.0118x | 0.50x |
| PN | TSER40+W4BFPA4 | 39.90% | 0.6010x | 0.6010x | 0.6010x | 0.6010x | 0.6010x | 1.66x |
| PL | HEAT-style W8A10/W4A2 | 0.00% | 2.0118x | 1.1000x | 0.6946x | 1.0000x | 2.0118x | 0.50x |
| PL | TSER40+W4BFPA4 | 42.36% | 0.5764x | 0.5765x | 0.5764x | 0.5764x | 0.5764x | 1.73x |
| AR | HEAT-style W8A10/W4A2 | 0.00% | 2.0118x | 1.1000x | 0.6946x | 1.0000x | 2.0118x | 0.50x |
| AR | TSER40+W4BFPA4 | 39.69% | 0.6031x | 0.6031x | 0.6031x | 0.6031x | 0.6031x | 1.66x |
| WK | HEAT-style W8A10/W4A2 | 0.00% | 2.0118x | 1.1005x | 0.6946x | 1.0000x | 2.0118x | 0.50x |
| WK | TSER40+W4BFPA4 | 38.90% | 0.6110x | 0.6111x | 0.6110x | 0.6110x | 0.6110x | 1.64x |

## Interpretation

- This is the correct speedup comparison for the local fixed-W4BFPA4 path: GRACE and HEAT-style are simulated through different hardware datapaths.
- HEAT-style is not represented by a single average bit-plane ratio. Its W8A10 key path serializes both weight and activation bits, while GRACE's W4BFPA4 array uses the measured local BFP trace.
- TSER40 reduces the number of encoder invocations. Therefore compute, weight streaming, activation loading, and intermediate output writes all shrink with the miss-node stream after compaction.
- The overlap model assumes compute can overlap with module weight/activation/output movement; the no-overlap model is a conservative upper bound.
