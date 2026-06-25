# GFMEngine-Style PQ Frontend Path Timing

## Important Correction

- `N * [0.1 * T_bitserial(W8A10) + 0.9 * T_bitserial(W4A2)]` is a HEAT-style topology-aware bit-serial model, not GFMEngine.
- GFMEngine's ASPDAC'25 path is PQ-based MatMul: online centroid search plus activation-book lookup. The paper ignores offline codebook/book construction, and this simulator follows that convention.
- The baseline therefore charges every node/token for centroid search, activation-book/index traffic, IU cycles, adder-tree accumulation, intermediate output traffic, and final embedding writes. There is no TSER-style node skip in GFMEngine-PQ.

## Configuration

- Local array: `W4BFPA4`, `500.0 MHz`.
- Local bandwidths: weight `25.6 GB/s`, activation/output `1024.0 GB/s`, embedding `25.6 GB/s`.
- GFMEngine: `500.0 MHz`, `16` PEs, each with one `4x16` SA and `2` `8`-lane ATs.
- GFMEngine peak model: `1024` centroid-search MAC-equivalent ops/cycle and `256` AT lanes/cycle before utilization.
- PQ: `256` centroids, `128` subvectors, input `8`b, centroid `8`b, activation-book `8`b, index `8`b.
- GFMEngine HBM bandwidth `64.0 GB/s`; GB bandwidth `1024.0 GB/s`; IU memory reduction `30.00%`.
- Codebook load mode: `per_call`.

## Aggregate Result

| Scope | Policy | Reuse | Drop / PQ Loss | Compute Norm | Memory Norm | Output Norm | Total Norm, No Overlap | Speedup, No Overlap | Total Norm, Pipelined | Speedup, Pipelined |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AVG6 | GFMEngine-PQ | 0.00% | 0.23% | 0.6462x | 22.0759x | 1.0000x | 1.1658x | 0.86x | 0.7335x | 1.36x |
| AVG6 | TSER40+W4BFPA4 | 40.03% | 1.39% | 0.6003x | 0.6010x | 0.6003x | 0.6003x | 1.67x | 0.6003x | 1.67x |

## Per-Task Timing

| Task | Policy | Reuse | Drop / PQ Loss | Compute Norm | Memory Norm | Total Norm, Pipelined | Speedup, Pipelined |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | GFMEngine-PQ | 0.00% | -0.86% | 0.6462x | 17.3723x | 0.7335x | 1.36x |
| CN | TSER40+W4BFPA4 | 39.90% | 0.98% | 0.6010x | 0.6012x | 0.6010x | 1.66x |
| CL | GFMEngine-PQ | 0.00% | 0.51% | 0.6462x | 17.3723x | 0.7335x | 1.36x |
| CL | TSER40+W4BFPA4 | 39.46% | 1.59% | 0.6054x | 0.6056x | 0.6054x | 1.65x |
| PN | GFMEngine-PQ | 0.00% | 1.37% | 0.6462x | 29.2566x | 0.7335x | 1.36x |
| PN | TSER40+W4BFPA4 | 39.90% | 1.67% | 0.6010x | 0.6010x | 0.6010x | 1.66x |
| PL | GFMEngine-PQ | 0.00% | 0.16% | 0.6462x | 29.2566x | 0.7335x | 1.36x |
| PL | TSER40+W4BFPA4 | 42.36% | 1.51% | 0.5764x | 0.5765x | 0.5764x | 1.73x |
| AR | GFMEngine-PQ | 0.00% | -0.09% | 0.6462x | 20.0757x | 0.7335x | 1.36x |
| AR | TSER40+W4BFPA4 | 39.69% | 1.47% | 0.6031x | 0.6031x | 0.6031x | 1.66x |
| WK | GFMEngine-PQ | 0.00% | 0.31% | 0.6462x | 28.5000x | 0.7335x | 1.36x |
| WK | TSER40+W4BFPA4 | 38.90% | 1.15% | 0.6110x | 0.6111x | 0.6110x | 1.64x |

## What Dominates GFMEngine-PQ Here

| Task | Search Compute (s) | Query/Add Compute (s) | Search Mem (s) | Query Mem After IU (s) | Raw Activation Book GB | IU-Reduced Activation Book GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 532.105 | 1270.971 | 17.218 | 1512.652 | 138276.83 | 96793.78 |
| CL | 532.105 | 1270.971 | 17.218 | 1512.652 | 138276.83 | 96793.78 |
| PN | 6667.494 | 15925.794 | 199.560 | 18954.154 | 1732665.79 | 1212866.05 |
| PL | 6667.494 | 15925.794 | 199.560 | 18954.154 | 1732665.79 | 1212866.05 |
| AR | 38643.651 | 92303.175 | 1219.302 | 109855.026 | 10042233.92 | 7029563.74 |
| WK | 3849.642 | 9195.151 | 115.583 | 10943.649 | 1000397.39 | 700278.17 |

## Interpretation

- GFMEngine-PQ removes full weight-GEMM MACs, but it still runs every token row. Its online cost is dominated by activation-book lookup and M-way accumulation when the activation book is modeled explicitly.
- TSER40 gets speedup from reducing the miss stream: compute, weight loading, activation loading, intermediate outputs, and final writes all shrink by roughly the miss rate.
- The GFMEngine paper reports cycle-level results from its own simulator, but it does not publish enough per-layer trace detail to reproduce exact cycles. This script is therefore a transparent path-level reconstruction using the paper's public parameters.
