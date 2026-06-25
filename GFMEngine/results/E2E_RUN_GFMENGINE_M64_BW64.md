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
- PQ: `256` centroids, `64` subvectors, input `8`b, centroid `8`b, activation-book `8`b, index `8`b.
- GFMEngine HBM bandwidth `64.0 GB/s`; GB bandwidth `1024.0 GB/s`; IU memory reduction `30.00%`.
- Codebook load mode: `per_call`.
- Attention residual: `True`; multiplier `2.0` for `QK^T` and `AV`.

## Aggregate Result

| Scope | Policy | Reuse | Drop / PQ Loss | Compute Norm | Attention Norm | Memory Norm | Output Norm | Total Norm, No Overlap | Speedup, No Overlap | Total Norm, Pipelined | Speedup, Pipelined |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AVG6 | GFMEngine-PQ | 0.00% | 0.23% | 0.4564x | 1.0588x | 11.1724x | 1.0000x | 0.7012x | 1.43x | 0.4352x | 2.30x |
| AVG6 | TSER40+W4BFPA4 | 40.03% | 1.39% | 0.6002x | 0.5994x | 0.6010x | 0.6003x | 0.6002x | 1.67x | 0.6002x | 1.67x |

## Per-Task Timing

| Task | Policy | Reuse | Drop / PQ Loss | Compute Norm | Attention s | Memory Norm | Total Norm, Pipelined | Speedup, Pipelined |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | GFMEngine-PQ | 0.00% | -0.86% | 0.4475x | 140.374 | 8.7957x | 0.4416x | 2.26x |
| CN | TSER40+W4BFPA4 | 39.90% | 0.98% | 0.6010x | 79.678 | 0.6012x | 0.6010x | 1.66x |
| CL | GFMEngine-PQ | 0.00% | 0.51% | 0.4475x | 140.374 | 8.7957x | 0.4416x | 2.26x |
| CL | TSER40+W4BFPA4 | 39.46% | 1.59% | 0.6054x | 80.261 | 0.6056x | 0.6054x | 1.65x |
| PN | GFMEngine-PQ | 0.00% | 1.37% | 0.4669x | 3027.079 | 14.8006x | 0.4276x | 2.34x |
| PN | TSER40+W4BFPA4 | 39.90% | 1.67% | 0.6010x | 1718.204 | 0.6010x | 0.6010x | 1.66x |
| PL | GFMEngine-PQ | 0.00% | 0.16% | 0.4669x | 3027.079 | 14.8006x | 0.4276x | 2.34x |
| PL | TSER40+W4BFPA4 | 42.36% | 1.51% | 0.5764x | 1647.875 | 0.5765x | 0.5764x | 1.73x |
| AR | GFMEngine-PQ | 0.00% | -0.09% | 0.4520x | 11839.366 | 10.1617x | 0.4383x | 2.28x |
| AR | TSER40+W4BFPA4 | 39.69% | 1.47% | 0.6031x | 6743.637 | 0.6031x | 0.6031x | 1.66x |
| WK | GFMEngine-PQ | 0.00% | 0.31% | 0.4657x | 1700.424 | 14.4184x | 0.4285x | 2.33x |
| WK | TSER40+W4BFPA4 | 38.90% | 1.15% | 0.6110x | 981.239 | 0.6111x | 0.6110x | 1.64x |

## What Dominates GFMEngine-PQ Here

| Task | Search Compute (s) | Query/Add Compute (s) | Search Mem (s) | Query Mem After IU (s) | Raw Activation Book GB | IU-Reduced Activation Book GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 532.105 | 635.486 | 17.218 | 756.326 | 69138.42 | 48396.89 |
| CL | 532.105 | 635.486 | 17.218 | 756.326 | 69138.42 | 48396.89 |
| PN | 6667.494 | 7962.897 | 199.560 | 9477.077 | 866332.89 | 606433.03 |
| PL | 6667.494 | 7962.897 | 199.560 | 9477.077 | 866332.89 | 606433.03 |
| AR | 38643.651 | 46151.587 | 1219.302 | 54927.513 | 5021116.96 | 3514781.87 |
| WK | 3849.642 | 4597.575 | 115.583 | 5471.824 | 500198.69 | 350139.09 |

## Interpretation

- GFMEngine-PQ removes full weight-GEMM MACs, but it still runs every token row. Its online cost is dominated by activation-book lookup and M-way accumulation when the activation book is modeled explicitly.
- TSER40 gets speedup from reducing the miss stream: compute, weight loading, activation loading, intermediate outputs, and final writes all shrink by roughly the miss rate.
- The GFMEngine paper reports cycle-level results from its own simulator, but it does not publish enough per-layer trace detail to reproduce exact cycles. This script is therefore a transparent path-level reconstruction using the paper's public parameters.
