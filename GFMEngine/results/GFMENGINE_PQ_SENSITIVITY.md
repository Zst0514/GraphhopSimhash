# GFMEngine-PQ Sensitivity Summary

Note: this is the older peak sensitivity without the full objective framing.
Use `GFMENGINE_PQ_OBJECTIVE_COMPARISON.md` as the primary comparison because it
includes the attention residual that PQ-based MatMul does not remove.

GFMEngine-PQ does not skip nodes.  The online path still processes every token
row through centroid search, activation-book lookup, indexing, accumulation,
and output writeback.

The fast peak result comes from a different mechanism: PQ replaces the original
weight-involving `D x D` GEMMs with `nc x D` centroid search plus lookup.  This
can be much faster than the original encoder, but the actual speed depends
strongly on two details that the ASPDAC'25 paper does not disclose:

- PQ subvector count `M`, which controls activation-book lookup and adder-tree
  accumulation.
- Effective indexed HBM bandwidth after fine-grained activation-book requests.

All rows below use the local fixed `W4BFPA4` baseline at 500MHz and the current
TSER40 reuse table.

| Scenario | GFM Clock | M | Effective GFM HBM | GFMEngine-PQ Pipelined Norm | GFMEngine-PQ Speedup | TSER40 Norm | TSER40 Speedup | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Peak-style upper bound | 500MHz | 16 | 256GB/s | 0.2485x | 4.02x | 0.6003x | 1.67x | GFMEngine may look much faster, but this assumes small `M` and peak indexed bandwidth. |
| Moderate indexed path | 500MHz | 64 | 64GB/s | 0.4625x | 2.16x | 0.6003x | 1.67x | GFMEngine is still faster, but margin is much smaller. |
| Conservative indexed path | 500MHz | 128 | 64GB/s | 0.7335x | 1.36x | 0.6003x | 1.67x | GFMEngine becomes slower than TSER40 because it runs every token row and pays heavy lookup/accumulation traffic. |

Interpretation: cite GFMEngine-PQ peak as an optimistic upper bound only.  For a
paper comparison against TSER40, the safer statement is that GFMEngine trades
full GEMM compute for all-node PQ lookup traffic, while TSER40 reduces the
number of encoder invocations.  Without GFMEngine's private cycle simulator or
its exact `M` and indexed-bandwidth measurements, the fair comparison should
show this sensitivity instead of a single speedup number.
