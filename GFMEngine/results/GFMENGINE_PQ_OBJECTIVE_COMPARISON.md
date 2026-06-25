# Objective GFMEngine-PQ Comparison

## What Was Reproduced

This comparison separates two things:

1. Algorithm reproduction: `reproduce_pq_matmul.py` actually trains PQ
   centroids, builds activation books, and evaluates `X @ W` through
   activation-book lookup and summation.
2. Architecture timing: `simulate_pq_frontend_paths.py` maps the reproduced
   PQ online path onto the local LLaMA2-7B module trace and compares it with
   `TSER40+W4BFPA4`.

The timing model now includes the Transformer attention residual (`QK^T` and
`AV`).  This matters because GFMEngine's PQ-based MatMul only replaces GEMMs
involving model weights; the attention quadratic term remains.  TSER40 skips
the whole encoder invocation for reused nodes, so it also scales the attention
term by the miss stream.

## Current Environment Limitation

The visible Python environment does not have `torch`, so the `.pt` embedding
pools cannot be loaded in this session.  The PQ MatMul reproduction therefore
ran on deterministic synthetic low-rank Gaussian inputs.  The script is ready
to load a real pool with `--pool_path` when run in the project's torch
environment.

## PQ MatMul Reproduction, D=4096

Command:

```bash
python GFMEngine/reproduce_pq_matmul.py \
  --rows 768 --train_rows 640 --rows_eval 128 \
  --in_features 4096 --out_features 4096 \
  --subvectors 16 64 128
```

| M | dsub | Rel RMSE | Mean Cosine | Online Compute / Dense | Activation-Book Bytes / Row | Offline Book Size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 1.0106 | 0.1331 | 0.0664x | 64.00 KiB | 16.00 MiB |
| 64 | 64 | 0.9808 | 0.2934 | 0.0781x | 256.00 KiB | 64.00 MiB |
| 128 | 32 | 0.9242 | 0.4428 | 0.0938x | 512.00 KiB | 128.00 MiB |

Read: for fixed `K=256` and `D=4096`, centroid-search compute stays near
`K/out_features = 0.0625x` of dense GEMM plus `M/D` accumulation overhead.
However, activation-book traffic grows linearly with `M`.

## Trace-Driven Timing With Attention Residual

All rows use equal 500MHz clock for the local `W4BFPA4` path and GFMEngine.
The TSER row uses the current `~40%` reuse table.

| Scenario | M | Effective GFM HBM | GFMEngine-PQ Pipelined Norm | GFMEngine-PQ Speedup | GFMEngine-PQ No-Overlap Norm | TSER40 Norm | TSER40 Speedup | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Optimistic upper bound | 16 | 256GB/s | 0.2338x | 4.28x | 0.3066x | 0.6002x | 1.67x | Fast, but `dsub=256` is a very coarse PQ split for `D=4096`. |
| Indexed-memory moderate | 64 | 64GB/s | 0.4352x | 2.30x | 0.7012x | 0.6002x | 1.67x | Faster only if indexed activation-book bandwidth remains decent. |
| Higher-quality / heavier lookup | 128 | 64GB/s | 0.6901x | 1.45x | 1.1596x | 0.6002x | 1.67x | Slower than TSER40 because all nodes still pay attention plus lookup/accumulation. |

## Conclusion

GFMEngine-PQ should not be modeled as skipped compute.  It processes every node
and trades full weighted GEMMs for:

- centroid search: `rows * K * D`;
- activation-book indexed reads: `rows * M * out_features`;
- adder-tree accumulation: `rows * M * out_features`;
- unchanged attention GEMMs: `QK^T` and `AV`.

The realistic comparison is therefore a range, not a single number.  If one
assumes small `M` and high effective indexed bandwidth, GFMEngine-PQ can be
faster than TSER40.  If `M` is increased to preserve PQ quality or indexed
bandwidth falls below peak HBM, GFMEngine-PQ's advantage shrinks and can become
slower than TSER40.  TSER40's advantage is structurally different: it reduces
the number of encoder invocations, so compute, weight loading, activation
loading, attention, output writes, and final embedding writes all shrink with
the miss stream.
