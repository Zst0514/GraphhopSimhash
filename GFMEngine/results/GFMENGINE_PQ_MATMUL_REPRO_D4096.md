# GFMEngine PQ MatMul Reproduction

## Scope

This is an algorithm-level reproduction of GFMEngine's PQ-based MatMul, not a formula-only estimate.
It trains PQ centroids, builds activation books, and evaluates `X @ W` through activation-book lookup and summation.

- Source: `synthetic_lowrank_gaussian`.
- Rows generated/loaded: `768`; eval rows: `128`.
- Input dim: `4096`; output dim: `4096`; centroids: `256`.

## Results

| M | dsub | Rel RMSE | Mean Cosine | Online Compute / Dense | Activation-Book Bytes / Row | Offline Book Size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 1.0106 | 0.1331 | 0.0664x | 64.00 KiB | 16.00 MiB |
| 64 | 64 | 0.9808 | 0.2934 | 0.0781x | 256.00 KiB | 64.00 MiB |
| 128 | 32 | 0.9242 | 0.4428 | 0.0938x | 512.00 KiB | 128.00 MiB |

## Read

- Search compute is independent of `M` for fixed `K` and `D`: `rows * K * D`.
- Activation-book traffic grows linearly with `M`: `rows * M * out_features * bits`.
- This is why a small `M` can make GFMEngine-PQ look strong, while a realistic larger `M` can become memory-bound.
