# GFMEngine PQ MatMul Reproduction

## Scope

This is an algorithm-level reproduction of GFMEngine's PQ-based MatMul, not a formula-only estimate.
It trains PQ centroids, builds activation books, and evaluates `X @ W` through activation-book lookup and summation.

- Source: `synthetic_lowrank_gaussian`.
- Rows generated/loaded: `768`; eval rows: `128`.
- Input dim: `1024`; output dim: `4096`; centroids: `256`.

## Results

| M | dsub | Rel RMSE | Mean Cosine | Online Compute / Dense | Activation-Book Bytes / Row | Offline Book Size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 0.9798 | 0.2955 | 0.0781x | 64.00 KiB | 16.00 MiB |
| 64 | 16 | 0.8028 | 0.6222 | 0.1250x | 256.00 KiB | 64.00 MiB |
| 128 | 8 | 0.5988 | 0.8045 | 0.1875x | 512.00 KiB | 128.00 MiB |

## Read

- Search compute is independent of `M` for fixed `K` and `D`: `rows * K * D`.
- Activation-book traffic grows linearly with `M`: `rows * M * out_features * bits`.
- This is why a small `M` can make GFMEngine-PQ look strong, while a realistic larger `M` can become memory-bound.
