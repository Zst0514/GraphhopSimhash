# Candidate Discovery Ablation

This experiment isolates the SimHash/CAM candidate-discovery stage before TSER and residual repair.
Lookup uses cheap online keys; LLaMA embeddings and labels are used only for offline quality measurement.

## Key Findings

This ablation supports the candidate-discovery part of graph-aware encoder reuse:

1. Exact text caching has essentially no coverage on the sampled anchor pools, so ordinary exact-match caching cannot explain the observed reuse opportunity.
2. Single-head SimHash can find many candidates, but graph-context keys improve candidate quality: average label agreement rises from `24.90%` to `33.29%`.
3. Multi-head graph-context SimHash exposes a useful support structure: `68.24%` of sampled queries reach the usable support region, with `0.8449` LLaMA embedding cosine and `63.73%` label agreement.

Paper-facing compact table:

| Method | Candidate Coverage | Candidate Cosine | Label Agreement |
| --- | ---: | ---: | ---: |
| Exact text cache | 0.00% | - | - |
| Self-only SimHash | 99.81% | 0.7488 | 24.90% |
| Graph-context SimHash | 99.86% | 0.7604 | 33.29% |
| Multi-head graph-context | 68.24% | 0.8449 | 63.73% |

For single-head methods, coverage means any candidate found within radius. For multi-head graph-context, coverage means the candidate reaches the usable support region (`support >= 3`).

## Setup

- query sample per dataset: `5000`
- anchor sample per dataset: `8192`
- SimHash heads: `8`
- bits per head: `16`
- Hamming radius: `2`
- graph-context key: `0.50 * self + 0.50 * neighbor_mean`
- usable multi-head support: `support >= 3`
- strong multi-head support: `support >= 5`

## Average Across Datasets

| Method | AnyHit | Usable | Strong | Fuzzy | CandCos | LabelHit | MeanSupport |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random anchor | 100.00% | 100.00% | - | - | 0.6930 | 16.25% | 1.00 |
| Exact text cache | 0.00% | 0.00% | - | - | - | - | - |
| Self-only SimHash | 99.81% | 99.81% | - | - | 0.7488 | 24.90% | 1.00 |
| Graph-context SimHash | 99.86% | 99.86% | - | - | 0.7604 | 33.29% | 1.00 |
| Multi-head graph-context | 100.00% | 68.24% | 23.44% | 44.80% | 0.8449 | 63.73% | 3.54 |

## Per-Dataset Results

| Dataset | Method | AnyHit | Usable | Strong | Fuzzy | CandCos | LabelHit | Support |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CR | Random anchor | 100.00% | 100.00% | - | - | 0.6451 | 17.98% | 1.00 |
| CR | Exact text cache | 0.00% | 0.00% | - | - | - | - | - |
| CR | Self-only SimHash | 98.89% | 98.89% | - | - | 0.7523 | 25.35% | 0.99 |
| CR | Graph-context SimHash | 99.22% | 99.22% | - | - | 0.7465 | 38.78% | 0.99 |
| CR | Multi-head graph-context | 100.00% | 84.64% | 41.69% | 42.95% | 0.8220 | 77.23% | 4.37 |
| PB | Random anchor | 100.00% | 100.00% | - | - | 0.8054 | 36.26% | 1.00 |
| PB | Exact text cache | 0.00% | 0.00% | - | - | - | - | - |
| PB | Self-only SimHash | 100.00% | 100.00% | - | - | 0.8358 | 44.46% | 1.00 |
| PB | Graph-context SimHash | 99.98% | 99.98% | - | - | 0.8458 | 49.81% | 1.00 |
| PB | Multi-head graph-context | 100.00% | 77.12% | 21.94% | 55.18% | 0.8904 | 69.58% | 3.56 |
| AR | Random anchor | 100.00% | 100.00% | - | - | 0.7356 | 7.78% | 1.00 |
| AR | Exact text cache | 0.00% | 0.00% | - | - | - | - | - |
| AR | Self-only SimHash | 99.98% | 99.98% | - | - | 0.7734 | 12.64% | 1.00 |
| AR | Graph-context SimHash | 99.98% | 99.98% | - | - | 0.7889 | 19.28% | 1.00 |
| AR | Multi-head graph-context | 100.00% | 64.94% | 8.74% | 56.20% | 0.8455 | 40.10% | 3.01 |
| WK | Random anchor | 100.00% | 100.00% | - | - | 0.7591 | 14.76% | 1.00 |
| WK | Exact text cache | 0.00% | 0.00% | - | - | - | - | - |
| WK | Self-only SimHash | 100.00% | 100.00% | - | - | 0.8076 | 26.70% | 1.00 |
| WK | Graph-context SimHash | 99.98% | 99.98% | - | - | 0.8232 | 38.23% | 1.00 |
| WK | Multi-head graph-context | 100.00% | 90.60% | 39.28% | 51.32% | 0.8776 | 66.64% | 4.28 |
| PR | Random anchor | 100.00% | 100.00% | - | - | 0.4906 | 8.52% | 1.00 |
| PR | Exact text cache | 0.00% | 0.00% | - | - | - | - | - |
| PR | Self-only SimHash | 100.00% | 100.00% | - | - | 0.5659 | 20.76% | 1.00 |
| PR | Graph-context SimHash | 100.00% | 100.00% | - | - | 0.5936 | 30.78% | 1.00 |
| PR | Multi-head graph-context | 100.00% | 63.48% | 24.50% | 38.98% | 0.7901 | 77.95% | 3.56 |
| A23 | Random anchor | 100.00% | 100.00% | - | - | 0.7220 | 12.20% | 1.00 |
| A23 | Exact text cache | 0.00% | 0.00% | - | - | - | - | - |
| A23 | Self-only SimHash | 100.00% | 100.00% | - | - | 0.7577 | 19.48% | 1.00 |
| A23 | Graph-context SimHash | 100.00% | 100.00% | - | - | 0.7645 | 22.84% | 1.00 |
| A23 | Multi-head graph-context | 100.00% | 28.66% | 4.50% | 24.16% | 0.8436 | 50.87% | 2.46 |

## Reading The Metrics

- `AnyHit`: the lookup found at least one candidate in the sampled anchor pool.
- `Usable`: candidate evidence reaches the method-specific usable threshold. For multi-head, this is `support >= soft_support`.
- `Strong` and `Fuzzy`: the high-support and medium-support regions used by the frontend policy.
- `CandCos`: cosine similarity between query and selected anchor in the LLaMA target embedding space.
- `LabelHit`: offline label agreement sanity check; labels are not used by lookup.

The key comparison is not only hit rate. Exact text caching has little coverage, single-head SimHash can find candidates but lacks repeated evidence, and multi-head graph-context SimHash exposes support structure that can feed TSER/residual decisions.
