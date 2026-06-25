# Candidate Discovery Ablation

This experiment isolates the SimHash/CAM candidate-discovery stage before TSER and residual repair.
Lookup uses cheap online keys; LLaMA embeddings and labels are used only for offline quality measurement.

## Setup

- query sample per dataset: `5000`
- anchor sample per dataset: `8192`
- SimHash heads: `16`
- bits per head: `16`
- Hamming radius: `2`
- valid-anchor threshold: `cosine >= 0.80`
- graph-context key: `0.50 * self + 0.50 * neighbor_mean`
- usable multi-head support: `support >= 3`
- strong multi-head support: `support >= 5`

## Average Across Datasets

| Method | Lookup Yield | Valid Anchor | Valid Yield | Emb. Cos. | Label Agree. | Mean Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-only SimHash | 99.81% | 44.08% | 44.00% | 0.7488 | 24.90% | 1.00 |
| Graph-context SimHash (1H) | 99.86% | 49.74% | 49.67% | 0.7604 | 33.29% | 1.00 |
| Graph-context SimHash (2H) | 50.09% | 69.23% | 35.94% | 0.8171 | 50.73% | 1.50 |
| Graph-context SimHash (4H) | 32.98% | 82.01% | 27.05% | 0.8560 | 65.70% | 2.28 |
| Graph-context SimHash (8H) | 68.24% | 78.90% | 54.71% | 0.8449 | 63.73% | 3.54 |
| Graph-context SimHash (16H) | 42.42% | 87.95% | 36.88% | 0.8740 | 75.46% | 5.85 |

## Per-Dataset Results

| Dataset | Method | Lookup Yield | Valid Anchor | Valid Yield | Emb. Cos. | Label Agree. | Support |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CR | Self-only SimHash | 98.89% | 44.36% | 43.87% | 0.7523 | 25.35% | 0.99 |
| CR | Graph-context SimHash (1H) | 99.22% | 47.79% | 47.42% | 0.7465 | 38.78% | 0.99 |
| CR | Graph-context SimHash (2H) | 62.74% | 65.98% | 41.40% | 0.7982 | 57.15% | 1.63 |
| CR | Graph-context SimHash (4H) | 48.34% | 74.48% | 36.00% | 0.8271 | 73.34% | 2.56 |
| CR | Graph-context SimHash (8H) | 84.64% | 75.57% | 63.96% | 0.8220 | 77.23% | 4.37 |
| CR | Graph-context SimHash (16H) | 70.01% | 80.06% | 56.06% | 0.8343 | 86.34% | 7.93 |
| PB | Self-only SimHash | 100.00% | 74.72% | 74.72% | 0.8358 | 44.46% | 1.00 |
| PB | Graph-context SimHash (1H) | 99.98% | 79.22% | 79.20% | 0.8458 | 49.81% | 1.00 |
| PB | Graph-context SimHash (2H) | 56.82% | 87.93% | 49.96% | 0.8663 | 59.87% | 1.57 |
| PB | Graph-context SimHash (4H) | 35.72% | 91.66% | 32.74% | 0.8835 | 67.30% | 2.34 |
| PB | Graph-context SimHash (8H) | 77.12% | 93.28% | 71.94% | 0.8904 | 69.58% | 3.56 |
| PB | Graph-context SimHash (16H) | 48.86% | 96.32% | 47.06% | 0.9054 | 78.06% | 6.03 |
| AR | Self-only SimHash | 99.98% | 38.51% | 38.50% | 0.7734 | 12.64% | 1.00 |
| AR | Graph-context SimHash (1H) | 99.98% | 47.55% | 47.54% | 0.7889 | 19.28% | 1.00 |
| AR | Graph-context SimHash (2H) | 46.70% | 67.58% | 31.56% | 0.8222 | 30.36% | 1.47 |
| AR | Graph-context SimHash (4H) | 20.50% | 79.51% | 16.30% | 0.8495 | 41.46% | 2.12 |
| AR | Graph-context SimHash (8H) | 64.94% | 78.50% | 50.98% | 0.8455 | 40.10% | 3.01 |
| AR | Graph-context SimHash (16H) | 19.04% | 88.03% | 16.76% | 0.8754 | 52.10% | 4.44 |
| WK | Self-only SimHash | 100.00% | 59.08% | 59.08% | 0.8076 | 26.70% | 1.00 |
| WK | Graph-context SimHash (1H) | 99.98% | 67.63% | 67.62% | 0.8232 | 38.23% | 1.00 |
| WK | Graph-context SimHash (2H) | 70.80% | 83.79% | 59.32% | 0.8554 | 52.40% | 1.71 |
| WK | Graph-context SimHash (4H) | 55.30% | 90.34% | 49.96% | 0.8756 | 65.21% | 2.69 |
| WK | Graph-context SimHash (8H) | 90.60% | 91.02% | 82.46% | 0.8776 | 66.64% | 4.28 |
| WK | Graph-context SimHash (16H) | 65.38% | 95.41% | 62.38% | 0.8947 | 77.85% | 7.11 |
| PR | Self-only SimHash | 100.00% | 10.34% | 10.34% | 0.5659 | 20.76% | 1.00 |
| PR | Graph-context SimHash (1H) | 100.00% | 15.42% | 15.42% | 0.5936 | 30.78% | 1.00 |
| PR | Graph-context SimHash (2H) | 43.42% | 48.36% | 21.00% | 0.7491 | 66.79% | 1.43 |
| PR | Graph-context SimHash (4H) | 31.64% | 68.65% | 21.72% | 0.8295 | 82.74% | 2.26 |
| PR | Graph-context SimHash (8H) | 63.48% | 58.13% | 36.90% | 0.7901 | 77.95% | 3.56 |
| PR | Graph-context SimHash (16H) | 42.74% | 72.34% | 30.92% | 0.8413 | 84.37% | 6.02 |
| A23 | Self-only SimHash | 100.00% | 37.50% | 37.50% | 0.7577 | 19.48% | 1.00 |
| A23 | Graph-context SimHash (1H) | 100.00% | 40.84% | 40.84% | 0.7645 | 22.84% | 1.00 |
| A23 | Graph-context SimHash (2H) | 20.06% | 61.71% | 12.38% | 0.8112 | 37.79% | 1.20 |
| A23 | Graph-context SimHash (4H) | 6.36% | 87.42% | 5.56% | 0.8705 | 64.15% | 1.73 |
| A23 | Graph-context SimHash (8H) | 28.66% | 76.90% | 22.04% | 0.8436 | 50.87% | 2.46 |
| A23 | Graph-context SimHash (16H) | 8.48% | 95.52% | 8.10% | 0.8929 | 74.06% | 3.55 |

## Reading The Metrics

- `Lookup Yield`: the lookup returns an anchor satisfying the method's support rule.
- `Valid Anchor`: among returned anchors, the fraction whose LLaMA embedding cosine is at least `0.80`.
- `Valid Yield`: returned valid anchors as a fraction of all queried nodes.
- `Emb. Cos.`: cosine similarity between query and selected anchor in the LLaMA target embedding space.
- `Label Agree.`: offline label agreement sanity check; labels are not used by lookup.

The key comparison is not raw lookup yield alone. Single-head lookup can return many anchors, but multi-head graph-context consensus improves valid-anchor precision and embedding similarity, giving the later reuse filter a cleaner candidate set.
