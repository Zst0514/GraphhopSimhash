# Matched Replacement Motivation Table

Each task replaces the same node budget with real SimHash-CAM anchors. High/low groups are matched by support and Hamming-distance buckets, so drop differences reflect graph-position sensitivity rather than a larger replacement count or looser candidate-distance distribution.

| Task | Metric | Replaced | Support | Ham. | High-P | Low-P | High-C | Low-C | High-U | Low-U | Random |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | Node Acc. | 10.01% | 3.63 | 1.23 | 0.74% | 0.75% | 0.76% | 0.30% | 0.76% | 0.59% | 0.18% |
| CL | Link AUC | 10.01% | 3.63 | 1.23 | 2.40% | 0.25% | 1.65% | 0.60% | 0.58% | 2.15% | 1.34% |
| PN | Node Acc. | 10.00% | 3.69 | 1.20 | 0.62% | 0.98% | 0.89% | 0.71% | 0.84% | 0.65% | 0.65% |
| PL | Link AUC | 10.00% | 3.67 | 1.20 | 5.04% | 0.55% | 2.63% | 1.59% | 0.80% | 3.89% | 2.19% |
| AR | Node Acc. | 10.00% | 3.44 | 1.27 | 0.60% | 0.68% | 0.70% | 0.49% | 0.60% | 0.59% | 0.59% |
| WK | Node Acc. | 9.95% | 3.53 | 1.25 | 0.50% | 0.38% | 0.72% | 0.12% | 0.29% | 0.40% | 0.22% |

Reading guide:

- `P`: propagation / fanout-related position.
- `C`: graph-context boundary / neighborhood mismatch.
- `U`: rare-tail / low-redundancy position.
- `Drop` is accuracy drop for node tasks and AUC drop for link tasks; smaller is better.

This table is intended for Motivation. It should be used to state that semantic candidate quality is not sufficient by itself; downstream damage changes with graph position, and degree alone is not a complete explanation.

## Arxiv Check

The Arxiv row is complete: it contains 5 runs and all seven groups
(`High-P`, `Low-P`, `High-C`, `Low-C`, `High-U`, `Low-U`, and `Random`).
The strongest Arxiv separation is the context pair: `High-C` causes `0.70%`
drop while the matched `Low-C` group causes `0.49%`. Propagation alone is not
monotonic on Arxiv under matched candidate quality (`High-P` is `0.60%` and
`Low-P` is `0.68%`), which is useful for the Motivation text: graph-side
vulnerability is multi-dimensional and cannot be reduced to degree only.
