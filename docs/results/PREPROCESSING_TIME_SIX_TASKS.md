# Six-Task Preprocessing Time

Scope: graph-side offline preprocessing only. This excludes data loading, LLaMA/AWQ/BFPA pool generation, and online CAM query execution.

- SimHash heads: `8`
- bits/head: `16`
- graph-context key: `0.50 * self + 0.50 * neighbor_mean`
- measured on: `pimarch`
- torch threads: `64`

## Six Evaluation Tasks

| Task | Dataset | Nodes | Edges | Graph Key (s) | SimHash Table (s) | P Risk (s) | C Risk (s) | U Risk (s) | Method Total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | cora | 2708 | 10858 | 0.062 | 0.019 | 0.007 | 0.269 | 0.008 | 0.365 |
| CL | cora | 2708 | 10858 | 0.062 | 0.019 | 0.007 | 0.269 | 0.008 | 0.365 |
| PN | pubmed | 19717 | 88670 | 0.247 | 0.045 | 0.019 | 0.274 | 0.006 | 0.590 |
| PL | pubmed | 19717 | 88670 | 0.247 | 0.045 | 0.019 | 0.274 | 0.006 | 0.590 |
| AR | arxiv | 169343 | 1166243 | 2.323 | 0.317 | 0.195 | 0.560 | 0.127 | 3.522 |
| WK | wikics | 11701 | 431206 | 0.679 | 0.029 | 0.024 | 0.301 | 0.001 | 1.034 |

## Notes

- CN and CL share the same Cora preprocessing artifact.
- PN and PL share the same PubMed preprocessing artifact.
- `Method Total` excludes graph loading and cheap-feature loading; those are still kept in the raw JSON/TSV for reproducibility.
- `SimHash Table` builds both self-only and graph-context multi-head signatures for profiling/reuse support.
- `P Risk`, `C Risk`, and `U Risk` separately time propagation, graph-context, and low-degree uniqueness metadata construction.

## Cheap Feature Generation

The graph-context key currently uses DistilBERT layer-1 cheap semantic features.
The original implementation requested layer-1 features but still executed the
full 6-layer DistilBERT forward pass. A true layer-1 early-exit path preserves
the feature exactly while reducing extraction time:

| Dataset | Nodes | Full Forward | L1 Early Exit | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Cora | 2708 | 5.012s | 2.985s | 1.68x |
| PubMed | 19717 | 11.134s | 4.245s | 2.62x |
| OGBN-Arxiv | 169343 | 89.841s | 29.759s | 3.02x |
| Wiki-CS | 11701 | 8.513s | 3.488s | 2.44x |

Detailed timing and correctness check: `docs/results/DISTILBERT_L1_EARLY_EXIT_TIME.md`.
