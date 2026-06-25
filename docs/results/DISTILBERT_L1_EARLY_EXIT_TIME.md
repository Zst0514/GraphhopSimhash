# DistilBERT-L1 Early-Exit Timing

Scope: timing check for the cheap semantic feature used by the graph-context
SimHash key. The previous implementation requested `hidden_states[1]` but still
executed the full 6-layer DistilBERT forward pass. The optimized path executes
only the embedding layer and the first Transformer layer, then returns the same
layer-1 CLS feature.

## Correctness Check

On 64 Cora texts, the early-exit output exactly matches the full-forward
`hidden_states[1]` output:

| Check | Value |
| --- | ---: |
| max absolute difference | 0 |
| full forward, CPU | 0.5719s |
| early exit, CPU | 0.0901s |

## End-to-End Extraction Time

Measured on `pimarch`, `cuda:0`, batch size 128.

| Dataset | Nodes | Full DistilBERT Forward | True L1 Early Exit | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Cora | 2,708 | 5.012s | 2.985s | 1.68x |
| PubMed | 19,717 | 11.134s | 4.245s | 2.62x |
| OGBN-Arxiv | 169,343 | 89.841s | 29.759s | 3.02x |
| Wiki-CS | 11,701 | 8.513s | 3.488s | 2.44x |

## Artifacts

- Full-forward baseline: `output/preprocessing_time_six_tasks/distilbert_l1_time_six_tasks.md`
- Early-exit run: `output/distilbert_l1_early_exit_check/early/distilbert_l1_time_six_tasks.md`

## Interpretation

This optimization keeps the current DistilBERT-L1 cheap feature definition
unchanged. It only avoids executing unused DistilBERT layers. Compared with
switching to BoW/TF-IDF or TinyBERT, this is the safest preprocessing speedup
because it preserves the SimHash input feature exactly.
