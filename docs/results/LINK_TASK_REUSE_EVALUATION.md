# Link-Task Reuse Evaluation

This note records the current link-task sanity checks for the frontend reuse path.
The results are not mixed with the main TAG node-classification table because the
tasks have different downstream objectives.

## PubMed Link-Transfer Check

PubMed is still a TAG node-classification graph, so the check reuses the same
LLaMA2-7B frontend setting as the main node task:

```text
dataset: PubMed
target:  W4BFPA8_B128
policy:  Full TSER + residual reuse
T:       24
runs:    3
```

The script reconstructs reused embeddings, trains a sampled link predictor on
baseline node representations, and evaluates the same predictor on baseline
versus reused representations. AUC is the primary metric; AP is reported as a
secondary metric.

| Dataset | Policy | Effective Changed Nodes | Base AUC | Reuse AUC | AUC Drop | Base AP | Reuse AP | AP Drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PubMed | Full TSER + residual, T=24 | 18.64% | 0.9192 | 0.9144 | 0.48% | 0.9137 | 0.9088 | 0.50% |

Source:

```text
output/node_link_reuse_transfer/pubmed_tser_T24_link_reuse.md
output/node_link_reuse_transfer/pubmed_T24_runs3.log
```

Interpretation:

```text
The PubMed reuse point that is useful for node classification also transfers
reasonably to a sampled link-prediction proxy. The AUC/AP drops are both about
0.5%, much smaller than the KG fuzzy-reuse drops below.
```

## KG Link-Prediction Proxy

FB15K237 and WN18RR are knowledge-graph link-prediction datasets. They are
evaluated with a separate proxy that learns relation prototypes from train
triples and reports sampled TransE-style AUC. This is not official KG MRR.

| Dataset | Config | Reuse | AvgErr | HitErr | BaseAUC | ReuseAUC | AUCDrop | Alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15K237 | DirectReuse | 63.97% | 0.09705 | 0.15170 | 0.9375 | 0.8724 | 6.51% | - |
| FB15K237 | SoftDirectReuse | 84.79% | 0.15220 | 0.17951 | 0.9375 | 0.8428 | 9.47% | - |
| FB15K237 | ResidualReuse | 84.79% | 0.15219 | 0.17950 | 0.9375 | 0.8429 | 9.46% | 0.031 |
| WN18RR | DirectReuse | 7.36% | 0.01869 | 0.25395 | 0.9749 | 0.9636 | 1.13% | - |
| WN18RR | SoftDirectReuse | 82.84% | 0.29347 | 0.35428 | 0.9749 | 0.7408 | 23.41% | - |
| WN18RR | ResidualReuse | 82.84% | 0.29512 | 0.35627 | 0.9749 | 0.7410 | 23.39% | 0.500 |

Source:

```text
docs/results/EXTRA_GRAPH_DATASETS.md
output/kg_frontend_reuse/kg_frontend_reuse_summary.md
output/kg_frontend_reuse_shared/kg_frontend_reuse_summary.md
```

Interpretation:

```text
FB15K237 has many hash-near entity anchors, but relation prediction is sensitive
to wrong entity substitutions, so high reuse produces large AUC drops.

WN18RR is stable only under very conservative direct reuse. Opening the fuzzy
bucket creates many noisy entity substitutions and collapses sampled AUC.

Thus, current GRACE reuse should remain scoped to TAG node graphs in the main
paper. KG link prediction would need relation-aware TSER/residual logic before
it can become a main result.
```
