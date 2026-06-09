# Extra Graph Datasets

本文档记录计划补充的四个额外图数据集，以及它们和当前 `Cora / PubMed / Arxiv` 主线实验的关系。

## Dataset Summary

| Dataset | Task | Nodes / Entities | Edges / Triples | Classes / Relations | Text / Feature Status | Current Local Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Wiki-CS | Node classification | 11,701 | 216,123 | 10 classes | Wikipedia CS article title/tokens; OFA generator builds node text from `metadata.json` | present: `data/single_graph/wikics` |
| FB15K237 | Link prediction | 14,541 | 310,116 | 237 relations | entity label / aliases / description from `entity2wikidata.json` | present: `data/KG/FB15K237` |
| WN18RR | Link prediction | 40,943 | 93,003 | 11 relations | entity text from `entity2text.txt` | present: `data/KG/WN18RR` |
| ogbn-products | Node classification | 2,449,029 | 123,718,280 directed edge records | 47 classes | OGB provides 100-d BoW/PCA node features; raw product text is not part of the default local GraphhopSimhash path | downloaded; feature proxy checked |

## Why Add Them

These datasets cover different stress points:

```text
Wiki-CS:
    text-rich medium-size node graph.
    Useful for checking whether SimHash / CAM reuse generalizes beyond citation graphs.

FB15K237 and WN18RR:
    relation-heavy link prediction KGs.
    Useful for evaluating whether entity-text reuse and graph-risk scoring transfer to KG workloads.

ogbn-products:
    million-scale product co-purchase graph.
    Useful for scalability, miss-node compaction, CAM capacity, and NPU batch-formation analysis.
```

## Integration Status

### Wiki-CS

OFA already has:

```text
configs/data_config.yaml: wikics
configs/task_config.yaml: wikics
data/single_graph/wikics/gen_data.py
data/single_graph/wikics/metadata.json
```

`gen_data.py` creates node text in the form:

```text
feature node. wikipedia entry name: <title>. entry content: <tokens>
```

Therefore Wiki-CS is the easiest fourth node-classification TAG dataset to add after Cora / PubMed / Arxiv.

### FB15K237 / WN18RR

OFA already has KG configs and local raw files:

```text
configs/data_config.yaml: FB15K237, WN18RR
configs/task_config.yaml: FB15K237, WN18RR
data/KG/FB15K237/{train,valid,test}.txt
data/KG/WN18RR/{train,valid,test}.txt
```

`data/KG/gen_data.py` builds entity text and relation text. These are link-prediction workloads, so they should not be merged blindly into the node-classification accuracy tables. They need a separate `link prediction / KG` section.

### ogbn-products

`ogbn-products` is available locally through OGB. It should be treated in two phases:

```text
Phase 1:
    Use official OGB 100-d node features for graph-scale proxy analysis.
    Measure degree/risk distribution, CAM pressure, miss-node compaction, and batching.

Phase 2:
    If raw product title/description text is available, build a true LLM encoder pool.
```

The default OGB dataset is:

```text
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name="ogbn-products", root="data")
```

Because it has 2.4M nodes, downloads and full embedding generation must be explicit, not automatic.

## ogbn-products Feature Proxy

`ogbn-products` does not ship raw product text in the standard OGB release, so the first check uses the built-in 100-d node features as a graph-scale proxy. This does not replace LLaMA/ST text-embedding results on Cora / PubMed / Arxiv / Wiki-CS; it tests CAM pressure, degree-risk distribution, and miss-node batching at million-node scale.

Command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/profile_ogbn_products_proxy.py
```

Output:

```text
output/ogbn_products_proxy/summary.json
output/ogbn_products_proxy/summary.md
```

Dataset and split:

| Dataset | Nodes | Directed edge records | Feature dim | Classes | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ogbn-products | 2,449,029 | 123,718,280 | 100 | 47 | 196,615 | 39,323 | 2,213,091 |

Degree distribution:

| Mean incidence degree | P50 | P90 | P95 | P99 | Max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101.03 | 52.0 | 228.0 | 326.0 | 756.0 | 34,962 |

Sampled SimHash/CAM proxy:

| Query sample | Anchor sample | Heads x bits | Radius | Hard reuse | Soft reuse | Miss | Hard label agreement | Soft label agreement |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4,096 | 8 x 16 | 2 | 4.80% | 19.05% | 80.95% | 71.67% | 60.47% |

Interpretation:

```text
ogbn-products is useful as a scalability and batching stress test.
The standard feature-only proxy gives low reuse under the current SimHash/CAM setting,
so it should not be compared directly with text-rich Cora/PubMed/Arxiv/Wiki-CS.
Its strongest role is evaluating degree-risk skew, miss-node compaction,
CAM capacity, and NPU batch formation at 2.4M-node scale.
```

## TAPE ogbn-products Text Subset

TAPE provides a text-attributed `ogbn-products` subset. This is not the full 2.45M-node OGB graph; it is a 54K-node product-text subset aligned to OGB node ids through ASIN.

The source CSV is available in the TAPE repository:

```text
dataset/ogbn_products_orig/ogbn-products_subset.csv
```

It contains:

```text
uid, nid, title, content
```

Local preparation command:

```bash
mkdir -p /home/zhangshangtong/Transformer/OFA/data/tape_ogbn_products_orig
curl -L \
  https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/dataset/ogbn_products_orig/ogbn-products_subset.csv \
  -o /home/zhangshangtong/Transformer/OFA/data/tape_ogbn_products_orig/ogbn-products_subset.csv

/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/prepare_tape_products_text.py
```

Generated files:

```text
data/tape_ogbn_products_orig/ogbn-products_subset_text.tsv
output/tape_products_text/summary.json
output/tape_products_text/summary.md
```

Current local conversion:

| Rows | Unique node ids | Non-empty title | Non-empty content | Non-empty title or content | Missing both | ASIN match |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54,025 | 54,025 | 49,122 | 50,179 | 53,446 | 579 | 100.00% |

Text format:

```text
Product: <title>; Description: <content>
```

Current status:

```text
TAPE subset text is prepared and aligned to OGB node ids.
It is ready to be consumed by an ST/LLaMA embedding generation script.
It is not yet registered as a first-class GraphhopSimhash dataset key.
```

The next integration step is to construct a `products_tape` dataset entry by:

```text
1. reading ogbn-products_subset_text.tsv as nid -> raw_text
2. loading OGB products labels and edges
3. inducing the subgraph over the 54,025 TAPE node ids
4. remapping node ids to a compact 0..N-1 range
5. generating ST/LLaMA embeddings for raw_text
6. running the same SimHash / TSER / residual frontend as Cora/PubMed/Arxiv
```

This path is the closest to TAPE's products setting and can be used for text-encoder experiments on a products subset. The full OGB products graph remains a feature-only scalability proxy unless external raw product text is added for all ASINs.

## Readiness Script

Use:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/inspect_extra_graph_datasets.py
```

This writes:

```text
output/dataset_readiness/extra_graph_datasets.json
output/dataset_readiness/extra_graph_datasets.md
```

To explicitly download/check `ogbn-products`:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/inspect_extra_graph_datasets.py \
  --download-products
```

## Recommended Next Steps

1. Run the shared TSER/residual configuration on `wikics`, then decide whether to generate LLaMA/BFP pools.
2. Extend the KG proxy path toward official link-prediction metrics if these datasets become main-table workloads.
3. Add an `ogbn-products` graph-scale proxy path before attempting LLaMA embedding generation.
4. Keep node-classification and link-prediction result tables separate.

## Wiki-CS Initial Sanity Result

Wiki-CS has been minimally connected to the existing node-classification hash-reuse path.

Command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -m GraphhopSimhash \
  --datasets wikics \
  --runs 1 \
  --experiment_suite single \
  --learned_hash_epochs 2 \
  --max_test 2000
```

Result:

| Dataset | Suite | Runs | Baseline Acc | Reuse | Acc | Drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Wiki-CS | single / R2 | 1 | 0.7960 | 22.9% | 0.7895 | 0.65% |

This confirms that Wiki-CS can enter the current node-classification pipeline. The next formal step is to run the shared TSER/residual setting and then, if needed, generate LLaMA/BFP pools.

## Wiki-CS Shared TSER + Residual-Gate Result

The current shared frontend setting is:

```text
8 hash heads x 16 bit
radius = 2
score gate = 3 / 1 / 1
score_reuse_threshold = 31
support >= 5 -> direct reuse
support = 3..4 -> residual-gate path
support < 3 -> compute
```

Command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -m GraphhopSimhash \
  --runs 3 \
  --seed 42 \
  --experiment_suite residual_reuse \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hash_heads_per_route 8 \
  --hamming_only_acceptor \
  --disable_structure_check \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 31 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --score_pair_confidence_discount 1 \
  --radius 2 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 3 \
  --residual_rank 64 \
  --residual_epochs 200 \
  --residual_max_train_pairs 4096 \
  --residual_min_dist 1.0 \
  --residual_alpha_grid 0 0.03125 0.0625 0.125 0.25 0.5 \
  --residual_support_aware_alpha \
  --residual_adapter_type mlp \
  --residual_dropout 0.05 \
  --residual_loss_cosine_weight 1.0 \
  --residual_loss_mse_weight 0.5 \
  --residual_loss_delta_weight 0.75 \
  --residual_bucket_mode support_dist \
  --residual_offline_extra_anchors_per_node 8 \
  --residual_offline_extra_query_nodes 4096 \
  --residual_train_split train_val \
  --residual_gate_loss_weight 0.5 \
  --residual_gate_error_scale 0.25 \
  --residual_gate_error_max 0.45 \
  --datasets wikics \
  --residual_embedding_source data_x \
  --residual_fit_profile st \
  --residual_accept_mode separate \
  --residual_positive_error_max -1 \
  --residual_offline_negative_anchors_per_node 0 \
  --residual_negative_gate_weight 0.0 \
  --residual_accept_loss_weight 1.0 \
  --residual_gate_sparsity_weight 0.0 \
  --residual_gate_accept_threshold 0.575
```

Log:

```text
output/t31_shared_frontend_reuse_wikics/logs/st_wikics_separate_runs3.log
```

Result:

| Dataset | Config | Runs | Baseline Acc | Reuse | Acc | Drop | AvgErr | HitErr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wiki-CS | DirectReuse | 3 | 0.7913 | 10.1% | 0.7888 | 0.25% | 0.01105 | 0.10851 |
| Wiki-CS | SoftDirectReuse | 3 | 0.7913 | 36.9% | 0.7732 | 1.81% | 0.04710 | 0.12734 |
| Wiki-CS | ResidualReuse | 3 | 0.7913 | 36.9% | 0.7779 | 1.33% | 0.03769 | 0.10185 |

Interpretation:

```text
T31 shared TSER/residual-gate raises Wiki-CS reuse from the initial 22.9% sanity run to 36.9%.
Residual-gate keeps the same reuse as SoftDirectReuse and recovers about 0.48% accuracy drop.
The result is close to the 40% reuse target, but not yet a 40%+ setting.
```

## KG Frontend Reuse Proxy

FB15K237 and WN18RR are link-prediction datasets, so they are evaluated with a separate KG proxy rather than the node-classification GNN accuracy path.

Command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/validate_kg_frontend_reuse.py \
  --datasets FB15K237 WN18RR \
  --encoder ST \
  --batch-size 128 \
  --hash-heads 8 \
  --head-bits 16 \
  --radius 2 \
  --support-threshold 5 \
  --max-anchors 2048 \
  --max-eval-triples 5000 \
  --negatives 20 \
  --output-dir output/kg_frontend_reuse
```

The script caches ST entity embeddings under:

```text
cache_data/kg_frontend_reuse/
```

It reports sampled TransE-style AUC with relation prototypes learned from train triples. This is a frontend perturbation proxy, not official KG MRR.

| Dataset | Entities | Relations | Triples | Anchors | Reuse | AvgErr | HitErr | BaseAUC | ReuseAUC | AUCDrop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15K237 | 14,541 | 237 | 310,116 | 2,048 | 63.97% | 0.09705 | 0.15170 | 0.9375 | 0.8724 | 6.51% |
| WN18RR | 40,943 | 11 | 93,003 | 2,048 | 7.36% | 0.01869 | 0.25395 | 0.9749 | 0.9636 | 1.13% |

Interpretation:

```text
FB15K237:
    Text descriptions create many hash-near entity anchors.
    Reuse is high, but relation prediction is sensitive to wrong entity substitutions.

WN18RR:
    Entity text is more dispersed under the current anchor cache.
    Reuse is low, but the proxy AUC perturbation is smaller.
```

This confirms that KG workloads need a KG-specific residual / relation-aware gate before being used as main-table reuse results.

## KG Shared Support-Split Proxy

To mirror the shared frontend split used by node-classification datasets, the KG proxy was also run with:

```text
8 hash heads x 16 bit
radius = 2
support >= 5 -> DirectReuse
support >= 3 -> SoftDirectReuse
support = 3..4 -> residual bucket-delta proxy
```

The KG residual proxy is intentionally lightweight: it learns an average embedding delta per support bucket from entity pairs and selects `alpha` on validation triples. It is not the full node-classification residual-gate model.

Command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/validate_kg_frontend_reuse.py \
  --datasets FB15K237 WN18RR \
  --encoder ST \
  --batch-size 128 \
  --hash-heads 8 \
  --head-bits 16 \
  --radius 2 \
  --hard-support-threshold 5 \
  --soft-support-threshold 3 \
  --max-anchors 2048 \
  --max-eval-triples 5000 \
  --negatives 20 \
  --residual-max-train-pairs 4096 \
  --residual-alpha-grid 0 0.03125 0.0625 0.125 0.25 0.5 \
  --output-dir output/kg_frontend_reuse_shared
```

Result:

| Dataset | Config | Reuse | AvgErr | HitErr | BaseAUC | ReuseAUC | AUCDrop | Alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15K237 | DirectReuse | 63.97% | 0.09705 | 0.15170 | 0.9375 | 0.8724 | 6.51% | - |
| FB15K237 | SoftDirectReuse | 84.79% | 0.15220 | 0.17951 | 0.9375 | 0.8428 | 9.47% | - |
| FB15K237 | ResidualReuse | 84.79% | 0.15219 | 0.17950 | 0.9375 | 0.8429 | 9.46% | 0.031 |
| WN18RR | DirectReuse | 7.36% | 0.01869 | 0.25395 | 0.9749 | 0.9636 | 1.13% | - |
| WN18RR | SoftDirectReuse | 82.84% | 0.29347 | 0.35428 | 0.9749 | 0.7408 | 23.41% | - |
| WN18RR | ResidualReuse | 82.84% | 0.29512 | 0.35627 | 0.9749 | 0.7410 | 23.39% | 0.500 |

Interpretation:

```text
FB15K237:
    support>=5 already gives high reuse but a noticeable AUC drop.
    Opening support=3..4 pushes reuse to 84.8%, but link prediction quality degrades sharply.

WN18RR:
    support>=5 is very conservative and stable.
    support=3..4 is too noisy; direct fuzzy reuse collapses the proxy AUC.

Residual proxy:
    simple bucket-delta correction does not rescue KG fuzzy hits.
    A relation-aware accept gate is needed before KG link-prediction datasets can use the shared fuzzy bucket.
```
