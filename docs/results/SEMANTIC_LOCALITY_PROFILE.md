# Semantic Locality Profile for Bypass Motivation

该实验用于支撑 Motivation 中的 bypass opportunity：在不使用最终 reuse policy 的前提下，直接比较图相邻节点和随机节点的语义相似度。

核心问题是：

```text
TAG / GFM workload 中，相邻节点文本 embedding 是否天然更相似？
SimHash 是否能观测到这种 locality？
```

因此这里统计的是 raw opportunity，不是最终安全 reuse 率。最终 reuse/drop 需要在 SimHash/CAM/TSER/residual gate 后单独报告。

## Experimental Setup

Pair types:

```text
neighbor:
    graph edge pairs

random:
    uniformly sampled unrelated node pairs

same_label:
    same-class pairs when labels are available
```

Metrics:

```text
cosine similarity:
    higher means closer embedding

normalized SimHash Hamming distance:
    lower means closer hash code
```

Main command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  scripts/profile_semantic_locality.py \
  --datasets cora pubmed arxiv \
  --embedding-source llama \
  --sample-pairs 30000 \
  --hash-bits 128 \
  --chunk-size 512 \
  --output-dir output/semantic_locality_profile
```

Supplemental ST/products-proxy command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  scripts/profile_semantic_locality.py \
  --datasets cora pubmed arxiv wikics products \
  --embedding-source st \
  --sample-pairs 50000 \
  --hash-bits 128 \
  --output-dir output/semantic_locality_profile
```

Output files:

```text
output/semantic_locality_profile/llama/summary.md
output/semantic_locality_profile/llama/cdf.tsv
output/semantic_locality_profile/llama/cdf.png

output/semantic_locality_profile/st/summary.md
output/semantic_locality_profile/st/cdf.tsv
output/semantic_locality_profile/st/cdf.png
```

## LLaMA Embedding Results

Reference embedding source:

```text
W4A8 LLaMA-7B embedding pools
```

Summary:

| Dataset | Pair | Cos mean | Cos p50 | Ham mean | Ham <=0.25 | Ham <=0.30 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora | neighbor | 0.7391 | 0.8495 | 0.2211 | 70.3% | 76.5% |
| Cora | random | 0.6585 | 0.7465 | 0.2631 | 59.1% | 72.5% |
| PubMed | neighbor | 0.8978 | 0.9089 | 0.1381 | 99.1% | 99.9% |
| PubMed | random | 0.8079 | 0.8130 | 0.1994 | 86.9% | 98.0% |
| Arxiv | neighbor | 0.8801 | 0.8921 | 0.1426 | 98.1% | 99.7% |
| Arxiv | random | 0.7394 | 0.7466 | 0.2193 | 77.2% | 94.9% |

Neighbor-vs-random gap:

| Dataset | Cos mean lift | Hamming mean reduction | Ham <=0.30 lift |
| --- | ---: | ---: | ---: |
| Cora | 0.0806 | 0.0420 | 4.0% |
| PubMed | 0.0899 | 0.0614 | 1.9% |
| Arxiv | 0.1407 | 0.0767 | 4.7% |

Interpretation:

```text
Graph-neighbor pairs are consistently closer than random pairs.
The gap is visible in both cosine similarity and SimHash Hamming distance.
This supports a fuzzy bypass path: graph-local nodes often carry similar semantic embeddings.
```

LLaMA cosine values are globally high, so the SimHash Hamming gap is the cleaner signal for the Motivation figure.

## ST / Products Proxy Results

This run includes `wikics` and standard `ogbn-products`.

Important distinction:

```text
Cora/PubMed/Wiki-CS:
    ST text embeddings.

Arxiv:
    cached processed feature path in the current repository.

ogbn-products:
    OGB 100-d feature proxy, not raw product text.
```

Neighbor-vs-random gap:

| Dataset | Cos mean lift | Hamming mean reduction | Ham <=0.30 lift |
| --- | ---: | ---: | ---: |
| Cora | 0.3005 | 0.1076 | 31.4% |
| PubMed | 0.2777 | 0.1079 | 57.8% |
| Arxiv | 0.0549 | 0.0374 | 8.9% |
| Wiki-CS | 0.1070 | 0.0614 | 5.0% |
| ogbn-products | 0.2997 | 0.1054 | 17.9% |

Interpretation:

```text
The products proxy also shows strong graph-local feature similarity.
Because standard OGB products lacks raw product text, this should be used as a scalability/locality proxy rather than a direct LLaMA text-encoder result.
```

## How to Use in Motivation

Recommended figure:

```text
Figure: Semantic locality CDF

(a) LLaMA cosine CDF:
    neighbor vs random pairs on Cora/PubMed/Arxiv

(b) LLaMA SimHash Hamming CDF:
    neighbor vs random pairs on Cora/PubMed/Arxiv

optional supplement:
    products feature-proxy CDF
```

Narrative:

```text
1. Graph-local nodes are semantically closer than random nodes.
2. Exact cache matching cannot exploit this because texts are rarely identical.
3. SimHash/CAM can expose fuzzy semantic locality.
4. TSER/residual gate is needed later to make this bypass safe.
```
