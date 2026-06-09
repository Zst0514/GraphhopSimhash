# Graph-Context SimHash Retrieval Key

## Core Idea

SimHash/CAM 前端不直接只对节点自身文本特征做 hash。当前实现中有一条 topology-aware retrieval route，把节点自己的 cheap semantic feature 和一跳邻居的平均 feature 混合后再做 SimHash：

```text
f_hash(v) = normalize(0.5 * f_self(v) + 0.5 * mean_{u in N(v)} f_self(u))
```

代码位置：

```text
GraphhopSimhash/features.py
build_topology_hash_features()
```

对应实现：

```python
neighbor_mean = _compute_neighbor_mean(verify_features, edge_index)
topology_feat = 0.5 * verify_features + 0.5 * neighbor_mean
return F.normalize(topology_feat, p=2, dim=1)
```

## Why Not Hash Self Feature Only

只 hash 节点自身文本特征时，CAM 找到的是：

```text
text-semantically similar nodes
```

但 GFM 的最终预测不是纯文本分类。LLM encoder 输出 embedding 后，还要经过 GNN propagation。一个 reuse candidate 是否安全，不只取决于目标节点和锚点节点自己的文本是否相似，也取决于它们的局部图上下文是否相似。

纯 self hash 的问题是 graph-blind：

```text
两个节点文本相似，但邻域主题/结构不同，
直接把它们作为 reuse anchor 可能污染后端 GNN propagation。
```

## Why Mix Neighbor Context

一跳邻居均值提供了局部图上下文：

```text
mean_{u in N(v)} f_self(u)
```

它近似回答：

```text
这个节点周围是什么语义环境？
```

混合 self 和 neighbor context 后，CAM 查找的候选不再只是文本相似，而是：

```text
文本语义相似 + 局部图上下文相似
```

这更接近 GNN 后端看到的输入状态，因为 GNN 本身也会把 self feature 和 neighbor information 组合起来。

## Why 0.5 / 0.5

`0.5 self + 0.5 neighbor` 是一个保守的第一版设计：

```text
self term:
    保留目标节点自己的文本身份，避免 hash 完全被邻居平滑。

neighbor term:
    注入一跳图上下文，避免 hash 退化成普通 semantic hashing。
```

如果 neighbor 权重过高，hub 或 dense community 中的节点容易被过度平滑，导致不同节点被压到相近 hash key。
如果 neighbor 权重过低，retrieval key 又会回到 graph-blind self hash。

因此 0.5 / 0.5 的意义不是最优调参，而是作为一个低复杂度、无需学习的 graph-context key：

```text
保留 self semantics，同时让 CAM lookup 对局部图上下文敏感。
```

## Relation to Motivation Profiling

Motivation 的 semantic locality CDF 显示：

```text
graph-neighbor pairs have lower SimHash Hamming distance than random pairs.
```

这说明图局部结构中确实存在 semantic locality。但这种 locality 是 noisy 的，neighbor 和 random 分布仍然重叠。

Graph-context SimHash key 的作用是把 raw semantic locality 变成更接近 GNN execution semantics 的 candidate search：

```text
raw text similarity:
    candidate nodes may be semantically similar but graph-context mismatched.

graph-context SimHash:
    candidate nodes are encouraged to be similar in both self semantics and local context.
```

## Paper Wording

English draft:

```text
Instead of hashing only the node text feature, we construct a graph-context-aware retrieval key by mixing the node's own cheap semantic feature with the mean feature of its one-hop neighbors. This makes CAM lookup sensitive to both semantic similarity and local graph context, better matching the downstream GNN propagation behavior.
```

Chinese explanation:

```text
我们不直接对节点自身文本特征做 SimHash，而是对 self feature 和一跳邻居均值的混合表示做哈希。这样 CAM 查找的候选节点不仅文本语义接近，而且局部图上下文也接近，从而更符合 GNN 后端传播后的误差敏感性。
```

## Takeaway

Graph-context SimHash key 不是普通 semantic hashing。它把图结构以极低成本注入 CAM retrieval key，使前端 bypass candidate search 从：

```text
find text-similar anchors
```

变成：

```text
find text- and context-similar anchors
```

这是 TSER/residual gate 之前的第一层 graph-aware reuse design。
