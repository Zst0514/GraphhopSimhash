# TSER 分数定义

本文档说明当前 `GraphhopSimhash` 中的两类分数：

1. **TSER reuse score**：用于 hash reuse gate，判断一个节点能不能安全复用候选节点的 embedding。
2. **TSER quant score**：用于 W4A8/W4A4 固定预算路由，判断哪些节点应该走 W4A8，哪些节点可以走 W4A4。

这两者共享图结构和 cheap feature，但风险含义不同：

```text
reuse score:
    风险来自“拿别人的 embedding 替代自己”。

quant score:
    风险来自“自己的 embedding 被低精度计算扰动”。
```

因此 reuse 分数不能直接等价为量化误差分数。

## 1. 公共输入

### `verify_features`

用于打分和验证的 cheap feature。当前实验通常使用：

```text
DistilBERT layer-1 feature
+ centering
+ L2 normalization
```

它不是最终 ST/LLaMA embedding，而是低成本代理特征。

### `hash_features`

用于 SimHash 的图上下文特征，常见形式是：

```text
hash_features(v) =
    0.3 * self_feature(v)
  + 0.7 * neighbor_mean(v)
```

这样 hash 不只看节点文本，也看局部图上下文。

### 4-bit 分数

大多数风险项量化到 `[0, 15]`：

```text
q(x) = round(clamp(x, 0, 1) * 15)
```

好处是硬件友好：小寄存器、LUT、比较器就能实现。

## 2. 三个核心风险项

### 2.1 `propagation_q`

高 degree 节点会影响更多邻居，错误复用或低精度扰动更容易传播。

```text
propagation_risk(v) =
    log(1 + degree(v)) / log(1 + max_degree)

propagation_q(v) =
    q(propagation_risk(v))
```

使用 log 是为了避免少数 hub 节点把范围拉爆。

### 2.2 `graph_context_q`

只看 degree 不够。低度节点如果处在语义边界或结构边界，也可能很敏感。

当前定义：

```text
graph_context_risk(v) =
    max(boundary_risk(v), context_shift(v))

graph_context_q(v) =
    q(graph_context_risk(v))
```

`boundary_risk` 来自邻居 hash 差异：

```text
boundary_risk(v) =
    average Hamming(hash_context(v), hash_context(u)) / sketch_bits
    for u in N(v)
```

它是逐边比较：看 `v` 和每个邻居是否相似，用来捕捉局部边界混杂。`sketch_bits` 是 hash bit 数，除以它是为了把 Hamming 距离归一化到约 `[0, 1]`。

`context_shift` 来自 self feature 和 context signature 的 cosine 偏移：

```text
context_signature(v) =
    normalize(0.5 * self(v) + 0.5 * neighbor_mean(v))

context_shift(v) =
    0.5 * clamp(1 - cosine(self(v), context_signature(v)), 0, 2)
```

它是整体上下文比较：先把邻居聚合成 `neighbor_mean`，再看节点自己和局部上下文中心是否偏离。加入 `0.5 * self` 是为了让低度节点的上下文估计更平滑。

二者相关但不重复：`boundary_risk` 抓逐边异质性，`context_shift` 抓整体上下文偏移。取 `max` 表示任一风险高，都应提高复用保护。

简单例子：

```text
boundary_risk 高：
    v 的一半邻居像自己，另一半邻居属于别的语义簇。
    逐边 Hamming 平均会变大，说明 v 位于混杂边界。

context_shift 高：
    每个邻居和 v 的单边差异不一定很大，
    但邻居整体平均方向一致地偏离 v。
    neighbor_mean 会把这种整体偏移累积出来。
```

### 2.3 `low_degree_unique_q`

低 degree 节点传播影响小，但如果它语义很稀有，错误复用会直接破坏自身预测。

先用 SimHash bucket 估计稀有度：

```text
similar_count(v) = bucket_count(hash_self(v)) - 1
```

映射：

```text
similar_count >= 8  -> rarity_q = 0
similar_count >= 4  -> rarity_q = 4
similar_count >= 2  -> rarity_q = 8
similar_count >= 1  -> rarity_q = 12
similar_count == 0  -> rarity_q = 15
```

再和低 degree 因子相乘：

```text
low_degree_factor_q(v) =
    15 - propagation_q(v)

low_degree_unique_q(v) =
    round(low_degree_factor_q(v) * rarity_q(v) / 15)
```

这个项只有在“低度 + 稀有”同时满足时才高。

## 3. 综合分数 `sensitivity_q`

```text
sensitivity_q(v) =
    w_prop * propagation_q(v)
  + w_ctx  * graph_context_q(v)
  + w_low  * low_degree_unique_q(v)
```

命令行参数：

```text
--score_propagation_weight
--score_graph_context_weight
--score_low_unique_weight
```

代码默认值目前是：

```text
3 / 2 / 2
```

近期主要探索的轻量版本：

```text
3/0/0  degree-only
3/1/0  propagation + context
3/0/1  propagation + low-degree uniqueness
3/1/1  light TSER
2/1/1  降低 propagation 权重
1/1/1  三项等权
0/1/1  去掉 degree，只看 context + low-degree uniqueness
3/2/2  更强 graph/context 修正
```

当前观察：

```text
Cora:
    score gate 明显能把 NoScore 的高 reuse / 高掉点拉回来。
    3/1/1, T=30 往往更保守，drop 更低，但 reuse 也降低。

PubMed:
    T=45 偏松，reuse 很高但 drop 仍大。
    更值得扫 T=15/20/25/30/35。
```

## 4. Reuse Gate 如何决策

候选节点的基础复用误差来自 Hamming 距离：

```text
dist <= 0 -> reuse_error_q = 1
dist == 1 -> reuse_error_q = 2
dist == 2 -> reuse_error_q = 4
dist >  2 -> reuse_error_q = max(4, 2 * dist)
```

多 head 支持会降低风险：

```text
route_hit_count >= 4 -> reuse_error_q -= 2
route_hit_count >= 2 -> reuse_error_q -= 1
```

最终风险：

```text
reuse_risk(v, candidate) =
    sensitivity_q(v) * reuse_error_q(candidate)
```

候选会被拒绝的三种情况：

```text
1. propagation_q(v) >= T_hub and hamming_dist > 0
2. low_degree_unique_q(v) >= T_rare and hamming_dist > 0
3. reuse_risk(v, candidate) > T_reuse
```

当前默认阈值：

```text
T_reuse = 45
T_hub   = 12
T_rare  = 10
```

其中 `T_reuse` 是主要旋钮：

```text
T 变小 -> gate 更严格 -> reuse 降低 -> drop 通常降低
T 变大 -> gate 更宽松 -> reuse 升高 -> drop 通常升高
```

## 5. 量化路由中的 TSER

在 `real_quant_ablation + w4a8_budget` 中：

```text
W4A8 = 安全路径
W4A4 = 激进低成本路径
```

例如：

```text
--real_quant_int8_ratio 0.20
```

表示：

```text
20% 节点走 W4A8
80% 节点走 W4A4
```

主表策略：

```text
AllFP
UniformW4A8
UniformW4A4
RandomTopK_W4A8
DegreeTopK_W4A8
TSERTopK_W4A8
```

### `DegreeTopK_W4A8`

按 `propagation_q` 排序：

```text
高 propagation_q -> W4A8
低 propagation_q -> W4A4
```

这是强 baseline，尤其在 PubMed 这种传播主导数据集上经常很强。

### `TSERTopK_W4A8`

按 `sensitivity_q` 排序：

```text
高 sensitivity_q -> W4A8
低 sensitivity_q -> W4A4
```

相比 degree-only，它额外考虑：

```text
graph context boundary
low-degree uniqueness
```

量化路由这里经过实验发现：

```text
DegreeTopK_W4A8 优于 TSERTopK_W4A8。
```

原因是量化损伤主要由两个因素决定：

```text
1. 节点自身的量化误差大小
2. 这个误差沿图传播的范围
```

第 1 项需要真实 FP/quant embedding 对比，在线不可得；第 2 项最直接的代理就是
`propagation_q` / degree。因此在不使用 oracle error 的前提下，DegreeTopK_W4A8
比加入 context / low-unique 的 TSERTopK_W4A8 更符合当前实验结果。

TSER 的优势边界主要在 hash reuse gate：reuse 是“用别的节点 embedding 替代自己”，
语义边界和稀有低度节点会显著影响错复用风险；而量化是“自己的 embedding 被扰动”，
更多受量化误差大小和传播范围控制。

## 6. Oracle / Error-aware 行

代码里还保留一些 error-aware helper：

```text
DegreeErrorTopK
TSERErrorTopK
QuantTSERTopK
```

它们会用真实 embedding error：

```text
err4(v) = 1 - cosine(FP(v), W4A4(v))
err8(v) = 1 - cosine(FP(v), W4A8(v))
```

这类策略不是严格 deployable，因为如果全图每个节点都知道 `err4/err8`，说明已经生成并比较过 FP 与 quant embedding。

因此它们只适合作为：

```text
oracle upper bound
debug / profiling baseline
说明真实量化误差信息确实有价值
```

因此不能把这类 error-aware 策略当成可部署系统策略；它们只能用于离线分析和上界参照。

## 7. 常用命令

Cora score ablation：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite score_ablation \
  --radius 2 \
  --hash_heads_per_route 4 \
  --main_hash_head_bits 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor
```

Cora 参数扫：

```bash
RUNS=5 bash GraphhopSimhash/run_cora_tser_reuse_sweep.sh
```

PubMed 参数扫：

```bash
RUNS=5 bash GraphhopSimhash/run_pubmed_tser_reuse_sweep.sh
```
