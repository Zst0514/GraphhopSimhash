# TSER 与 TSER-Q 分数定义说明

本文档解释当前 `GraphhopSimhash` 代码中两套核心分数：

1. **TSER reuse score**：用于 GraphHop SimHash 的复用判断。它回答的问题是：一个节点能否安全复用另一个节点的 embedding？
2. **TSER-Q quantization score**：用于 W4A8/W4A4 混合精度路径选择。它回答的问题是：在固定预算下，哪些节点应该保留在安全的 W4A8 路径，哪些节点可以走更激进的 W4A4 路径？

这两套分数共享一部分图结构和语义上下文信号，但风险含义不同：

```text
TSER reuse score:
    估计 hash 复用带来的替代风险。

TSER-Q quantization score:
    估计低精度执行带来的任务风险。
```

所以不能简单地把 reuse 分数直接等同于量化分数。reuse 的风险来自“拿别人的 embedding 替代自己”，量化的风险来自“自己的 embedding 被 W4A4/W4A8 计算路径扰动”。

## 0. 当前主线

当前 `fixed_aggressive_budget` 主表使用的是**不依赖逐节点真实量化误差**的版本。

主表策略是：

```text
AllW4A8
AllW4A4
RandomBudget
DegreeBudget
TSERBudget
GraphHopSafeBudget
```

其中 `GraphHopSafeBudget` 是当前更适合部署叙事的量化路由策略。它不使用逐节点真实误差：

```text
int4_err_q / int8_err_q
```

而是使用 graph/hash stability score：

```text
W4A4-safe score =
    hash bucket density
  + hash agreement proxy
  + self/context consistency
  + low-propagation safety
  + non-rare-tail safety
```

等价地说，它倾向于把 W4A4 分给这些节点：

```text
1. hash bucket 更密集，周围有相似节点；
2. 多个 hash/head 的判断更一致；
3. 自身语义和邻居上下文更一致；
4. degree / propagation risk 更低；
5. 不是低度且语义稀有的 tail 节点。
```

这样做的优点是：路由只依赖 cheap feature、图结构和 hash 统计，不需要给全图每个节点都先跑 FP/W4A8/W4A4 三套前端再比较误差。

`ErrorBudget / TSERQBudget / Calib*Budget` 更适合作为 oracle 或历史消融，本文档只在最后保留简短说明。

## 1. 公共输入

### 1.1 `verify_features`

`verify_features` 是用于验证和打分的 cheap feature。当前实验里通常来自：

```text
DistilBERT layer-1 feature
中心化 + 归一化
```

它不是最终 LLaMA/ST 的高质量 embedding，而是一个便宜代理，用来构造风险分数、SimHash、上下文差异等。

### 1.2 `hash_features`

`hash_features` 是用于 SimHash 的特征视图，一般由 self feature 和邻居 feature 混合得到。例如：

```text
hash_features(v) = 0.3 * self(v) + 0.7 * neighbor_mean(v)
```

具体权重由命令行中的 hash view / mix weight 配置控制。

### 1.3 `edge_index`

`edge_index` 是图连接关系，用于计算：

- degree；
- 邻居均值；
- 图上下文变化；
- 传播影响；
- 1-hop / 2-hop 风险。

### 1.4 `context_signature`

`context_signature` 表示节点的局部语义上下文：

```text
neighbor_mean(v) =
    mean({verify_features(u) | u in N(v)})

context_signature(v) =
    normalize(0.5 * verify_features(v) + 0.5 * neighbor_mean(v))
```

它用于判断一个节点自己的语义是否和邻居上下文一致。

### 1.5 4-bit 分数寄存器

大多数风险子项都会被量化到 `[0, 15]` 的整数区间：

```text
q(x) = round(clamp(x, 0, 1) * 15)
```

这样做的原因是：

- 分数可以用 4-bit 寄存器保存；
- 硬件上可以用小 LUT / comparator 实现；
- 在线执行时不需要保留高精度浮点风险值。

## 2. TSER Reuse Score

TSER reuse score 主要用于 SimHash/CAM 检索之后的安全判断。

流程是：

```text
1. CAM / SimHash 找到候选复用节点 u
2. 计算目标节点 v 和候选 u 的 Hamming 距离
3. 计算目标节点 v 的图语义风险 sensitivity_q(v)
4. 判断 v 是否允许复用 u 的 embedding
```

### 2.1 `propagation_q`

高 degree 节点会影响更多下游消息传播，因此它的 embedding 如果被错误复用，误差更容易扩散。

```text
propagation_risk(v) =
    log(1 + degree(v)) / log(1 + max_degree)

propagation_q(v) =
    q(propagation_risk(v))
```

其中：

```text
degree(v):
    节点 v 在当前图中的无向度数。

max_degree:
    当前图中所有节点 degree 的最大值。
```

这里用 `log(1 + degree)`，是为了避免少数超高度节点把分数范围完全拉爆。

### 2.2 `graph_context_q`

仅看 degree 不够。一个节点即使 degree 不高，如果处在语义边界或结构边界附近，错误复用也可能带来明显影响。

当前定义是：

```text
graph_context_risk(v) =
    max(boundary_risk(v), context_shift(v))

graph_context_q(v) =
    q(graph_context_risk(v))
```

`boundary_risk` 用 hash 空间里的邻居差异来衡量：

```text
boundary_risk(v) =
    average_{u in N(v)}
        Hamming(hash_context(v), hash_context(u)) / sketch_bits
```

`context_shift` 用连续向量 cosine 衡量 self feature 和邻居上下文之间的偏移：

```text
context_shift(v) =
    0.5 * clamp(
        1 - cosine(verify_features(v), context_signature(v)),
        0,
        2
    )
```

如果后续要做更硬件友好的版本，可以把 `context_shift` 替换成 hash 版本：

```text
context_shift_hash(v) =
    Hamming(hash_self(v), hash_context(v)) / sketch_bits
```

这样整个 graph context 计算都可以依赖 bit sketch，而不需要在线 cosine。

### 2.3 `rarity_q`

`rarity_q` 衡量一个节点在全局 hash 空间中是否常见。

代码会对 `verify_features` 生成额外的 SimHash bucket，然后统计每个 bucket 中有多少相似节点：

```text
similar_count(v) =
    bucket_count(hash_self(v)) - 1
```

映射规则是：

```text
similar_count >= 8  -> rarity_q = 0
similar_count >= 4  -> rarity_q = 4
similar_count >= 2  -> rarity_q = 8
similar_count >= 1  -> rarity_q = 12
similar_count == 0  -> rarity_q = 15
```

含义是：

- bucket 里有很多相似节点：节点不稀有，复用更安全；
- bucket 里几乎没有相似节点：节点很稀有，复用风险高。

### 2.4 `low_degree_unique_q`

如果只用 degree，高度节点会被保护，但低度节点容易被认为不重要。可是有些低度节点虽然传播范围小，语义却很独特，错误复用会直接破坏该节点自身预测。

因此代码加入：

```text
low_degree_factor_q(v) =
    15 - propagation_q(v)

low_degree_unique_q(v) =
    round(low_degree_factor_q(v) * rarity_q(v) / 15)
```

这个分数只有在两个条件同时满足时才高：

```text
1. 节点 degree 低；
2. 节点语义/hash 稀有。
```

### 2.5 综合风险 `sensitivity_q`

TSER reuse 的综合风险分数是：

```text
sensitivity_q(v) =
    w_prop * propagation_q(v)
  + w_ctx  * graph_context_q(v)
  + w_low  * low_degree_unique_q(v)
```

当前默认权重是：

```text
w_prop = 3
w_ctx  = 1
w_low  = 1
```

也就是：

```text
sensitivity_q =
    3 * propagation_q
  + 1 * graph_context_q
  + 1 * low_degree_unique_q
```

这个默认值来自 Cora / LLaMA-7B 上的固定预算调参。它比旧的 `3/2/2` 更简洁，并在 20% W4A4 + 80% W4A8 的设置下略优。因此当前推荐把 TSER 理解为：

```text
degree-dominant risk score
+ lightweight graph-context correction
+ lightweight low-degree uniqueness correction
```

对应命令行参数：

```text
--score_propagation_weight
--score_graph_context_weight
--score_low_unique_weight
```

### 2.6 Reuse Gate

候选复用还会结合 Hamming 距离和多 head 支持度。

基础候选误差：

```text
if hamming_dist <= 0:
    reuse_error_q = 1
elif hamming_dist == 1:
    reuse_error_q = 2
elif hamming_dist == 2:
    reuse_error_q = 4
else:
    reuse_error_q = max(4, 2 * hamming_dist)
```

如果多个 hash head 都支持同一个候选，会降低误差：

```text
route_hit_count >= 4:
    reuse_error_q -= 2

route_hit_count >= 2:
    reuse_error_q -= 1
```

最终复用风险是：

```text
reuse_risk(v, candidate) =
    sensitivity_q(v) * reuse_error_q(candidate)
```

候选会被三类规则拒绝：

```text
1. propagation_q(v) >= T_hub and hamming_dist > 0
2. low_degree_unique_q(v) >= T_rare and hamming_dist > 0
3. reuse_risk(v, candidate) > T_reuse
```

默认阈值：

```text
T_hub   = 12
T_rare  = 10
T_reuse = 120
```

要测试 TSER-gated reuse，需要显式打开：

```text
--enable_score_gate
```

## 3. TSER-Q Quantization Score

TSER-Q 服务于真实 embedding pool 的 W4A8/W4A4 路由。

问题定义是：

```text
给定 FP / W4A8 / W4A4 三套 embedding pool，
以及固定 W4A8 预算，
选择哪些节点走 W4A8，哪些节点走 W4A4。
```

在 `fixed_aggressive_budget` 策略下：

```text
safe path       = W4A8
aggressive path = W4A4
FP path         = 不参与 budget 对比
```

例如：

```text
--real_quant_int8_ratio 0.80
```

表示：

```text
80% 节点使用 W4A8；
20% 节点使用 W4A4。
```

### 3.1 `DegreeBudget`

`DegreeBudget` 按 degree / propagation risk 保护高传播节点。

直觉是：

```text
高度节点或传播影响大的节点走 W4A8；
低传播节点走 W4A4。
```

这个策略非常简单，而且在 PubMed 这类传播主导的数据集上通常很强。

### 3.2 `TSERBudget`

`TSERBudget` 使用上面的 `sensitivity_q` 排序：

```text
sensitivity_q =
    3 * propagation_q
  + 1 * graph_context_q
  + 1 * low_degree_unique_q
```

路由规则是：

```text
sensitivity_q 高的节点 -> W4A8
sensitivity_q 低的节点 -> W4A4
```

它相比 `DegreeBudget` 多考虑了：

- 图上下文边界；
- 低度但稀有的节点；
- hash/语义上下文稳定性。

但当前实验也说明，TSER 不应被描述成“总是优于 Degree”。在 PubMed 和 ST 后端上，degree-only 往往更强；在 Cora / LLaMA-7B 的 20% W4A4 + 80% W4A8 设置下，`3/1/1` 的 TSER 有轻微优势。

### 3.3 `GraphHopSafeBudget`

`GraphHopSafeBudget` 是当前推荐的 deployable 量化路由策略。它回答的问题不是：

```text
这个节点真实 W4A4 error 是多少？
```

而是：

```text
从图结构和 hash 稳定性看，这个节点是否适合走 aggressive W4A4？
```

它使用 W4A4-safe score：

```text
safe_q(v) =
    density_q(v)
  + agreement_q(v)
  + consistency_q(v)
  + low_prop_q(v)
  + non_unique_q(v)
```

各项含义：

```text
density_q:
    hash bucket 越密集，说明附近相似节点越多，W4A4 更安全。

agreement_q:
    多个 hash/head 越一致，说明节点位置越稳定，W4A4 更安全。

consistency_q:
    self feature 和 context signature 越一致，说明节点不在语义边界，W4A4 更安全。

low_prop_q:
    propagation risk 越低，扰动扩散越少，W4A4 更安全。

non_unique_q:
    节点越不稀有，越不需要保护，W4A4 更安全。
```

路由规则是：

```text
safe_q 高的节点 -> W4A4
safe_q 低的节点 -> W4A8
```

这个策略的核心优点是：

```text
不用逐节点真实量化误差；
不用额外训练 predicted error model；
只依赖 cheap graph/hash statistics；
更容易解释成 NDP/NPU/CAM 路由逻辑。
```

## 4. 当前调参结论

### 4.1 Cora / LLaMA-7B

在 20% W4A4 + 80% W4A8 的固定预算下，当前观察到：

```text
3/1/1 略优于 3/2/2，也略优于 Degree/Random。
```

因此当前默认值设置为：

```text
--score_propagation_weight 3
--score_graph_context_weight 1
--score_low_unique_weight 1
```

这说明在 Cora / LLaMA-7B 上，传播风险仍然是主项，但轻量 graph context 和 low-degree uniqueness 可以提供一点修正。

### 4.2 Cora / ST

在 ST 后端下，W4A4 本身损伤更重，且 degree-only 往往更稳定。已有调参结果显示：

```text
Degree-only 3/0/0:
    通常优于 3/1/1。

Context-only add 3/1/0:
    接近 Degree-only，但不一定更好。

Unique-only add 3/0/1:
    经常伤精度。
```

因此 ST 上不能强行宣称 TSER 一定优于 Degree。更合理的结论是：

```text
TSER correction 是否有收益，与 embedding backend 和数据集结构有关。
```

### 4.3 PubMed

PubMed 上 DegreeBudget 经常优于 TSERBudget 和 GraphHopSafeBudget。原因是 PubMed 的任务更传播主导，高传播节点保护比语义边界修正更关键。

这给论文叙事提供了一个重要提醒：

```text
Graph-aware quant routing 需要 dataset/backend-aware；
TSER 是可调的图语义修正项，不是无条件压过 degree 的万能分数。
```

## 5. 实验命令

Cora / LLaMA-7B：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite fixed_aggressive_budget \
  --real_quant_model_name llama2_7b \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_LLAMA7B_PTQ_TEST \
  --real_quant_int4_tag W4A4_LLAMA7B_W4A4O_R2 \
  --real_quant_error_norm 1.0 \
  --real_quant_int8_ratio 0.80 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

Cora / ST：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite fixed_aggressive_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_PTQ_TEST \
  --real_quant_int4_tag W4A4_PTQ_TEST2 \
  --real_quant_error_norm 1.0 \
  --real_quant_int8_ratio 0.80 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

PubMed / LLaMA-7B：

```bash
python -m GraphhopSimhash \
  --datasets pubmed \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite fixed_aggressive_budget \
  --real_quant_model_name llama2_7b \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_LLAMA7B_PTQ_TEST \
  --real_quant_int4_tag W4A4_LLAMA7B_W4A4O_R2 \
  --real_quant_error_norm 1.0 \
  --real_quant_int8_ratio 0.80 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

## 6. 历史 / Oracle 消融说明

早期版本里还有一些依赖逐节点真实量化误差的策略：

```text
ErrorBudget
DegreeErrorBudget
TSERQBudget
CalibErrorBudget
CalibDegreeErrorBudget
CalibTSERQBudget
```

它们会使用：

```text
err8(v) = 1 - cosine(FP(v), W4A8(v))
err4(v) = 1 - cosine(FP(v), W4A4(v))
```

这类分数的问题是：如果要得到全图精确 `err4/err8`，必须先为每个节点生成 FP、W4A8、W4A4 三套 embedding，再逐节点比对。这对最终部署来说开销过高。

因此当前主线不再把它们作为核心策略，只把它们看成：

```text
1. oracle upper bound；
2. profiling baseline；
3. 历史消融，用来说明真实量化误差确实有信息量。
```

论文主系统应该优先讲：

```text
GraphHop SimHash reuse
+ TSER reuse gate
+ fixed outlier-preserved low-precision backend
+ deployable graph/hash stability routing
+ NDP/NPU/CAM pipeline
```
