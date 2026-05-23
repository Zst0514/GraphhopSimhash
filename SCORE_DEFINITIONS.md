# TSER 与 TSER-Q 分数定义说明

本文档解释当前 `GraphhopSimhash` 代码中两套核心分数的定义和作用：

1. **TSER reuse score**：用于 GraphHop SimHash 的复用判断。它回答的问题是：一个节点能否安全复用另一个节点的 embedding？
2. **TSER-Q quantization score**：用于 W4A8/W4A4 混合精度量化路径选择。它回答的问题是：在固定预算下，哪些节点应该走安全的 W4A8 路径，哪些节点可以走更激进的 W4A4 路径？

这两套分数共享一部分图结构和语义上下文信号，但它们的风险含义不同：

```text
TSER reuse score:
    估计 hash 复用带来的替代风险。

TSER-Q quantization score:
    估计低精度量化执行带来的任务风险。
```

所以不能简单地把 reuse 的分数直接拿来指导量化。reuse 的误差来自“拿别人的 embedding 替代自己”，量化的误差来自“自己的 embedding 被 W4A4/W4A8 计算路径扰动”。

## 0. 当前主线更新

当前 `fixed_aggressive_budget` 主表已经回到**不依赖逐节点真实量化误差**的版本。

主表策略是：

```text
AllW4A8
AllW4A4
RandomBudget
DegreeBudget
TSERBudget
GraphHopSafeBudget
```

其中 `GraphHopSafeBudget` 是当前推荐的 deployable 量化路由策略。它不使用：

```text
int4_err_q / int8_err_q
```

来决定节点路径，而是使用 graph/hash stability score：

```text
W4A4-safe score =
    hash bucket density
  + hash agreement proxy
  + self/context consistency
  + low-propagation safety
  + non-rare-tail safety
```

等价于惩罚：

```text
propagation risk
graph boundary/context risk
rare low-degree penalty
```

`ErrorBudget / TSERQBudget / Calib*Budget` 更适合作为 oracle 或历史消融，不再作为当前 fixed-budget 主系统表的核心策略。

## 1. 公共输入

两套分数都会使用一些预计算的图结构和语义特征。

### 1.1 `verify_features`

`verify_features` 是用于验证和打分的 cheap feature。

当前实验中，它通常来自：

```text
DistilBERT layer-1 feature
中心化 + 归一化
```

它不是最终 LLaMA/ST 的高质量 embedding，而是一个便宜的代理特征，用来构造风险分数、SimHash、上下文差异等。

### 1.2 `hash_features`

`hash_features` 是用于 SimHash 的特征视图，一般由 self feature 和邻居 feature 混合得到。例如：

```text
hash_features(v) = 0.3 * self(v) + 0.7 * neighbor_mean(v)
```

具体权重由命令行中的 hash view / mix weight 配置控制。

### 1.3 `edge_index`

`edge_index` 是图连接关系，用于计算：

- 度数；
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

TSER reuse score 的实现主要在：

```text
GraphhopSimhash/scoring.py
```

它用于 SimHash/CAM 检索之后的安全判断。

完整流程是：

```text
1. CAM / SimHash 找到候选复用节点 u
2. 计算候选和目标节点 v 的 Hamming 距离
3. 结合 v 的图语义风险 sensitivity_q(v)
4. 判断是否允许 v 复用 u 的 embedding
```

### 2.1 传播风险 `propagation_q`

高 degree 节点通常会影响更多下游消息传播，因此它的 embedding 被错误复用时，误差更容易扩散。

代码中的定义是：

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

所以 `propagation_q` 是一个图内归一化的 degree 风险。

直观理解：

```text
propagation_q = 0:
    几乎没有传播影响。

propagation_q = 15:
    当前图中传播影响最大的节点。
```

这里用了 `log(1 + degree)`，而不是直接用 degree，是为了避免极少数超高度节点把分数范围完全拉爆。

### 2.2 图上下文风险 `graph_context_q`

仅仅看 degree 不够。一个节点即使 degree 不高，如果它处在语义边界附近，错误复用也可能带来明显影响。

当前代码定义：

```text
graph_context_risk(v) =
    max(boundary_risk(v), context_shift(v))

graph_context_q(v) =
    q(graph_context_risk(v))
```

也就是说，它取两个上下文风险的较大值。

#### 2.2.1 `boundary_risk`

`boundary_risk` 用 hash 空间里的邻居差异来衡量。

先对每个节点的 `hash_features` 生成 context hash，然后计算节点和邻居之间的平均 Hamming 距离：

```text
boundary_risk(v) =
    average_{u in N(v)}
        Hamming(hash_context(v), hash_context(u)) / sketch_bits
```

如果一个节点和邻居的 context hash 差异很大，说明它可能位于语义边界或结构边界上，复用风险更高。

#### 2.2.2 `context_shift`

`context_shift` 用连续向量 cosine 衡量节点自身语义和邻居上下文之间的偏移：

```text
context_shift(v) =
    0.5 * clamp(
        1 - cosine(verify_features(v), context_signature(v)),
        0,
        2
    )
```

如果节点自己的 feature 和邻居上下文很接近，`context_shift` 低。

如果节点自己的 feature 和邻居上下文差异大，`context_shift` 高。

#### 2.2.3 是否必须同时存在？

不是必须。

当前代码取：

```text
max(boundary_risk, context_shift)
```

这是为了让两种风险互相补充：

- `boundary_risk` 更适合硬件，因为它基于 Hamming 距离；
- `context_shift` 更细，但需要连续向量 cosine，开销更高。

如果后续要做更硬件友好的版本，可以把 `context_shift` 替换成 hash 版本：

```text
context_shift_hash(v) =
    Hamming(hash_self(v), hash_context(v)) / sketch_bits
```

这样整个 graph context 计算都可以依赖 bit sketch，而不需要在线 cosine。

### 2.3 稀有性分数 `rarity_q`

`rarity_q` 衡量一个节点在全局 hash 空间中是否常见。

代码会对 `verify_features` 生成一个额外的 SimHash bucket，然后统计每个 bucket 中有多少相似节点：

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

- bucket 里有很多相似节点：这个节点不稀有，复用更安全；
- bucket 里几乎没有相似节点：这个节点很稀有，复用风险高。

### 2.4 低度独特节点保护 `low_degree_unique_q`

如果只用 degree，高度节点会被保护，但低度节点很容易被认为不重要。

这个逻辑对很多图任务是不够的，因为有些低度节点虽然传播范围小，但语义非常独特，错误复用会直接破坏该节点自身预测。

所以代码加入：

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

它的作用是保护“低度但独特”的节点，避免 degree-only 策略把它们粗暴丢给复用路径。

### 2.5 综合敏感度 `sensitivity_q`

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

这个默认值来自 Cora / LLaMA-7B 上的固定预算消融。它比 `3/2/2`
更简洁，并在 20% W4A4 + 80% W4A8 的设置下略优。因此当前推荐把
TSER 理解为：

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

直观上：

- `propagation_q`：保护高传播节点；
- `graph_context_q`：保护语义/结构边界节点；
- `low_degree_unique_q`：保护低度但稀有节点。

### 2.6 复用误差 `reuse_error_q`

TSER reuse 不只看节点本身风险，还要看候选复用是否可靠。

候选可靠性由 Hamming 距离和多 head 支持度决定。

基础 Hamming 误差：

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

最低不会低于 1。

这相当于：

```text
Hamming 越近，候选越可靠；
多个 head 同时命中，候选更可靠。
```

### 2.7 最终复用风险 `reuse_risk`

最终复用风险定义为：

```text
reuse_risk(v, candidate) =
    sensitivity_q(v) * reuse_error_q(candidate)
```

这很关键：

```text
节点本身越敏感，越不能接受 hash 近似误差；
候选 Hamming 距离越大，越容易被拒绝。
```

### 2.8 Reuse Gate 判断规则

候选会被以下三类规则拒绝。

#### 规则 1：高传播节点 fuzzy reuse 保护

```text
propagation_q(v) >= T_hub
and hamming_dist > 0
```

默认：

```text
T_hub = 12
```

含义：高度/高传播节点默认不允许 fuzzy reuse，除非显式打开 `--allow_hub_fuzzy`。

#### 规则 2：低度稀有节点 fuzzy reuse 保护

```text
low_degree_unique_q(v) >= T_rare
and hamming_dist > 0
```

默认：

```text
T_rare = 10
```

含义：低度但独特的节点不允许 fuzzy reuse，避免 tail 节点被错误近似。

#### 规则 3：综合风险阈值

```text
reuse_risk(v, candidate) > T_reuse
```

默认：

```text
T_reuse = 120
```

### 2.9 相关命令行参数

```text
--enable_score_gate
--disable_score_gate
--score_reuse_threshold
--score_hub_threshold
--score_rare_threshold
--allow_hub_fuzzy
--allow_rare_fuzzy
--disable_score_support_discount
--score_rarity_bits
--score_rarity_seed
--score_propagation_weight
--score_graph_context_weight
--score_low_unique_weight
```

注意：当前代码中 score gate 默认关闭。要测试 TSER-gated reuse，需要显式加：

```text
--enable_score_gate
```

## 3. TSER-Q Quantization Score

TSER-Q 的实现主要在：

```text
GraphhopSimhash/real_quant.py
```

它服务于真实 embedding pool 的 W4A8/W4A4 路由。

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
--real_quant_int8_ratio 0.70
```

表示：

```text
70% 节点使用 W4A8；
30% 节点使用 W4A4。
```

### 3.1 量化误差 `int8_err_q` / `int4_err_q`

TSER-Q 首先需要知道 W4A8 和 W4A4 对每个节点 embedding 的真实损伤。

代码用 cosine error：

```text
err8(v) =
    1 - cosine(FP(v), W4A8(v))

err4(v) =
    1 - cosine(FP(v), W4A4(v))
```

然后量化成 4-bit error：

```text
int8_err_q(v) =
    q(err8(v) / real_quant_error_norm)

int4_err_q(v) =
    q(err4(v) / real_quant_error_norm)
```

相关参数：

```text
--real_quant_error_norm
--real_quant_error_space
```

如果 `real_quant_error_space=encoded`，误差是在 GNN encoder 后的空间计算。

如果 `real_quant_error_space=raw`，误差是在原始 embedding 空间计算。

当前很多实验使用：

```text
--real_quant_error_norm 1.0
```

这样 AvgErr 和 error q 的解释更直观。

#### 3.1.1 这个误差是不是必须先完整跑 LLM 前端？

如果要得到上面这个**精确的 per-node quantization error**，答案是：是的。

也就是说，当前代码中的：

```text
err8(v) = 1 - cosine(FP(v), W4A8(v))
err4(v) = 1 - cosine(FP(v), W4A4(v))
```

前提是已经为同一批节点生成了：

```text
FP embedding pool
W4A8 embedding pool
W4A4 embedding pool
```

因此，`int4_err_q / int8_err_q` 在当前实验里属于：

```text
offline profiling signal
```

而不是一个无需代价即可在线获得的硬件运行时信号。

这一点非常重要。它意味着：

```text
ErrorBudget:
    基本是 quantization-damage oracle / profiling baseline。

TSERQBudget:
    当前实现也使用了 profiled quantization damage，
    因此它是 graph/task-aware profiling routing，
    不是完全零 profiling 的 online-only routing。
```

所以在论文里不能把 `int4_err_q / int8_err_q` 描述成完全免费的在线分数。更严谨的说法应该是：

```text
We use a small offline calibration/profiling stage to estimate per-node
or per-group quantization vulnerability, and then combine it with graph-task
risk for routing.
```

#### 3.1.2 部署时是否需要所有节点都跑三遍？

不应该。

如果部署时对每个节点都完整生成 FP/W4A8/W4A4 三套 embedding，再决定用哪个精度，那么节省计算的意义就被抵消了。

更合理的系统设计应该区分两种模式。

第一种是实验分析 / 上界评估模式：

```text
为所有节点生成 FP / W4A8 / W4A4 pools；
计算真实 per-node quantization error；
用于分析 TSER-Q 是否选对节点，以及和 Degree/Error baseline 对比。
```

这就是当前代码主要在做的事情。

第二种是实际部署模式：

```text
只在小规模 calibration set 上生成多精度 embedding；
学习或统计一个 quantization vulnerability proxy；
对全图节点只计算便宜的 score；
最后只执行被路由到的那一条路径。
```

也就是说，部署时理想流程应是：

```text
1. calibration/profiling stage:
   少量节点跑 FP/W4A8/W4A4，估计 W4A4 损伤模式。

2. scoring stage:
   对所有节点计算便宜的 graph/hash/task proxy。

3. routing stage:
   每个节点只走一条路径：
       W4A4 或 W4A8。
```

#### 3.1.3 后续更合理的误差 proxy

为了避免全图三套 LLM 前端，`int4_err_q / int8_err_q` 可以从精确误差改成预测误差。

可选 proxy 包括：

```text
1. calibration set 上拟合的误差预测器
   输入：degree、graph_context_q、rarity_q、hash bucket、activation stats
   输出：predicted_int4_err_q / predicted_int8_err_q

2. group-level error
   按 hash bucket / degree bin / TSER bin 分组，
   每组只 profiling 少量节点，
   同组节点共享平均 quantization error。

3. activation-side proxy
   使用每层 activation clipping rate、outlier channel ratio、
   saturation count 等统计预测 W4A4 是否危险。

4. hybrid sampling
   对高风险区域多采样，对低风险区域少采样，
   用少量 profiled nodes 估计全图 routing。
```

这样 `TSERQBudget` 可以从当前的：

```text
profiled TSER-Q
```

进一步演化为：

```text
calibrated/predicted TSER-Q
```

后者更适合硬件论文中的真实部署叙事。

### 3.2 图传播影响 `graph_impact_q`

量化误差不是所有节点都一样重要。一个节点如果在 GNN 消息传播中影响范围更大，它的量化误差更危险。

TSER-Q 使用一个 1-hop + 2-hop 的传播影响代理。

先把图做对称化，并加入 self-loop。

对于边：

```text
source -> target
```

定义归一化传播权重：

```text
weight(source, target) =
    1 / sqrt(degree(source) * degree(target))
```

然后计算：

```text
impact_1hop(source) += weight(source, target)

impact_2hop(source) +=
    weight(source, target) * impact_1hop(target)
```

最终：

```text
graph_impact(source) =
    log(1 + impact_1hop(source) + impact_2hop(source))

graph_impact_q(source) =
    round(graph_impact(source) / max(graph_impact) * 15)
```

相比单纯 degree，这个指标更接近 GNN 中归一化 message passing 的传播影响。

### 3.3 分类边界风险 `margin_risk_q`

同样大小的 embedding 误差，对不同节点造成的任务影响不同。

如果一个节点的分类 margin 很小，它本来就接近决策边界，量化扰动更容易改变预测。

代码使用 FP baseline GNN 的 logits：

```text
margin(v) =
    top1_logit(v) - top2_logit(v)
```

margin 越小，风险越高：

```text
margin_risk(v) =
    1 - clamp(margin(v) / tserq_margin_norm, 0, 1)

margin_risk_q(v) =
    q(margin_risk(v))
```

默认：

```text
tserq_margin_norm = 1.0
```

相关参数：

```text
--tserq_margin_norm
--tserq_margin_weight
```

### 3.4 量化敏感度 `quant_sensitivity_q`

TSER-Q 的图任务敏感度定义为：

```text
quant_sensitivity_q(v) =
    a * graph_impact_q(v)
  + b * margin_risk_q(v)
  + c * graph_context_q(v)
  + d * low_degree_unique_q(v)
```

当前默认权重：

```text
a = 4   # graph impact
b = 2   # margin risk
c = 1   # graph context
d = 1   # low-degree unique
```

相关参数：

```text
--tserq_graph_impact_weight
--tserq_margin_weight
--tserq_graph_context_weight
--tserq_low_unique_weight
```

这个分数和 reuse 的 `sensitivity_q` 不完全一样。

reuse 的 `sensitivity_q` 更强调：

```text
复用替代是否安全。
```

TSER-Q 的 `quant_sensitivity_q` 更强调：

```text
量化扰动是否会通过 GNN 和分类边界放大。
```

### 3.5 W4A8 保护收益 `tserq_protect_gain_q`

TSER-Q 最重要的不是“谁风险大”，而是：

```text
把某个节点从 W4A4 提升到 W4A8，能减少多少风险？
```

代码先计算：

```text
int4_quant_risk_q(v) =
    quant_sensitivity_q(v) * int4_err_q(v)

int8_quant_risk_q(v) =
    quant_sensitivity_q(v) * int8_err_q(v)
```

然后定义保护收益：

```text
tserq_protect_gain_q(v) =
    max(int4_quant_risk_q(v) - int8_quant_risk_q(v), 0)
```

在 fixed budget 下：

```text
默认所有节点先放到 W4A4；
然后选择 tserq_protect_gain_q 最大的前 K 个节点放到 W4A8。
```

这里的 `K` 由：

```text
--real_quant_int8_ratio
```

决定。

例如 Cora 有 2708 个节点，`--real_quant_int8_ratio 0.70` 表示：

```text
约 1896 个节点走 W4A8；
约 812 个节点走 W4A4。
```

## 4. Fixed Budget 表格中各策略含义

当前 `fixed_aggressive_budget` suite 中常见结果行如下。

### 4.1 `AllW4A8`

所有节点都走 W4A8。

用途：

```text
衡量 safe low-precision path 本身是否稳定。
```

如果 `AllW4A8` 已经明显掉点，说明 W4A8 pool 本身不够好，后续混合精度策略没有意义。

### 4.2 `AllW4A4`

所有节点都走 W4A4。

用途：

```text
衡量 aggressive path 的最坏情况。
```

如果 `AllW4A4` 掉点很大，但不是 NaN，也说明 W4A4 可以作为“少量节点使用”的激进路径，而不是全局替代路径。

### 4.3 `RandomBudget`

随机选固定比例节点走 W4A8，其余走 W4A4。

用途：

```text
随机 baseline。
```

任何有意义的策略都应该明显优于它。

### 4.4 `DegreeBudget`

按照 `propagation_q` 从高到低排序，选前 K 个节点走 W4A8。

用途：

```text
测试 degree / topology-only 策略是否足够。
```

这是很多已有图量化工作的常见基准。

### 4.5 `TSERBudget`

按照 reuse 风格的 `sensitivity_q` 从高到低排序，选前 K 个节点走 W4A8。

用途：

```text
测试图语义风险本身是否能指导量化。
```

注意：它不使用真实量化误差。

所以如果 `TSERBudget` 不如 `DegreeBudget`，不一定说明 TSER 没价值，而是说明：

```text
reuse 风险不能直接等价为量化风险。
```

### 4.6 `GraphHopSafeBudget`

`GraphHopSafeBudget` 是当前 fixed-budget 主表中的推荐策略。

它不再问：

```text
这个节点真实 W4A4 error 是多少？
```

而是问：

```text
这个节点是否图语义稳定、哈希空间常见、低传播、非稀有？
```

如果答案是 yes，则该节点更适合走 W4A4 aggressive path。

当前代码中的 W4A4-safe 子项是：

```text
bucket_density_q =
    15 - rarity_q

context_consistency_q =
    15 - graph_context_q

low_propagation_q =
    15 - propagation_q

non_unique_q =
    15 - low_degree_unique_q

hash_agreement_proxy_q =
    min(hash_support_q, context_consistency_q)
```

其中 `hash_agreement_proxy_q` 是 real-quant ablation 中对 multi-head agreement 的轻量代理。因为该实验不执行完整 reuse retrieval，所以没有真实 route hit count；因此用“hash bucket support + local context consistency”近似表示哈希稳定性。

最终：

```text
w4a4_safe_q =
    a * bucket_density_q
  + b * hash_agreement_proxy_q
  + c * context_consistency_q
  + d * low_propagation_q
  + e * non_unique_q
```

默认权重：

```text
a = 2   # hash bucket density
b = 1   # hash agreement proxy
c = 2   # self/context consistency
d = 3   # low propagation
e = 2   # non-rare-tail safety
```

在固定预算下，系统会：

```text
1. 默认所有节点走 W4A8；
2. 选择 w4a4_safe_q 最高的前 (1 - real_quant_int8_ratio) 节点走 W4A4；
3. 其余节点保持 W4A8。
```

这条策略的意义是：

```text
不用逐节点真实量化误差；
不用额外训练 predicted error model；
直接复用 GraphHop/TSER 的 graph/hash metadata 做 aggressive precision routing。
```

### 4.7 `ErrorBudget`

按照纯量化误差收益排序：

```text
error_gain_q(v) =
    max(int4_err_q(v) - int8_err_q(v), 0)
```

用途：

```text
量化误差 oracle / profiling baseline。
```

它是 graph-agnostic 的，只看这个节点从 W4A4 换成 W4A8 能减少多少 embedding 损伤。

### 4.8 `DegreeErrorBudget`

按照 degree 加权量化误差排序：

```text
degree_error_gain_q(v) =
    propagation_q(v) * max(int4_err_q(v) - int8_err_q(v), 0)
```

用途：

```text
测试 degree + quant error 是否足够。
```

这是比纯 degree 更强的 baseline。

### 4.9 `TSERQBudget`

按照 TSER-Q 保护收益排序：

```text
tserq_protect_gain_q(v) =
    max(
        quant_sensitivity_q(v) * int4_err_q(v)
      - quant_sensitivity_q(v) * int8_err_q(v),
        0
    )
```

用途：

```text
测试 graph + task + quant error 联合建模是否优于 degree/error-only。
```

它是当前最接近论文主张的量化路由策略。

### 4.10 `CalibErrorBudget`

`CalibErrorBudget` 是 `ErrorBudget` 的可部署 proxy 版本。

区别是：

```text
ErrorBudget:
    使用全图每个节点真实 int4_err_q / int8_err_q。

CalibErrorBudget:
    只在少量 calibration nodes 上读取真实 int4_err_q / int8_err_q；
    然后用 score-bin LUT 预测全图节点的量化误差。
```

当前实现中，calibration nodes 默认按 `quant_sensitivity_q` 分层采样：

```text
--calib_proxy_size 256
--calib_proxy_strategy score_stratified
```

然后用以下 4 个 4-bit score 组成 LUT bucket：

```text
graph_impact_q
margin_risk_q
graph_context_q
low_degree_unique_q
```

每个 bucket 统计 calibration nodes 的平均误差：

```text
pred_int4_err_q(bucket) = mean(int4_err_q of calibration nodes in bucket)
pred_int8_err_q(bucket) = mean(int8_err_q of calibration nodes in bucket)
```

全图节点通过查表得到：

```text
calib_proxy_error_gain_q(v) =
    max(pred_int4_err_q(v) - pred_int8_err_q(v), 0)
```

然后按这个 gain 选择 W4A8 节点。

### 4.11 `CalibDegreeErrorBudget`

这是 `DegreeErrorBudget` 的 calibration proxy 版本：

```text
calib_proxy_degree_error_gain_q(v) =
    propagation_q(v) * calib_proxy_error_gain_q(v)
```

它用于测试：

```text
degree + predicted quant error
```

是否足够。

### 4.12 `CalibTSERQBudget`

这是 `TSERQBudget` 的 calibration proxy 版本，也是更接近部署系统的策略：

```text
calib_proxy_tserq_protect_gain_q(v) =
    quant_sensitivity_q(v) * calib_proxy_error_gain_q(v)
```

它不再依赖全图逐节点真实量化误差，而是依赖：

```text
少量 calibration profiling
+ graph/task score-bin LUT
```

论文中可以把它作为主系统结果，把 `ErrorBudget/TSERQBudget` 作为 oracle upper bound。

## 5. TSER 和 TSER-Q 的核心区别

### 5.1 TSER reuse 的误差项来自 hash 近似

reuse 的风险形式是：

```text
reuse_risk =
    sensitivity_q * reuse_error_q
```

其中：

```text
reuse_error_q =
    f(Hamming distance, route_hit_count)
```

它关心的是：

```text
目标节点 v 能不能复用候选节点 u 的 embedding？
```

### 5.2 TSER-Q 的误差项来自真实量化损伤

quantization 的风险形式是：

```text
int4_quant_risk =
    quant_sensitivity_q * int4_err_q

int8_quant_risk =
    quant_sensitivity_q * int8_err_q

protect_gain =
    int4_quant_risk - int8_quant_risk
```

它关心的是：

```text
节点 v 自己用 W4A4 是否危险？
把它提升到 W4A8 是否值得？
```

### 5.3 为什么不能只用 TSERBudget？

因为 TSERBudget 只知道节点图语义上是否敏感，但不知道该节点的 W4A4 embedding 是否真的被严重破坏。

例如：

```text
节点 A 图风险很高，但 W4A4 和 W4A8 embedding 几乎一样；
节点 B 图风险中等，但 W4A4 embedding 严重损坏，W4A8 很稳定。
```

如果只用 TSERBudget，可能会优先保护 A。

如果用 TSER-Q，则会优先保护 B，因为保护 B 的收益更大。

这也是为什么当前实验中：

```text
ErrorBudget / TSERQBudget
通常会比 TSERBudget 更稳。
```

## 6. 复杂度和硬件实现开销

### 6.1 离线或预计算开销

大多数分数可以离线或随图快照预计算。

#### Degree / propagation

```text
O(|E|)
```

只需要扫描边表统计 degree。

#### Neighbor mean / context signature

```text
O(|E| * d)
```

其中 `d` 是 cheap feature 维度。

这一步如果用连续向量实现会比较贵，但它是离线预计算。

#### SimHash sketch

```text
O(|V| * d * b)
```

其中 `b` 是 hash bits。

生成后就可以用 bit vector 表示。

#### Boundary Hamming

```text
O(|E| * b)
```

如果 `b` 很小，例如 16/32/64 bits，这一步硬件上很轻。

#### Rarity bucket

```text
O(|V|)
```

前提是 hash 已经生成，只需要统计 bucket occupancy。

#### Quantization error

需要已经生成 FP / W4A8 / W4A4 embedding pools。

开销是：

```text
O(|V| * embedding_dim)
```

但这是 profiling/calibration 阶段，不是在线执行。

#### Margin risk

需要一次 FP baseline GNN forward，得到 logits。

这也是离线 calibration/profiling 阶段完成。

### 6.2 在线硬件需要保存什么？

对于 reuse gate，在线只需要：

```text
propagation_q        4 bits
graph_context_q      4 bits
low_degree_unique_q  4 bits
sensitivity_q        small int
候选 Hamming distance
route_hit_count
```

对于 quant routing，在线可以只保存：

```text
tserq_protect_gain_q
或预先排序后的 routing bit
```

如果已经离线决定每个节点是 W4A8 还是 W4A4，那么在线只需要一个 routing bit：

```text
0 -> W4A4 aggressive path
1 -> W4A8 safe path
```

### 6.3 和 Degree-only 相比的额外开销

Degree-only 只需要：

```text
degree(v)
```

TSER/TSER-Q 额外需要：

```text
1. SimHash sketch
2. hash bucket count
3. 邻居上下文差异
4. 量化误差 profiling
5. baseline margin profiling
```

但这些额外开销大部分都可以离线完成。

论文中可以这样表述：

```text
Degree-only 是最低开销 baseline；
TSER/TSER-Q 通过少量离线 profiling 和 4-bit metadata，
换取更准确的 reuse/quant routing 决策。
```

### 6.4 建议的硬件表述

可以把系统拆成两个阶段：

```text
Offline / graph snapshot stage:
    计算 TSER metadata 和 TSER-Q routing score。

Online inference stage:
    CAM/SimHash 做候选检索；
    小寄存器 + LUT/comparator 做 reuse gate；
    routing bit 控制 W4A8 / W4A4 执行路径。
```

这样可以避免让审稿人觉得在线计算分数太复杂。

## 7. 推荐论文命名

建议把两个分数明确命名为：

```text
TSER:
    Topology-Semantic Execution Risk
    用于 GraphHop SimHash reuse safety gate。

TSER-Q:
    Quantization-aware TSER
    用于 W4A8/W4A4 mixed-precision routing。
```

核心贡献可以表述为：

```text
TSER captures whether a node is safe for semantic reuse.
TSER-Q extends TSER with task margin and measured quantization damage,
so that the system protects nodes where safe precision brings the largest
graph-task benefit.
```

中文表述：

```text
TSER 衡量节点是否适合被哈希复用。
TSER-Q 在 TSER 的图语义风险基础上，引入任务边界风险和真实量化误差，
用于判断哪些节点最值得从 W4A4 提升到 W4A8。
```

## 8. 当前实验结果应该如何解读

如果看到：

```text
TSERBudget < DegreeBudget
```

说明：

```text
单纯 reuse 风格的图语义风险不一定适合直接指导量化。
```

如果看到：

```text
TSERQBudget > DegreeErrorBudget
```

说明：

```text
相比 degree + quant error，加入 graph context / low-degree unique /
margin / graph impact 后，量化路由更有效。
```

如果看到：

```text
ErrorBudget >= TSERQBudget
```

这并不奇怪，因为 ErrorBudget 更接近纯 profiling oracle，只看真实量化损伤。

TSER-Q 的价值在于：

```text
它不是只看 embedding error，
而是把 embedding error 放到图传播和任务边界中加权。
```

因此论文中应该重点比较：

```text
TSERQBudget vs RandomBudget
TSERQBudget vs DegreeBudget
TSERQBudget vs DegreeErrorBudget
```

而不是只和 ErrorBudget 比。
