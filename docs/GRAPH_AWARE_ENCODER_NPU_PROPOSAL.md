# Graph-Aware Encoder NPU Design Proposal

本文档基于当前 GraphHopSimhash 实验、TSER/reuse 文档、precision-depth 模拟、FFN gating 原型和 LLM 加速器综述，整理一套更适合写成体系结构贡献的 NPU 设计方案。

核心判断先说清楚：

```text
不要把主创新写成 degree-guided W4A8/W4A4 quant routing。

更有价值的主线是：
Graph-Bit NPU: graph-conditioned precision-depth execution

也就是：
图后端风险信息不只用于 GNN 或 reuse gate，
而是直接进入 LLM encoder NPU 的 datapath，
控制每个 graph-text node 在 GEMM 内部到底算到多少 bit-plane。
```

## 0. Graph-Bit NPU：当前最值得主打的架构线

### 0.1 一句话定义

Graph-Bit NPU 的目标是：

```text
Graph risk controls arithmetic precision depth inside the LLM encoder NPU.
```

它不是普通的量化路由，也不是简单决定某些节点走 W4A8、某些节点走 W4A4。它更深入一层：当一个节点已经被判定必须执行 LLM encoder 时，NPU 内部的 bit-serial / bit-grained GEMM 不必对所有节点都完整执行 A8 bit-plane，而是由图任务风险决定执行深度：

```text
high-risk node:
    P8 / P6，保留更多 activation bit-plane

medium-risk node:
    P6 / P5，少算部分低位 bit-plane

low-risk node:
    P5 / P4，更早停止低位计算
```

因此 Graph-Bit 的核心不是“跳过一个模块”，而是让图后端信息控制 NPU array 的算术努力。

### 0.2 为什么它比 degree-guided quant routing 更适合做主贡献

Degree-guided W4A8/W4A4 routing 的问题是：它仍然停留在系统层选择路径。

```text
degree routing:
    选择哪个节点用 W4A8，哪个节点用 W4A4

Graph-Bit:
    在 W4A8-style encoder NPU 内部，
    控制 bit-plane sequencer、partial-sum accumulator、early-stop mask
```

前者更像硬件友好的 baseline；后者真正落到了：

```text
datatype
bit-serial datapath
activation bit-plane traffic
partial-sum execution depth
array utilization
```

这更符合 HPCA / ISCA / MICRO 体系结构论文需要回答的问题：不是只说“谁更重要”，而是说明“重要性如何改变硬件执行方式”。

### 0.3 为什么它适合 graph-text encoder workload

Graph-text encoder 与普通 LLM serving 的差别在于，每个节点 embedding 的任务价值不同：

```text
hub / high-degree node:
    embedding 误差会传播到更多邻居，低位误差也可能放大

boundary node:
    节点位于语义或结构边界，embedding 扰动更容易跨分类边界

rare low-degree node:
    传播范围小，但自身语义稀有，错误 embedding 可能直接毁掉自身预测

homogeneous low-risk node:
    位于同质社区，邻域冗余强，对小数值扰动更鲁棒
```

普通 bit-serial accelerator 对所有样本使用统一 stopping rule。Graph-Bit 的新点是：

```text
同样是 bit-serial early termination，
但 tolerance 不是固定常数，
而是由 graph risk 动态设置。
```

### 0.4 建议的硬件抽象

Graph-Bit NPU 内部保留一个 W4 weight path，activation 以 A8 逻辑格式进入，但执行时按 bit-plane 渐进计算：

```text
Activation A8:
    b7 b6 b5 b4 b3 b2 b1 b0

P8:
    compute b7..b0

P6:
    compute b7..b2

P5:
    compute b7..b3

P4:
    compute b7..b4
```

硬件模块：

```text
1. graph-risk register
   每个 node batch 保存 risk mode / precision depth

2. bit-plane sequencer
   控制 activation bit-plane 从 MSB 到 LSB 送入 PE array

3. partial-sum scoreboard
   记录当前 bit-plane 已累加的 partial sum

4. remaining-error bound estimator
   可选，用于 true early termination

5. early-stop mask generator
   达到 P6/P5/P4 或满足 bound 后停止低位 bit-plane

6. mode-aware scheduler
   把节点按 P8/P6/P5/P4 分 batch，提高阵列利用率
```

### 0.5 两阶段落地路线

第一阶段使用固定 precision depth，先验证方向：

```text
P8 / P6 / P5 / P4 offline embedding pools
    ~= bit-serial early termination 的离线近似
```

这正对应当前 `precision_depth_ablation`：

```text
FullP8
AllP6 / AllP5 / AllP4
RandomDepthBudget
DegreeDepthBudget
TSERDepthBudget
ContextDepthBudget
```

`PredictorDepthBudget` 和 `OracleDamageBudget` 只作为 debug/oracle 行：

```text
PredictorDepthBudget:
    需要 calibration nodes 学 predictor，不作为主策略。

OracleDamageBudget:
    需要真实 reference damage，只能作为不可部署上界。
```

第二阶段再做真正的 bit-level bound：

```text
for each bit-plane:
    update partial sum
    estimate remaining low-bit contribution
    if bound < graph-conditioned tolerance:
        stop
```

这样可以从“离线多精度 embedding pool 实验”自然过渡到“真实 NPU bit-serial datapath 设计”。

### 0.6 和现有系统路径的关系

Graph-Bit 不替代 SimHash/reuse，而是补齐必须执行 encoder 的那部分：

```text
P0: exact hash reuse
    不跑 encoder

P1: fuzzy hash reuse + residual correction
    不跑 full encoder，只做轻量 adapter

P2: Graph-Bit precision-depth encoder
    必须跑 encoder，但低风险节点少算 bit-plane

P3: full W4A8 encoder
    高风险节点完整执行
```

这条线把整个系统组织成：

```text
减少 encoder 调用次数:
    SimHash reuse / residual correction

降低单次 encoder 执行成本:
    Graph-Bit precision-depth NPU
```

### 0.7 论文中应该主打的贡献表述

推荐写法：

```text
We propose Graph-Bit, a graph-conditioned precision-depth NPU for graph-text LLM encoders.
Graph-Bit maps graph semantic risk to bit-plane execution depth, allowing low-risk nodes to terminate activation bit-serial GEMM earlier while preserving high-risk nodes with full W4A8 execution.
```

不要写成：

```text
We route high-degree nodes to W4A8 and low-degree nodes to W4A4.
```

后者会显得像已有 degree-aware quantization 的变体；前者强调的是图任务风险进入 NPU 内部算术路径。

## 1. 当前场景的本质

GraphHopSimhash 的 workload 和普通 LLM serving 不一样：

```text
普通 LLM serving:
    一个或多个 prompt/token stream
    重点是 decode / KV cache / latency

Graph-text encoder:
    一个图上有大量节点文本
    每个节点需要一个 embedding
    embedding 后面还要进入 GNN / label classifier
```

因此每个节点的 encoder 计算价值并不相同：

```text
某些节点可以直接复用相似节点 embedding；
某些节点可以轻量修正复用结果；
某些低风险节点可以少算一部分 bit-plane / mantissa / channel；
少数高风险节点必须完整 W4A8 encoder。
```

这就是图后端引入后的核心机会：**不是让 GNN 更快，而是让图结构和图风险反过来指导 LLM encoder 本体少算。**

## 2. 推荐主架构：四级 Encoder Execution Hierarchy

当前最合理的系统路径是：

```text
P0: Exact hash reuse
    cost ~= embedding cache read

P1: Fuzzy hash reuse + residual correction
    cost ~= embedding cache read + tiny low-rank adapter

P2: Graph-Bit precision-depth encoder
    graph risk controls bit-plane / mantissa execution depth

P3: Full W4A8 encoder
    high-risk fallback
```

对应硬件图可以写成：

```text
Node text + graph metadata
        |
GraphHash / TSER Router
        |
+----------------+----------------+----------------------+---------------+
| P0 exact reuse | P1 residual    | P2 Graph-Bit encoder | P3 full W4A8 |
| cache read     | correction     | P6/P5/P4 depth       | encoder      |
+----------------+----------------+----------------------+---------------+
        |
Final node embedding -> GNN / classifier
```

这条线的好处是：P0/P1 减少 encoder 调用次数，P2/P3 解决必须执行 encoder 的节点怎么在 NPU datapath 上更便宜地跑。FFN/channel gating 可以作为 P2 的一个补充模式，但不再是唯一或最高优先级的 P2 定义。

## 3. P0/P1：Hash Reuse + Residual Engine

### 3.1 P0 exact reuse

当 SimHash/CAM 找到 exact hit：

```text
E_hat(v) = E(u)
```

硬件上只需要：

```text
1. hash/CAM lookup
2. source node id
3. embedding cache read
```

这条路径的价值是极低成本。当前实验也说明 exact hit 不适合再强行 residual correction，因为 exact hit 本来就是最低风险路径，额外修正可能引入扰动。

### 3.2 P1 fuzzy reuse + residual correction

当 fuzzy hit 通过 TSER gate：

```text
E_hat(v) = normalize(E(u) + alpha * R(z_vu))
```

这里 `R(.)` 是低秩 residual adapter，输入不是 hash bits，而是目标节点和锚点节点之间的差异：

```text
cheap feature delta
graph context delta
hamming distance
route support
candidate confidence
degree / sensitivity metadata
```

硬件意义：

```text
SimHash/CAM:
    找锚点，不生成 embedding

TSER gate:
    判断复用风险

Residual engine:
    小型 low-rank GEMM/vector unit
    修正 fuzzy reuse 的 embedding 偏差
```

当前 Cora/ST 结果说明这条路径是站得住的：

```text
TSER 3/1/1, T=45:
    DirectReuse    Drop 3.84%
    ResidualReuse  Drop 3.16%

TSER 3/1/1, T=30:
    DirectReuse    Drop 2.76%
    ResidualReuse  Drop 2.43%
```

结论不是“residual 单点救很多”，而是：**在相同 reuse 率下，residual 把 reuse-drop 曲线整体向下推。**

### 3.3 这部分的创新点

这比普通 cache reuse 更强，因为它利用了图场景特有的信息：

```text
1. 图节点之间有语义相似性，可以通过 hash/CAM 找 anchor；
2. 图上下文可以判断 anchor 是否危险；
3. reuse 错误会沿图传播，因此需要 TSER gate；
4. fuzzy anchor 不直接用，而是用 cheap feature + graph context 做残差修正。
```

这条路径非常适合做系统级贡献的一部分。

## 4. P2：Graph-Bit Precision-Depth Encoder

### 4.1 为什么 P2 主线应从 FFN gating 升级到 Graph-Bit

FFN/channel gating 已经证明“graph-aware scheduler 能让近似路径更安全”，但它仍然有两个限制：

```text
1. 它只作用在 FFN channel，覆盖范围窄；
2. channel 是否可跳过不一定和 graph risk 强绑定，精度扰动比较硬。
```

Graph-Bit 更适合作为 P2 主线，因为它不改变 Transformer 结构，也不删除特定 token/channel，而是控制 GEMM 的数值执行深度：

```text
full W4A8:
    所有节点都计算完整 activation bit-plane

Graph-Bit:
    高风险节点完整计算 P8
    中风险节点计算 P6/P5
    低风险节点计算 P4 或更早停止
```

这更像硬件论文的核心机制：图风险不是只影响路径选择，而是直接控制 bit-serial datapath。

### 4.2 机制：从 A8 逻辑输入到 variable precision-depth execution

NPU 仍然使用 W4 weight path，activation 在逻辑上是 A8，但执行时按 bit-plane 进入阵列：

```text
A8 activation bit-plane:
    b7 b6 b5 b4 b3 b2 b1 b0

P8:
    b7..b0 全部计算

P6:
    b7..b2 计算，b1..b0 跳过

P5:
    b7..b3 计算，b2..b0 跳过

P4:
    b7..b4 计算，b3..b0 跳过
```

这等价于把低位 bit-plane 当作 residual precision。低风险节点可以丢掉更多 residual precision；高风险节点保留完整数值路径。

### 4.3 路由：graph risk 映射到 precision depth

P2 不依赖 oracle error。它使用图上可部署的风险代理：

```text
propagation_q:
    误差会传播到多少邻居

graph_context_q:
    节点是否处在语义/结构边界

low_unique_q:
    低度但语义稀有节点是否需要保护自身预测

hash/reuse confidence:
    复用或候选匹配是否可靠
```

一个简单、稳定的 budget router 是：

```text
top 20% risk nodes:
    P8

next 30% risk nodes:
    P6

remaining 50% nodes:
    P4
```

也可以加入 P5 作为中间深度：

```text
P8 / P6 / P5 / P4
```

这样避免把论文结论绑死在某个绝对阈值 `T` 上，而是转成固定成本预算下的排序问题。

### 4.4 NPU 数据流

Graph-Bit 的执行流程：

```text
for each node micro-batch:
    read precision_depth_mode from graph-risk buffer
    load W4 weight tile
    for activation bit-plane from MSB to LSB:
        feed bit-plane to PE array
        update partial sum
        if reached selected depth:
            stop remaining low-bit planes
    write output activation
```

对应新增硬件状态：

```text
precision-depth mode register:
    P8 / P6 / P5 / P4

bit-plane sequencer:
    控制当前送入哪个 activation bit-plane

partial-sum accumulator:
    累加已计算 bit-plane 的贡献

early-stop mask:
    达到目标 depth 后关闭低位 bit-plane 的 fetch / MAC

mode-aware batch queue:
    把相同 depth 的节点聚成 micro-batch，减少控制发散
```

### 4.5 从固定 depth 到 true early termination

当前离线实验用 P8/P6/P5/P4 embedding pool 模拟不同 bit-depth，这是第一阶段。

更硬件化的第二阶段是 true bit-serial early termination：

```text
for each bit-plane k:
    partial_sum += contribution(k)
    remaining_bound = estimate_low_bit_bound(k)
    if remaining_bound < graph_tolerance(v):
        stop
```

其中：

```text
graph_tolerance(v) = f(propagation_q, graph_context_q, low_unique_q, confidence)
```

这和 PADE/BETA 的区别是：PADE/BETA 主要从数值 bound 自身决定能否停止；Graph-Bit 把图任务风险加入 tolerance，因此同样的数值 bound 对高风险节点和低风险节点会触发不同停止行为。

### 4.6 FFN/channel gating 的位置

FFN/channel gating 仍然有价值，但应作为 Graph-Bit 之外的补充候选路径：

```text
Graph-Bit:
    优先级更高，控制 bit-plane / mantissa depth

FFN gating:
    可作为低风险节点上的进一步 channel-level skip
```

已有实验可以保留为旁证：

```text
FullW4A8        Cost 0.500  Drop 0.08%
Uniform FFN75   Cost 0.419  Drop 6.06%
Uniform FFN50   Cost 0.338  Drop 9.75%

TSER20_FFN75     Cost 0.484  Drop 0.44%
Degree20_FFN75   Cost 0.484  Drop 0.52%
TSER40_FFN75     Cost 0.468  Drop 1.08%
Degree60_FFN75   Cost 0.451  Drop 1.82%
Random60_FFN75   Cost 0.451  Drop 3.16%
```

这说明 graph-aware scheduler 确实能让 NPU 内部近似路径更安全，但最终主线应提升为更通用的 Graph-Bit precision-depth execution。

## 5. P3：Full W4A8 Encoder 的底层数值路径

P3 是精度兜底路径。这里不建议把 FlashAttention / exact attention dataflow 当主创新，因为这类 exact IO-aware attention 已有大量工作。

但 P3 必须做得硬件上可信：

```text
1. W4A8 / BFP-INT linear array
2. exact attention dataflow as baseline
3. LayerNorm / pooling / output normalization 支持
4. activation scale / outlier / dequant / repack 成本说明
```

可以借鉴的方向：

```text
FIGNA:
    FP activation x INT weight 可以用 integer datapath 保持数值精度。

Anda:
    activation 不一定固定 bit，可以 variable-length grouped activation。

Harmonia:
    BFP activation 可以扩展到 linear 和 attention，不只 linear layer。

BitMod:
    不同 group 可选择不同 low-bit datatype。

FIGLUT / Panacea / AxCore:
    低风险路径可考虑 LUT/RAC、bit-slice sparsity 或 approximate GEMM。
```

但这些应该作为 P3/P2 的数值执行支撑，而不是抢掉 GraphHop 的主线。

## 6. 最有创新性的图后端切入点

下面这些点是真正和 graph 场景绑定的，建议重点写。

### 6.1 Replacement risk 和 quantization risk 分离

当前最重要的理论边界：

```text
reuse risk:
    用别人的 embedding 替代自己。
    关键是语义边界、hash 置信、上下文偏移、低度稀有节点。

quant / gating risk:
    自己的 encoder 计算被扰动。
    关键是量化/剪枝误差大小，以及误差会传播到多少邻居。
```

因此：

```text
TSER 更适合 reuse gate；
Degree / propagation risk 更适合 quant/gating 路由。
```

这能避免把所有分数混在一起，也让论文逻辑更清楚。

### 6.2 Graph-aware execution effort

普通 NPU 会对每条文本等价处理：

```text
every node -> full encoder
```

GraphHop 的新视角是：

```text
node importance is graph-dependent
```

同样的 embedding error，在不同节点上造成的任务损伤不同：

```text
高传播节点:
    错误会影响更多邻居/下游分类，应该 full W4A8。

低传播且高置信节点:
    可以 reuse 或 gated FFN。

语义边界节点:
    hash 相似也要谨慎，因为错复用可能跨 topic。

低度但稀有节点:
    传播小但自身预测敏感，不能简单按低 degree 放松。
```

这个是 LLM-for-GNN 特有的，不是普通 LLM accelerator 会考虑的。

### 6.3 Graph-aware cache hierarchy

embedding cache 不只是普通 LRU cache。图场景下 cache 可以按以下方式组织：

```text
hot anchor cache:
    高 reuse 支持、多次作为 anchor 的节点放 SRAM。

community / bucket cache:
    同 hash bucket 或同 graph community 的 embedding 连续放置。

risk-aware cache:
    高风险节点的 full embedding 优先保留，
    因为它们更可能作为安全 anchor 或 fallback reference。
```

这能把 SimHash/CAM 和 memory hierarchy 绑定起来：

```text
CAM 找 source id
cache controller 预取 E(source)
residual engine 直接消费 anchor embedding
```

### 6.4 Path-aware batching

多路径系统如果逐节点乱跑，array utilization 会很差。因此 scheduler 应该先分类再批处理：

```text
Batch P0:
    cache read only

Batch P1:
    residual vector GEMM

Batch P2:
    Graph-Bit W4A8-style encoder with P8/P6/P5/P4 depth

Batch P3:
    full W4A8 encoder
```

这个调度点很重要，因为它把算法路径变成硬件可执行的数据流，而不是只在软件里 if-else。

### 6.5 Risk-conditioned numeric format

在 P2/P3 内部，还可以进一步做 finer-grained precision：

```text
high-risk node/channel:
    exact W4A8 / BFP safer format

low-risk node/channel:
    approximate GEMM / shorter mantissa / cheaper datatype
```

这可以借鉴：

```text
BitMod:
    per-group datatype adaptation

Anda:
    variable-length grouped activation

Panacea:
    asymmetric activation + bit-slice sparsity

AxCore:
    low-risk approximate GEMM
```

但要注意：这是第二阶段扩展，不建议一开始就把它做成主线，否则系统太散。

## 7. 不建议作为主线的方向

### 7.1 Degree-guided W4A8/W4A4 quant routing

实验上 DegreeTopK_W4A8 比 TSERTopK_W4A8 更稳定，但这不适合当主创新：

```text
1. degree-guided quant routing 已经接近 HEAT/CATOR 等图硬件论文思路；
2. 它更像 baseline / hardware-friendly policy；
3. 量化路由缺少 graph semantics 的强新意。
```

建议定位：

```text
Degree = deployable quant/gating baseline
TSER quant = semantic correction ablation
```

### 7.2 Naive partial-depth encoder

已经验证：

```text
直接拿第 K 层 hidden state mean-pooling 当 final embedding 效果很差。
```

因此不建议保留这条路径作为主线。除非后续加专门 distillation / projector，否则不要写成可用方案。

### 7.3 Token truncation / token budget 作为主贡献

token budget routing 在一些数据上结果不错，但作为主线风险较大：

```text
1. 强依赖文本格式，例如 title/abstract 是否 front-loaded；
2. 容易被质疑只是输入裁剪工程；
3. 又增加额外 token/chunk scorer 和 preprocessing；
4. 与 NPU array 本体创新关系弱于 Graph-Bit precision-depth datapath。
```

建议作为补充实验或 appendix，不要抢主线。

### 7.4 Graph-aware FlashAttention tile reuse

需要谨慎。相似节点聚到一起并不意味着 Q/K/V 可以直接复用：

```text
每个节点的 Q/K/V 仍由自己的 token 和权重计算得到；
不能因为 graph/hash 相似就拿别人的 Q/K/V 替代。
```

因此 FlashAttention-style exact dataflow 可以作为 P3 baseline，但不要包装成核心新意。

### 7.5 Oracle error-aware routing

凡是需要全图 `FP embedding vs quant embedding` 的真实误差，都不能作为 deployable strategy：

```text
DegreeErrorTopK
TSERErrorTopK
OracleDamageBudget
```

它们只能作为上界和 debug 工具。

同理，任何需要额外 calibration nodes 学习 damage predictor 的 `Predictor*Budget` 也不进入主线策略；它只能用来诊断手写 graph proxy 还有多少提升空间。

## 8. 建议的最终论文贡献组织

建议把贡献写成三层，而不是堆一堆 tricks。

### Contribution 1: GraphHash reuse hierarchy

```text
SimHash/CAM finds reusable anchors.
TSER gate filters unsafe reuse.
Residual adapter corrects fuzzy reuse.
```

重点指标：

```text
reuse rate
drop
hit error
cost
exact/fuzzy/reject breakdown
```

### Contribution 2: Graph-aware NPU execution paths

```text
For non-reused nodes, graph risk routes nodes to:
    Graph-Bit P8/P6/P5/P4 precision-depth path
    or full W4A8 path.
```

重点指标：

```text
P8/P6/P5/P4 node ratio
average executed bit-depth
MAC reduction
weight traffic reduction
activation traffic reduction
array utilization
drop under same cost
```

### Contribution 3: Unified scheduler and memory hierarchy

```text
Path-aware batching
embedding anchor cache
residual engine
W4A8/BFP array
FFN mask buffer
```

重点是把 graph metadata 从“离线分数”变成硬件执行控制流：

```text
metadata -> route -> cache/prefetch -> compute path -> final embedding
```

## 9. 推荐硬件模块

### 9.1 GraphHash Front-End

负责：

```text
hash bucket lookup
candidate source id
hamming distance
route support counters
score gate input
```

输出：

```text
hit type = exact / fuzzy / reject / miss
candidate id
confidence metadata
```

### 9.2 Risk Router

输入：

```text
hit type
reuse risk
propagation risk
graph context risk
low-unique risk
support confidence
```

输出：

```text
P0 / P1 / P2 / P3
```

建议强调：不同路径用不同风险 proxy，不强行一个 TSER 总分管所有事情。

### 9.3 Embedding Anchor Cache

存：

```text
frequent anchor embeddings
high-confidence bucket embeddings
recent full W4A8 outputs
```

功能：

```text
P0 direct output
P1 residual source vector
P3 full output writeback
```

### 9.4 Residual Correction Engine

小型 vector / low-rank GEMM：

```text
input  = pair feature z_vu
output = residual vector
```

适合独立小阵列或复用 vector unit，不要占用 full W4A8 array。

### 9.5 W4A8 / BFP Encoder Array

P2/P3 共享：

```text
QKV / attention / FFN W4A8 compute
LayerNorm / pooling support
activation format conversion
```

底层可选：

```text
FIGNA-like FP-INT integer path
Harmonia-like BFP activation path
Anda-like grouped activation path
```

### 9.6 FFN Channel-Gated Datapath

需要：

```text
channel-group mask SRAM
grouped weight layout
gated FFN activation buffer
grouped GEMM scheduler
```

核心是规则 group，不做 unstructured sparsity。

## 10. 实验验证路线

### Stage A: 单独验证每条路径

```text
P0/P1:
    residual_reuse sweep
    direct vs residual
    CAM anchor vs random anchor

P2:
    precision_depth_ablation
    FullP8 / AllP6 / AllP5 / AllP4
    random / degree / TSER / context / predictor / oracle depth routing
    optional: ffn_channel_gating as secondary comparison

P3:
    W4A8 full path accuracy
    compare FP16 / W4A8 / W4A4
```

### Stage B: 组合验证

```text
P0 + P1:
    reuse hierarchy

P2 + P3:
    Graph-Bit precision-depth routing

P0 + P1 + P2 + P3:
    full hierarchy
```

组合实验必须保证：

```text
FullP8 / P6 / P5 / P4 来自同源 backend；
否则 drop 差异可能来自 embedding pool 生成方式，而不是 precision-depth 本身。
```

### Stage C: 硬件指标

除了 Acc / Drop，还必须报：

```text
full encoder invocation reduction
residual corrected node ratio
P8 / P6 / P5 / P4 node ratio
average executed bit-plane depth
MAC reduction
weight traffic reduction
activation traffic reduction
SRAM metadata overhead
embedding cache hit rate
array utilization
energy/cost model
```

这样才像 HPCA/ISCA/MICRO 风格的系统评估。

## 11. 最推荐的论文主线表述

可以把整篇论文主线压成一句话：

```text
GraphHopSimhash turns graph-aware semantic risk into a hardware execution hierarchy for LLM encoders:
exact reuse, corrected fuzzy reuse, graph-conditioned precision-depth execution, and full W4A8 fallback.
```

中文表述：

```text
本文不是单纯加速 GNN，也不是单纯量化 LLM；
而是利用图结构和节点语义风险，
为每个 graph-text node 选择不同强度的 LLM encoder 执行路径，
并进一步控制 NPU 内部 bit-plane / mantissa 计算深度，
从而减少 full encoder 调用和单次 encoder 的算术/访存成本，同时保持 GNN 任务精度。
```

## 12. 当前最靠谱的建议

如果现在要收敛到一条最稳路线，我建议：

```text
主线:
    P0 exact reuse
    P1 fuzzy reuse + residual correction
    P2 Graph-Bit precision-depth W4A8-style encoder
    P3 full W4A8

主打创新:
    Graph-Bit NPU
    graph-risk-conditioned bit-plane execution
    residual correction engine
    risk-separated scheduler

Baseline:
    full W4A8
    direct reuse
    random routing
    degree routing
    TSER routing
    uniform P6/P5/P4
    oracle damage routing
    optional uniform FFN gating

不要主打:
    degree-guided W4A8/W4A4 quant routing
    naive partial-depth encoder
    pure token truncation
    oracle error-aware routing
    FFN channel gating as the only NPU contribution
    graph-aware FlashAttention reuse
```

这套方案的优点是：每个模块都有当前实验支撑，也能自然接到 NPU 设计，不会只停留在软件启发式。

## 13. Beyond FFN Gating: NPU 内部设计空间

前面的 P0/P1/P2/P3 更像系统执行层级，还不够深入到 NPU 内部。真正面向 HPCA/ISCA/MICRO 的设计，需要回答：

```text
当一个节点必须跑 LLM encoder 时，
NPU 内部的 array、datatype、bit-serial datapath、activation format、outlier path、tile schedule
如何利用 graph 后端信息进一步减少计算和访存？
```

因此 FFN channel gating 只能算一个候选点，不应该限制整个设计空间。下面是更深入的候选机制。

## 14. Graph-Conditioned Bit-Serial Early Termination

### 14.1 思路来源

参考 PADE / BETA / BitMod 这类 bit-serial 或 bit-grained accelerator：

```text
不是先完整计算再判断是否重要，
而是在 bit-plane 逐步计算过程中提前判断：
    当前 partial sum 是否已经足够确定？
    后续低位 bit 是否不可能改变最终重要性？
```

PADE 的关键是 predictor-free：用 bit-level upper/lower bound 控制 early termination，避免额外 predictor 成本。

### 14.2 迁移到 graph-text encoder

普通 bit-serial early termination 对所有 token/node 使用同一停止规则。但在 graph-text workload 中，不同节点对数值误差的容忍度不同：

```text
high-risk node:
    高 degree / 高 propagation / 边界节点 / 低置信 reuse
    -> 需要更严格 bit-bound
    -> 多算低位 bit

low-risk node:
    低传播风险 / 高置信 / 同质社区内节点
    -> 可以更早停止
    -> 少算低位 bit
```

这会形成一个新的机制：

```text
Graph-conditioned bit-plane termination
```

核心不是“跳过某个 FFN channel”，而是在 GEMM 内部让每个 node-batch 使用不同 bit-depth / termination bound。

### 14.3 硬件实现

NPU 内部增加：

```text
1. bit-serial W4A8 / BFP datapath
2. partial-sum scoreboard
3. upper/lower bound estimator
4. per-batch risk tolerance register
5. early-stop mask generator
```

执行流程：

```text
for each node batch:
    load graph_risk_tolerance
    for each bit-plane:
        update partial sum
        estimate remaining error bound
        if bound < tolerance:
            stop remaining low-bit planes
```

这里的 `tolerance` 不是 oracle error，而是由 graph risk 映射得到：

```text
tolerance = f(propagation_q, graph_context_q, low_unique_q, confidence)
```

### 14.4 为什么这是图场景的新点

普通 LLM accelerator 的 early termination 是 sequence/token 级数值优化。这里变成：

```text
graph risk controls arithmetic precision at runtime
```

也就是图后端信息直接控制 NPU 内部 bit-plane 计算深度。

这是比 FFN gating 更“内部”的设计点，值得优先尝试。

## 15. Graph-Adaptive Mixed Datatype / Mantissa Allocation

### 15.1 思路来源

参考 BitMod / Anda / Harmonia：

```text
BitMod:
    per-group datatype adaptation, FP3/FP4/INT mixed datatype

Anda:
    variable-length grouped activation mantissa

Harmonia:
    all-layer BFP activation, BFP-INT + BFP-BFP PE
```

这些工作说明：低比特不应该只写成 W4A8/W4A4 二选一，而应该细到 group / datatype / mantissa。

### 15.2 迁移到 graph-text encoder

对 graph-text node，可以让风险决定 activation/weight group 的数据格式：

```text
high-risk node / high-risk layer / high-risk channel group:
    safer datatype
    longer mantissa
    exact W4A8 / BFP8

low-risk node / low-risk channel group:
    cheaper datatype
    shorter mantissa
    FP4/INT4/sub-W4
```

也就是说，routing 不再只是：

```text
node -> W4A8 or W4A4
```

而是：

```text
node/layer/channel-group -> datatype mode
```

### 15.3 硬件实现

需要的 NPU 模块：

```text
1. mixed-datatype PE
    支持 INT4 / FP4 / BFP mantissa 等格式

2. per-group datatype tag buffer
    每个 channel group 或 activation group 存 2-3 bit mode

3. runtime activation compressor
    把 FP/W4A8 activation 压到 variable mantissa / BFP group

4. mode-aware scheduler
    把相同 datatype mode 的节点聚成 batch，减少 mode switch
```

### 15.4 图后端的新意

普通 BitMod/Anda 是按 weight/activation distribution 自适应；我们的新点是加入 graph semantics：

```text
datatype selection = numerical distribution + graph task risk
```

例如：

```text
同样 activation 分布下，
高传播节点保守；
低传播且高置信节点激进。
```

这个比 degree-guided W4A4 更细，也更靠近 NPU 内部。

## 16. Graph-Aware Outlier Channel Protection

### 16.1 思路来源

W4A4/W4A8 的问题经常不是平均误差，而是少量 outlier channel / outlier node 破坏 embedding。Harmonia、llm.npu、Oaken 等工作都强调 outlier/hot channel 需要特殊路径。

### 16.2 迁移到 graph-text encoder

当前可以设计：

```text
Graph-risk-aware outlier preservation
```

机制：

```text
1. offline calibration 找出每层 activation outlier channel group
2. online 根据 node graph risk 决定 outlier group 的保护强度
```

例如：

```text
high-risk nodes:
    outlier channels use A8/BFP8
    normal channels use A4/BFP4

low-risk nodes:
    fewer outlier channels protected
    or all channels use cheaper format
```

### 16.3 硬件实现

```text
outlier channel table:
    per layer, top-k channel group ids

risk-conditioned precision mask:
    high-risk node -> protect more groups
    low-risk node  -> protect fewer groups

dual-path PE:
    protected groups -> safer precision lane
    normal groups    -> low precision lane
```

### 16.4 为什么有意义

这比全局 outlier protection 更有新意：

```text
不是所有节点都为最坏 outlier 付出代价；
只有图上重要/敏感节点保护更多 outlier channel。
```

这条线特别适合和 W4A4/W4A8 的经验结合，因为你已经观察到 W4A4 的损伤主要来自 backend/outlier，而不是分数机制本身。

## 17. Graph-Routed Approximate GEMM

### 17.1 思路来源

参考 AxCore / FIGLUT / Panacea：

```text
AxCore:
    approximate FP multiplication, multiplier-free GEMM

FIGLUT:
    LUT/RAC 替代低比特乘法

Panacea:
    asymmetric quantization + bit-slice sparsity + compensation
```

这些工作说明：低风险计算不一定要使用 exact MAC。

### 17.2 迁移到 graph-text encoder

可以设计两条 GEMM lane：

```text
Exact lane:
    high-risk nodes
    exact W4A8 / BFP-INT

Approx lane:
    low-risk nodes
    approximate GEMM / LUT-RAC / bit-slice skip
```

图风险控制 lane selection：

```text
propagation high -> exact lane
confidence high + propagation low -> approximate lane
```

### 17.3 硬件实现

NPU PE cluster 支持：

```text
1. exact low-bit MAC lane
2. approximate / LUT / RAC lane
3. correction / compensation unit
4. per-batch lane mode register
```

执行时：

```text
batch low-risk nodes:
    route to approximate lane

batch high-risk nodes:
    route to exact lane
```

### 17.4 风险

这条线很有硬件味，但实验风险也更高：

```text
1. approximate GEMM 是否损伤 embedding 需要单独验证；
2. 低风险 proxy 是否真的能筛出可近似节点；
3. 如果近似误差和图风险不相关，收益会不稳定。
```

建议作为第二阶段创新，而不是当前唯一主线。

## 18. Graph-Aware Tile and Dataflow Scheduling

### 18.1 不是复用别人的 Q/K/V

需要明确：graph/hash 相似并不意味着可以直接拿别人的 Q/K/V 替代自己的 Q/K/V。每个节点的 Q/K/V 仍由自己的 token 和权重生成。

因此不要把 Graph-aware FlashAttention 写成：

```text
similar nodes reuse Q/K/V tiles
```

这个说法不靠谱。

### 18.2 真正可做的是 dataflow scheduling

图信息可以控制 execution ordering 和 mode grouping：

```text
1. path-aware batching:
    P0/P1/P2/P3 分别聚成 batch

2. risk-aware mode grouping:
    同 precision mode / datatype mode 的节点聚成 batch

3. length-aware within graph bucket:
    在同一社区/同一 hash bucket 内按 token length 排序，减少 padding

4. cache-aware anchor ordering:
    让使用同一 anchor 或同一 bucket 的 P1 节点相邻执行，提高 embedding cache locality
```

这不是复用 Q/K/V，而是减少：

```text
mode switch
padding waste
cache miss
metadata fetch
```

### 18.3 硬件实现

```text
graph-aware work queue:
    queue[P0], queue[P1], queue[P2_mode0], queue[P2_mode1], queue[P3]

batch builder:
    packs nodes with same path/mode/length bucket

NPU controller:
    configures PE mode once per batch
    streams grouped weights/activations
```

这个模块比较系统，但非常必要。否则多路径设计会因为调度混乱导致 array utilization 掉下去。

## 19. Graph-Aware Attention Early Exit / Sparse Attention

### 19.1 为什么不能直接主打

attention skipping 很吸引人，但比 FFN/channel 更危险：

```text
1. attention softmax 对局部误差敏感；
2. 不规则 sparse attention 硬件复杂；
3. encoder 文本长度不一定长到 attention 成为主要瓶颈；
4. 当前 graph-text embedding 对 attention sparsity 的容忍性还没验证。
```

### 19.2 可行的保守方案

更合理的是采用 predictor-free / bounded 方案：

```text
high-risk nodes:
    exact attention

low-risk nodes:
    bounded bit-serial early termination
    or top-k attention with strict error bound
```

核心是：

```text
graph risk controls attention approximation tolerance
```

而不是 graph/hash 直接决定哪些 token 互相 attend。

## 20. 推荐的 NPU 内部创新优先级

如果从“最靠谱 + 最像硬件论文 + 最能体现 graph 场景新意”排序，我建议：

### Priority 1: Graph-conditioned bit-serial / precision-depth execution

```text
机制:
    bit-plane early termination / variable mantissa / mixed datatype

图场景新意:
    graph risk controls arithmetic effort

优点:
    深入 NPU datapath
    不局限 FFN
    可作用于 QKV/FFN/attention GEMM
```

### Priority 2: Graph-risk-aware outlier channel protection

```text
机制:
    high-risk nodes protect more outlier channels
    low-risk nodes use cheaper activation format

图场景新意:
    outlier protection budget is allocated by graph task sensitivity

优点:
    与 W4A4/W4A8 实验经验强相关
    硬件实现清晰
```

### Priority 3: Graph-routed approximate GEMM lane

```text
机制:
    low-risk nodes use AxCore/FIGLUT/Panacea-like cheaper GEMM
    high-risk nodes use exact W4A8/BFP-INT

图场景新意:
    approximate computing is no longer uniform, but graph-risk conditioned

优点:
    硬件味强
    可作为 NPU 内部 array 创新
```

### Priority 4: FFN/channel gating

```text
机制:
    grouped FFN channel skip

图场景新意:
    graph-aware scheduler selects safe nodes

优点:
    已有实验支撑
    实现最直接

不足:
    如果只写它，NPU 内部创新略窄
```

### Priority 5: Graph-aware batching/dataflow scheduler

```text
机制:
    path/mode/length/cache-aware work queue

图场景新意:
    graph/hash metadata controls NPU execution order

优点:
    能提高实际 utilization

不足:
    更偏系统调度，单独作为主创新不够硬
```

## 21. 建议重新组织最终架构

更强的版本可以写成：

```text
Graph-conditioned adaptive arithmetic NPU for LLM encoders
```

而不是：

```text
Graph-aware FFN gating NPU
```

推荐最终结构：

```text
Frontend:
    SimHash/CAM + TSER risk engine

Scheduler:
    maps node -> path + arithmetic mode

Datapath:
    mode-adaptive W4A8/BFP/bit-serial PE array
    supports:
        exact W4A8
        variable mantissa / mixed datatype
        outlier protected mode
        approximate low-risk mode

Side engine:
    residual correction engine

Memory:
    anchor embedding cache
    channel/outlier/mode metadata buffer
```

这套设计里，图后端不只是决定“跑不跑 encoder”，而是决定：

```text
1. 算多少 bit-plane
2. 用多长 mantissa
3. 保护多少 outlier channel
4. 走 exact 还是 approximate lane
5. 是否启用 FFN/channel gating
6. 如何 batch 和 cache
```

这才真正深入到了 NPU 内部。

## 22. 下一步最该验证什么

不要一口气全做。建议按风险最小、硬件味最强的顺序：

```text
Step 1:
    先做 graph-risk-conditioned activation precision / mantissa allocation。
    用已有 embedding pool 模拟：
        high-risk -> P8 / full W4A8
        mid-risk  -> P6 / P5
        low-risk  -> P4
    观察 Degree/TSER/risk 是否能稳定筛出可激进节点。

Step 2:
    做 outlier channel protection sweep。
    比较：
        uniform outlier protection
        graph-risk-aware outlier protection
        random outlier budget

Step 3:
    做 bit-plane early termination 的软件仿真。
    不必先写 RTL，先在 GEMM/embedding 层模拟：
        full bit-plane
        low-risk fewer bit-plane
        high-risk full bit-plane

Step 4:
    保留 FFN gating 作为已验证路径，
    但把它升级成 mode-adaptive PE array 的一个实例。
```

如果 Step 1/2 能跑通，论文的新意会比单纯 FFN gating 强很多。

当前 Step 3/4 的正式设计定义已经整理到：

```text
docs/GRAPH_BIT_NPU_DESIGN.md
```

其中 Step 3 对应 bit-plane early-termination simulator，Step 4 对应 mode-adaptive PE array。FFN gating 只作为 mode-adaptive array 的可选执行模式，不再作为 P2 主线。
