# Project Roadmap

本文档整理当前论文主线，供 introduction、motivation 和 contribution 组织使用。核心目标是把 text-attributed graph / GFM 推理中的前端 LLM encoder 从默认“全图逐点执行”，重构为一个由图后端信息驱动的分层执行系统。

---

## 1. 核心命题

在当前 LLM-for-GNN / graph foundation model 场景中，节点文本首先经过 LLM encoder 生成 embedding，随后再进入 GNN / graph head 做传播和分类。profiling 显示，端到端推理时间主要被前端 LLM encoder 主导，后端 GNN 通常只占很小比例。

因此，单纯加速后端 GNN 的收益有限。更关键的问题是：

```text
后端图任务信息能否反向撬动前端 LLM encoder 的执行？
```

当前主线回答是：

```text
可以。

图结构、图传播风险、语义边界风险、SimHash 近邻关系
共同决定：
    1. 哪些节点可以跳过 encoder，直接复用已有 embedding；
    2. 哪些 fuzzy hit 需要轻量 residual 修正；
    3. 哪些 miss nodes 必须进入 encoder；
    4. miss nodes 在 NPU encoder path 内应该使用多少 BFP refinement。
```

这使系统从普通 Transformer accelerator 变成 graph-aware encoder execution hierarchy。

---

## 2. 挑战一：LLM Encoder 成本压倒 GNN 后端

### 2.1 问题

GFM / TAG 推理里，每个节点都携带文本。若直接对每个节点运行 LLM encoder，成本随节点数线性增长：

```text
node text -> LLM encoder -> node embedding -> GNN head
```

在 Cora / PubMed / Arxiv 这类任务上，GNN 后端传播计算远小于 LLM encoder 的矩阵计算和权重访存。已有 GNN accelerator 或 NPU-NDP 图加速工作通常重点优化 graph aggregation / combination / graph sampling，但当 LLM encoder 成本占主导时，只优化后端会被前端吞没。

### 2.2 现有 Transformer 加速器的盲点

通用 Transformer / LLM accelerator 通常把输入看成一批独立 sequence：

```text
sequence batch -> Transformer encoder -> output embedding
```

它们可以优化 attention dataflow、GEMM array、quantization、BFP format 或 memory tiling，但通常不知道：

```text
1. 哪些节点在图中高度相似；
2. 哪些节点的 embedding 可以复用；
3. 哪些节点错误后会通过图传播放大；
4. 哪些节点已经被前端 cache / CAM 命中，不需要进入 encoder。
```

如果把 GFM 前端当成普通 LLM batch 来执行，就会忽视这些图任务特有的信息。

### 2.3 当前切入点

本项目的第一层贡献不是更快地执行所有节点，而是减少需要执行 LLM encoder 的节点数量：

```text
SimHash + LRU/CAM:
    快速发现可复用节点 embedding。

TSER score:
    判断复用是否会带来图传播风险。

Residual-Gate:
    对 fuzzy hit 做轻量修正或拒绝。
```

---

## 3. 挑战二：Hash Reuse 需要“安全性”而不只是命中率

### 3.1 问题

SimHash / CAM 可以找到相似节点，但 fuzzy hit 直接复用会带来两类风险：

```text
semantic error:
    anchor 和目标节点语义相近但不完全一致。

propagation error:
    目标节点如果在图中传播影响大，embedding 误差会被 GNN 放大。
```

因此，不能只追求 hash 命中率。系统需要知道：

```text
这个 fuzzy hit 是否值得复用？
如果直接复用不安全，是否可以用很小代价修正？
```

### 3.2 TSER：图风险打分

TSER 是当前复用安全判断的核心分数：

```text
TSER = 3 * propagation_q
     + 1 * graph_context_q
     + 1 * low_unique_q
```

其中：

```text
propagation_q:
    节点错误是否容易通过图传播扩大，核心与 degree / propagation 范围相关。

graph_context_q:
    节点是否处在语义或结构边界区域。

low_unique_q:
    低度但语义稀有节点是否容易因错误复用破坏自身预测。
```

TSER 的定位：

```text
不是替代 SimHash。
SimHash/CAM 负责找候选 anchor；
TSER 负责决定候选是否足够安全。
```

### 3.3 Residual-Gate：拯救 fuzzy match

CAM 输出 support count 后，当前主线在线状态机为：

```text
support >= 5:
    direct reuse

support = 3..4:
    residual-gate candidate

support < 3:
    encoder path
```

Residual-Gate 使用少量离线校准 pair 训练一个轻量 MLP：

```text
input:
    cheap feature difference
    hash / CAM confidence
    graph risk signal

output:
    delta embedding
    accept / reject score
```

在线时：

```text
if accept:
    E(v) = E(anchor) + alpha * delta
else:
    send to encoder path
```

它的作用是把 fuzzy hit 从“全收或全拒”变成中间路径：

```text
可靠 fuzzy hit:
    轻量修正后复用。

不可靠 fuzzy hit:
    打回 encoder。
```

### 3.4 当前前端稳定点

当前统一在线配置：

```text
SimHash:
    8 heads x 16 bit
    radius = 2

TSER:
    weights = 3 / 1 / 1
    threshold = dataset-level policy register T

support split:
    >=5 -> direct
    3..4 -> residual-gate
    <3 -> encoder
```

当前推荐 policy register：

| Dataset | T | Reuse | Direct | Residual | FullP8 Drop | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Cora | 31 | 39.0% | 18.5% | 20.5% | 1.56% | 稳健主点 |
| PubMed | 31 | 42.3% | 42.2% | 0.0% | 1.95% | 保守 fuzzy gate |
| Arxiv | 22 | 46.2% | 20.4% | 25.8% | 2.02% | 当前折中点 |

这里 `FullP8 Drop` 表示 reuse/residual 前端误差；miss nodes 仍走高精度 BFPA8 reference。

---

## 4. 挑战三：Miss Nodes 仍然需要昂贵 Encoder

### 4.1 问题

即使前端 reuse/residual 已经跳过约 40% 左右节点，剩余 miss / reject nodes 仍然需要运行 LLaMA encoder。若这些节点全部走 W4A8 / BFPA8，后端 NPU 仍然有较大成本。

直接使用普通 W4A8 的问题是：

```text
精度稳，但 miss-node encoder 成本高。
```

直接使用全 BFPA4 的问题是：

```text
成本低，但某些 activation block 的 shared exponent 会损伤小值精度。
```

因此后端需要一个比“固定 BFPA4 / 固定 BFPA8”更细的机制。

### 4.2 BFP 的机会和问题

BFP activation 的核心形式：

```text
block values share one exponent
each value keeps its own mantissa
```

当前采用 rowwise block：

```text
1 token row x 128 hidden values
```

BFPA4 相比普通 A4 更稳，因为 shared exponent 保留了 block 动态范围；但 BFP 的风险也来自 shared exponent：

```text
block 内存在 outlier
    -> exponent 被 outlier 拉高
    -> 小值 mantissa 有效位减少
```

这引出 block-level stress：

```text
activation_stress(block)
    描述 block 内动态范围是否不均衡。
```

### 4.3 Graph-Aware Dynamic BFP Refinement

当前后端主线不是固定 BFPA4，也不是静态 top-k BFPA6，而是 block-wise dynamic refinement：

```text
base:
    BFPA4 always compute

refinement:
    selected blocks execute extra BFPA6 mantissa planes
```

选择信号：

```text
priority(block, node)
    = graph_risk(node) * activation_stress(block)
```

也就是说：

```text
普通 Transformer accelerator:
    只能看到 activation block stress。

Graph-aware encoder NPU:
    同时看到 graph risk 和 activation stress。
```

这使 refinement 不再是单纯数值格式迁移，而是图任务风险参与 NPU 内部 block 选择：

```text
高图风险 + 高 activation stress:
    refine to BFPA6

低图风险 或 低 stress:
    keep BFPA4
```

### 4.4 Progressive BFP Array

硬件路径：

```text
BFPA4 base lane:
    计算 4-bit mantissa partial sum

BFPA6 refinement lane:
    对 selected blocks 追加 2-bit mantissa correction

output:
    base partial sum + optional refinement partial sum
```

它不是两套完整阵列，而是 BFPA6-capable progressive array：

```text
默认按 BFPA4 执行；
只有 selected blocks 触发额外 mantissa-plane compute。
```

当前 dynamic pool：

```text
W4GraphBFPA4to6_B128_deg_t0.20
```

含义：

```text
W4:
    weight 使用 AWQ W4。

GraphBFPA4to6:
    activation 默认 BFPA4，部分 block refine 到 BFPA6。

B128:
    BFP block size = 128 values。

deg_t0.20:
    degree risk x activation stress threshold = 0.20。
```

---

## 5. 挑战四：前端 Reuse 与后端 NPU 必须作为一个系统评估

### 5.1 为什么不能只看后端量化

如果只看 encoder pool：

```text
BFPA4 / BFPA6 / dynamic BFP 的 drop 可能很小。
```

但实际 full-stack 中还叠加：

```text
1. direct reuse 误差；
2. residual correction 误差；
3. residual reject 后的 encoder path；
4. miss-node BFP 误差。
```

因此论文主表必须固定前端 policy，再评估后端 NPU：

```text
FullP8:
    前端相同，miss nodes 走 BFPA8。

Dynamic BFP:
    前端相同，miss nodes 走 GraphBFPA4to6。
```

两者差值才是后端 dynamic BFP 的额外代价。

### 5.2 当前 full-stack 推荐组织

最终系统路径：

```text
Graph text node
    |
    v
SimHash + LRU/CAM
    |
    +-- direct hit -------> embedding cache
    |
    +-- fuzzy hit --------> residual-gate correction / reject
    |
    +-- miss/reject ------> Graph-aware Dynamic BFP NPU
                                |
                                v
                           LLaMA encoder embedding
    |
    v
GNN classifier
```

主表建议列：

```text
Dataset
T
Reuse
Direct / Residual / Miss
FullP8 Drop
Dynamic BFP Drop
Extra Drop
Effective Mantissa Bits
Refined Block Ratio
Cost / Cycle Proxy
```

---

## 6. NDP-NPU Heterogeneous System View

### 6.1 NDP 侧职责

NDP / near-memory 侧适合处理低成本、高并发、数据局部性强的前端逻辑：

```text
SimHash generation / lookup
LRU-CAM cache management
support count
route tag generation
embedding cache read
residual-gate lightweight correction
```

输出 route tags：

```text
direct
residual
miss
```

### 6.2 NPU 侧职责

NPU 侧只处理真正需要 encoder 的 miss / reject nodes：

```text
miss-node compaction buffer
Graph-aware Dynamic BFP encoder
progressive BFPA4/BFPA6 array
output merge buffer
```

这样 front-end reuse 不是简单算法优化，而是改变 NPU 输入流：

```text
普通 encoder NPU:
    dense node batch enters encoder.

本系统:
    direct/residual nodes bypass encoder;
    only compacted miss nodes enter Graph-BFP NPU.
```

### 6.3 系统收益来源

收益由两层组成：

```text
1. Encoder bypass:
       direct/residual nodes 不进入 LLaMA encoder。

2. Miss-node low-cost execution:
       进入 encoder 的节点默认 BFPA4，
       仅高 graph-risk x high-stress blocks refine to BFPA6。
```

---

## 7. Paper Contribution Structure

### Contribution 1: Graph-Aware Encoder Bypass

提出 SimHash + LRU/CAM 的 graph-text embedding reuse 前端。它利用节点文本和图上下文构造 hash route，在运行时为节点找到可复用 anchor embedding。

核心区别：

```text
不是缓存完整 query response；
而是缓存 graph-text node embedding。
```

### Contribution 2: TSER-Guided Safe Reuse

提出 TSER 风险分数和 residual-gate 机制，使 fuzzy hash hit 从直接全收变成可控复用：

```text
TSER:
    判断 graph propagation / context / rarity 风险。

Residual-Gate:
    对中风险 fuzzy hit 做 delta correction 和 accept/reject。
```

核心区别：

```text
reuse 决策不只看 hash distance；
还看图传播风险和下游分类敏感性。
```

### Contribution 3: Graph-Aware Dynamic BFP NPU

提出面向 miss nodes 的 progressive BFP encoder array：

```text
BFPA4 base compute
optional BFPA6 block refinement
selection = graph risk x activation stress
```

核心区别：

```text
BFP 不是简单搬到 GFM；
图风险进入 block refinement 决策，
使 NPU 内部计算资源优先服务图任务敏感节点。
```

### Contribution 4: Full-Stack NDP-NPU Pipeline

将前端 cache/reuse、residual correction、miss-node compaction 和 dynamic BFP encoder 组合成完整异构执行路径：

```text
NDP:
    cache / CAM / route / residual

NPU:
    compacted miss-node encoder
```

核心区别：

```text
系统不是单点优化 LLM 或 GNN；
而是让图后端信息控制前端 LLM encoder 的执行层级。
```

---

## 8. Current Result Anchors

当前可作为 introduction / evaluation motivation 的结果锚点：

### 8.1 Unified Front-End Policy

| Dataset | T | Reuse | Direct | Residual | FullP8 Drop |
|---|---:|---:|---:|---:|---:|
| Cora | 31 | 39.0% | 18.5% | 20.5% | 1.56% |
| PubMed | 31 | 42.3% | 42.2% | 0.0% | 1.95% |
| Arxiv | 22 | 46.2% | 20.4% | 25.8% | 2.02% |

### 8.2 Progressive BFP Full-Stack

| Dataset | Policy | Cost | Drop |
|---|---|---:|---:|
| Cora | 25% BFPA6 + 75% BFPA4 | 0.193 | 2.10%-2.21% |
| PubMed | 25% BFPA6 + 75% BFPA4 | 0.181 | 2.52%-2.69% |
| Arxiv | 25% BFPA6 + 75% BFPA4 | 0.170 | 2.06%-2.11% |

### 8.3 Dynamic BFP

Cora dynamic BFP metadata:

```text
refined ratio:
    20.79%

effective mantissa bits:
    4.416

dynamic / BFPA4 cycles:
    1.102x

dynamic / BFPA6 cycles:
    0.735x

dynamic / BFPA8 cycles:
    0.551x
```

PubMed / Arxiv dynamic full-stack results are being completed with `array_trace`.

---

## 9. Introduction Writing Skeleton

Introduction 可以按以下逻辑展开：

```text
Paragraph 1:
    GFM / text-attributed graph increasingly relies on LLM encoder.
    Encoder dominates runtime; GNN back-end is no longer the main bottleneck.

Paragraph 2:
    Existing GNN accelerators optimize aggregation / propagation.
    Existing Transformer accelerators optimize a batch of independent sequences.
    Both miss the bidirectional opportunity:
        graph information can control LLM encoder execution.

Paragraph 3:
    Challenge 1:
        How to avoid redundant encoder calls?
    Solution:
        SimHash + LRU/CAM embedding reuse.

Paragraph 4:
    Challenge 2:
        How to make reuse safe under graph propagation?
    Solution:
        TSER + residual-gate.

Paragraph 5:
    Challenge 3:
        How to reduce remaining miss-node encoder cost?
    Solution:
        graph-aware dynamic BFP refinement NPU.

Paragraph 6:
    System:
        NDP handles cache/reuse/residual;
        NPU handles compacted miss-node dynamic BFP encoder.

Paragraph 7:
    Contributions and results.
```

---

## 10. Current Experiment Priorities

### A. Complete Dynamic BFP Full-Stack

Need:

```text
PubMed:
    regenerate dynamic pool with array_trace
    run T31 full-stack

Arxiv:
    generate dynamic pool with array_trace
    run T22 full-stack
```

### B. Update Main Tables

Required tables:

```text
1. Front-end reuse table:
       direct / residual / miss / FullP8 drop

2. Dynamic BFP backend table:
       refined block ratio / effective bits / cycle proxy

3. Full-stack table:
       FullP8 vs static BFPA4/BFPA6 vs dynamic GraphBFPA4to6
```

### C. Keep Archived Paths Out of Mainline

The following have been explored but should not appear as main contributions:

```text
partial-depth encoder:
    naive layer truncation drops too much.

token compaction:
    not central to current hardware story.

FFN gating:
    weaker than dynamic BFP path.

cross-row BFP packing:
    rowwise BFP is more stable.

old predictor-free bit-plane early stop:
    useful background, not current mainline.
```

