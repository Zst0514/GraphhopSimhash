# Paper Roadmap

本文档整理当前工作的论文叙事主线。它不是模块实现清单，而是围绕 introduction / motivation / contribution 的写作逻辑展开：先说明 GFM 场景为什么不同于普通 Transformer 或 GNN 推理，再凝练三个核心挑战，最后给出对应机制和贡献点。

---

## 1. 论文核心命题

Text-attributed graph / Graph Foundation Model 推理通常由两段组成：

```text
node text
    -> LLM / Transformer encoder
    -> semantic embedding
    -> GNN / graph classifier
```

随着前端从轻量 LM 变成更重的 LLM encoder，端到端瓶颈发生变化：GNN 后端传播和分类计算通常只占很小比例，而前端 LLM encoder 的矩阵计算、权重访存和 embedding 生成成为主导成本。

这带来一个与 HEAT / CATOR 类工作不同的中心问题：

```text
当 LLM encoder 占据绝大多数推理时间时，
GFM 加速的关键不再只是优化后端 GNN，
而是利用图后端信息反向控制前端 LLM encoder 的执行。
```

因此本文的核心命题是：

```text
Graph information should not only be consumed after embeddings are generated.
It should guide which node embeddings are reused, which fuzzy matches are corrected,
and how much computation the remaining LLM encoder path should spend.
```

对应到系统上，本文不是单纯设计一个 Transformer NPU，也不是单纯设计一个 GNN 加速器，而是一个面向 GFM 的分层执行系统：

```text
SimHash + LRU/CAM:
    找到可复用 embedding，跳过一批 LLM encoder 调用。

TSER + Residual-Gate:
    判断 fuzzy match 是否安全，并用轻量 correction 拯救中风险命中。

Graph-aware Dynamic BFP NPU:
    对剩余 miss nodes，用图风险和 activation stress 控制 BFPA4/BFPA6 refinement。
```

---

## 2. 背景段落应该如何写

### 2.1 从 GFM / TAG 推理讲起

可以从 Text-Attributed Graphs 和 Graph Foundation Models 的应用动机切入：

```text
论文节点分类、医疗知识图谱、金融风控、推荐系统等场景中，
节点通常携带丰富文本属性。
LLM encoder 能提取高质量语义 embedding；
GNN 能进一步结合拓扑结构进行传播和分类。
```

这一组合提升了模型能力，但也引入了系统问题：

```text
每个节点都需要文本编码；
大图有成千上万到数十万节点；
本地部署时无法把所有节点都逐个送入大模型 encoder。
```

### 2.2 与 CATOR / HEAT 的定位区别

CATOR / HEAT 这类工作重点讨论 cascaded GFM 中 Transformer 与 GNN 的协同、混合精度、NPU-NDP 异构等问题。它们的叙事常常建立在：

```text
frontend TF 和 backend GNN/Comb 都是重要瓶颈；
需要在二者之间分配硬件资源和精度。
```

本文的 profiling 则强调另一种更重的 LLM 前端场景：

```text
LLM encoder 时间占比极高；
GNN 后端计算占比很小；
因此只优化后端 GNN 难以带来端到端收益。
```

因此，本文不把问题定义为“如何分别加速 TF 和 GNN 两段”，而是定义为：

```text
如何利用后端图任务信息减少和重塑前端 LLM encoder 执行。
```

这个定位可以避免与 HEAT/CATOR 的 degree-guided quantization 或 cascaded mixed-precision array 叙事正面重叠。

---

## 3. 三个核心挑战

### Challenge 1: LLM Encoder Redundancy in Graph Workloads

普通 Transformer accelerator 把输入看成一批独立 sequence：

```text
sequence batch -> encoder -> embeddings
```

但在图文本任务中，节点之间存在大量语义近邻和结构近邻：

```text
同社区节点文本相似；
同主题论文标题和摘要相似；
相邻节点或同 hash bucket 节点可能映射到相近 embedding。
```

如果仍然逐节点运行完整 LLM encoder，会产生大量冗余：

```text
很多节点的 embedding 可以由已有 anchor embedding 近似复用；
但普通 LLM 加速器没有图级相似性和缓存命中信息。
```

现有工作不足：

```text
GNN accelerators:
    主要优化 aggregation / combination / sampling，对 LLM encoder 冗余无能为力。

Transformer accelerators:
    优化 GEMM、attention、quantization，但默认每个 sequence 都必须完整编码。

KV-cache / decoder serving work:
    面向 autoregressive decoding，不适合 encoder-only graph embedding 生成。
```

本文对应方案：

```text
Graph-aware SimHash + LRU/CAM embedding reuse。
```

核心思想：

```text
用 SimHash 将节点语义和局部图上下文映射到多头 hash signature；
用 LRU/CAM 在近邻 bucket 中查找 anchor embedding；
exact / high-confidence hit 直接从 embedding cache 读取，不进入 LLM encoder。
```

写作时应强调：

```text
这不是普通结果缓存；
缓存对象是 graph-text node embedding；
命中关系由语义 hash、图上下文和 CAM support 共同决定。
```

---

### Challenge 2: Unsafe Fuzzy Reuse under Graph Propagation

仅有 hash 命中率不足以保证下游精度。原因是图任务会放大某些节点错误：

```text
低风险节点:
    embedding 误差主要影响自身，传播范围有限。

高风险节点:
    hub / boundary / semantically rare 节点一旦复用错误，
    误差会通过 GNN message passing 影响更多节点。
```

因此 fuzzy hit 面临两难：

```text
全部接受:
    reuse 高，但 drop 可能明显变大。

全部拒绝:
    精度稳，但 reuse 太低。
```

现有工作不足：

```text
LSH / ANN / CAM:
    更关注相似性检索，不理解图传播风险。

Degree-guided quantization:
    可以保护 hub，但不能判断一个具体 fuzzy anchor 是否适合复用。

普通 confidence gate:
    只看 hash distance / cosine margin，缺少 graph-aware sensitivity。
```

本文对应方案：

```text
TSER-guided Residual-Gate fuzzy reuse。
```

TSER 的作用：

```text
TSER = topology-aware semantic error risk

它把传播风险、上下文边界风险、低度稀有风险组合成 reuse gate 分数。
```

当前主线分流：

```text
support >= 5:
    direct reuse

support = 3..4:
    residual-gate candidate

support < 3:
    compute / encoder path
```

Residual-Gate 的作用：

```text
对 fuzzy anchor 训练一个轻量 MLP，
根据 pair feature 预测 delta embedding 和 accept/reject。

accepted:
    E(v) = E(anchor) + alpha * delta

rejected:
    send to encoder path
```

这里要避免把 Residual-Gate 写成一个过大的贡献点。它更适合放在 Challenge 2 的解决方案内部：

```text
SimHash/CAM provides candidate anchor.
TSER controls graph-aware reuse safety.
Residual-Gate repairs or rejects medium-confidence fuzzy hits.
```

当前可作为结果锚点：

| Dataset | Policy | Reuse | Direct | Residual | FullP8 Drop |
|---|---:|---:|---:|---:|---:|
| Cora | T31 | 39.0% | 18.5% | 20.5% | 1.56% |
| PubMed | T31 | 42.3% | 42.2% | 0.0% | 1.95% |
| Arxiv | T22 | 46.2% | 20.4% | 25.8% | 2.02% |

`FullP8 Drop` 的含义：

```text
reuse/residual 前端固定；
所有 miss nodes 仍走 high-quality BFPA8 encoder。
因此该 drop 主要反映前端 reuse/residual 本身的误差。
```

---

### Challenge 3: Miss-Node Encoder Cost Remains Dominant

即使前端跳过约 40% 节点，剩余 miss / rejected nodes 仍要进入 LLM encoder。直接使用 W4A8 或 BFPA8 仍然昂贵。

一个直接想法是把 miss nodes 全部降到 BFPA4：

```text
BFPA4:
    成本低；
    但 shared exponent 可能牺牲 block 内小值精度。
```

BFP 的关键问题：

```text
一个 block 共享 exponent。
如果 block 内存在 outlier，
共享 exponent 被大值拉高，
小值的 mantissa 有效位被压缩。
```

这说明 BFP 不能只作为普通 Transformer 低精度格式搬过来。图场景提供了新的选择信号：

```text
普通 Transformer accelerator:
    可以看到 activation block stress；
    但不知道该 block 对图任务是否重要。

Graph-aware GFM accelerator:
    同时知道 node graph risk 和 activation block stress。
```

现有工作不足：

```text
通用 BFP / mixed datatype / FP-INT GEMM:
    主要从数值格式和 GEMM datapath 出发，缺少 graph task sensitivity。

HEAT 类 degree-guided precision:
    用拓扑信息指导节点级精度，但更像 coarse-grained node precision assignment。

普通 W4A8 / AWQ / QServe:
    PTQ 很强，但 graph-blind；无法把有限高精度预算投给图任务敏感 block。
```

本文对应方案：

```text
Graph-aware Dynamic BFP Refinement NPU。
```

核心机制：

```text
Base path:
    all miss-node activation blocks compute with BFPA4.

Refinement path:
    selected blocks execute extra two mantissa planes, becoming BFPA6.

Selection:
    priority(block, node) = graph_risk(node) * activation_stress(block)
```

其中：

```text
graph_risk(node):
    来自 degree / TSER / propagation sensitivity，
    表示该节点 embedding 误差对图任务的影响。

activation_stress(block):
    来自 BFP exponent selection 过程，
    表示该 activation block 是否容易因 shared exponent 损失小值精度。
```

这条线的关键创新表述：

```text
BFPA4/BFPA6 本身不是贡献；
贡献是把 graph risk 引入 BFP block refinement 决策，
使 NPU 内部 mantissa refinement 预算优先服务图任务敏感节点和数值脆弱 block。
```

硬件结构可以写成：

```text
BFPA4 base lane:
    执行基础 4-bit mantissa MAC。

Optional refinement lane:
    对 selected blocks 追加 2-bit mantissa-plane correction。

Shared W4 tile:
    BFPA4 和 BFPA6 refinement 共享同一份 W4 weight tile。
```

当前 dynamic BFP 结果锚点：

| Dataset | Refined Blocks | Effective Bits | Dynamic / BFPA4 | Dynamic / BFPA6 | Dynamic / BFPA8 |
|---|---:|---:|---:|---:|---:|
| Cora | 20.79% | 4.416 | 1.102x | 0.735x | 0.551x |
| PubMed | 13.88% | 4.278 | 1.070x | 0.713x | 0.535x |
| Arxiv | 21.46% | 4.429 | 1.105x | 0.737x | 0.553x |

解释：

```text
Dynamic / BFPA4 > 1:
    dynamic refinement 比纯 BFPA4 多算少量 extra planes。

Dynamic / BFPA6 < 1:
    相比全 BFPA6，dynamic 只 refine 少量 blocks，节省约 26%-29% cycles。

Dynamic / BFPA8 < 1:
    相比全 BFPA8，节省约 45%-47% cycles。
```

---

## 4. 现有工作分析如何放入 Introduction

### 4.1 GNN Accelerators

可讨论对象：

```text
传统 GNN accelerator:
    优化 aggregation、sampling、sparse-dense execution。

局限:
    当 LLM encoder 占端到端绝大多数时间时，
    后端 GNN 加速无法解决主要瓶颈。
```

写法重点：

```text
不是说 GNN 加速器无用；
而是说在 LLM-for-GNN encoder-heavy setting 中，
主要成本前移到了文本 encoder。
```

### 4.2 Transformer / LLM Accelerators

可讨论对象：

```text
FlashAttention / IO-aware attention:
    优化 attention tile 数据流。

AWQ / QServe / PTQ:
    优化 LLM 量化。

BFP / mixed datatype / FP-INT GEMM:
    优化数值格式和矩阵单元。

PADE / FACT / bit-serial early termination:
    优化 attention 或 GEMM 内部计算路径。
```

局限：

```text
它们默认输入是一批独立 sequence/token rows；
缺少 graph-level route、reuse、propagation-risk 信息；
无法知道哪些 node embedding 可以 bypass，
也无法知道哪些 BFP refinement 对图任务更关键。
```

### 4.3 Cascaded GFM Accelerators: HEAT / CATOR

可讨论对象：

```text
HEAT:
    NPU-NDP heterogeneous architecture for Transformer-empowered GNN。

CATOR:
    topology-aware mixed precision and backend redundancy elimination。
```

本文差异：

```text
HEAT/CATOR 更关注 cascaded TF-GNN 两段共同执行和 degree-guided precision。

本文面向更重 LLM frontend：
    LLM encoder 占主导；
    GNN 后端更像提供 control signal；
    graph information 被用于 encoder bypass、safe fuzzy reuse 和 BFP block refinement。
```

避免碰撞的表述：

```text
不是简单“degree 指导量化”；
而是 graph signal 进入三层执行控制：
    1. 是否运行 encoder；
    2. fuzzy hit 是否修正/拒绝；
    3. miss-node encoder 内部哪些 activation blocks 需要 BFPA6 refinement。
```

---

## 5. 推荐的 Introduction 结构

### Paragraph 1: GFM 能力和代价

目标：

```text
介绍 TAG/GFM 依赖 LLM semantic understanding + GNN topology reasoning。
说明 LLM encoder 让节点文本 embedding 质量提升，但本地推理成本巨大。
```

可写要点：

```text
Graph-text workloads contain many nodes with rich textual attributes.
Modern GFM pipelines first encode node text with an LLM/Transformer,
then apply a graph head or GNN classifier.
This architecture improves semantic generalization but introduces a dominant encoder cost.
```

### Paragraph 2: 瓶颈转移

目标：

```text
强调与传统 GNN / 轻量 GFM 不同，现在瓶颈主要是 LLM encoder。
```

可写要点：

```text
Our profiling shows that the frontend LLM encoder dominates execution,
while the backend GNN contributes a small fraction of runtime.
Therefore, accelerating only graph propagation offers limited end-to-end gains.
```

### Paragraph 3: 机会

目标：

```text
提出中心观察：图后端信息可以反向指导前端 encoder。
```

可写要点：

```text
Although the GNN backend is lightweight, it exposes valuable control information:
node similarity, propagation risk, semantic boundary, and downstream sensitivity.
These signals are invisible to standalone Transformer accelerators.
```

### Paragraph 4: Challenge 1

主题：

```text
Encoder redundancy。
```

写法：

```text
Many node texts are semantically/topologically close.
Running a full LLM encoder for every node wastes computation.
However, ordinary caches cannot directly decide graph-text embedding reuse.
```

对应方案：

```text
SimHash + LRU/CAM embedding reuse。
```

### Paragraph 5: Challenge 2

主题：

```text
Reuse safety。
```

写法：

```text
Fuzzy hash hits are not always safe.
Graph propagation can amplify embedding errors on important nodes.
```

对应方案：

```text
TSER + Residual-Gate。
```

### Paragraph 6: Challenge 3

主题：

```text
Miss-node encoder cost。
```

写法：

```text
Even after reuse, miss nodes still enter the LLM encoder.
Uniform BFPA8 is expensive; uniform BFPA4 may lose accuracy on sensitive BFP blocks.
```

对应方案：

```text
Graph-aware Dynamic BFP Refinement NPU。
```

### Paragraph 7: System Integration

主题：

```text
NDP-NPU heterogeneous pipeline。
```

写法：

```text
NDP handles cache/CAM/routing/residual correction.
NPU handles compacted miss-node dynamic BFP encoding.
The route information changes the input stream seen by the encoder array.
```

### Paragraph 8: Contributions

建议压成三个主要贡献：

```text
1. Graph-aware encoder bypass using SimHash + LRU/CAM.

2. TSER-guided safe fuzzy reuse with residual correction and accept/reject.

3. Graph-aware dynamic BFP refinement NPU for miss-node LLM encoder execution.
```

Full-stack NDP-NPU pipeline 可以作为 contribution 4，也可以并入 contribution 3 的系统实现里。若篇幅和审稿定位允许，建议作为第 4 点：

```text
4. Full-stack implementation and evaluation across Cora/PubMed/Arxiv.
```

---

## 6. 推荐 Contributions 写法

### Contribution 1: Graph-Aware Encoder Bypass

```text
We propose a SimHash- and CAM-based graph-text embedding reuse frontend.
It maps node text and graph context into multi-head signatures,
uses LRU/CAM to locate reusable anchor embeddings,
and bypasses LLM encoder execution for high-confidence hits.
```

强调点：

```text
reuse object = node embedding
reuse condition = semantic hash + graph context + support
benefit = fewer encoder calls
```

### Contribution 2: TSER-Guided Residual Reuse

```text
We introduce TSER, a topology-aware semantic error risk score,
and a lightweight residual-gate mechanism for fuzzy matches.
TSER prevents graph-sensitive nodes from unsafe reuse,
while the residual adapter corrects medium-confidence hits instead of rejecting them all.
```

强调点：

```text
not just hash distance
graph propagation risk controls reuse safety
residual-gate is a middle path between direct reuse and full recompute
```

### Contribution 3: Graph-Aware Dynamic BFP NPU

```text
We design a progressive BFPA4/BFPA6 encoder datapath for miss nodes.
The array executes BFPA4 by default and selectively refines activation blocks to BFPA6
according to graph risk and BFP activation stress.
```

强调点：

```text
BFP format itself is not the novelty.
Novelty = graph risk enters block-level refinement decision inside the encoder NPU.
```

### Contribution 4: Full-Stack GFM Accelerator

```text
We integrate encoder bypass, safe fuzzy reuse, and dynamic BFP encoding into a full-stack NDP-NPU pipeline.
The NDP side performs cache/CAM/routing/residual correction,
while the NPU side processes only compacted miss nodes with graph-aware dynamic BFP.
```

强调点：

```text
front-end graph information changes both algorithmic routing and hardware execution.
```

---

## 7. 当前结果如何服务论文

### 7.1 前端结果用于证明 Challenge 1/2

| Dataset | T | Reuse | Direct | Residual | FullP8 Drop |
|---|---:|---:|---:|---:|---:|
| Cora | 31 | 39.0% | 18.5% | 20.5% | 1.56% |
| PubMed | 31 | 42.3% | 42.2% | 0.0% | 1.95% |
| Arxiv | 22 | 44.9%-46.2% | 19.8%-20.4% | 25.1%-25.8% | 2.02%-2.29% |

如何解释：

```text
前端可以 bypass 约 40%-46% 节点；
drop 控制在约 2% 左右；
不同数据集用 dataset-level T policy register，不需要改硬件。
```

### 7.2 后端结果用于证明 Challenge 3

Dynamic BFP array trace:

| Dataset | Refined Blocks | Effective Bits | Dynamic/BFPA6 | Dynamic/BFPA8 |
|---|---:|---:|---:|---:|
| Cora | 20.79% | 4.416 | 0.735x | 0.551x |
| PubMed | 13.88% | 4.278 | 0.713x | 0.535x |
| Arxiv | 21.46% | 4.429 | 0.737x | 0.553x |

如何解释：

```text
大多数 blocks 只需 BFPA4；
少量 high-priority blocks refine 到 BFPA6；
相对 full BFPA6/BFPA8 显著减少 mantissa compute。
```

### 7.3 full-stack 表用于证明系统闭环

建议主表固定以下对比：

```text
FullP8:
    reuse/residual 前端固定；
    miss nodes 全部 BFPA8。

AllP4:
    reuse/residual 前端固定；
    miss nodes 全部 BFPA4。

Dynamic Graph-BFP:
    reuse/residual 前端固定；
    miss nodes 默认 BFPA4；
    selected blocks refine to BFPA6。
```

这样能拆清：

```text
FullP8 Drop:
    前端 reuse/residual 误差。

AllP4 - FullP8:
    uniform BFPA4 后端额外误差。

Dynamic - AllP4:
    refinement 带来的精度恢复与额外计算。
```

---

## 8. 论文主线中应避免的表述

### 8.1 不要把 BFPA4 本身写成创新

不推荐：

```text
We apply BFPA4 to GFM.
```

推荐：

```text
We use BFPA4 as a low-cost base path and introduce graph-aware block refinement,
where graph risk determines which numerically stressed activation blocks receive BFPA6 correction.
```

### 8.2 不要把 Degree-guided precision 写成主创新

不推荐：

```text
Degree controls precision.
```

推荐：

```text
Graph risk participates in multi-level encoder control:
reuse safety, residual accept/reject, and BFP block refinement.
```

### 8.3 不要把 Residual-Gate 单独拔太高

Residual-Gate 是重要机制，但更适合归入 safe fuzzy reuse：

```text
TSER decides safety.
Residual-Gate repairs medium-confidence hits.
```

### 8.4 不要保留已验证较弱路径作为主线

以下可以放到 appendix / design exploration，不放 intro 主贡献：

```text
partial-depth encoder
token compaction
FFN channel gating
cross-row BFP packing
old predictor-free bit-plane early stop
```

---

## 9. One-Sentence Thesis

可以作为 introduction 末尾或 abstract 的核心句：

```text
This work turns backend graph information from a passive consumer of LLM embeddings
into an active control signal that bypasses redundant encoder calls,
guards fuzzy embedding reuse, and allocates BFP refinement effort inside the LLM encoder NPU.
```

中文含义：

```text
本文把图后端信息从“使用 LLM embedding 的消费者”，
变成“控制 LLM encoder 是否执行、如何复用、以及如何分配 NPU 计算精度的主动信号”。
```

---

## 10. Suggested Paper Title Directions

可以围绕以下关键词组合：

```text
Graph-Guided LLM Encoder Acceleration
Graph-Aware Encoder Bypass
Dynamic BFP Refinement
Text-Attributed Graph Inference
NDP-NPU Heterogeneous Execution
```

示例：

```text
Graph-Guided LLM Encoder Acceleration for Text-Attributed Graph Inference

GraphBFP: Graph-Aware Encoder Bypass and Dynamic BFP Refinement for GFM Inference

GFMAcc: Graph-Guided LLM Encoder Bypass and Dynamic BFP NPU for Text-Attributed Graphs
```

