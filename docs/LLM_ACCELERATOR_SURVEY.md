# Encoder / General Transformer Accelerator Survey

本文档聚焦 **encoder-only / 通用 Transformer / NPU 计算阵列** 相关的加速器论文，主要服务 GraphHopSimhash 后续的硬件架构设计：W4A8/W4A4 数值路径、FFN/channel gating、attention dataflow、near-memory / in-memory compute、低比特与压缩执行等。

为了让分类更清晰，decoder / serving / KV-cache / prefill-decode overlap / decode-stage MVM 等内容已经拆到：[LLM_DECODER_ACCELERATOR_SURVEY.md](./LLM_DECODER_ACCELERATOR_SURVEY.md)。当前文件只保留：

```text
1. Encoder / BERT / ViT / 通用 Transformer block 加速器
2. Attention / FFN / QKV 的动态稀疏与 exact dataflow
3. 低比特量化、outlier、scale alignment、dequant path
4. PIM / PNM / IMC / 3D / in-memory 或 near-core 数据流
5. 可迁移到 LLM-for-GNN encoder NPU 的通用设计思想
```

阅读时需要注意：部分论文原本面向 LLM prefill 或通用 Transformer，但只要其核心机制不是 decoder-only KV cache / serving scheduling，就保留在这里作为 encoder NPU 的参考。

---

## 1. FIGNA: 保持数值精度的 FP-INT GEMM

**发表信息**：HPCA 2024；Sungkyunkwan University 等团队；定位是面向 weight-only / FP-INT GEMM 的数值等价整数计算加速器。

### 核心问题

Weight-only quantization 中：

```text
weight = INT4 / INT8
activation = FP16 / BF16 / FP32
```

常规 GPU kernel 通常需要先把 INT weight 反量化成 FP，再用 FP MAC 运算。这带来：

```text
1. FP multiplier 面积/功耗高；
2. dequant overhead 高；
3. 直接 BFP/近似整数化又可能损失精度。
```

### 核心 idea

FIGNA 试图在不损失传统 FP-INT GEMM 数值结果的前提下，把主要 MAC 转成整数运算。

### 细粒度创新

**Integer-based dynamic prealignment**

对一个 block 内的 FP activation：

```text
1. 提取指数；
2. 找最大指数；
3. 对齐尾数；
4. 转成共享指数下的整数 mantissa。
```

之后和 INT weight 的乘加可用纯整数 MAC。

**Truncation theorem**

对齐后的 mantissa 可能很长。FIGNA 证明只需保留有限位宽即可保持和传统 FP 运算一致的数值结果。

**0-less INT format**

为避免 weight 为 0 时导致截断精度崩塌，提出 0-less signed integer 表示，使极端截断仍保持精度。

### 加速对象

```text
weight-only quantized GEMM
FP activation × INT weight
dequant-free compute
integer MAC replacement
```

### 核心价值

它的价值在于“数值等价”的低功耗计算路径，而不是靠任务精度容忍误差。

---

## 2. FAS-Trans: FFN 与 Attention 联合动态稀疏

**发表信息**：ICCAD 2024；复旦大学等团队；定位是同时挖掘 FFN 和 attention 稀疏性的 Transformer 加速器。

### 核心问题

很多 Transformer 稀疏加速器只看 self-attention，但在常规 token 长度下：

```text
QKV generation + FFN 往往占主要计算和能耗。
```

只优化 attention score 计算会出现木桶效应。

### 核心 idea

FAS-Trans 用预测-执行机制同时挖掘：

```text
1. QKV generation sparsity
2. attention top-k sparsity
3. FFN activation sparsity
4. approximate result reuse
```

### 细粒度创新

**AP / EC phase**

AP 阶段用 LBD / shift-add 低成本预测注意力和 FFN 结果分布；EC 阶段只对必要部分做精确计算。

**Cross-stage QKV pruning**

先预测 attention top-k，再反推哪些 Q/K/V 生成没有贡献，从源头减少 QKV linear layer 计算。

**Approximate result reuse**

传统 predictor 只产生 mask，预测值本身丢弃。FAS-Trans 把 AP 阶段的近似结果作为 EC 阶段的部分积，EC 只补残差。

**FFN sparsity**

利用 GELU 对负输入输出接近 0 的性质：

```text
FC1 预测为负 -> FC2 对应项低精度或跳过
```

### 加速对象

```text
QKV generation
attention score
FFN FC1/FC2
predictor reuse
```

### 核心价值

它代表了“跨阶段稀疏预测”：不是只跳 attention，而是从预测结果反向影响前后多个模块。

---

## 3. Ayaka: 低秩估计与异构数据流 Transformer 加速器

**发表信息**：IEEE JSSC 2024；清华大学等团队；定位是面向 Transformer 的低秩估计、注意力预测和异构数据流加速器。

### 核心问题

Transformer 的瓶颈随 token 长度变化：

```text
短序列: QKV / FFN linear 更重要
长序列: attention score 更重要
```

单一数据流难以适配不同矩阵形状和稀疏模式。

### 核心 idea

Ayaka 用随机投影提前估计 attention 稀疏性，并用异构数据流 PE 支持不同阶段。

### 细粒度创新

**RPAS**

Random Projection Attention Speculation 在完整 QKV 生成前预测 attention sparsity，然后反向跳过不重要 token 的 Q/K/V 生成。

**IPRM**

利用 softmax 平移不变性，在 `QK^T` 中检测并去除相同部分积，降低 bit-level 冗余计算。

**HDPE**

Heterogeneous Dataflow PE 可在不同 stationary dataflow 间切换，以适配线性层、attention、不同稀疏率。

### 加速对象

```text
QKV generation
attention score
bit-level multiply redundancy
dataflow utilization
```

### 核心价值

Ayaka 的重点是“预测 + 数据流自适应”，比单纯 attention sparsity 更全面。

---

## 4. STAR: Cross-Stage Tiling 的 Sparse Attention Spatial Architecture

**发表信息**：MICRO 2024 / arXiv 扩展线；清华大学、上海交通大学等团队；定位是 sparse attention 的 cross-stage tiling spatial architecture。

### 核心问题

长序列 LLM 中动态 sparse attention 常有三个孤立阶段：

```text
prediction -> sorting/top-k -> exact attention
```

阶段之间不协同会带来：

```text
1. predictor 开销大；
2. top-k sorting 慢；
3. 中间结果存储和访存大；
4. 长序列空间架构扩展困难。
```

### 核心 idea

STAR 通过 cross-stage coordinated tiling，让预测、排序、精确计算细粒度流水化，并扩展到 spatial multi-core 架构。

### 细粒度创新

**DLZS**

Differential Leading-Zero Summation 用 log-domain / leading-zero 近似替代乘法预测 attention 分数。

**SADS**

Sphere-search Aided Distributed Sorting 把长行拆成子段排序，并用 softmax 特性提前排除不可能进入 top-k 的元素。

**SU-FA**

Sorted-Updating FlashAttention 利用排序结果简化块间 max/softmax 更新，把 top-k 与 FlashAttention-style tiling 结合。

**Spatial extension**

针对超长序列，多核空间架构用 DRAttention dataflow 和 MRCA 通信策略扩展。

### 加速对象

```text
long-context attention
top-k sorting
softmax tiling
multi-core sparse attention
```

### 核心价值

STAR 是从算法到 spatial architecture 的长序列 sparse attention 全链路设计。

---

## 5. SOFA: Compute-Memory Optimized Sparse Attention Accelerator

**发表信息**：MICRO 2024；清华大学、上海交通大学等团队；定位是面向 LTPP sparse attention 的 compute-memory 协同 tiling 加速器。

### 核心问题

动态 sparse attention 的预测和排序可能省了计算，却引入大量中间结果访存，导致 memory wall。

### 核心 idea

SOFA 更强调单核/局部加速器内部的 compute-memory co-optimization：

```text
tile-level prediction -> sorting -> KV generation -> exact compute
```

尽量让一个 tile 在片上走完整流程。

### 细粒度创新

**Fine-grained tiled pipeline**

把预测、排序、精确计算按 tile 串成流水线，缩短中间结果生命周期，减少 DRAM 访问。

**RASS**

Reuse-Aware Schedule Scheme 针对不同 query 共享 K/V 的情况进行乱序重排，把同一 K/V 的使用聚集，提高片上复用。

**Configurable LZ / sort hardware**

硬件支持不同精度的 leading-zero 编码与变长输入排序。

### 加速对象

```text
sparse attention memory traffic
KV on-chip reuse
tile-level pipeline
```

### 核心价值

SOFA 的亮点是把动态稀疏从“算得少”推进到“搬得少”。

---

## 6. SALO2 / Static-Dynamic Sparse Attention Co-Design

**发表信息**：SALO / SALO2 sparse attention accelerator 论文线；上海交通大学等团队；定位是同时支持 static 与 dynamic sparse attention 的软硬件协同设计。

### 核心问题

静态稀疏 attention 硬件友好但不灵活；动态稀疏 attention 自适应强但开销和不规则访存大。

### 核心 idea

在同一套硬件上同时支持静态和动态 sparse attention，把动态 pattern 映射成硬件友好的静态 pattern 组合。

### 细粒度创新

**Static pattern reordering**

对 sliding window、global、random 等静态模式做数据重排和切割，映射到统一硬件格式。

**Dynamic pattern matching**

用低精度近似 matmul 快速得到粗略 attention score，再把高分区域拟合成硬件支持的 local/global/random pattern。

**Diagonal dataflow**

Query 水平流动，Key/Value 对角线流动，使相邻 query 自然复用重叠窗口的 K/V。

### 加速对象

```text
static sparse attention
dynamic sparse attention
pattern mapping
K/V reuse
```

### 核心价值

这类工作说明：硬件不一定要支持任意不规则稀疏，可以把动态稀疏约束到 pattern library。

---

## 7. ESACT: 基于 Local Similarity 的端到端稀疏 Transformer

**发表信息**：arXiv 2025；Hongxiang Liu、Zhifang Deng、Tong Pu、Shengli Lu 等作者团队；定位是利用 local similarity 做端到端 sparse Transformer 加速。

### 核心问题

很多 sparse attention 工作只关注 attention matrix 内部稀疏，预测开销大，且不能端到端覆盖 QKV 和 FFN。

### 核心 idea

利用局部相似性和低成本 HLog 量化，在局部窗口内预测稀疏性，并端到端作用于 QKV、attention、FFN。

### 细粒度创新

**HLog-based SPLS**

用 HybridLog 量化保留局部相似性分布，以较低成本预测 attention。

**Bit-level prediction**

用位级相关性把乘法替换为 shift-add。

**Progressive generation**

让预测和 QKV 实际生成重叠，隐藏 predictor latency。

**Dynamic allocation**

多头 sparsity 不同会负载不均，ESACT 动态分配计算路径提高 PE 利用率。

### 加速对象

```text
QKV
attention
FFN
predictor latency
multi-head load balance
```

### 核心价值

ESACT 是端到端 sparse Transformer 思路，强调低开销 predictor 和 pipeline overlap。

---

## 8. IMCsim: 面向深度学习的全系统 IMC 模拟框架

**发表信息**：DAC 2025；UIUC 等团队；定位是面向 LLM / DiT 等 workload 的 full-system IMC simulator。

### 核心问题

IMC 设计缺少能支持 LLM/DiT 等大模型的全系统、可编程、周期级模拟工具。

### 核心 idea

提供可扩展 IMC simulator，把模型算子映射到自定义 ISA 和 micro-ops，支持 SRAM/eNVM/analog/digital 多种 IMC。

### 细粒度创新

**Full-system modeling**

不仅模拟 matmul macro，还模拟指令、缓存、主存交互和执行周期。

**Hardware library**

支持 SRAM digital/analog IMC、MRAM/ReRAM 等多介质。

**SNR model**

对 analog IMC 建模噪声和计算精度。

**Design-space exploration**

可用于分析 LLM/DiT 在不同 bank / array size 下的真实利用率。

### 加速对象

它本身不是一个加速器，而是用于 IMC-based accelerator 的设计空间探索和验证。

---

## 9. Balanced Systolic Array Attention Accelerator

**发表信息**：DAC 2025；作者团队围绕 balanced systolic array 与 multi-row interleaved attention dataflow 展开；定位是 attention 的 exact dataflow 与高利用率阵列设计。

### 核心问题

传统 attention 加速器中：

```text
1. 常规 systolic array 难以同时兼顾复用、寄存器开销和利用率；
2. QK^T -> softmax -> PV 的分阶段执行产生大量中间矩阵读写。
```

### 核心 idea

用 balanced systolic array 和 multi-row interleaved ordering 优化 attention 数据流。

### 细粒度创新

**BSA**

设计 inner-outer mixed product 的阵列结构，在外积复用和内积累加器开销间平衡。

**Array shape modeling**

数学推导阵列形状和 PE 乘法器配置，优化 SRAM 与累加能耗。

**Booth encoder sharing**

外积数据共享使外围 Booth 编码可复用，减少编码器数量。

**Multi-row interleaving**

不完整物化 `QK^T` 和 `P` 矩阵，而是行粒度交织执行 `QK^T` 和 `PV`，快速消费 P。

### 加速对象

```text
attention matrix multiplication
softmax intermediate storage
systolic array utilization
```

### 核心价值

这是偏 exact dataflow 的硬件优化，不依赖模型近似或任务容忍度。

---

## 10. DESA: Dataflow Efficient Systolic Array for Transformers

**发表信息**：IEEE Transactions on Computers 2025；Z. Wang、H. Fan、G. He 等作者团队；定位是面向 encoder-only Transformer 的 dataflow-efficient systolic array。

### 核心问题

Transformer end-to-end inference 包含多种矩阵形状和非线性算子。Softmax / LayerNorm 的依赖容易造成阵列停顿和中间结果搬运。

### 核心 idea

DESA 用混合数据流 systolic array、算子解耦和 attention 融合来提高端到端利用率。

### 细粒度创新

**Nonlinear decoupling**

把 softmax/layernorm 拆为统计特征提取与归一化，隐藏归一化延迟。

**Fused attention mapping**

把 QKV generation 与 attention process 流水融合，降低 `O(N^2)` 中间存储。

**Hybrid stationary SA**

运行时在 weight-stationary / output-stationary 间切换，适配不同矩阵。

**Embedded VPU/IPU**

把 transpose、post-processing 等向量操作嵌入阵列附近，减少显式通信。

### 加速对象

```text
end-to-end Transformer encoder/block
softmax/layernorm dependency
attention dataflow
matrix workload diversity
```

---

## 11. H3D-Transformer: 异构 3D Transformer 平台

**发表信息**：ACM TODAES 2024；Georgia Institute of Technology 等团队；定位是面向 edge Transformer 的 heterogeneous 3D / CIM + digital TPU 平台。

### 核心问题

边缘端部署 Transformer 面临参数容量、内存墙和功耗墙。

### 核心 idea

用异构 3D/2.5D 集成，把不同计算范式映射到不同 die：

```text
dense MM -> FeFET D-CIM
sparse attention -> digital TPU
approximate score -> SRAM CIM
```

### 细粒度创新

**Heterogeneous integration**

逻辑层、SRAM CIM、FeFET D-CIM 多层堆叠，提高片上存储密度。

**Workload-specific assignment**

稠密 linear/FFN 用高密度 CIM；稀疏 attention 用数字 TPU 跳零。

**Approximate top-k**

先用低精度 SRAM CIM 估计 attention score，筛选 top-k，再交给数字 TPU 精算。

### 加速对象

```text
edge Transformer
dense linear layer
sparse attention
on-chip model residency
```

### 核心价值

它代表异构集成路线：不是单一阵列，而是根据算子类型分配不同计算介质。

---

## 12. PADE: Predictor-Free Sparse Attention

**发表信息**：arXiv 2025 / HPCA 2026 论文线；清华大学、上海交通大学等团队；定位是 predictor-free sparse attention accelerator。

### 核心问题

随着低位计算执行器越来越便宜，单独的稀疏 predictor 可能占据大量能耗，甚至抵消稀疏收益。

### 核心 idea

PADE 放弃显式 predictor，用 bit-serial 计算过程本身实现 early termination 和 result reuse。

### 细粒度创新

**BSF**

Bit-serial stage fusion 从 MSB 到 LSB 逐步计算 `QK^T`。当高位部分已经足以判定某 key 不可能重要，就停止后续低位计算。

**BUI-GF**

Bit Uncertainty Interval 计算当前 partial sum 的理论上下界，只有当上界不可能达到阈值时才安全丢弃。

**BS-OOE**

Bit-level out-of-order execution 在等待某个 token 低位数据时处理其他 token 高位，隐藏访存。

**ISTA**

把 bit-level early termination 与 FlashAttention-like tiling 结合，避免全行物化。

### 加速对象

```text
sparse attention
predictor overhead
bit-serial early termination
safe pruning bound
```

### 核心价值

PADE 的重要性在于“预测器无关”的安全早停，精度控制比纯 heuristic predictor 更强。

---

## 13. Energon / PNM Transformer Throughput Maximization

**发表信息**：Energon / PNM Transformer inference 论文线；作者团队聚焦 processing-near-memory 的 Transformer throughput mapping；定位是 PNM/NDP 路线的系统建模与映射。

### 核心问题

Transformer 推理在传统 CPU/GPU 上常受内存墙限制，尤其权重大、复用低时。

### 核心 idea

在 Processing-Near-Memory 架构上做映射和调度，以最大化吞吐。

### 细粒度创新

**XY-aligned pipeline mapping**

根据 PNM 的物理布局做流水线划分，减少跨节点数据迁移。

**Throughput-oriented scheduling**

同时支持离线 batch 和在线延迟敏感场景的任务划分。

**PNM-aware mapping**

尽量把中间数据保留在近存节点本地缓存。

### 加速对象

```text
Transformer throughput
near-memory mapping
data movement
```

### 核心价值

它强调“映射框架”和系统吞吐，而不是单个算子的稀疏/量化。

---

## 14. Low-Power ViT Accelerator with Hardware-Aware Pruning

**发表信息**：近年 edge ViT / Transformer accelerator 方向论文；作者团队聚焦 hardware-aware pruning 与 low-power dataflow；定位是 FFN/token pruning 路线的对照。

### 核心问题

短 token ViT 中，FFN 往往比 attention 更关键。只优化 attention 无法获得端到端收益。

### 核心 idea

通过硬件感知 pruning、简化激活函数和优化 dataflow 降低 ViT 推理功耗。

### 细粒度创新

**Dynamic token pruning**

跳过不重要 token 的后续计算。

**GELU -> ReLU**

替换激活函数简化硬件逻辑。

**Output-oriented row dataflow**

避免矩阵转置和复杂寻址。

### 加速对象

```text
ViT / encoder-style Transformer
FFN-heavy workload
edge low-power inference
```

### 核心价值

它再次说明短序列 encoder 里 FFN 是主要优化对象。

---

## 15. BETA: Bit-Grained Transformer Attention Accelerator

**发表信息**：IEEE TCAS-II 2025；清华大学团队；定位是 bit-grained attention early termination accelerator。

### 核心问题

显式 attention sparsity predictor 可能能耗高，且误判会影响精度。

### 核心 idea

用 bit-grained multi-round filtering 在计算过程中逐步过滤不重要 token，实现 early termination。

### 细粒度创新

**BMF**

Bit-grained multi-round filtering 用位平面信息逐轮缩小候选范围。

**MTS**

Max-value-driven threshold 根据当前最大值自适应调整早停阈值。

**BOOE**

Bit-level out-of-order execution 解决不同 token 早停程度不同导致的负载不均。

### 加速对象

```text
attention score computation
predictor-free early termination
bit-level sparse execution
```

### 核心价值

BETA 和 PADE 类似，代表 bit-level bounded / early-stop attention 加速路线。

# 顶会新论文补充：DAC / ISCA / MICRO / HPCA / ASPLOS

这一节保留近年顶会中更贴近 encoder / 通用 Transformer / NPU 数据流的工作。decoder-only 的 KV cache、serving 调度、decode MVM 和 prefill-decode overlap 已移动到 decoder 专门文档。

## 16. Tender: Tensor Decomposition + Runtime Requantization

**发表信息**：ISCA 2024；作者团队围绕 LLM low-bit tensor decomposition 与 tensor compute hardware；定位是低比特 scale alignment / runtime requantization。

来源：[ISCA 2024 / arXiv](https://arxiv.org/abs/2406.12930)

### 核心问题

低比特 LLM 推理不仅是把权重量化到 INT4/INT8。难点在于：

```text
1. outlier 破坏低比特量化精度；
2. 多个低比特 partial sum 合并时需要 dequant / requant；
3. 如果硬件路径频繁回到高精度，低比特计算收益会被抵消。
```

### 核心 Idea

Tender 用 tensor decomposition 把难量化矩阵拆成若干低比特子矩阵，并让不同子矩阵的 scale factor 之间保持 power-of-two 关系。

这样做的关键好处是：

```text
partial sum 对齐可以用 shift 完成，
不需要显式 dequantize -> accumulate -> requantize。
```

### 硬件意义

Tender 不是另造一套全新阵列，而是在已有 tensor compute pipeline 上加少量 scale/shift 支持，让低比特矩阵乘可以更自然地落到硬件里。

### 对我们场景的启发

它说明 W4A8/W4A4 的难点往往不是 MAC 本身，而是：

```text
scale alignment
outlier handling
partial sum accumulation
```

如果后续设计 Graph-aware W4A8 NPU，应该把 graph routing 和数值路径分开：routing 决定谁走近似路径，数值路径必须保证 scale/outlier 的硬件开销足够低。

---

## 17. LLMCompass: LLM Inference Hardware Design Model

**发表信息**：ISCA 2024；Princeton University 团队；定位是 LLM inference accelerator 的 cost / performance / area design-space model。

来源：[ISCA 2024 / Princeton page](https://collaborate.princeton.edu/en/publications/llmcompass-enabling-efficient-hardware-design-for-large-language-)

### 核心问题

LLM 硬件设计空间巨大：array 规模、buffer 容量、memory bandwidth、parallelism、operator mapping 都会影响性能。没有可靠 cost model 时，很多架构探索只能凭经验。

### 核心 Idea

LLMCompass 是面向 LLM inference 的硬件设计建模框架，建模 latency、area、memory、operator mapping，并用真实硬件校准模型误差。

### 核心价值

它不是一个单点 accelerator，而是设计空间探索工具：

```text
给定模型和硬件参数，
预测不同 operator / layer / mapping 的性能和面积代价。
```

### 对我们场景的启发

如果要把 GraphHopSimhash 讲成体系结构论文，最终也需要一个类似的 cost model：

```text
P0 exact reuse      cost = cache read
P1 residual reuse   cost = low-rank adapter
P2 gated W4A8       cost = partial FFN/channel/token compute
P3 full W4A8        cost = full encoder
```

这比只报 accuracy/drop 更像 HPCA/ISCA 的完整架构评估。

---

## 18. LUT Tensor Core: LUT-Based Low-Bit LLM Inference

**发表信息**：ISCA 2025；Imperial College London、Microsoft Research 等团队；定位是 LUT-based low-bit LLM inference tensor core。

来源：[ISCA 2025 project page](https://hamerlate.github.io/publications/old/LUT_Tensor_Core/)

### 核心问题

低比特 LLM 推理不一定要完全沿用传统乘法器路径。极低比特下，很多乘法可以转化为查表、编码、重排和简单累加。

### 核心 Idea

LUT Tensor Core 是 soft/hardware co-design：用 LUT-based 方法支持 low-bit LLM inference，把部分低比特乘法/组合计算转化为 lookup-friendly execution。

### 硬件意义

```text
传统 Tensor Core:
    低比特矩阵乘仍围绕 multiplier / MAC 展开。

LUT Tensor Core:
    更重视 lookup table、bit packing、低比特组合复用。
```

### 对我们场景的启发

如果我们后面做 FFN channel gating 或 graph-guided low-bit path，LUT/bit-serial 方向可以作为替代阵列方案：不一定只设计 W4A8 MAC，也可以设计“低比特查表 + 小规模精确补偿”的混合路径。

---

## 19. FuseMax: Extended Einsums for Attention Accelerator Design

**发表信息**：MICRO 2024；UIUC / MIT 等团队；定位是基于 extended einsum 的 attention accelerator dataflow。

来源：[MICRO 2024 / arXiv](https://arxiv.org/abs/2406.10491)

### 核心问题

Attention 的高效实现不只是少算，还包括：

```text
1. 避免 materialize attention matrix；
2. 避免 off-chip traffic bottleneck；
3. 保持 PE utilization 接近满载；
4. 降低 on-chip buffer 对 sequence length 的依赖。
```

### 核心 Idea

FuseMax 使用 extended einsum 表达和重组 attention 计算，把传统分裂的 attention/softmax/value 计算重排成更适合 accelerator 的 fused dataflow。

### 细粒度创新

```text
把 attention kernel 当成 einsum graph 优化问题：
    哪些维度该 tile；
    哪些中间结果不需要显式存；
    哪些 reduction 可以和后续算子融合；
    如何让 compute utilization 接近 100%。
```

### 对我们场景的启发

FuseMax 是 exact dataflow 优化，适合做我们 full W4A8 encoder path 的 baseline。它不能替代 GraphHopSimhash 的 reuse，但可以作为：

```text
P3 full encoder 的底层精确执行引擎。
```

---

## 20. DECA: Near-Core LLM Decompression Accelerator

**发表信息**：MICRO 2025；Intel、UIUC 等团队；定位是 near-core dequantization / desparsification accelerator。

来源：[MICRO 2025 / UIUC page](https://experts.illinois.edu/en/publications/deca-a-near-core-llm-decompression-accelerator-grounded-on-a-3d-r/)

### 核心问题

LLM 权重常以 quantized / sparsified 格式存储，但进入 GEMM engine 前要先：

```text
dequantize
de-sparsify
repack tile
feed matrix engine
```

如果这些操作由软件 vector kernel 完成，compressed GEMM 的收益会被解压和格式转换吃掉。

### 核心 Idea

DECA 在 core 附近加入 ML-model decompression accelerator，把 tile 级 dequantization / de-sparsification 从 CPU vector 单元卸载出去，直接生成 GEMM engine 可消费的 tile。

### 细粒度创新

```text
1. Roof-Surface 3D performance model：
   同时建模 memory、vector unit、matrix engine。

2. near-core decompression accelerator：
   负责 compressed tile -> compute-ready tile。

3. out-of-order invocation ISA：
   允许 core compute 和 decompression overlap。
```

### 对我们场景的启发

如果我们采用 W4A8 / W4A4 embedding pool 或在线 W4A8 encoder，DECA 提醒我们：硬件论文不能只写“量化省了多少 bit”，还要写清楚：

```text
scale / zero-point / sparsity metadata 如何取；
dequant/repack 在哪里做；
是否会阻塞 tensor array。
```

---

## 21. LLM.265 / VcLLM: Video Codecs as Tensor Codecs

**发表信息**：MICRO 2025；Duke University、Carnegie Mellon University 等团队；定位是把 video codec 思路迁移到 LLM tensor compression。

来源：[MICRO 2025 project page](https://fact-lab.hkust.edu.hk/publications/conference-paper/2025/xu-2025-llm/)

### 核心问题

LLM 权重和中间张量越来越大，传统量化主要关注数值精度，但没有充分利用张量在空间/维度上的局部相关性。

### 核心 Idea

LLM.265 / VcLLM 把 video codec 的思想迁移到 tensor compression：把 tensor 看成具有局部结构的数据块，用成熟视频压缩中的预测、变换、编码思想压缩 LLM 张量。

### 硬件意义

这类工作代表一个新方向：

```text
不是只设计更快 MAC，
而是把 tensor 当作可压缩信号，
用 codec-style pipeline 降低内存和带宽。
```

### 对我们场景的启发

Graph-text workloads 里的 embedding / cheap feature / neighbor context 也有局部相关性。未来可以考虑：

```text
hash bucket 内 embedding delta compression
reuse anchor + residual 的压缩存储
graph cluster 内 tensor codec
```

---

## 22. MCBP: Bit-Slice Sparsity and Repetitiveness

**发表信息**：MICRO 2025 / arXiv preprint；清华大学、上海交通大学等团队；定位是 bit-slice sparsity / repetitiveness LLM inference accelerator。

来源：[arXiv / MICRO 2025 preprint](https://arxiv.org/abs/2509.10372)

### 核心问题

很多 LLM accelerator 在 value-level 做稀疏或量化，但忽略了 bit-slice 层面的重复和稀疏。低比特推理里，bit-plane / bit-slice 的分布本身就有结构。

### 核心 Idea

MCBP 从 bit-slice 级别同时挖掘：

```text
1. repetitiveness：重复 bit-slice 计算可以复用；
2. sparsity：高位 bit-slice 中零/稀疏结构可以编码；
3. progressive prediction：逐 bit 预测减少 KV cache access。
```

### 细粒度创新

```text
BRCR:
    利用 bit-slice 重复性减少 GEMM 计算。

BSTC:
    利用高位 bit-slice 稀疏降低权重访问。

BGPP:
    bit-grained progressive prediction 减少 KV cache 访存。
```

### 对我们场景的启发

这和我们讨论的 FFN/channel gating 是同一个大方向：从粗粒度节点路由进一步下沉到 bit/channel/tile 层级，让 NPU 本体也能受益。

---

## 23. Efficient Transformer Inference with Statically Structured Sparse Attention

**发表信息**：DAC 2023；NVIDIA Research、UC Berkeley 等团队；定位是硬件友好的 static structured sparse attention。

来源：[DAC 2023 / NVIDIA Research](https://research.nvidia.com/publication/2023-07_efficient-transformer-inference-statically-structured-sparse-attention)

### 核心问题

动态稀疏 attention 虽然灵活，但硬件不规则；完全 dense attention 又浪费。静态稀疏如果设计得好，能在硬件上获得更稳定收益。

### 核心 Idea

设计 static structured sparse attention masks，把 attention matrix 切成硬件友好的 dense regions，只计算有意义区域，跳过其他区域。

### 细粒度创新

```text
1. 静态结构化 mask：
   稀疏模式对硬件友好，减少不规则索引。

2. entropy-aware finetuning：
   训练时鼓励 attention 稀疏，同时保持任务精度。

3. accelerator extension：
   在 dense accelerator 上加少量结构稀疏支持。
```

### 对我们场景的启发

它说明“稀疏模式硬件友好性”非常重要。Graph-aware routing 也不能只追求最高 sparsity/reuse；必须让路径规则足够简单：

```text
node route buckets
fixed FFN channel groups
structured residual adapter
```

---

## 24. TF-MVP: Mixed-Length Vector Pruning Transformer Accelerator

**发表信息**：DAC 2023；POSTECH、Naver Cloud 等团队；定位是 mixed-length vector pruning 与 reconfigurable PE transformer accelerator。

来源：[DAC 2023 / KAIST page](https://pure.kaist.ac.kr/en/publications/tf-mvp-novel-sparsity-aware-transformer-accelerator-with-mixed-le)

### 核心问题

Transformer 剪枝后的稀疏模式常常不规则，导致硬件利用率下降。单一向量长度的 pruning 不能很好匹配不同层的稀疏方向和大小。

### 核心 Idea

TF-MVP 提出 mixed-length vector pruning，根据不同层的 pruning pattern 选择不同 vector granularity，并设计 reconfigurable PE 结构支持这些模式。

### 细粒度创新

```text
direction strength:
    分析每层 pruning pattern 的主要方向和大小。

mixed-length vector pruning:
    让 pruning pattern 更硬件友好。

reconfigurable PE:
    支持不同长度 vector sparse execution。
```

### 对我们场景的启发

如果后续做 graph-guided FFN channel gating，不要做完全 unstructured channel mask。更适合硬件的是：

```text
group-level channel gating
fixed group size
少数可重配置 group shape
```

---

## 25. APTQ: Attention-Aware Mixed-Precision PTQ

**发表信息**：DAC 2024；Southern University of Science and Technology、University of Hong Kong 等团队；定位是 attention-aware post-training mixed-precision quantization。

来源：[DAC 2024 / arXiv](https://arxiv.org/abs/2402.14866)

### 核心问题

传统 post-training quantization 多看 weight Hessian 或 layer sensitivity，但 LLM 中 attention output 的非线性影响会传到后续层，单看权重误差不够。

### 核心 Idea

APTQ 用 Hessian trace 建模敏感度，同时考虑 attention output 对整体模型的非线性影响，做 mixed-precision quantization。

### 加速对象

```text
LLM post-training mixed precision
layer/weight sensitivity estimation
attention-aware precision assignment
```

### 对我们场景的启发

APTQ 说明“精度分配”必须看 downstream sensitivity。我们在 graph-text encoder 中类似地发现：

```text
量化路由不能只看 embedding L2 error；
还要看 degree / propagation risk / GNN 分类边界。
```

---

## 26. OPAL: Outlier-Preserved Microscaling Quantization

**发表信息**：arXiv 2024；KAIST 等团队；定位是 outlier-preserved microscaling quantization accelerator。

来源：[arXiv](https://arxiv.org/abs/2409.05902)

### 核心问题

LLM 低比特量化的主要阻碍之一是 activation outlier。只量化 weight 不够，activation 也要低比特，才可能真正降低 compute 和 bandwidth。

### 核心 Idea

OPAL 做 outlier-preserved microscaling quantization：普通值走 microscaling low-bit path，outlier 用专门机制保留，避免 outlier 被低比特 scale 抹平。

### 硬件意义

```text
main path:
    low-bit microscaling compute

outlier path:
    preserve rare large values

最终目标:
    兼顾 activation quantization 和模型精度。
```

### 对我们场景的启发

这和我们目前 W4A8/W4A4 的经验一致：W4A4 出问题通常不是平均误差，而是少量节点/维度/通道的异常误差。硬件上必须给 outlier 或 high-risk channel 留一条保护路径。

---

## 27. PIMoE: NPU-PIM for MoE Transformer

**发表信息**：DAC 2025；作者团队围绕 NPU-PIM heterogeneous MoE Transformer deployment；定位是 throttle-aware NPU/PIM offloading。

来源：[DAC 2025 / DOI record](https://colab.ws/articles/10.1109%2Fdac63849.2025.11132528)

### 核心问题

MoE Transformer 的 expert 激活稀疏，但带来两个硬件问题：

```text
1. expert workload 不均衡；
2. NPU 和 PIM 之间 sparse data layout 不匹配；
3. 数据搬运和调度开销可能抵消 MoE 稀疏收益。
```

### 核心 Idea

PIMoE 用 NPU + PIM 异构系统执行 MoE Transformer，并提出 throttle-aware task offloading，把任务在 NPU/PIM 之间动态分配。

### 细粒度创新

```text
throttle-aware offloading:
    根据 NPU/PIM 当前瓶颈决定任务分配。

near-memory-controller data condenser:
    解决 sparse data layout mismatch，提高搬运效率。
```

### 对我们场景的启发

它和我们的层级执行路线很接近：不是所有节点/算子都走同一个引擎，而是根据风险、稀疏、负载，把任务分配给不同硬件路径。

---

## 28. Anda: Variable-Length Grouped Activation Format

**发表信息**：HPCA 2025；南京大学、KU Leuven 等团队；定位是 variable-length grouped activation data format for LLM inference。

来源：[HPCA 2025 / arXiv](https://arxiv.org/abs/2411.15982)

### 核心问题

LLM activation 的数值范围和精度需求不是固定的。统一位宽会浪费：简单 activation 给太多 bit，困难 activation 又可能精度不够。

### 核心 Idea

Anda 提出 variable-length grouped activation data format，用 group-shared exponent 和动态 mantissa bit allocation 表示 activation。

### 硬件意义

```text
固定 INT/FP 格式:
    简单，但不能适应不同 activation 分布。

variable-length grouped format:
    让不同 group 使用不同有效精度，
    在压缩和精度之间做更细粒度权衡。
```

### 对我们场景的启发

Graph-aware FFN gating / mixed precision 不一定只靠“跳过通道”，也可以做：

```text
high-risk nodes/channels -> longer mantissa
low-risk nodes/channels  -> shorter mantissa
```

这比 W4A8/W4A4 二选一更细。

---

## 29. llm.npu: Fast On-Device LLM Inference with NPUs

**发表信息**：ASPLOS 2025；Peking University 等团队；定位是面向移动端 NPU 的 LLM inference offloading / hot-channel 管理。

来源：[ASPLOS 2025 PDF](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf)

### 核心问题

移动端 NPU 对 MatMul 友好，但 LLM 里还有 LayerNorm、Attention、outlier channel、数据同步等不适合 NPU 的部分。很多系统只把部分 MatMul offload 到 NPU，结果 CPU/GPU/NPU 之间同步和数据复制成为瓶颈。

### 核心 Idea

llm.npu 目标是让 on-device NPU 更完整地承担 LLM prefill。它关注：

```text
1. float operators 如何避免拖慢 NPU critical path；
2. hot/outlier channels 如何单独处理；
3. CPU/GPU/NPU 之间如何减少同步与重复权重存储。
```

### 对我们场景的启发

这篇对我们最直接：如果要写 Graph-aware encoder NPU，不能只说 W4A8 MatMul。需要回答：

```text
LayerNorm / pooling / normalization 怎么做？
residual adapter 放在哪里？
outlier channel 是否保留高精度？
hash/reuse engine 和 W4A8 array 如何同步？
```

---

---

---

# 分类总结

## A. Encoder / 通用 Transformer 主线

当前主文件保留的论文大致分成六类：

```text
1. Attention / FFN / QKV 联合优化
   FAS-Trans, Ayaka, ESACT, FACT-like eager prediction, TF-MVP

2. Sparse attention / early termination / exact attention dataflow
   STAR, SOFA, SALO2, PADE, BETA, FuseMax, Balanced SA, DESA

3. 低比特数值路径与量化硬件
   FIGNA, Tender, LUT Tensor Core, APTQ, OPAL, Anda, DECA

4. Tensor / weight / activation 压缩
   LLM.265 / VcLLM, MCBP

5. PIM / PNM / IMC / 3D / memory-side compute
   IMCsim, H3D-Transformer, PIMoE, Energon

6. 设计空间建模与 NPU 落地参考
   LLMCompass, llm.npu
```

## B. 对 GraphHopSimhash Encoder NPU 最相关的点

```text
1. FFN/channel gating：
   Graph/risk 信息可以指导哪些节点、哪些 FFN group 走轻量路径。

2. W4A8/W4A4 数值路径：
   需要关注 scale alignment、outlier、dequant/repack，而不只是 MAC 位宽。

3. Exact attention dataflow：
   Full encoder path 应该用 FlashAttention/FuseMax/DESA/Balanced-SA 这类精确数据流作为底层执行方式。

4. Predictor / early termination：
   FACT/FAS-Trans/PADE/BETA 的关键启发是：预测必须便宜，且最好能复用预测阶段的部分结果。

5. Memory hierarchy：
   hash reuse cache、residual adapter、W4A8 encoder、FFN-gated encoder 应该被建模成多路径层级执行。
```

## C. 已拆出的 Decoder / Serving 内容

以下内容已经移动到 [LLM_DECODER_ACCELERATOR_SURVEY.md](./LLM_DECODER_ACCELERATOR_SURVEY.md)：

```text
1. LoRA adapter serving / heterogeneous GPU serving / thermal-power scheduling
2. KV cache quantization / KV cache pruning / sparsity-aware KV placement
3. decode-stage MVM / in-flash GEMV / PIM-NDP LLM serving
4. prefill-decode overlap attention kernel
5. decoder-oriented FPGA mapping / long-context LLM serving
```

这样主文件可以专注回答：**如果 LLM 是 graph-text encoder，NPU 本体还能怎么加速？**
