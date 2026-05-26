# Recent LLM / Transformer Accelerator Survey

本文档基于当前整理的多篇 Transformer / LLM 加速器论文，重点分析这些加速器本身的核心 idea、技术挑战、细粒度创新点，以及它们大致可以分成哪些类别。

这里的“LLM 加速器”包含几类相关但不完全相同的对象：

```text
1. LLM serving / adapter / KV-cache 系统
2. Decoder LLM 推理加速器
3. Encoder / BERT / ViT / Transformer 加速器
4. 稀疏 attention / FFN 动态稀疏硬件
5. 量化 / 近存 / 存内计算 / 数据流优化
```

因此阅读时需要注意：有些工作主要服务 autoregressive decode，有些更适合 encoder-only Transformer，有些是通用 Transformer block 级优化。

---

## 1. Chameleon: 多 LoRA Adapter LLM Serving 的自适应缓存与调度

### 核心问题

云端 LLM serving 往往共享一个 base model，同时为不同任务动态加载不同 LoRA adapter。传统方案的问题是：

```text
1. Adapter 在 CPU/GPU 间频繁搬运，造成 PCIe / GPU memory bandwidth 压力；
2. 不同 adapter rank 不同，执行时间差异大；
3. 请求长度和 adapter 大小共同导致队头阻塞和长尾延迟。
```

### 核心 idea

Chameleon 把 LoRA adapter 看成 LLM serving 中的动态资源，围绕 adapter 做：

```text
1. GPU adapter cache
2. cost-aware eviction
3. adapter-aware multi-level queue scheduling
```

它不是优化单个 Transformer 算子，而是优化多 adapter serving 系统的吞吐和尾延迟。

### 细粒度创新

**Adapter Cache**

利用 GPU 中动态空闲的显存缓存热门 adapter，而不是每次用完就丢弃。Cache manager 根据当前显存压力动态调整 cache 大小。

**Cost-aware eviction**

驱逐策略不只看 LRU / LFU，还考虑 adapter 重载成本：

```text
eviction cost ~= adapter size + recent/frequent usage
```

大的 adapter 重新加载更贵，因此不应只按时间驱逐。

**Adapter-aware MLQ scheduler**

定义加权请求大小 WRS：

```text
WRS = f(input length, predicted output length, adapter size/rank)
```

然后把请求分到不同优先级队列，短请求走 fast lane，降低 head-of-line blocking，同时通过资源再分配避免长请求饥饿。

### 加速对象

```text
LLM serving scheduling
LoRA adapter loading
GPU memory residency
tail latency
```

### 局限

它不改变 Transformer block 内部计算，也不减少单次 encoder/decoder 的 MAC；核心收益来自系统调度和缓存。

---

## 2. FlightLLM: FPGA 上的完整 LLM 映射流

### 核心问题

LLM 压缩技术理论上能降低计算和内存开销，但 FPGA/GPU 很难直接高效利用：

```text
1. 动态稀疏导致计算阵列负载不均；
2. Decode 阶段逐 token 生成，activation 是小向量，访存利用率差；
3. 混合精度需要灵活数据通路；
4. 可变长度输入如果逐长度编译，会产生巨大指令存储。
```

### 核心 idea

FlightLLM 提供完整 FPGA mapping flow，使压缩后的 LLM 能真正落到 FPGA 上：

```text
1. configurable sparse DSP chain
2. always-on-chip decode
3. length-adaptive compilation
4. mixed precision dequant support
```

### 细粒度创新

**CSD-Chain**

通过 Sparse MUX 和 reduction node 灵活重连 DSP48 级联路径，支持分块稀疏、N:M 稀疏等多种模式。

关键是让 DSP 不因稀疏模式变化而空转。

**Always-On-Chip Decode**

Decode 阶段 activation 是向量，体积较小。FlightLLM 让中间 activation 常驻 FPGA SRAM，并融合矩阵计算引擎 MPE 和特殊函数单元 SFU，避免中间结果反复写回片外。

**长度自适应编译**

把连续 token 长度分 bucket，共享一套指令模板：

```text
TB-level instruction storage -> MB-level
```

### 加速对象

```text
LLM FPGA inference
decode activation movement
sparse / mixed-precision linear layers
instruction storage
```

### 局限

主要面向 decoder LLM 与 FPGA 部署；对 encoder-only batch encoding 的直接迁移需要重新定义数据流和 batch scheduling。

---

## 3. FIGNA: 保持数值精度的 FP-INT GEMM

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

## 4. FAS-Trans: FFN 与 Attention 联合动态稀疏

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

## 5. MECLA: 缩放子矩阵划分的 LLM 加速器

### 核心问题

Autoregressive LLM decode 阶段每生成一个 token 都要读一遍巨大权重矩阵，典型 memory-bound：

```text
batch 小
MVM 为主
权重复用差
HBM bandwidth 成为瓶颈
```

### 核心 idea

MECLA 用 scaled sub-matrix partition 压缩权重并重组计算：

```text
Derived Sub-Matrix ~= scalar × Source Sub-Matrix
```

不完整存储所有权重块，而是用少量源子矩阵和缩放标量表示大量派生子矩阵。

### 细粒度创新

**SSMP compression**

把权重矩阵划分为：

```text
SS: source sub-matrix
DS: derived sub-matrix = scalar × SS
```

DS 只存 scalar，从而显著减少权重存储。

**Fine-tuning recovery**

这种强线性相关约束会损伤模型，因此通过知识蒸馏式微调恢复精度，只训练少量 SS 参数和 DS scalar。

**PSum reuse**

对于外积复用：

```text
input × SS -> PSum
input × DS -> scalar × PSum
```

硬件不需要恢复 DS，也不需要重复完整 MAC。

### 加速对象

```text
LLM decode MVM
weight storage
weight bandwidth
partial-sum reuse
```

### 局限

需要模型压缩和微调。适合 decoder memory-bound 场景，对 encoder batch GEMM 场景需要重新评估。

---

## 6. Ayaka: 低秩估计与异构数据流 Transformer 加速器

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

## 7. STAR: Cross-Stage Tiling 的 Sparse Attention Spatial Architecture

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

## 8. SOFA: Compute-Memory Optimized Sparse Attention Accelerator

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

## 9. SALO2 / Static-Dynamic Sparse Attention Co-Design

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

## 10. ESACT: 基于 Local Similarity 的端到端稀疏 Transformer

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

## 11. AccLLM: 长上下文 LLM 的 FPGA 协同设计

### 核心问题

长上下文 LLM 在边缘设备上受限于：

```text
1. 模型参数大；
2. decode memory-bound；
3. KV cache 随上下文长度膨胀；
4. 混合精度和稀疏不易映射到 FPGA。
```

### 核心 idea

用激进压缩算法和可重构 FPGA 引擎协同支持 long-context inference。

### 细粒度创新

**Compression stack**

包括 2:4 pruning、W2A8KV4 量化、A-shape attention，用来同时压缩 weight、activation、KV cache。

**RCE**

Reconfigurable Computing Engine 支持 prefill MMM 和 decode MVM 两种模式切换。

**DSP packing**

针对 2/4/8-bit 混合精度，把多个低位运算映射到同一个 DSP，提高利用率。

**Kernel/layer fusion**

融合 attention 和 decode 层间中间结果，减少片外访问。

### 加速对象

```text
long-context LLM
KV cache
prefill/decode dual mode
mixed precision FPGA execution
```

### 局限

主要服务 decoder 和 long-context，不直接适用于没有 KV-cache 的 encoder-only workload。

---

## 12. CENT / PIM Is All You Need

### 核心问题

LLM decode 阶段 operational intensity 很低，GPU 算力利用率差，性能受内存带宽限制。

### 核心 idea

构建 GPU-free LLM inference system，用 CXL 扩展容量，并用 PIM/PNM 提供高内存带宽。

### 细粒度创新

**CXL-scaled memory**

通过 CXL 交换网络连接多个 memory device，突破单设备容量限制。

**Hierarchical PIM-PNM**

PIM 负责大量 MAC，PNM 负责 softmax、SiLU、sqrt 等复杂但低频操作。

**Parallel mapping**

支持 pipeline parallelism、tensor parallelism 及混合映射。

### 加速对象

```text
memory-bound LLM decode
large model capacity
weight bandwidth
distributed memory system
```

### 核心价值

它从系统结构上回答：当 LLM 主要瓶颈是访存时，是否还需要 GPU。

---

## 13. IMCsim: 面向深度学习的全系统 IMC 模拟框架

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

## 14. Balanced Systolic Array Attention Accelerator

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

## 15. DESA: Dataflow Efficient Systolic Array for Transformers

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

## 16. H3D-Transformer: 异构 3D Transformer 平台

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

## 17. PADE: Predictor-Free Sparse Attention

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

## 18. Oaken: Online-Offline Hybrid KV Cache Quantization

### 核心问题

LLM decode 阶段 KV cache 巨大，KV quantization 可降内存，但在线检测 outliers 成本高。

### 核心 idea

把 outlier 阈值从在线检测改为离线统计，并在线快速分类和量化。

### 细粒度创新

**Offline thresholds**

观察到不同层 KV 分布与 prompt 关系弱，离线统计 outer/middle/inner 阈值。

**Group-shift quantization**

对 outer/inner outlier 先用阈值平移到近原点，再低位量化，避免 FP16 outlier path。

**Dense-sparse fused encoding**

Middle dense 紧凑存储，outer/inner 稀疏编码融合到 dense 矩阵空位中，减少索引和访存开销。

### 加速对象

```text
KV cache memory
decode bandwidth
online outlier detection
mixed-format storage
```

### 局限

纯 encoder 场景没有 autoregressive KV cache，因此不能直接迁移。

---

## 19. Hybrid Systolic Array for Edge LLM

### 核心问题

边缘 LLM 同时有：

```text
prefill: MMM, compute-intensive
decode: MVM, memory-bound, batch small
```

传统 2D systolic array 在 decode batch=1 时利用率很低。

### 核心 idea

设计 Hybrid Systolic Architecture，在 prefill 和 decode 两种模式间切换。

### 细粒度创新

**HSA mode switching**

Prefill 时 PE 组成大阵列处理矩阵-矩阵；decode 时拆成多个 cluster 处理矩阵-向量，提高 batch=1 利用率。

**MXINT4 dequant dataflow**

利用 decode 阶段空闲 PE 执行反量化，隐藏低位权重恢复开销。

**Fused RMSNorm/RoPE**

优化后处理和位置编码，减少额外 buffer 和 DRAM 表访问。

### 加速对象

```text
edge LLM prefill/decode
systolic utilization
low-bit dequant
post-processing ops
```

---

## 20. MATA: Look-Back KV Cache Pruning

### 核心问题

长上下文 LLM decode 中，KV cache 随序列长度线性增长，带来巨大 DRAM 存取和容量压力。

### 核心 idea

在线回顾历史 token 的重要性，把低价值 KV 丢弃，使 KV cache 接近固定预算。

### 细粒度创新

**Look-back pruning**

动态评估历史 token 是否仍重要，并从 KV cache 中移除不重要 token。

**Lightweight scoring**

用轻量综合评分避免 pruning 决策本身成为瓶颈。

**Hardware memory manager**

支持剪枝后不规则 KV 存取和压缩存储。

### 加速对象

```text
long-context decode
KV cache capacity
KV memory bandwidth
```

### 局限

主要适用于 decoder KV cache；encoder-only workload 没有持续增长的 KV cache。

---

## 21. Energon / PNM Transformer Throughput Maximization

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

## 22. Low-Power ViT Accelerator with Hardware-Aware Pruning

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

## 23. BETA: Bit-Grained Transformer Attention Accelerator

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

---

# 分类总结

## A. Serving / Scheduler / Cache 类

代表：

```text
Chameleon
Oaken
MATA
```

主要加速：

```text
adapter loading
request scheduling
KV cache storage
online memory management
tail latency
```

核心创新：

```text
把 LLM serving 中的动态资源显式建模：
adapter、KV cache、请求长度、rank、显存容量。
```

适合场景：

```text
multi-tenant LLM serving
LoRA-as-a-service
long-context decode
```

不适合直接作为 encoder-only NPU 核心，因为它们很多依赖 KV cache 或 serving queue。

## B. FPGA / Reconfigurable Mapping 类

代表：

```text
FlightLLM
AccLLM
Hybrid Systolic Array Edge LLM
```

主要加速：

```text
prefill MMM
decode MVM
mixed precision
sparsity mapping
low-bit dequant
variable length compilation
```

核心创新：

```text
用可重构阵列 / DSP packing / mode switching 适配 LLM 不同阶段。
```

适合场景：

```text
FPGA deployment
edge LLM
mixed prefill/decode workload
```

## C. Quantization / Numeric Compute 类

代表：

```text
FIGNA
MECLA
Oaken
```

主要加速：

```text
weight-only GEMM
MVM weight bandwidth
KV quantization
dequant overhead
```

核心创新：

```text
用新的数值表示、权重结构或离线阈值降低 bit-width 和访存。
```

可分两类：

```text
1. numerical-equivalent:
   FIGNA 追求与 FP-INT GEMM 数值一致。

2. model-compression/recovery:
   MECLA/Oaken 通过压缩、阈值、微调或任务容忍保持精度。
```

## D. Dynamic Sparse Attention / Eager Prediction 类

代表：

```text
FACT
FAS-Trans
Ayaka
STAR
SOFA
SALO2
ESACT
PADE
BETA
```

主要加速：

```text
QKV generation
attention score
softmax/top-k
attention output
sometimes FFN
```

核心创新：

```text
提前或边算边判断哪些 token / pair / row / column 重要，
然后跳过无贡献或低贡献计算。
```

内部又可分为：

```text
1. predictor-based:
   FACT, FAS-Trans, Ayaka, ESACT

2. pattern/tile optimized:
   STAR, SOFA, SALO2

3. predictor-free / bit-serial early termination:
   PADE, BETA
```

这类是当前最接近 HPCA/ISCA/MICRO 风格 Transformer NPU 创新的主流方向。

## E. FFN / Token Mixed Precision 类

代表：

```text
FACT
FAS-Trans
Low-Power ViT Accelerator
```

主要加速：

```text
FFN FC1 / FC2
token-wise precision
activation sparsity
channel or token pruning
```

核心创新：

```text
不要只盯 attention；
短序列或 encoder 场景中 FFN 往往是更大的瓶颈。
```

典型依据：

```text
token 被 attention top-k 选中的次数
GELU 后接近 0 的 activation
hidden/activation importance
```

## F. Exact Dataflow / Systolic / Softmax Fusion 类

代表：

```text
Balanced Systolic Array Attention
DESA
部分 FuseMax / FlashAttention-style accelerator
```

主要加速：

```text
attention dataflow
softmax intermediate storage
LayerNorm/Softmax dependency
systolic array utilization
```

核心创新：

```text
不依赖近似，不改变模型数学语义；
通过 tiling、fusion、stationary dataflow、interleaving 减少 IO 和提高利用率。
```

优点：

```text
精度最稳。
```

缺点：

```text
如果只是迁移已有 exact attention dataflow，论文主创新可能不足。
```

## G. PIM / PNM / 3D / IMC 类

代表：

```text
CENT
Energon
H3D-Transformer
IMCsim
```

主要加速：

```text
weight bandwidth
large model capacity
near-memory matrix compute
heterogeneous compute placement
```

核心创新：

```text
把计算靠近存储，或者把不同 Transformer 算子映射到不同存储/计算介质。
```

适合场景：

```text
memory-bound LLM inference
edge model residency
large model serving
```

## H. 总体趋势

这些工作可以归纳成一个趋势：

```text
早期 Transformer accelerator:
    重点优化 attention matrix / softmax dataflow。

近期 LLM accelerator:
    更关注端到端瓶颈：
        QKV generation
        FFN
        weight bandwidth
        KV cache
        mixed precision
        runtime scheduling
```

另一个趋势是：

```text
只做 static low-bit 已经不够；
新的工作更强调 input-dependent / layer-dependent / runtime-adaptive execution。
```

也就是说，现代 LLM 加速器的核心不只是“算得更快”，而是：

```text
1. 哪些计算可以不做？
2. 哪些计算可以低精度做？
3. 哪些中间结果可以不存？
4. 哪些数据可以留在片上？
5. 哪些请求/adapter/token 应该被优先调度？
```

## I. 对 Encoder 场景最相关的方向

如果只看 LLM encoder / BERT / ST / graph-text encoder，最相关的是：

```text
1. FFN / QKV / attention joint optimization
   FACT, FAS-Trans, Ayaka, ESACT

2. predictor-free bounded early termination
   PADE, BETA

3. exact dataflow and softmax fusion
   DESA, Balanced SA, FuseMax-style

4. FFN token/channel mixed precision
   FACT, Low-Power ViT, FAS-Trans
```

相对不直接相关的是：

```text
1. KV-cache-only optimization
2. adapter serving cache
3. decode MVM-only accelerator
```

这些可以作为系统背景，但不应作为 encoder NPU 的核心路线。

