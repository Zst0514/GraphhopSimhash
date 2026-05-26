# Decoder / Serving LLM Accelerator Survey

本文档从 [LLM_ACCELERATOR_SURVEY.md](./LLM_ACCELERATOR_SURVEY.md) 中拆出 decoder / serving / KV-cache 相关论文，避免和 encoder / 通用 Transformer NPU 路线混在一起。

这里保留的工作主要优化：

```text
1. autoregressive decode 阶段的 KV cache / MVM / memory bandwidth
2. prefill-decode 混合调度
3. LoRA adapter / heterogeneous GPU / thermal-power serving scheduler
4. PIM / NDP / in-flash 等面向大模型 decode 的 memory-side execution
5. decoder-oriented FPGA mapping / long-context LLM serving
```

这些工作对 GraphHopSimhash 的直接关系较弱，因为当前场景是 **LLM encoder for graph-text nodes**，没有 autoregressive KV cache。但它们仍然可以作为系统调度、存储层级和 memory-bound LLM inference 的背景参考。

---

## 1. Chameleon: 多 LoRA Adapter LLM Serving 的自适应缓存与调度

**发表信息**：MICRO 2025；UIUC / IBM Research 等团队；定位是多 LoRA adapter LLM serving 的缓存与调度系统。

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

**发表信息**：FPGA 2024；清华大学、上海交通大学等团队；定位是 FPGA 上的 LLM inference 完整映射流与混合精度执行。

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

## 3. AccLLM: 长上下文 LLM 的 FPGA 协同设计

**发表信息**：arXiv 2025；Yanbiao Liang、Huihong Shi、Haikuo Shao、Zhongfeng Wang 等作者团队；定位是 FPGA 上 long-context LLM 的 algorithm-hardware co-design。

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

## 4. MECLA: 缩放子矩阵划分的 LLM 加速器

**发表信息**：ISCA 2024；Microsoft Research、Virginia Tech 等团队；定位是利用 scaled sub-matrix partition 做 LLM 权重压缩与部分和复用。

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

## 5. CENT / PIM Is All You Need

**发表信息**：arXiv 2025；University of Michigan 等团队；定位是 CXL-enabled GPU-free / PIM-PNM LLM inference system。

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

## 6. Oaken: Online-Offline Hybrid KV Cache Quantization

**发表信息**：ISCA 2025；KAIST、POSTECH 等韩国高校团队；定位是 online-offline hybrid KV cache quantization 与硬件量化引擎。

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

## 7. Hybrid Systolic Array for Edge LLM

**发表信息**：ISLPED 2025 / arXiv 2025；Cornell Tech / Cornell University 团队；定位是面向 edge LLM 的 hybrid systolic array 与 MXINT4 dataflow。

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

## 8. MATA: Look-Back KV Cache Pruning

**发表信息**：近年 LLM KV-cache pruning / memory management 方向论文；作者团队聚焦 long-context / decode-stage KV cache 压缩；定位是 KV-cache-only 路线的代表性对照。

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

## 9. ALISA: Sparsity-Aware KV Caching

**发表信息**：ISCA 2024；Youpeng Zhao、Di Wu、Jun Wang 等作者团队；定位是 decode-stage sparsity-aware KV caching。

来源：[ISCA 2024 / NSF repository](https://par.nsf.gov/biblio/10538084-alisa-accelerating-large-language-model-inference-via-sparsity-aware-kv-caching)

### 核心问题

自回归 LLM decode 的瓶颈主要在 KV cache：

```text
每生成一个 token，都要访问历史 K/V；
context 越长，KV cache 访存越重；
GPU/NPU compute 可能闲着等内存。
```

### 核心 Idea

ALISA 利用 attention sparsity 管理 KV cache，把更可能被访问的重要 KV 留在更快路径，把低价值 KV 迁到更低成本层级。

### 加速对象

```text
decode-phase KV cache access
attention sparsity-aware cache placement
memory bandwidth and capacity pressure
```

### 对 encoder 场景的边界

我们的 graph-text encoder 没有 autoregressive KV cache，所以 ALISA 不能直接迁移。但它的“稀疏性指导存储层级”的思想可以迁移到：

```text
embedding cache placement
hash bucket cache locality
reuse anchor 的 SRAM/DRAM 分层管理
```

---

## 10. AiF: Accelerator-in-Flash for On-Device LLM

**发表信息**：ISCA 2025；Seoul National University 团队；定位是 on-device LLM 的 in-flash processing。

来源：[ISCA 2025 / SNU page](https://snu.elsevierpure.com/en/publications/aif-accelerating-on-device-llm-inference-using-in-flash-processin/)

### 核心问题

on-device LLM 经常放不进 DRAM，只能把模型参数 offload 到 SSD/flash。普通 SSD offloading 的瓶颈是外部读带宽：

```text
model weights 在 flash 里；
每层 GEMV/GEMM 都要把大量权重搬到计算侧；
外部接口带宽远小于 flash 内部并行读能力。
```

### 核心 Idea

AiF 把 GEMV 操作直接集成到 flash chip 里，利用 flash 内部高并行读带宽做 in-flash processing。

### 细粒度创新

```text
1. 在 flash 内部执行 GEMV，减少外部数据搬运；
2. 针对 LLM 参数读取优化 flash read path；
3. 简化 ECC / read 过程，提高内部有效带宽。
```

### 对我们场景的启发

AiF 对 decode MVM 更直接，但对 graph-text encoder 也有一个重要启发：当 embedding 模型太大时，系统瓶颈可能不是 PE array，而是“模型权重从哪里来”。如果 GraphHopSimhash 通过 reuse/gating 减少 encoder 调用次数，本质上也在减少权重流量。

---

## 11. HPCA 2025 PIM/NDP LLM 系列：PAISE / FACIL / Lincoln / NDP-DIMM

**发表信息**：HPCA 2025；Samsung SDS、Seoul National University、清华大学、中国科学院等多个团队；定位是 PIM / NDP / compute-enabled memory for LLM inference 的系列工作。

来源：[HPCA 2025 main program](https://hpca-conf.org/2025/main-program/)

### 共同背景

HPCA 2025 专门有一组 LLM + PIM/NDP 工作，说明一个趋势非常明确：

```text
LLM inference 正在从“算力问题”变成“存储层级 + 带宽 + 数据搬运问题”。
```

### 代表工作

**PAISE**

```text
PIM-accelerated inference scheduling engine。
重点在 PIM 环境下如何调度 Transformer-based LLM inference。
```

**FACIL**

```text
SoC-PIM cooperative on-device LLM inference。
重点是 flexible DRAM address mapping，
让 SoC 与 PIM 协同执行时数据布局更适合 PIM 访问。
```

**Lincoln**

```text
LPDDR-interfaced, compute-enabled flash memory。
目标是在 consumer devices 上支持 50~100B LLM inference。
```

**NDP-DIMM**

```text
用 near-data-processing DIMM 扩展 GPU memory。
核心是让大模型推理不完全受 GPU 显存容量限制。
```

### 对我们场景的启发

这些工作更偏 generative decode，但它们对 GraphHopSimhash 的系统设计有直接参考价值：

```text
embedding cache 放哪里？
reuse anchor 放 SRAM 还是 DRAM？
低风险节点是否可以只读 cache，不触发 encoder 权重流？
高风险节点是否集中调度，减少权重反复加载？
```

---

## 12. POD-Attention: Prefill-Decode Overlap Attention Kernel

**发表信息**：ASPLOS 2025；Microsoft Research、University of Washington 等团队；定位是 hybrid batch prefill/decode overlapped attention kernel。

来源：[ASPLOS 2025 project page](https://akkamath.github.io/publication/ASPLOS25-POD-Attention)

### 核心问题

LLM serving 中 prefill 是 compute-bound，decode 是 memory-bandwidth-bound。Hybrid batching 把不同请求的 prefill/decode 放一起，但普通 attention kernel 仍然把 prefill 和 decode 分开优化，导致 SM 资源利用不充分。

### 核心 Idea

POD-Attention 在同一个 GPU SM 上并发执行 prefill 和 decode attention，让 compute-heavy 和 bandwidth-heavy 工作互补。

### 细粒度创新

```text
1. 面向 hybrid batch 的 attention kernel；
2. SM-aware CTA scheduling；
3. 在同一 multiprocessor 上同时利用 compute 和 memory bandwidth；
4. 与 serving scheduler 集成。
```

### 对 encoder 场景的边界

我们的 encoder 没有 decode phase，因此 POD-Attention 不能直接照搬。但它的“异质 workload overlap”很重要：

```text
reuse cache read / residual adapter / full W4A8 encoder
可以被调度到不同硬件单元并行执行。
```

---

## 13. Helix: Heterogeneous GPU LLM Serving via Max-Flow

**发表信息**：ASPLOS 2025；Carnegie Mellon University 团队；定位是 heterogeneous GPU / network LLM serving 的 max-flow / MILP 调度。

来源：[ASPLOS 2025 / CMU page](https://pdl.cmu.edu/PDL-FTP/BigLearning/helix_abs.shtml)

### 核心问题

现实集群不是同构 GPU，GPU 型号、显存、网络带宽都不同。手工放置模型和调度请求很难同时优化 throughput 和 latency。

### 核心 Idea

Helix 把 heterogeneous GPU + network 的 LLM serving 建成一个 directed weighted graph，用 max-flow / MILP 联合优化 model placement 和 request scheduling。

### 细粒度创新

```text
nodes:
    GPU instances

edges:
    network / transfer capacity

optimization:
    model placement + request scheduling jointly solved
```

### 对我们场景的启发

Helix 不是 NPU array 论文，但它提醒我们：当系统存在多条执行路径时，调度不能靠固定规则。GraphHopSimhash 的 P0/P1/P2/P3 层级路径最终也可以建模成 flow / capacity / cost 优化问题。

---

## 14. TAPAS: Thermal- and Power-Aware LLM Scheduling

**发表信息**：ASPLOS 2025；Microsoft Azure Research、UIUC 等团队；定位是 cloud LLM inference 的 thermal- and power-aware scheduling。

来源：[ASPLOS 2025 author page](https://haoran-qiu.com/publication/asplos-2025/)

### 核心问题

云端 LLM inference 不只是性能问题，还有热和功耗限制。高吞吐调度如果忽略 thermal / power，会导致频率下降、尾延迟上升或能耗不可控。

### 核心 Idea

TAPAS 把 thermal 和 power 纳入 LLM inference scheduling，在云平台上做更稳的请求调度。

### 对我们场景的启发

如果我们的 NPU 有多条路径：

```text
P0 cache
P1 residual
P2 gated FFN
P3 full W4A8
```

那么调度目标也不应只有 accuracy/cost，还可以加入：

```text
power budget
thermal headroom
array utilization
memory bandwidth
```

---

---

---

# 分类总结

## A. Serving / Scheduler / Adapter Cache

```text
Chameleon -> 多 LoRA adapter cache 与 multi-level queue scheduling
Helix     -> heterogeneous GPU serving 的 max-flow / MILP placement
TAPAS     -> thermal / power-aware cloud LLM scheduling
```

这类工作优化的是请求级系统吞吐、尾延迟和资源调度，不直接减少单个 encoder block 的 MAC。

## B. KV Cache / Decode Memory Path

```text
Oaken -> online-offline hybrid KV cache quantization
MATA  -> look-back KV cache pruning
ALISA -> sparsity-aware KV cache placement
```

这些工作高度依赖 autoregressive decode 的历史 K/V 访问。Graph-text encoder 没有 KV cache，因此只能迁移“缓存层级管理”的思想，不能直接作为核心机制。

## C. Decode MVM / Memory-Side Execution

```text
MECLA               -> scaled sub-matrix partition for memory-bound decode MVM
CENT / PIM          -> CXL + PIM/PNM GPU-free LLM inference
AiF                 -> in-flash GEMV for on-device LLM
Hybrid Systolic     -> prefill/decode 双模式 edge LLM array
HPCA PIM/NDP 系列   -> PIM scheduling、compute-enabled flash、NDP-DIMM
```

这类工作说明 decode 往往是 memory-bound，适合 PIM/NDP/in-flash。对 encoder 场景的启发是：减少 full encoder 调用次数或改善权重流 locality，比只优化 GNN 后端更关键。

## D. Prefill-Decode / FPGA / Long-Context Mapping

```text
POD-Attention -> 同一 SM 上 overlap compute-bound prefill 和 bandwidth-bound decode
FlightLLM     -> FPGA 上完整 LLM mapping flow，重点包含 always-on-chip decode
AccLLM        -> long-context FPGA LLM 的压缩与映射
```

这类工作适合 serving 系统和长上下文生成模型；对 encoder NPU 更像调度背景，不是主线。
