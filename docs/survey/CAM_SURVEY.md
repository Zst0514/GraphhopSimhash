# CAM / Approximate Search Survey

本文档整理 `docs/paper/CAM` 目录下的 CAM 相关论文，主要服务 GraphhopSimhash 后续的硬件架构判断：

```text
1. 普通 CAM / TCAM 的基本电路与功耗优化
2. Hamming-distance tolerant approximate CAM
3. Edit-distance tolerant CAM 与 genomics-oriented CAM
4. XNOR + popcount / associative processor / PIM 路线
5. 对 8x16-bit、HD<=2、多头 support 聚合架构的启发
```

阅读时需要注意：这个目录里的论文来源比较混合，既有经典综述，也有精确匹配电路论文、近似匹配 CAM、PIM / associative processor、以及面向基因组和多模态检索的系统论文。本文会按“和 GraphhopSimhash 的相关性”重新组织，而不是简单按年份罗列。

---

## 1. 如何看这批 CAM 论文

如果只从 GraphhopSimhash 的需求出发，这批论文大致可以分成四层：

```text
Layer 1: 基础 CAM / TCAM 电路
Layer 2: 直接支持 HD 阈值比较的 approximate CAM
Layer 3: 用 XNOR+popcount / PIM 计算相似度或精确 HD
Layer 4: 面向 DNA / few-shot / multimodal retrieval 的应用系统
```

对我们最关键的是 Layer 2 和 Layer 3：

- Layer 2 回答：能不能像 CAM 一样直接做 `HD <= T`？
- Layer 3 回答：能不能先算出精确 HD，再做阈值或排序？

这也是后面普通 CAM 分块粗筛 + `XOR/popcount` 和 `HD-CAM` 路线比较的核心。

---

## 2. 基础综述与经典精确 CAM / TCAM 电路

这一组论文的重点不是 approximate search，而是帮助建立 CAM 的基本电路直觉：NOR/NAND bitcell、ML/SL sensing、功耗、面积、密度、PVT、TCAM compiler 设计。

---

## 2.1 Pagiamtzis & Sheikholeslami: CAM Tutorial and Survey

**发表信息**：IEEE JSSC 2006；经典 CAM 教程与综述。

### 核心问题

在大容量 CAM 中，速度、面积和功耗三者如何平衡？

### 核心内容

- 给出了最经典的 CAM 分类：
  - `BCAM`
  - `TCAM`
  - `NOR CAM`
  - `NAND CAM`
- 系统梳理了：
  - matchline sensing
  - searchline 驱动
  - 预充电
  - 功耗降低
  - 架构级分段和过滤

### 对我们的价值

这是整个目录里理解 CAM 电路最好的入口。  
后面读 `HD-CAM`、`DASH-CAM`、`TAP-CAM` 时，里面大量 bitcell / ML / sensing 术语，都可以回到这篇打底。

---

## 2.2 Karam et al.: Emerging Trends in Design and Applications of Memory-Based Computing and CAMs

**发表信息**：Proceedings of the IEEE 2015；CAM/AM/HTM 与 emerging device 综述。

### 核心问题

除了传统 CMOS CAM，哪些新型 memory / associative 结构值得看？

### 核心内容

- 从宏观上梳理了：
  - binary CAM
  - ternary CAM
  - associative memory
  - neuromorphic memory
  - transactional memory
- 也系统盘点了：
  - STT-MRAM / MTJ
  - ReRAM
  - FeRAM / FeFET
  - memristor / DWM

### 对我们的价值

如果要把 GraphhopSimhash 的 CAM 设计从 `CMOS` 扩展到 `FeFET / MRAM / resistive CAM`，这篇是最好的器件级路线图。

---

## 2.3 Yang et al.: Low Swing Search Lines

**发表信息**：IEEE TCAS-I 2011；低摆幅 search line CAM。

### 核心问题

CAM 的大头功耗往往来自：

```text
search line (SL)
match line (ML)
```

如何把 SL 摆幅降下来？

### 核心 idea

用 CAM cell 自身作为放大器，以低摆幅 search line 完成比较，并进一步降低 ML 功耗。

### 对我们的价值

这篇不直接做 approximate search，但它说明：  
**CAM 的功耗优化不一定只发生在 ML 判决，也可以从 SL 驱动路径下手。**

---

## 2.4 Do et al.: Parity Bit and Power-Gated ML Sensing

**发表信息**：IEEE TVLSI 2013；高速低功耗 CAM。

### 核心问题

如何提高 sensing 速度、压低平均功耗和峰值电流？

### 核心 idea

- 引入 parity bit 改善 sensing
- 用 power-gated ML sensing 降功耗和峰值电流

### 对我们的价值

这篇更偏 exact CAM circuit optimization，不是通用 approximate CAM。  
如果后续要在近似前端之外保留一个 exact CAM baseline，它是很好的参考。

---

## 2.5 Hayashi et al.: Full TCAM With Low-Voltage Matchline Sensing

**发表信息**：2013；65nm、18Mb 级 full TCAM。

### 核心问题

在大容量 TCAM 上，如何同时保住：

```text
容量
速度
低电压 sensing
```

### 对我们的价值

这篇代表的是“大规模 industrial TCAM macro”的一侧。  
它对 approximate matching 的直接帮助不大，但有助于理解：为什么 TCAM compiler 和近似匹配 CAM 的设计约束差异很大。

---

## 2.6 Abbas et al.: Multiple Cell Upsets Tolerant CAM

**发表信息**：IEEE Transactions on Computers 2014。

### 核心问题

软错误、多 bit upset 下，CAM 如何做 MCU 容错？

### 核心 idea

利用 parity / ECC 的组织方式，降低 MCU 保护开销。

### 对我们的价值

它不是近似搜索论文，但适合补充一个认知：

```text
“容错型 Hamming 码 CAM” 和 “相似度检索型 Hamming threshold CAM”
是两条不同路线。
```

---

## 2.7 Arsovski et al.: High-Performance TCAM Compiler

**发表信息**：IEEE JSSC 2018；14nm TCAM compiler。

### 核心问题

高性能 TCAM 在大并行搜索下，电源噪声与 Ldi/dt 怎么压？

### 核心 idea

- two-phase pre-charge ML sensing
- power-grid pre-conditioning

### 核心结果

- `1.4 Gsearch/s`
- `2 Mb/mm^2`
- within-cycle / multi-cycle power-supply noise 显著降低

### 对我们的价值

如果后续考虑把 CAM 前端做成真正高吞吐 macro，这篇提供的是：

```text
compiler / macro / power-delivery 级约束
```

不是 approximate 路线，但很工程。

---

## 2.8 Taco et al.: Precharge-Free CAM

**发表信息**：IEEE TVLSI 2024。

### 核心问题

传统 NOR / NAND CAM 都严重依赖预充电：

- NOR：快但耗电
- NAND：省电但慢、可扩展性差

有没有一种 precharge-free CAM？

### 核心 idea

提出 `PCAM` 类 precharge-free CAM，在能耗和时延之间取得更好的平衡。

### 对我们的价值

这篇的意义在于：  
**它试图从 CAM 基本工作模式本身改写能耗结构，而不是只在原有 precharge CAM 上做打补丁式优化。**

对 GraphhopSimhash，如果未来希望 exact / threshold CAM 共用一类更低能耗底座，这篇值得追。

---

## 3. Hamming-Distance Approximate CAM 主线

这一组论文最直接对应我们的核心问题：

```text
能不能让 CAM 直接比较 HD <= T，
而不是先粗筛再 XOR+popcount？
```

---

## 3.1 Imani et al.: MASC TCAM

**发表信息**：DATE 2016；近似计算导向 TCAM。

### 核心问题

TCAM 搜索能耗太高，能不能通过近似容忍换能耗？

### 核心 idea

- Multiple-Access Single-Charge TCAM
- 用更长 refresh 周期换近似匹配能力
- 支持小范围 `1-HD / 2-HD` 近似

### 对我们的价值

MASC 很重要，因为它代表了：

```text
“低能耗近似 TCAM” 的早期路线
```

但它更像 approximate computing，而不是我们要的通用 HD-threshold search engine。  
对 `HD<=2` 有一定启发，但对多头哈希复用不够完整。

---

## 3.2 CAMsure: Secure Approximate Search

**发表信息**：ACM TECS 2017。

### 核心问题

如果要在不可信服务器上做 approximate search，CAM 能不能兼顾相似度与隐私？

### 核心 idea

- 先用 secure LSH / hash embedding
- 再在 hash 域里用 CAM 做近似查找

### 对我们的价值

这篇不是直接在原始数据上做 HD threshold。  
它是：

```text
先映射到 hash 域
再用 CAM 做 Hamming-space 检索
```

如果以后考虑“相似度缓存 + 隐私”或“hash-based secure retrieval”，它值得看；  
但它不是 `HD-CAM` 的直接替代。

---

## 3.3 Garzon et al.: HD-CAM

**发表信息**：IEEE Access 2022 / arXiv 2021 线。

### 核心问题

能不能直接在 CAM 里实现：

```text
HD <= T
```

而且支持比早期采样时间调节方案更大的阈值？

### 核心 idea

- 在 NOR-type CAM bitcell 中加入 `Meval`
- 通过 matchline charge redistribution / discharge rate 映射 Hamming distance
- 通过 `Veval` / sensing threshold 做可调 mismatch threshold

### 核心价值

这篇是目录里最关键的近似 CAM 起点。  
它的贡献不在于把 HD 精确算出来，而在于：

```text
不显式 popcount，
直接把“mismatch 数量”变成 ML 电学量，
再做 thresholded hit/miss。
```

### 对我们的价值

这正是和 `普通 CAM + XOR/popcount` 对比的那条主线。  
对于 `8x16-bit, HD<=2`，这篇提供了最直接的 conceptual baseline。

---

## 3.4 Garzon et al.: Low-Complexity Sensing Scheme

**发表信息**：IEEE TCAS-II 2023。

### 核心问题

原始 HD-CAM 虽然方向对，但 sensing 较复杂、外围较重。  
能不能保留 HD-threshold 思路，同时把 sensing 做薄？

### 核心 idea

- replica line
- 12T positive-feedback SA
- 用低复杂度 sensing 提供更稳定的 threshold control

### 对我们的价值

如果 2022 HD-CAM 是“概念原型”，这篇是：

```text
把概念往可落地电路推进一步
```

它对我们最大的启发是：  
**HD-CAM 的关键不只是 bitcell，还有 sensing path 的复杂度。**

---

## 3.5 Jahshan et al.: DASH-CAM

**发表信息**：MICRO 2023。

### 核心问题

HD-CAM 虽然能做大 HD threshold，但基于 SRAM 的密度不够高。  
在 genome classification 这类大库场景里，容量会成为瓶颈。

### 核心 idea

- dynamic approximate search CAM
- 用动态存储换更高密度
- 保留可调 Hamming threshold
- 通过 one-hot DNA 编码和 refresh 机制适配 genomics

### 核心结果

- 相比 state-of-the-art SRAM approximate CAM，密度提升 `5.5x`

### 对我们的价值

这篇是 HD-CAM 路线的“高密度分支”。  
如果我们以后更看重容量而不是最简单的数字实现，DASH-CAM 比原始 HD-CAM 更值得注意。

---

## 3.6 Garzon et al.: 128-kbit Tunable HD-CAM

**发表信息**：IEEE JSSC 2025。

### 核心问题

前面 HD-CAM 多是仿真和电路概念，能不能做成完整硅宏？

### 核心 idea

- 把 HD-CAM 做成 `128-kbit` 研究芯片宏
- 用低复杂度 sensing + 可调 `Veval/Vref`
- 在 PVT 条件下验证阈值稳定性

### 核心结果

- `125 MHz`
- `0.76 mW`
- user-programmable HD tolerance threshold
- 硅片级 approximate CAM 验证

### 对我们的价值

这是目录里最接近“真硬件答案”的一篇。  
如果要给 `HD-CAM` 路线找一个最强工程基准，这篇就是。

---

## 3.7 Ni et al.: TAP-CAM

**发表信息**：ICCAD 2024 / 2025 线；FeFET-based tunable approximate matching engine。

### 核心问题

CMOS HD-CAM 面积偏大，FeFET approximate CAM 又常常阈值控制不够细。

### 核心 idea

- `2FeFET-2R` TCAM cell
- exact + tunable approximate matching
- bit-by-bit tunable threshold control
- 面向 KNN / similarity search

### 核心结果

- 相比 16T CMOS exact CAM，给出显著能效提升
- 相比已有 FeFET approximate TCAM，也给出能效改进

### 对我们的价值

这是目录里最值得关注的 **FeFET approximate CAM** 之一。  
如果未来想从 `CMOS HD-CAM` 往 `NVM/FeFET HD-threshold engine` 迁移，TAP-CAM 是重要候选。

---

## 3.8 Vardar et al.: 28nm FeFET CAM for Similarity Search and Few-Shot Learning

**发表信息**：JEDS 2025 author version；28nm FeFET CAM。

### 核心问题

能不能把 exact / approximate associative search 和 edge-AI few-shot learning 结合起来？

### 核心 idea

- `2FeFET` CAM bitcell
- 直接在 memory 中计算 Hamming distance
- 通过可编程串联电阻改善 variability 与 sensing

### 应用验证

- genome read mapping
- few-shot learning classification

### 对我们的价值

这篇说明 FeFET CAM 已经不只是器件论文，而是开始直接进入：

```text
few-shot / associative inference
```

这和 GraphhopSimhash 的“基于近似匹配做复用决策”在任务形态上有一定共鸣。

---

## 3.9 Ryu et al.: Time-Domain CAM With Ferroelectric Memcapacitor

**发表信息**：2025；time-domain CAM。

### 核心问题

传统 voltage-domain CAM 在距离计算上常常是非线性的，导致：

- sensing margin 差
- 精度差
- 外围复杂

### 核心 idea

- 用单 ferroelectric memcapacitor 做 time-domain CAM
- 让输出传播延迟和 Hamming distance 近线性相关

### 对我们的价值

这篇代表一条很不一样的路线：

```text
不是 voltage-domain，不是显式 popcount，
而是 time-domain 距离编码
```

如果以后要研究“多阈值 / 线性距离编码 / 高精度 ANN”，这篇值得单独跟进。

---

## 4. Edit-Distance / Genomics 定向 CAM

这组论文的目标不是普通 Hamming threshold，而是更接近：

```text
DNA sequencing
text / sequence matching
insertions + deletions + substitutions
```

对 GraphhopSimhash 不一定直接可用，但它们说明 CAM 可以被拉向更复杂的距离度量。

---

## 4.1 Kaplan et al.: Resistive CAM PRinS for DNA Sequence Alignment

**发表信息**：IEEE Micro 2017。

### 核心问题

Smith-Waterman 这类 DNA alignment 访存和并行需求极高，能否用 resistive CAM + processing-in-storage 加速？

### 核心 idea

- ReCAM
- processing-in-storage
- 面向 DNA sequence alignment

### 对我们的价值

它不是直接做 `HD<=T` 的 CAM，但它强调了：

```text
CAM 不只是查表，也可以是 in-storage accelerator
```

这是后面 `AM4`、`PPAC` 这类 associative processor 路线的一个背景。

---

## 4.2 Hanhan et al.: EDAM

**发表信息**：ISCA 2022。

### 核心问题

Hamming distance 只适合 substitution，不适合 insert / delete。  
DNA reads 和文本序列里，单个 indel 就可能造成巨大的 Hamming distance。

### 核心 idea

- edit-distance tolerant CAM
- 不再只看共位比较
- 同时考虑左右邻居，来吸收插入/删除

### 对我们的价值

EDAM 的重要性在于：  
它证明了 CAM 的“距离度量”是可以被重新定义的。

对 GraphhopSimhash 不是直接所需，因为我们现在的 hash 距离就是 Hamming；  
但如果将来要从二值 hash 走向序列级复用，EDAM 是一条完全不同的参考线。

---

## 4.3 Merlin et al.: DIPER

**发表信息**：2024；edit-distance tolerant resistive CAM。

### 核心问题

在 pathogen detection / identification 里，edit distance 比 Hamming distance 更符合生物学数据噪声。

### 核心 idea

- resistive CAM
- edit-distance tolerant search
- pathogen detection / identification

### 对我们的价值

它基本可以视作 EDAM 的 resistive / system-oriented 延伸。  
如果我们的应用以后从 hash-level 复用扩展到 sequence-level 检索，可以把这组论文单独拿出来再看。

---

## 5. XNOR + Popcount / Associative Processor / PIM 路线

这一组论文是理解 `普通 CAM + XOR/popcount` 路线的关键。

它们的共同点是：

```text
不把阈值判定直接埋进 ML sensing，
而是显式计算 similarity / distance / inner product。
```

---

## 5.1 del Mundo et al.: NCAM

**发表信息**：MEMSYS 2015。

### 核心问题

kNN / nearest neighbor 在视觉应用里数据搬运太重，CPU/GPU 带宽不足。

### 核心 idea

- 用 near-data associative memory 做 kNN
- 不是精确 CAM 查表，而是 memory-adjacent distance computation

### 对我们的价值

NCAM 的意义在于说明：

```text
相似度检索问题完全可以从“memory-centric nearest neighbor”
而不是“exact CAM lookup”来定义。
```

---

## 5.2 Castañeda et al.: PPAC

**发表信息**：ASAP 2019。

### 核心问题

能不能做一个 fully-digital、标准单元可综合的 associative in-memory processor，同时支持：

- Hamming similarity
- exact CAM
- similarity-threshold matching
- matrix-vector-like operations

### 核心 idea

- bitcell 做 `XNOR`
- subrow / row ALU 做 `population count`
- 阈值 `delta` 做 similarity match

也就是：

\[
h(a,x)=\mathrm{popcount}(\mathrm{XNOR}(a,x))
\]

然后判断：

\[
h(a,x)\ge \delta
\]

这等价于：

\[
HD(a,x)\le N-\delta
\]

### 对我们的价值

这是目录里最明确的：

```text
XNOR + popcount + 阈值比较
```

路线代表。  
如果我们以后需要：

- 精确 HD
- top-k
- 最近邻排序
- 多阈值数字判决

那 PPAC 这条路线比 HD-CAM 更自然。

---

## 5.3 Garzon et al.: AM4

**发表信息**：JETCAS 2023。

### 核心问题

能不能基于 Samsung MRAM crossbar，同一个底座同时支持：

- CAM
- TCAM
- approximate CAM
- associative processor

### 核心 idea

AM4 把 MRAM crossbar 变成一个多模式 associative architecture。

### 对我们的价值

这篇的关键不是某一个具体电路点，而是：

```text
同一底层 crossbar 同时支持 exact / approximate / AP / search
```

它对我们启发很大，因为 GraphhopSimhash 未来也可能需要：

- exact cache hit
- HD-threshold hit
- 更复杂的 support aggregation / reuse policy

如果想把这些统一到一个底层 memory fabric，AM4 很值得参考。

---

## 5.4 MIRACLE

**发表信息**：DAC 2025；multimodal information retrieval 系统。

### 核心问题

多模态检索里，Transformer 特征提取之后的 retrieval stage 仍然有很重的数据搬运和排序开销。

### 核心 idea

- PIM 做随机 hashing / hash mapping
- CAM 做 threshold filtering / Hamming retrieval
- 最后再做 cosine similarity rerank

### 核心结果

- 相比 CPU-based cosine retrieval
  - 延迟降低 `9.45x`
  - 能耗降低 `30.20x`

### 对我们的价值

这篇和 GraphhopSimhash 的系统形态最像：

```text
feature extraction
-> hashing
-> CAM / Hamming-space retrieval
-> 最终 rerank / decision
```

它说明：

- 纯 CAM-based Hamming retrieval 可以作为高效前端
- 但后面仍然可以保留更精细的 rerank 阶段

这和我们当前：

```text
多头 hash
-> CAM / HD-CAM 命中
-> support 聚合
-> direct / residual / compute
```

在系统结构上非常接近。

---

## 5.5 CAMformer

**发表信息**：arXiv 2025；Transformer attention accelerator。

### 核心问题

Transformer attention 的瓶颈在于：

```text
QK^T 相似度计算是 dense similarity search
复杂度随序列长度平方增长
传统实现依赖大量 MatMul 和数据搬运
```

这篇工作提出的核心问题不是“如何做一般的 HD-threshold retrieval”，而是：

```text
能不能把 attention 本身重解释为 associative memory operation，
直接用 CAM 做 similarity search？
```

### 核心 idea

CAMformer 把 attention 解释成一种“query 在 key memory 中做相似性检索”的过程：

- query 向量像 key-lock 里的 “key”
- key matrix 像 associative memory
- attention score 像 similarity search result

在硬件上，CAMformer提出：

- `BA-CAM`：Binary Attention CAM
- 用 voltage-domain charge sharing 直接感知 Hamming similarity
- 用 shared SAR ADC 把 matchline 电压数字化
- 结合 hierarchical two-stage top-k filtering 和 pipelined execution

它的关键点不是显式 `XNOR+popcount`，而是：

```text
把 binary attention score 直接映射成 CAM 内的物理 similarity sensing
```

### 细粒度创新

**BA-CAM cell / array**

- 10T1C CAM cell
- 每个 bit-cell 做 binary matching
- matchline 电压与 Hamming similarity 线性相关

**BIMV engine**

- 把 binary vector-matrix multiplication 实现为 associative retrieval
- 避免传统数字乘加与 popcount

**Top-k filtering**

- attention 并不需要保留所有相似度
- CAMformer 在 CAM 前端直接做 top-k 候选筛选

**Contextualization**

- 最终仍保留高精度 contextualization 路径，兼顾精度与能效

### 核心结果

论文在 BERT 和 ViT 上报告：

- `>10x` energy efficiency
- `up to 4x` throughput
- `6-8x` lower area
- 同时保持 near-lossless accuracy

### 对我们的价值

CAMformer 和 GraphhopSimhash 不在同一个任务层级：

- GraphhopSimhash 更像 `hash retrieval / reuse front-end`
- CAMformer 是 `attention compute accelerator`

但这篇论文有一个非常重要的启发：

```text
CAM 不一定只服务“检索之后的系统”，
它也可以直接成为 similarity compute primitive。
```

也就是说，CAMformer 把 CAM 从：

```text
search / filter / retrieval hardware
```

进一步推进成：

```text
attention / similarity compute engine
```

对于我们当前项目，它的直接价值主要有两点：

1. 它强化了 “Hamming similarity 可以直接在 CAM 中感知” 这一认知，这对理解 `HD-CAM / BA-CAM / TD-CAM` 路线很有帮助。
2. 如果未来 GraphhopSimhash 不只是做缓存复用，而是希望把部分 encoder-side similarity 计算也下沉到 memory fabric，那么 CAMformer 是非常值得跟的方向。

---

## 6. 目录中最值得优先看的论文

如果只考虑和 GraphhopSimhash 当前问题最相关的论文，我会把优先级排成这样。

### 第一优先级：直接相关

1. `Pagiamtzis和Sheikholeslami - 2006 - CAM circuits and architectures`
2. `Garzón 等 - 2022 - HD-CAM`
3. `Garzón 等 - 2023 - Low-Complexity Sensing Scheme`
4. `Garzón 等 - 2025 - 128-kbit Tunable HD-CAM`
5. `Castañeda 等 - 2019 - PPAC`
6. `Jahshan 等 - 2023 - DASH-CAM`

这组基本覆盖了：

- 基础 CAM 电路
- HD-CAM 路线
- XNOR + popcount 路线
- 高密度近似 CAM 路线

### 第二优先级：器件与未来替代底座

1. `Ni 等 - 2025 - TAP-CAM`
2. `Vardar 等 - 2025 - FeFET-Based CAM`
3. `Ryu 等 - 2025 - Time-Domain CAM`
4. `Garzón 等 - 2023 - AM4`
5. `Taco 等 - 2024 - Precharge-Free CAM`

这组适合在后续研究中回答：

```text
如果不满足于 65nm CMOS HD-CAM，
下一步往 FeFET / MRAM / TD-CAM / precharge-free 方向怎么走？
```

### 第三优先级：系统与应用启发

1. `MIRACLE - 2025`
2. `CAMsure - 2017`
3. `NCAM - 2015`
4. `Kaplan 等 - 2017 - ReCAM`
5. `CAMformer - 2025`

这组适合回答：

```text
CAM 前端如何和 hashing、retrieval、PIM、few-shot、multi-modal、attention 等系统模块拼起来？
```

---

## 7. 对 GraphhopSimhash 的直接结论

结合这批论文，可以得到几个比较明确的判断。

### 7.1 对 `8x16-bit, HD<=2`

最相关的是两条路线：

```text
Route A: 普通 CAM chunk coarse filter + XOR/popcount exact verify
Route B: HD-CAM 直接做 HD<=2 threshold compare
```

这两条路线在目录中分别由：

- `PPAC / XNOR+popcount`
- `HD-CAM / DASH-CAM / TAP-CAM`

代表。

### 7.2 如果目标是 strict bit-exact

优先：

```text
普通 CAM + XOR/popcount
```

因为：

- 结果是精确 HD
- 容易做多阈值、排序、top-k
- 纯数字电路更容易验证

### 7.3 如果目标是固定低时延的 threshold hit

优先：

```text
HD-CAM
```

因为：

- 不需要显式 popcount
- 不需要大 candidate queue
- 更像真正单级 CAM hit/miss

### 7.4 对当前文档中的主判断

结合 `16-bit, HD<=2` 的 survivor 理论分析，这批论文支持的结论依然是：

```text
64bit/radius-2: 普通 CAM coarse filter 路线很有竞争力
16bit/head, HD<=2: HD-CAM 路线整体上更自然
```

---

## 8. Paper Coverage

本文覆盖了 `docs/paper/CAM` 下的如下论文：

```text
Pagiamtzis and Sheikholeslami 2006
Yang 2011
Do 2013
Hayashi 2013
Abbas 2014
Karam 2015
del Mundo 2015
Imani 2016
Kaplan 2017
Riazi 2017
Arsovski 2018
Castañeda 2019
Garzón 2022
Hanhan 2022
Garzón 2023 (ML sensing)
Garzón 2023 (AM4)
Jahshan 2023
Taco 2024
Merlin 2024
Garzón 2025
Ni 2025
Vardar 2025
Ryu 2025
Liu 2025 (MIRACLE)
Molom-Ochir 2025 (CAMformer)
```

如果后续继续在 `docs/paper/CAM` 中新增论文，建议按下面顺序补充：

```text
1. 先判断它是 exact CAM / HD-CAM / edit-distance / XNOR-popcount / system application 哪一类
2. 再写“核心问题 / 核心 idea / 对我们的价值”
3. 最后决定它更支持 Route A 还是 Route B
```
