# CAM / Approximate Search Survey

本文档整理 `docs/paper/CAM` 目录下的 CAM 相关论文，主要服务 GraphhopSimhash 后续的硬件架构判断：

```text
1. 普通 CAM / TCAM 的基本电路与功耗优化
2. Hamming-distance tolerant approximate CAM
3. Edit-distance tolerant CAM 与 genomics-oriented CAM
4. 数字相似度计算 / associative processor / PIM / hybrid retrieval 路线
5. 对 8x16-bit、HD<=2、多头 support 聚合架构的启发
```

阅读时需要注意：这个目录里的论文来源比较混合，既有经典综述，也有精确匹配电路论文、近似匹配 CAM、PIM / associative processor、以及面向基因组和多模态检索的系统论文。  
这次整理不再只按年份或论文题目来分，而是优先按下面四类技术轴来组织：

```text
1. 器件底座：CMOS 还是 NVM/ReRAM/MRAM/FeFET/memcapacitor
2. 电路风格：数字动态 / mixed-signal / time-domain / PIM
3. 匹配语义：exact / Hamming distance / edit distance / general similarity
4. 距离实现机制：XNOR+popcount 还是 ML 放电阈值、time-domain、rerank 等
```

---

## 1. 先按技术路线看，不要先按年份看

如果只从 GraphhopSimhash 的需求出发，看 CAM 论文时最先该问的不是“这篇是哪一年”，而是：

```text
它的器件是什么？
它是数字的还是模拟/混合信号的？
它到底在做 exact、HD、edit-distance，还是更广义的 similarity compute？
```

这三件事决定了一篇论文能不能直接迁移到我们当前的：

```text
8 x 16-bit
HD <= 2
support >= 3
多头 hash 复用
```

场景中。

### 1.1 四个最重要的分类轴

**轴 A：器件底座**

- `CMOS`：最成熟，和标准数字/嵌入式存储流程兼容最好。
- `NVM`：包括 `ReRAM`、`MRAM/MTJ`、`FeFET`、`ferroelectric memcapacitor` 等，更强调密度、非易失性和能效潜力。

**轴 B：电路/信号风格**

- `数字动态 CAM`：典型是 NOR/NAND CAM，依赖 precharge/evaluate。
- `mixed-signal / analog CAM`：直接把 mismatch 数映射到电压、电流或时间。
- `time-domain CAM`：把距离编码到延迟，而不是静态电压。
- `PIM / associative processor`：不只做单纯 hit/miss，而是把 CAM 当作计算底座。

**轴 C：匹配语义**

- `exact match`
- `Hamming-distance threshold`
- `edit distance / sequence alignment`
- `hash-space ANN / similarity compute / rerank front-end`

**轴 D：距离实现机制**

- `bitwise exact compare`
- `XNOR + popcount + threshold/ranking`
- `ML discharge / voltage-current threshold sensing`
- `time-domain delay encoding`
- `edit-distance / neighborhood-aware comparison`
- `hash embedding + threshold filtering + rerank`

这里单独把 `XNOR + popcount` 拎出来，是因为它和下面几类机制在工程含义上明显不同：

- `XNOR + popcount`：显式算出 similarity / HD，再做阈值或排序
- `ML discharge threshold`：不显式 popcount，而是靠放电速度或电压窗口直接做判决
- `time-domain`：把距离编码成延迟
- `filter + rerank`：先做低成本筛选，再保留高精度后级

这也是为什么像 `PPAC` 和 `MIRACLE` 虽然都服务于 similarity retrieval，
但实现机制其实完全不同，不能简单归成同一类“XNOR+popcount 路线”。

### 1.2 这批论文的主技术路线

下面这张表是整份 survey 的“导航图”：

| Route | 器件底座 | 风格 | 主要匹配语义 | 典型距离实现机制 | 代表论文 | 一句话定位 |
|---|---|---|---|---|---|---|
| `R1` CMOS exact CAM/TCAM | `CMOS` | 以数字动态为主，少量 mixed-signal sensing | `exact match` / `TCAM` | `bitwise exact compare + ML sensing` | Pagiamtzis 2006, Yang 2011, Do 2013, Hayashi 2013, Arsovski 2018 | 先把经典 CAM/TCAM 电路和 compiler 路线吃透 |
| `R2` precharge-free / 低能耗 exact-or-near-exact CAM | `CMOS` 或 `NVM-oriented` | 仍偏数字，但重写 precharge/charge-retention 行为 | `exact` 或小范围 `HD` 容忍 | `precharge-free / charge-retention compare` | Taco 2024, Imani 2016 | 不直接做通用 HD 阈值，而是先改 CAM 的能耗结构 |
| `R3` CMOS mixed-signal HD-CAM | `CMOS` | `mixed-signal / analog threshold` | `HD <= T` | `ML discharge / voltage-current threshold sensing` | Garzón 2022/2023/2025, Jahshan 2023 | 直接把 Hamming threshold 变成 CAM 内部物理判决 |
| `R4` emerging-device approximate / similarity CAM | `FeFET / MRAM / memcapacitor` 等 | mixed-signal、time-domain 或 crossbar | `HD/similarity` 近似匹配 | `device-native threshold sensing` 或 `time-domain encoding` | Ni 2025, Vardar 2025, Ryu 2025 | 用新器件追求更高密度、更低能耗或更线性的距离编码 |
| `R5` edit-distance / genomics CAM | `CMOS-like` 或 `resistive CAM` | 架构级 + 定制比较逻辑 | `edit distance` / sequence matching | `edit-aware local comparison / sequence scoring` | Hanhan 2022, Kaplan 2017, Merlin 2024 | 目标不是 Hamming，而是插删改都能容忍 |
| `R6a` 显式数字相似度计算 / `XNOR + popcount` / associative processor | `CMOS`、`MRAM` 或可综合 AP | 数字计算、AP/PIM | `exact HD`、`similarity compute` | `XNOR+popcount`、`threshold / ranking / programmable verify` | PPAC 2019, AM4 2023 | 先把距离算出来，再做阈值、排序和验证 |
| `R6b` filter+rerank / similarity sensing / hybrid retrieval | `CMOS`、`MRAM` 或系统级 hybrid | 阈值筛选、physical similarity sensing、系统协同 | `hash retrieval`、`nearest-neighbor`、`similarity front-end` | `threshold filtering + rerank`、`similarity sensing`、`hash embedding + CAM lookup` | NCAM 2015, CAMsure 2017, MIRACLE 2025, CAMformer 2025 | 不把 CAM 限定为单级 hit/miss，而是把它嵌进更大的检索流水线 |

需要特别注意的是：  
这些路线不是严格互斥的，一篇论文经常同时落在“器件路线”和“匹配语义路线”的交叉处。比如：

- `TAP-CAM` 同时属于 `FeFET` 器件路线和 `approximate HD/similarity CAM`
- `CAMsure` 从电路上不强调新 bitcell，但从系统语义上很像 `R6b` 的 `hash-space approximate retrieval`
- `AM4` 既是 `MRAM` 路线，也属于 `R6a` 的 `associative processor / multi-mode memory fabric`

### 1.3 每种技术路线的优缺点

#### `R1` CMOS exact CAM/TCAM

优点：

- 语义最清楚，`match / mismatch` 定义最干净。
- 标准数字流程最成熟，PVT、compiler、macro 化经验最完整。
- 如果后续需要 strict bit-exact baseline，这条路线最稳。

缺点：

- 预充电和 ML/SL 翻转开销大，能耗往往最重。
- 不直接支持 `HD <= T`，通常要额外粗筛或后端 `XOR + popcount`。
- 大规模下搜索能耗和供电噪声管理都很硬。

#### `R2` precharge-free / 低能耗 exact-or-near-exact CAM

优点：

- 试图直接改写 CAM 的基本能耗结构，而不是只在 ML sensing 上修补。
- 保留较多“类数字 CAM”的工程直觉，比纯模拟 HD-CAM 更容易接入 exact 路线。
- 对 cache/tag/associative memory 这类 exact-dominant 任务很有吸引力。

缺点：

- 核心收益主要还是 exact search 的能效，不等于天然解决通用 `HD <= T`。
- 某些设计依赖 refresh、segmentation 或额外逻辑树，复杂度并不低。
- 对 GraphhopSimhash 这种多头近似阈值命中，通常还不是终局方案。

#### `R3` CMOS mixed-signal HD-CAM

优点：

- 最直接回答“能不能在阵列内做 `HD <= T`”。
- 不需要显式 `popcount`，前端阈值 hit/miss 很自然。
- 系统级延迟更接近“单级 associative hit”，适合固定低时延 front-end。

缺点：

- `Vref / Veval / sensing window / comparator offset / PVT` 都会变成工程风险。
- 做多阈值、精确排序、top-k 不如数字 `XOR + popcount` 自由。
- 它更像 threshold engine，而不是一套通用 distance computer。

#### `R4` emerging-device approximate / similarity CAM

优点：

- 非易失器件通常更有密度和待机能耗优势。
- 某些器件天然适合做可调阈值、时间域编码或 crossbar 并行。
- 对 few-shot、similarity search、可调近似匹配很有潜力。

缺点：

- 器件 variability、写入条件、耐久性、外设校准都更复杂。
- 工艺生态和编译器成熟度通常不如 CMOS。
- 很多论文“器件亮点很强”，但离大规模、可复用宏还隔着系统集成距离。

#### `R5` edit-distance / genomics CAM

优点：

- 能处理 insertion / deletion / substitution，不被 Hamming 假设绑死。
- 对 DNA / pathogen / sequence retrieval 这类任务非常贴题。
- 说明 CAM 的“距离度量”可以被重新定义，而不只是一位一位对齐比较。

缺点：

- 对 GraphhopSimhash 当前的二值 hash 复用并不直接。
- 控制逻辑和比较语义明显更复杂，难以拿来当通用二值 CAM baseline。
- 更偏垂直应用优化，而不是普适的 similarity search 前端。

#### `R6a` 显式数字相似度计算 / `XNOR + popcount` / associative processor

优点：

- 结果最可解释，拿到的是精确 HD 或更一般的 similarity score。
- 阈值、排序、top-k、rerank 都更灵活，适合系统级 pipeline。
- 更容易和现有数字后端、控制器、缓存策略对接。

缺点：

- 不是严格意义上的“单级 CAM hit”；经常要付 survivor verify 的代价。
- verified rows 一多，延迟和能耗会很快膨胀。
- 如果目标是稳定常数级 front-end latency，它通常不如 `R3` 自然。

#### `R6b` filter+rerank / similarity sensing / hybrid retrieval

优点：

- 更贴近真实检索系统，容易和 hashing、candidate filtering、rerank pipeline 对接。
- 可以利用 CAM 的物理感知能力先做低成本前端筛选，不必一上来显式算完整距离。
- 对 multi-modal、few-shot、attention、secure retrieval 这类系统任务更自然。

缺点：

- 前端筛选和后端 rerank 往往要一起评估，单看 CAM 宏本身很难得出完整结论。
- 精度、召回率和 latency 的 tradeoff 更依赖 workload 和系统协同，移植性不如 `R6a` 直接。
- 如果需求是 strict bit-exact `HD <= T`，它通常不如 `R3` 或 `R6a` 清晰。

### 1.4 对 GraphhopSimhash 最关键的是哪几条

如果只围绕当前任务：

```text
8 x 16-bit
HD <= 2
support >= 3
```

最关键的是四条，但关注重点不同：

- `R3`：直接做 `HD <= 2` threshold 命中
- `R6a`：显式 `HD` 计算，或普通 CAM 粗筛 + `XOR/popcount` 精确验证
- `R6b`：如果以后要把复用判断接进更长的 `hash -> filter -> rerank` 检索流水线
- `R2`：如果未来优先目标变成“exact hit 的更低能耗底座”，Taco 这类 precharge-free CAM 很值得跟

这也解释了为什么本文后面的重点会放在：

```text
HD-CAM
vs
普通 CAM + XOR/popcount
vs
filter+rerank / similarity-sensing front-end
```

而不是把所有 CAM 论文混在一起看。

## 2. Route `R1 / R2`：基础综述、经典 exact CAM / TCAM 与 precharge-free 路线

这一组论文的重点不是 approximate search，而是帮助建立 CAM 的基本电路直觉：NOR/NAND bitcell、ML/SL sensing、功耗、面积、密度、PVT、TCAM compiler 设计。  
其中：

- `2.1-2.7` 主要属于 `R1`：经典 `CMOS exact CAM/TCAM`
- `2.8` Taco 2024 更接近 `R2`：通过 precharge-free 思路重写 exact CAM 的能耗结构

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

**技术路线标签**：

- 器件：`65nm CMOS`
- 风格：`precharge-free`、仍以数字 CAM 语义为主
- 匹配类型：`exact match`
- 路线：`R2`

### 核心问题

传统 NOR / NAND CAM 都严重依赖预充电：

- NOR：快但耗电
- NAND：省电但慢、可扩展性差

有没有一种 precharge-free CAM？

### 核心 idea

提出 `PCAM`（precharge-free CAM）类设计，核心不是再去修补传统 precharge-evaluate 流程，而是尽量绕开 precharge 本身。

文中给出了两类代表性 cell：

- `N-PCAM`
- `TG-PCAM`

它的目标是把：

```text
NOR CAM 的速度优势
+ NAND CAM 的低能耗倾向
```

合并到一类新的 precharge-free 结构里。

### 核心结果

- `65nm CMOS`
- 做了 `Monte Carlo + process corner + layout parasitics` 评估
- 相比传统 `NAND CAM`
  - 搜索时间改善 `>30%`
  - 搜索能耗降低约 `15%`
- 相比传统 `NOR CAM`
  - 搜索能耗降低 `>75%`
- 在 fully associative cache 应用级评估里
  - 相比 `NOR CAM` 可做到约 `3.3x` 能耗下降
  - 同时保持相近的搜索运行时间

### 这篇论文在技术路线中的位置

它不是：

- 直接做 `HD <= T` 的 `HD-CAM`
- 也不是 `XNOR + popcount` 路线

它真正回答的是：

```text
exact CAM 的底座能不能不依赖重 precharge，
从而把 NOR 的速度和 NAND 的能耗折中得更好？
```

所以它更像：

```text
exact associative search 的低能耗底座路线
```

而不是 GraphhopSimhash 当前最关心的“通用 Hamming threshold engine”。

### 优点

- 不依赖传统 precharge，直接瞄准 CAM 的主能耗源。
- 仍保留 exact associative memory 的语义，工程直觉比纯模拟 HD-CAM 更接近常规 CAM。
- 论文把 circuit-level 和 cache-level 两层都补到了，工程味比较强。

### 局限

- 它解决的是 exact CAM 能耗问题，不是通用 `HD<=2` 阈值比较。
- 如果我们的目标是阵列内直接完成 Hamming threshold hit，它不是直接替代 `HD-CAM` 的方案。
- 新 cell / logic-tree / segmentation 结构虽然有效，但并不等于比标准 NOR/NAND compiler 更成熟。

### 对我们的价值

这篇的意义在于：  
**它试图从 CAM 基本工作模式本身改写能耗结构，而不是只在原有 precharge CAM 上做打补丁式优化。**

对 GraphhopSimhash，这篇最重要的启发不是“直接拿来做 `HD<=2`”，而是：

- 如果我们未来要保留一个 `exact CAM baseline`
- 又希望它的能耗结构比经典 NOR/NAND 更好

那么 `PCAM` 这条 precharge-free 路线值得单独跟进。

---

## 3. Route `R3 / R4`：Hamming-Distance Approximate CAM 主线

这一组论文最直接对应我们的核心问题：

```text
能不能让 CAM 直接比较 HD <= T，
而不是先粗筛再 XOR+popcount？
```

其中可以再分成两支：

- `3.1-3.6`：主要是 `CMOS mixed-signal HD-CAM`
- `3.7-3.9`：主要是 `FeFET / memcapacitor` 等 emerging-device approximate CAM

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

## 4. Route `R5`：Edit-Distance / Genomics 定向 CAM

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

## 5. Route `R6` 家族：数字相似度计算 / Associative Processor / PIM / Hybrid Retrieval 路线

这一组论文的共同点，不是都用了同一种距离实现机制，而是都不把 CAM 只当成单级 hit/miss 宏。

它们的共同点是：

```text
把 CAM / associative memory 嵌进更大的 similarity pipeline：
要么显式计算 distance，
要么先做 threshold filtering，
要么把 CAM 当作 associative compute primitive。
```

要特别强调：

- `R6a` 代表的是“显式把距离算出来”，最典型就是 `XNOR + popcount`
- `R6b` 代表的是“先筛选或先感知，再把高精度工作交给后级”
- `PPAC` 属于 `R6a`
- `CAMsure`、`NCAM`、`MIRACLE` 和 `CAMformer` 属于 `R6b`

---

### 5.1 `R6a`：显式 `XNOR + popcount` / 数字相似度计算

这一支的关键特征是：

```text
先把 similarity / HD 显式算出来，
再做阈值、排序、top-k 或 verify
```

如果我们以后想保留严格的数字可解释性，或者需要多阈值与排序控制，这一支最值得看。


#### 5.1.0 Manku et al.：Simhash Near-Duplicate Detection（粗筛 + 验证的软件源头）

**发表信息**：WWW 2007；Google simhash 近重复文档检测系统。

**技术路线标签**：

- 器件：不涉及硬件（纯软件 / 分布式系统）
- 风格：simhash fingerprint + Hamming distance 搜索
- 匹配类型：`HD ≤ k`（64-bit, k=3）
- 路线：`R6a`（两层粗筛 + 验证的策略源头）
- 距离实现机制：`block partition → exact prefix match → XOR/popcount verify`

### 核心问题

8B 网页，每页生成 64-bit simhash fingerprint。给定一个新 fingerprint F，如何快速找到已有 fingerprint 中与 F 汉明距离 ≤3 的条目？

### 核心 idea

把 "先粗筛、再精确验证" 提炼成一条清晰的两层策略：

1. **block partition + prefix match（粗筛）**：
   - 将 64-bit 分成多个 block（如 4×16-bit 或 6×11/11/11/11/10/10-bit）
   - 对每个 block 选择不同的 permutation，将选中的 block 移到 leading bits
   - 构建多张 sorted table，每张表按 leading bits 排序
   - 查询时，在 leading bits 上做 prefix match → 粗筛出候选集

2. **XOR + popcount（精确验证）**：
   - 对候选 fingerprint 做完整 XOR
   - 统计差异 bit 是否 ≤ 3 → 最终判断

关键见解（论文 §3.1）：

> "A single probe suffices to identify all fingerprints which match F in d' most significant bit-positions. For each matching fingerprint, we can easily figure out if it differs from F in at most k bit-positions or not."

### 在技术路线中的位置

这篇论文不是硬件 CAM 论文（没有 bitcell / matchline / sense amplifier）。它是：

```text
分布式系统的 sorted table + binary search
```

但它在概念上精确对应了硬件中 "chunk CAM 粗筛 + XOR/popcount 验证" 的两层结构：

| Manku 2007 软件版 | 数字引擎硬件版 |
|---|---|
| 64-bit 分 4×16-bit block | 16-bit 分 4×4-bit chunk |
| sorted table prefix match 粗筛 | CAM chunk exact match 粗筛 |
| XOR + 差异 bit 计数 ≤ 3 | XOR + popcount ≤ 2 |
| 多表覆盖所有 3-bit 差异情况 | 8-head 独立搜索 + support ≥ 3 |

### 优点

- 两层策略清晰，可解释性强
- 在 8B 网页规模上经过 Google 生产验证
- 多表 permutation 策略保障了 Hamming 球的覆盖完整性
- 对硬件设计有直接的 architecture-level 启发

### 局限

- 纯软件 / 分布式系统，不涉及 bitcell / ML / sensing 等电路层面
- 表是 sorted array，不是并行 CAM search
- 查询延迟受 binary search / index lookup 限制，不是单周期 associative hit

### 对我们的价值

这篇论文是 **"粗筛 + 验证"两层策略的明确软件源头**。对于 GraphhopSimhash：

1. 它为数字引擎（`chunk CAM coarse filter + XOR/popcount verify`）提供了**软件层面的 prior art**
2. 多表 permutation 覆盖完整 Hamming 球的思想，和 8-head 独立搜索 + support aggregation 在策略层面对齐
3. 在论文中引用它，可以从软件搜索策略自然过渡到硬件加速的动机——"这个方法在 8B 网页上效果很好，但每次查询需要多表 probe，延迟不恒定；如果把它搬到 CAM 硬件里做单周期粗筛，就能把 Hamming 搜索做成固定低时延前端"



#### 5.1.0b Shinde et al.：用商用 TCAM 做 simhash 近似搜索（Manku 路线的 CAM 硬件延伸）

**发表信息**：SIGMOD 2010；Stanford + Cisco。

**技术路线标签**：

- 器件：商用 `TCAM`（Ternary CAM，Cisco 交换机内置）
- 风格：`TLSH (Ternary LSH)` → TCAM wildcard `*` 模糊匹配 → Euclidean 距离验证
- 匹配类型：`(1,c)-NN` 近似最近邻搜索
- 路线：`R6b`（TCAM-based similarity search / hybrid retrieval）
- 距离实现机制：`ternary hash embedding → TCAM wildcard match → distance verification`（注意：**不是** XOR+popcount）

### 核心问题

LSH 在 c≈1 时空间需求接近 n²。能否用 TCAM 的 wildcard `*` 特性做硬件加速？

### 核心 idea

设计 **TLSH (Ternary Locality Sensitive Hashing)**：
- 将标准 LSH 的二元哈希 `{0, 1}` 扩展为三元哈希 `{0, 1, *}`
- 在随机方向上取超平面分区，交替区域标记为 `0, *, 1, *, 0, *, 1...`
- `*` 起到"模糊边界"作用 → 相邻点有更高概率在 TCAM 中匹配
- 将三元签名存入 TCAM，单次 O(1) TCAM lookup 返回所有 wildcard 匹配的候选
- 后端用 Euclidean 距离做精确验证

### 与我们的数字引擎的区别（重要）

Shinde 2010 **没有**使用 "chunk exact match 粗筛 + XOR/popcount 验证"。它的机制是：

| 要素 | Shinde 2010 | 我们的数字引擎 |
|---|---|---|
| 粗筛机制 | TCAM wildcard `*` 模糊匹配 | 普通 CAM chunk exact match |
| 验证方式 | Euclidean 距离计算 | XOR + popcount |
| 哈希方式 | TLSH 三元嵌入 (0/1/*) | 8×16-bit 二值 hash |
| 硬件 | 商用 TCAM (72/144/288-bit) | 定制 CAM + verifier lanes |

它和我们在"用 CAM 硬件做近似搜索"这个方向上**动机相同**，但实现机制不同。

### 与 Manku 2007 的关系

- 实验直接用 Manku 2007 的 simhash Wikipedia 数据集
- 但匹配机制是 TCAM wildcard，不是 Manku 的 block partition + prefix match
- 288-bit TCAM 在 1M 点数据集上做到 (1,2)-NN，F-score > 0.95
- 在 Cisco Catalyst 4500 交换机上实测 1.5M queries/s per 1Gbps port

### 对我们的价值

1. 证明了"商用 CAM（TCAM）+ simhash"这条路径在小阈值近似搜索上是可行的，给了我们做定制 CAM 的信心
2. TCAM 的 `*` wildcard 机制在概念上类似 HD-CAM 的 Veval/Vref 阈值放缩——都是通过放宽匹配条件来容忍小汉明距离
3. 但它的具体实现（ternary embedding + Euclidean verify）**和我们的 chunk CAM + XOR/popcount 是两条不同的技术路线**，引用时需要注意区分


#### 5.1.1 Castañeda et al.: PPAC

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

#### 5.1.2 Garzon et al.: AM4

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
它不一定是最教科书式的 `XNOR + popcount` 论文，但从“显式可编程相似度计算 / associative compute fabric”这个角度看，更接近 `R6a`。

---

### 5.2 `R6b`：filter+rerank / similarity sensing / hybrid retrieval

这一支的关键特征是：

```text
CAM 前端先做候选筛选、物理相似度感知或 hash-space retrieval，
再把高精度工作交给 rerank / verify / decision
```

如果我们以后要把 GraphhopSimhash 扩成更完整的检索流水线，这一支会比 `R6a` 更贴系统。

#### 5.2.1 CAMsure（交叉论文，正文见 §3.2）

`CAMsure` 的正文已经放在 [CAM_SURVEY.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/paper/survey/CAM_SURVEY.md:548)，这里仅在路线图里重新归类：

- 不是显式 `XNOR + popcount`
- 更像 `secure LSH embedding + CAM lookup`
- 从系统语义上属于 `R6b` 的 `hash-space approximate retrieval`

#### 5.2.2 del Mundo et al.: NCAM

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

#### 5.2.3 MIRACLE

**发表信息**：DAC 2025；multimodal information retrieval 系统。

**技术路线标签**：

- 器件：`STT-MRAM`
- 风格：`PIM + MRAM-CAM hybrid`
- 匹配类型：`hash-space Hamming filtering + cosine rerank`
- 路线：`R6b`
- 距离实现机制：`MRAM ML discharge-speed threshold filtering`

### 核心问题

多模态检索里，Transformer 特征提取之后的 retrieval stage 仍然有很重的数据搬运和排序开销。

### 核心 idea

- `STT-MRAM` PIM 做随机 hashing / hash mapping
- `STT-MRAM CAM` 做 segmented Hamming threshold filtering
- 最后再做 cosine similarity rerank

更准确地说，它不是用一个显式的：

```text
XNOR + popcount
```

数字后端去做完整的 Hamming 排序，而是利用 MRAM CAM 单元的：

```text
match-line discharge pattern / discharge speed
+ comparator threshold monitoring
```

来做筛选，再把高精度工作留给最后的 cosine rerank。

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

- 纯 CAM-based Hamming filtering 可以作为高效前端
- 但后面仍然可以保留更精细的 rerank 阶段

### 这篇在分类上的关键点

这篇最容易被误分到：

```text
XNOR + popcount
```

但更准确的说法应该是：

- 底层器件是 `STT-MRAM`
- CAM 端利用 `ML` 放电/放电速度做阈值筛选
- 系统上属于 `filter + rerank` 混合检索框架

所以它更像：

```text
MRAM-based threshold filtering CAM
+ hybrid retrieval pipeline
```

而不是一个标准的 `XNOR + popcount` 数字距离计算器。

这和我们当前：

```text
多头 hash
-> CAM / HD-CAM 命中
-> support 聚合
-> direct / residual / compute
```

在系统结构上非常接近。

---

#### 5.2.4 CAMformer

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
- `R6a` / XNOR + popcount 路线
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

最相关的是三条路线，但主次不同：

```text
Route A: 普通 CAM chunk coarse filter + XOR/popcount exact verify (`R6a`)
Route B: HD-CAM 直接做 HD<=2 threshold compare (`R3`)
Route C: hash/filter/rerank hybrid retrieval (`R6b`，当前是次相关)
```

其中前两条是当前主线，后一条更像系统外延。它们在目录中分别由：

- `PPAC / XNOR+popcount`
- `HD-CAM / DASH-CAM / TAP-CAM`
- `CAMsure / MIRACLE / CAMformer`

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

## 8. Paper Coverage（按技术路线索引）

为了后续继续维护，这里不再只给论文名单，而是把每篇论文映射到：

```text
器件底座
模拟/数字风格
匹配类型
所属技术路线
```

| Paper | 器件/底座 | 风格 | 主要匹配类型 | 距离实现机制 | Route | 备注 |
|---|---|---|---|---|---|---|
| Pagiamtzis & Sheikholeslami 2006 | `CMOS` | tutorial，覆盖数字与 sensing | `exact / TCAM` | `bitwise exact compare` | `R1` | 经典入门综述 |
| Yang 2011 | `CMOS` | 低摆幅 SL，偏数字 | `exact` | `bitwise exact compare + low-swing SL` | `R1` | SL 能耗优化 |
| Do 2013 | `CMOS` | 数字 + ML sensing | `exact` | `bitwise exact compare + power-gated sensing` | `R1` | parity / power-gated sensing |
| Hayashi 2013 | `CMOS TCAM` | mixed-signal sensing | `exact TCAM` | `bitwise exact compare + low-voltage ML sensing` | `R1` | 大容量工业风格 TCAM |
| Abbas 2014 | `CMOS` | 数字 | `exact` | `bitwise exact compare + ECC/parity` | `R1` | MCU 容错 CAM |
| Karam 2015 | `CMOS + NVM` | survey | `exact + associative variants` | `route map` | `route map` | 器件路线图 |
| del Mundo 2015 (NCAM) | `CMOS-like system` | 近数据 / 架构级 | `nearest-neighbor` | `distance compute near memory` | `R6b` | 系统级 associative retrieval |
| Imani 2016 (MASC) | `NVM-oriented TCAM` | charge-retention / 数字近似 | `exact + 1/2-HD` | `single-charge / refresh-tuned approximate compare` | `R2` | 多次搜索单次充电 |
| Kaplan 2017 | `Resistive CAM` | processing-in-storage | `sequence alignment / edit-like` | `sequence scoring in storage` | `R5` | genomics / DNA |
| Riazi 2017 (CAMsure) | `device-agnostic CAM system` | 数字 / 安全检索 | `hash-space approximate search` | `secure LSH embedding + CAM lookup` | `R6b` | secure LSH + CAM |
| Arsovski 2018 | `CMOS TCAM compiler` | 数字动态 | `exact TCAM` | `bitwise exact compare + two-phase ML sensing` | `R1` | compiler / power grid |
| Castañeda 2019 (PPAC) | `associative in-memory accelerator` | AP / PIM | `MVP-like similarity compute` | `XNOR + popcount + threshold` | `R6a` | 更像计算底座而非 threshold CAM |
| Garzón 2022 (HD-CAM) | `CMOS` | mixed-signal / analog threshold | `Hamming threshold` | `ML discharge / analog threshold sensing` | `R3` | 直接 HD-CAM |
| Hanhan 2022 (EDAM) | `conventional CAM-style` | 架构级近似比较 | `edit distance` | `neighbor-aware edit comparison` | `R5` | indel tolerant |
| Garzón 2023 (ML sensing) | `CMOS` | mixed-signal | `Hamming threshold` | `ML discharge sensing simplification` | `R3` | 低复杂度 sensing |
| Garzón 2023 (AM4) | `MRAM crossbar` | associative processor / multi-mode | `exact + approximate + AP` | `multi-mode associative crossbar compute` | `R6a` | 多模式 memory fabric，连接 AP 与 hybrid retrieval |
| Jahshan 2023 (DASH-CAM) | `CMOS-like CAM` | dynamic approximate search | `Hamming threshold` | `dynamic threshold search on CAM` | `R3` | genome classification |
| Taco 2024 | `65nm CMOS` | precharge-free，偏数字 | `exact` | `precharge-free compare` | `R2` | PCAM，低能耗 exact CAM 底座 |
| Merlin 2024 (DIPER) | `Resistive CAM` | system-oriented approximate CAM | `edit distance` | `resistive edit-distance filtering` | `R5` | pathogen detection |
| Garzón 2025 (128-kbit CAM) | `CMOS` | mixed-signal / tunable threshold | `tunable Hamming threshold` | `tunable ML discharge threshold` | `R3` | 工程最强 HD-CAM 基准 |
| Ni 2025 (TAP-CAM) | `FeFET` | mixed-signal | `tunable approximate / similarity` | `FeFET tunable threshold matching` | `R4` | FeFET approximate CAM |
| Vardar 2025 | `FeFET` | mixed-signal | `Hamming / similarity search` | `FeFET-based in-memory Hamming/similarity sensing` | `R4` | few-shot + retrieval |
| Ryu 2025 | `ferroelectric memcapacitor` | `time-domain` | `distance-coded similarity` | `time-domain delay encoding` | `R4` | 延迟近线性编码距离 |
| Liu 2025 (MIRACLE) | `STT-MRAM` | `PIM + MRAM-CAM hybrid` | `Hamming filtering + rerank` | `MRAM ML discharge-speed threshold filtering + cosine rerank` | `R6b` | 最像真实检索系统，但不是 XNOR+popcount |
| Molom-Ochir 2025 (CAMformer) | `CMOS CAM fabric` | analog / mixed-signal similarity sensing | `attention similarity / top-k` | `voltage-domain similarity sensing + top-k filtering` | `R6b` | 把 CAM 当 similarity compute primitive |

如果后续继续在 `docs/paper/CAM` 中新增论文，建议按下面顺序补充：

```text
1. 先判断它属于哪条技术路线：R1/R2/R3/R4/R5/R6a/R6b
2. 再标出器件、风格、匹配类型
3. 然后再写“核心问题 / 核心 idea / 对我们的价值”
4. 最后判断它更支持 exact baseline、HD-CAM，还是 `R6a` 的 `XOR+popcount` / `R6b` 的 hybrid retrieval
```
