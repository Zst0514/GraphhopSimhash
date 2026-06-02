# CAM 设计

## 1. 目标

本文说明我们在 GraphhopSimhash 中如何把普通 CAM 改造成 HD-CAM，并用于：

```text
8 个 16bit 哈希头
汉明距离阈值匹配
多头 support 聚合
direct / residual / compute 三段式复用
```

我们的目标不是只做精确检索，而是找“足够接近”的复用候选。

---

## 2. 普通 CAM 做什么

普通 CAM 的能力很直接：

```text
把查询值同时和所有存储行比较，
找出完全相同的条目。
```

对应架构如下：

![普通 CAM 架构](../figures/CAM.png)

图里可以看成三部分：

- `CAM Array`：每行一个存储 word，每列一个 bit 位
- `Search Data Registers / Drivers`：把查询值广播到所有 bitcell
- `ML sense amplifiers`：最终输出 `hit / miss`

普通 CAM 的本质是：

```text
逐行并行，逐位精确匹配
```

它适合 exact match，不适合直接回答“差 1 位、差 2 位是否也算命中”。

之前采用的普通 CAM 检索（CAM粗筛 + XOR/popcount）流程如下：

![普通 CAM 检索流程](../figures/CAM_forward.png)

---

## 3. 为什么要上 HD-CAM

在哈希复用里，真正有价值的候选往往不是完全相同，而是：

```text
汉明距离很小
```

我们现在是：

```text
8 个 head
每个 head 16bit
```

如果继续用普通 CAM，只能做“16bit 完全一样才命中”，太严格了。  
所以需要把 CAM 改成：

```text
找出所有 HD <= T 的行
```

这就是 HD-CAM 的作用。

---

## 4. HD-CAM cell

HD-CAM 的关键不是把 CAM 改成更复杂的存储单元，而是让它的 match line 能反映“差了几位”。

参考的 HD-CAM cell 如下：

![HD-CAM bitcell](../figures/HD-CAM_cell.png)

这里最关键的是新增的：

- `Meval`
- `Veval`

普通 CAM 只判断“有没有不匹配”；  
HD-CAM 则让“不匹配位的数量”影响 match line 的放电速度。

直觉上就是：

- 不匹配位少，ML 放电慢
- 不匹配位多，ML 放电快

因此，match line 不再只是二值信号，而是一个近似距离的物理载体。

---

## 5. 物理判决逻辑

HD-CAM 的基本判决可以抽象成：

```text
V_ML = VDD * exp(-G_total * t_eval / C_ML)
```

其中：

- `VDD`：供电电压
- `G_total`：总放电导通
- `t_eval`：评估时间
- `C_ML`：match line 电容

核心关系很简单：

```text
汉明距离越大 -> 放电越快 -> V_ML 越低
汉明距离越小 -> 放电越慢 -> V_ML 越高
```

然后用参考阈值 `Vref` 做比较：

```text
V_ML >= Vref -> hit
V_ML <  Vref -> miss
```

如果把 `Vref` 放在 `d = 2` 和 `d = 3` 之间，就能近似实现：

```text
HD <= 2 -> 命中
HD >= 3 -> 不命中
```

这比普通 CAM 的 exact match 更适合哈希复用。

---

## 6. 8 个 16bit head 的组织方式

我们不把一个长哈希直接丢进 CAM，而是采用：

```text
8 个独立 head
每个 head 16bit
```

对应到硬件上，就是 8 个并行 bank：

```text
Head0 -> HD-CAM bank 0
Head1 -> HD-CAM bank 1
...
Head7 -> HD-CAM bank 7
```

这样做有三个好处：

1. 单个 bank 字长短，阈值更容易调
2. 8 个 head 可以天然提供 support 置信度
3. 前端候选筛选和后端复用决策可以解耦

---

## 7. 我们的完整哈希复用架构

我们把上面的思想组合成一张完整架构图：

![HD-CAM 哈希复用架构图](../figures/HD-CAM_hash_reuse_architecture.png)

这张图表达的是整条链路：

1. 输入节点进入前端
2. 生成 8 个 16bit 哈希头
3. 8 个 HD-CAM bank 并行做阈值搜索
4. 候选按 `node_id` 聚合，统计 `support count`
5. 进入三段式决策：
   - `support 高` -> 直接复用
   - `support 中` -> 残差复用
   - `support 低` -> 重新计算
6. 结果写回缓存

这张图对应的不是单纯的 CAM 检索，而是一个完整的哈希复用前端。

---

## 8. 三段式决策

HD-CAM 前端返回的不是最终 embedding，而是候选和 support。

因此后端按 support 做三段式决策：

```text
support 高  -> hard direct reuse
support 中  -> residual reuse
support 低  -> compute
```

在当前 Cora/PubMed 共同参数探索后，主线固定切法是：

```text
R = 2
8 heads x 16 bits
score threshold T = 40
hard_direct >= 5
residual_soft = 4
compute < 4
```

含义很直接：

- `5~8` 个 head 同时支持：直接复用，风险低
- `4` 个 head 同时支持：进入 residual correction
- `0~3` 个 head 同时支持：置信度不够，重算

这组固定参数在 3-run 结果里得到：

```text
Cora:   reuse = 25.7%, drop = 0.45%
PubMed: reuse = 50.3%, drop = 2.52%
```

`hard_direct >= 6, residual_soft = 4` 的总 reuse 相同，但 PubMed drop 略高，说明 support=5 的节点已经足够可靠，直接复用比强制 residual correction 更稳。

所以，HD-CAM 只负责“找近似候选”，最终能不能复用，要看多头 support 聚合后的结果。

---

## 9. 这种设计为什么适合 GraphhopSimhash

这套设计和我们的任务是对齐的。

### 9.1 复用需要的是近似，不是精确

GraphhopSimhash 不追求“完全相同”，而是要找“足够相似”的旧节点。

### 9.2 多头 support 比单次命中更稳

一个候选如果在多个 head 上都命中，说明它不是偶然碰撞，而是更像真正可复用的锚点。

### 9.3 16bit/head 更适合做阈值感测

短 word 更容易把 `d=2` 和 `d=3` 拉开；这比直接做一个超长 128bit CAM 更现实。

### 9.4 前端与后端职责清晰

- HD-CAM 前端：负责近似搜索和候选筛选
- 数字后端：负责 support 聚合、残差修正、最终决策

这样系统更清楚，也更容易继续优化。

---

## 10. 频率估算与工艺边界

Garzón 等人的 HD-CAM 论文给出的硅片基线是：

```text
128-kbit approximate-search CAM
65nm CMOS
64bit query / CAM word 量级
最高测试频率 125MHz
```

这个结果可以作为我们的硬件外推锚点，但不能直接当成 GraphhopSimhash 的实测频率。原因是我们的前端不是单个长 word，而是：

```text
8 个并行 HD-CAM bank
每个 bank 只做 16bit/head
每个 head 的汉明距离阈值 R = 2
```

从 match line 物理行为看，16bit/head 比 64bit word 更容易跑高频：

```text
word 从 64bit 缩到 16bit
-> 单条 match line 上挂载的 bitcell 数量约降到 1/4
-> C_ML 显著下降
-> precharge / evaluation 的 RC 延迟下降
-> 允许更短搜索周期
```

`R = 2` 也是比较自然的阈值选择：

```text
16bit 下 HD = 2  -> 2 / 16 = 12.5%
64bit 下 HD = 8  -> 8 / 64 = 12.5%
```

也就是说，16bit/head, R=2 保留了和 64bit, R=8 类似的相对容错比例，同时把每条 match line 的负载大幅缩短。工程上可以把它理解成“更短的模拟判决线 + 合理的 HD 感测裕度”。

在没有重新版图和 SPICE/PVT 仿真的情况下，本文只采用如下估算口径：

| 设计点 | 频率结论 | 说明 |
|---|---:|---|
| 65nm, 64bit HD-CAM | 125MHz | Garzón 等人的硅片实测基线 |
| 65nm, 16bit/head, R=2 | 300-500MHz | 基于 match line 负载下降的工程外推 |
| 28nm, 16bit/head, R=2 | 0.8-1.5GHz | 同时考虑短 word 和工艺缩放后的工程外推 |

因此，如果把 GraphhopSimhash 的 HD-CAM 前端实现为 28nm 下的 `8 x 16bit` bank，并采用 `R=2`，那么把系统级仿真频率设在 `1GHz` 是一个合理但仍偏模型化的假设。它不是硅片实测结论，正式论文中应写成：

```text
We use a 1GHz HD-CAM frontend frequency as a technology-scaled modeling assumption
for a 28nm 8x16-bit multi-head design. This is extrapolated from the 65nm
64-bit HD-CAM silicon baseline, not measured silicon.
```

边界也要明确：

- 最高频率最终取决于版图后的 `C_ML`、search-line driver、MLSA、precharge、电源电压和 PVT corner。
- `16bit, R=2` 的 false match / false miss 需要用 Monte Carlo 和 extracted parasitics 验证。
- 本文可以 claim `16bit/head 降低 match-line 负载并支持更高频建模`，不要 claim `28nm 实测达到 1GHz`。

---

## 11. 一句话总结

我们这里的 HD-CAM 设计可以概括成一句话：

```text
把普通 CAM 的“完全相同才命中”，改成“match line 放电速度反映汉明距离”，
再让 8 个 16bit head 并行搜索，最后通过 support 聚合决定 direct / residual / compute。
```

---

## 12. 两条实现路线的评价总结

针对当前配置：

```text
8 个 head
每个 head 16bit
每个 CAM 的汉明距离阈值 HD <= 2
```

目前有两条可行路线：

### 12.1 路线 A：普通 CAM 分块粗筛 + XOR/Popcount 精确校验

思路如下：

```text
16bit query
-> 切成 n 个 chunk
-> 每个 chunk 用普通 CAM 做 exact match 粗筛
-> 保留满足“至少 n-2 个 chunk 相等”的候选
-> 对候选做 XOR + popcount
-> 最终判定 HD <= 2
```

这个路线的本质是：

```text
前端做 cheap filter
后端做 exact verify
```

它的优点是：

- 最终结果是 `bit-exact` 的，没有 false positive / false negative
- 可以直接得到精确汉明距离，而不只是 `hit/miss`
- 后续如果要做 top-k、最近邻排序、rerank，这条路线更自然
- 全数字实现，RTL、STA、功能验证和跨工艺迁移都更直接

它的缺点是：

- 16bit 很短，`HD<=2` 时分块粗筛的收益不如 64bit 场景明显
- 分块后会复制 CAM 子阵列、matchline、编码和候选聚合逻辑，外围开销偏大
- 如果粗筛不够锋利，仍会留下较多候选进入 `XOR + popcount` 二级校验
- 整体是两级路径，最坏情况延迟大于单级阈值比较

这里最关键的判断是：**图中的 `64bit -> 4 x 16bit chunk` 流程在 64bit/radius-2 上很合适，但缩到 16bit 后，同样的思路不再那么占优。**

原因是 16bit 太短：

- 如果切成 `4 x 4bit`，则真命中要求至少 `2/4` 个 chunk 完全相等，但随机候选通过粗筛的概率仍然不低
- 如果继续切细，例如切成 `8 x 2bit`，粗筛会更锋利，但 chunk 数、投票逻辑和外围管理成本也会进一步上升
- 如果切到 `1bit`，则已经接近直接做 popcount，本身就失去 coarse filter 的意义

换句话说：

```text
路线 A 的核心优势来自“大 bit-width + 小阈值”。
当 word length 缩到 16bit 时，这个优势会明显下降。
```

### 12.2 路线 B：HD-CAM 直接做汉明距离阈值比较

思路如下：

```text
16bit query
-> 直接送入 16bit HD-CAM bank
-> 利用 match line 放电速度/电压反映 mismatch 数量
-> 直接判定 HD <= 2 是否成立
```

这个路线的本质是：

```text
不显式求 HD 数值，
而是直接做 thresholded approximate match
```

它的优点是：

- 单级判定，路径更短，更接近真正 CAM 的单拍比较风格
- 不需要 chunk 切分、vote counter、candidate queue、二级 full-hash 读出
- 对于只关心 `HD<=2?` 的应用，面积和功耗通常更有优势
- `8 x 16bit` 的 bank 组织天然适合多头 support 聚合

它的缺点是：

- 本质上不是数字精确计数，而是 mixed-signal threshold sensing
- 结果受 `Veval`、`Vref`、PVT、版图寄生和 sensing margin 影响
- 如果后续需要精确 HD 数值、top-k 或最近邻排序，就不如 `XOR + popcount` 直接
- 设计和验证门槛高于普通数字逻辑

### 12.3 针对 `8 x 16bit, HD<=2` 的判断

对于当前这个具体设计点，我的结论是：

```text
如果目标是 strict bit-exact + 数字友好实现，选路线 A；
如果目标是低延迟、低外围开销、直接做阈值命中，选路线 B。
```

但如果必须给出一个更偏工程决策的主判断，那么我更倾向于：

```text
在 8 x 16bit, HD<=2 这个点上，
HD-CAM 直接阈值比较整体上更自然。
```

原因如下。

#### (1) 16bit 太短，路线 A 的 coarse filter 不够“锋利”

在 64bit / radius-2 时，分成 `4 x 16bit chunk` 后，随机数据很难同时命中多个 16bit chunk，因此粗筛非常有效。

但在 16bit / radius-2 时：

- 只能切成更小的 chunk
- 小 chunk 的 exact match 碰撞概率会上升
- 结果就是粗筛后的候选数量不会像 64bit 那样快速下降

所以：

```text
路线 A 在 16bit 上仍然可用，
但其“CAM coarse filter”的收益已经明显弱于 64bit 场景。
```

#### (2) 路线 A 的外围逻辑开始变得不太划算

对 8 个 head，如果每个 16bit head 再切 chunk，那么系统里除了 CAM 本体，还要承担：

- chunk 级 CAM 子阵列复制
- 多路 matchline 输出汇聚
- vote / support 统计
- 候选队列
- full-hash 读出
- XOR + popcount verifier

当 word 长度只有 16bit 时，这些外围会比长字长场景更显眼。

#### (3) 路线 B 更契合“只判断是否命中阈值”的任务

我们这里的前端职责不是：

```text
给出每个候选的精确 HD 排名
```

而是：

```text
快速筛出“足够近”的候选
```

这和 HD-CAM 的能力正好对齐：

- 它最擅长的是 `HD <= T ?`
- 而不是输出一个精确距离值

在这个任务定义下，路线 B 更专用，也更直接。

### 12.4 `16-bit, HD<=2` 的不同切法对比

为了更直观地说明普通 CAM 分块粗筛在 `16-bit, HD<=2` 下的筛选强度，我们以：

```text
数据库规模 N = 160,000
每个 survivor 只回读该 head 的 16-bit = 2 bytes
```

为基线，给出不同切法的理论比较结果。这里：

- `粗筛通过概率` 指随机候选通过第一级 chunk-CAM coarse filter 的概率
- `期望 survivor/head` = `N x 粗筛通过概率`
- `相对真命中放大量` 以随机 16-bit 下 `HD<=2` 的期望真命中数 `334.5` 为基准
- `二级回读/head` 表示每个 head 进入 `XOR + popcount` verifier 的数据读出量

| 切法 | 粗筛通过概率 | 期望 survivor/head | 相对真命中放大量 | 二级回读/head | 二级回读/8 heads |
|---|---:|---:|---:|---:|---:|
| `4+4+4+4` | `2.153%` | `3445` | `10.3x` | `6.73 KB` | `53.8 KB` |
| `4+3+3+3+3` | `1.157%` | `1851` | `5.53x` | `3.61 KB` | `28.9 KB` |
| `3+3+3+3+2+2` | `0.772%` | `1235` | `3.69x` | `2.41 KB` | `19.3 KB` |
| `3+3+2+2+2+2+2` | `0.578%` | `925` | `2.77x` | `1.81 KB` | `14.5 KB` |
| `2+2+2+2+2+2+2+2` | `0.423%` | `676` | `2.02x` | `1.32 KB` | `10.6 KB` |
| `2+2+2+2+2+2+2+1+1` | `0.391%` | `625` | `1.87x` | `1.22 KB` | `9.77 KB` |
| `1x16` | `0.209%` | `334.5` | `1.00x` | `0.65 KB` | `5.23 KB` |

这张表说明了三件事：

1. `4+4+4+4` 的 coarse filter 虽然已经有效，但仍会留下大约 `3445` 个候选，约为期望真命中数的 `10.3x`。
2. 如果继续沿着普通 CAM 路线优化，`4+3+3+3+3` 和 `3+3+3+3+2+2` 是更值得优先尝试的折中点。
3. 当切法继续细化到 `2-bit` 甚至 `1-bit` 级别时，survivor 数确实继续下降，但结构上已经越来越接近直接做 bitwise threshold compare，也就越来越接近 HD-CAM 的设计动机。

### 12.5 从不同维度的最终比较

#### 正确性

- 路线 A 更强：最终经过 `XOR + popcount` 精确校验，输出是 bit-exact
- 路线 B 较弱：依赖阈值感测，需要通过 PVT 和 Monte Carlo 来证明误判边界

#### 面积

- 路线 A：存储本体不复杂，但 chunk 化和二级校验带来明显外围开销
- 路线 B：如果只做阈值判定，通常省去大量显式计数逻辑和候选管理逻辑

对于当前 `16bit/head` 的配置，更倾向路线 B 的系统面积更优。

#### 功耗

- 路线 A：会消耗在 CAM chunk 搜索、候选聚合、full-hash 读出和 `XOR + popcount`
- 路线 B：直接单级阈值比较，没有二级 verifier 的数字切换开销

对于当前配置，更倾向路线 B 的功耗更低。

#### 延迟

- 路线 A：两级路径，粗筛后还要精确验证
- 路线 B：单级阈值判定

因此路线 B 的单次查询时延通常更好。

#### 可扩展性

- 路线 A 更适合 `L 大、r 小` 的情况，例如 `64bit, HD<=2`
- 路线 B 对短字长更自然，不依赖 chunk theorem 才能成立

所以在 `16bit, HD<=2` 这个点上，路线 B 更像“第一选择”，而不是路线 A。

#### 功能完整性

- 路线 A 可以自然拿到精确 HD，并支持排序、top-k、rerank
- 路线 B 更适合做 yes/no threshold match

如果未来需要精确距离输出，那么路线 A 的扩展性更好。

### 12.6 最终推荐

综合上面的比较，可以把选择标准归纳成下面一句话：

```text
如果前端目标是“快速筛出 HD<=2 的候选”，优先考虑 HD-CAM；
如果前端目标是“既筛候选又要拿到精确 HD”，优先考虑普通 CAM + XOR/Popcount。
```

对 GraphhopSimhash 当前的任务定义而言，前端更接近第一种：

```text
多头近似搜索
support 聚合
direct / residual / compute 三段式决策
```

因此，我对这两条路线的最终评价是：

```text
64bit/radius-2: 路线 A 很有竞争力；
16bit/head, HD<=2: 路线 B 整体上更合适。
```

---

## 13. 参考文献

Garzón E, Rechef E, Golman R, et al. A 128-kbit Approximate Search-Capable Content-Addressable Memory (CAM) With Tunable Hamming Distance[J]. IEEE Journal of Solid-State Circuits, 2025, 60(8): 3009-3019.
