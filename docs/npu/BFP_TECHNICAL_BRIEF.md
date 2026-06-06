# BFP Technical Brief

## 1. BFP 是什么

BFP, Block Floating Point, 是介于 fixed-point 和 floating-point 之间的一种数据格式。它把一组数划成一个 block，block 内所有数共享一个 exponent，每个数保留自己的 mantissa。

普通 FP16:

```text
x_i = sign_i * 2^{e_i} * mantissa_i

每个 value 都有自己的 exponent。
```

BFP:

```text
x_i ~= sign_i * 2^{E_block} * mantissa_i

一个 block 共享 E_block。
```

这样做的好处是：

```text
1. exponent 只存一份，格式更紧凑。
2. mantissa 可以做低 bit 计算。
3. 比普通 INT4/INT6 更能保留 activation 动态范围。
```

它的代价是：

```text
block 内如果有 outlier，shared exponent 会被 outlier 拉高，
小值在 mantissa 对齐时会损失有效位。
```

## 2. 一个简单例子

假设一个 block 里有两个数：

```text
A = 128 = 1.000 * 2^7
B =   1 = 1.000 * 2^0
```

BFP 需要为整个 block 选一个 shared exponent。为了不让最大值 `128` 溢出，通常选：

```text
E_block = 7
```

此时：

```text
A:
    1.000 * 2^7
    可以正常表示。

B:
    1.000 * 2^0
  = 0.0000001 * 2^7
```

如果 mantissa 只有 4 bit，那么 `B` 右移后的有效信息可能被截断，甚至变成 0。

这就是 BFP 的核心问题：

```text
shared exponent 保住了大值动态范围，
但可能牺牲同 block 内小值精度。
```

## 3. Transformer Activation 中常见的 BFP Scale 选择

Transformer linear/GEMM 中，activation 通常按 block 量化。当前实现采用 rowwise `1 x 128` block：

```text
一个 token row 的连续 128 个 hidden values 组成一个 block。
```

例如 LLaMA hidden size = 4096，则一个 token row 可以划成：

```text
4096 / 128 = 32 个 BFP blocks
```

每个 block 独立选择 shared exponent：

```text
E_block = ceil(log2(max(abs(block))))
```

然后每个 activation value 用低 bit mantissa 表示：

```text
BFPA8:
    shared exponent + 8-bit mantissa

BFPA6:
    shared exponent + 6-bit mantissa

BFPA4:
    shared exponent + 4-bit mantissa
```

当前主线仍使用 AWQ 的 W4 权重量化；BFP 只改变 activation 表示：

```text
W:
    AWQ W4, group size 128

A:
    BFP activation, block size 128
```

因此 `W4BFPA4_B128` 表示：

```text
W4:
    AWQ 4-bit weight

BFPA4:
    BFP activation with 4-bit mantissa

B128:
    128 activation values share one exponent
```

## 4. 为什么 BFP 适合 Miss-Node Encoder

在当前系统里，SimHash / CAM / Residual-Gate 已经把部分节点从 full encoder 路径中移走：

```text
direct hit:
    embedding cache read

fuzzy hit:
    residual-gate correction

miss / reject:
    run LLaMA encoder
```

BFP NPU 只服务最后一类 miss / rejected nodes。它的目标不是重新计算全图，而是降低剩余 encoder 调用的成本。

如果所有 miss nodes 都走 W4A8，精度稳，但 activation compute 和片上活动偏高。BFPA4 能大幅降低 activation-side mantissa 计算，但在某些 block 上可能因为 shared exponent 造成精度损失。

因此当前后端路径采用：

```text
BFPA4 base compute
optional BFPA6 refinement
```

## 5. 图场景带来的新信息

普通 Transformer accelerator 只看到一批 token rows。它可以看到 activation block 的数值分布，但不知道这些 token rows 对图任务的重要性。

GFM / text-attributed graph 场景多了节点级图信息：

```text
degree / propagation risk:
    节点误差会影响多少邻居。

TSER / graph semantic risk:
    节点是否处在结构或语义边界。

reuse route:
    节点是 direct hit、fuzzy hit 还是 miss。
```

这些信息让后端 encoder 可以区分：

```text
低风险 miss node:
    BFPA4 误差更容易被接受。

高风险 miss node:
    embedding 误差更容易通过 GNN 传播，
    更值得追加 BFPA6 refinement。
```

这不是单纯把 BFP 搬到 Transformer 上，而是把图任务风险引入 BFP refinement 决策。

## 6. Activation Stress

BFP 的数值风险来自 shared exponent。为了定位哪些 block 容易受 shared exponent 影响，当前实现定义 activation stress：

```text
max_abs    = max(abs(block))
median_abs = median(abs(block))

stress = log2(max_abs / median_abs)
stress_norm = clamp(stress / stress_scale, 0, 1)
```

含义：

```text
stress 小:
    block 内数值比较均匀，
    BFPA4 通常足够。

stress 大:
    block 内有 outlier，
    shared exponent 可能牺牲小值，
    更需要 BFPA6 refinement。
```

Transformer accelerator 本身也可以观测 activation stress，因为 exponent selector 本来就需要看 block 的最大值。图场景新增的是：

```text
stress 只说明这个 block 数值上危险；
graph risk 说明这个 block 所属节点在图任务上是否重要。
```

## 7. Graph-Aware Dynamic BFP Refinement

当前主线不是离线固定某些节点永远 BFPA4 或 BFPA6，而是在 block 执行时做动态判断：

```text
BFPA4 base always compute

for each activation block:
    stress = activation_block_stress(block)
    graph_risk = node_risk(node)
    priority = graph_risk * stress

    if priority >= threshold:
        execute extra 2 mantissa planes
    else:
        keep BFPA4 result
```

对应硬件含义：

```text
BFPA4:
    compute mantissa bits m[3:0]

BFPA6 refinement:
    additionally compute mantissa bits m[5:4]

final:
    BFPA6 result = BFPA4 partial sum + extra 2-bit correction partial sum
```

因此不需要两套完整阵列。阵列默认跑 BFPA4；只有被选中的 block 才进入 refinement lane。

## 8. NPU 通路

整体路径：

```text
Graph node text
      |
      v
SimHash + LRU/CAM
      |
      +-- direct hit -----> embedding cache
      |
      +-- fuzzy hit ------> residual-gate unit
      |
      +-- miss/reject ----> Dynamic BFP Encoder
                                |
                                v
                        BFPA4 base path
                                |
                  +-------------+-------------+
                  |                           |
          priority < threshold        priority >= threshold
                  |                           |
             keep BFPA4          execute extra BFPA6 planes
                  |                           |
                  +-------------+-------------+
                                |
                                v
                         output embedding
```

Dynamic BFP encoder 内部：

```text
1. Load W4 tile.
2. Stream activation block.
3. Select shared exponent.
4. Compute BFPA4 base partial sum.
5. Compute stress from activation block.
6. Combine stress with graph risk.
7. If selected, execute extra BFPA6 mantissa planes.
8. Merge partial sums.
```

## 9. 设计取舍

### BFPA4 Only

```text
优点:
    成本最低，阵列简单。

缺点:
    对 high-stress block 和高图风险节点不够稳。
```

### BFPA6 Everywhere

```text
优点:
    精度更稳。

缺点:
    所有 block 都多算 2 个 mantissa bits，
    没有利用图任务风险差异。
```

### Graph-Aware BFPA4-to-BFPA6 Refinement

```text
优点:
    BFPA4 作为低成本底座；
    只对 graph-risk × activation-stress 高的 block 追加计算；
    refinement ratio 可控。

代价:
    需要 stress 统计逻辑；
    需要 refinement lane 或额外 bit-plane execution cycles；
    需要 output merge / partial-sum accumulation 控制。
```

## 10. 当前实现映射

当前代码生成的 dynamic pool 使用：

```text
base:
    BFPA4

refine:
    BFPA6

block:
    rowwise 1 x 128

priority:
    graph risk * activation stress
```

典型 tag：

```text
W4GraphBFPA4to6_B128_deg_t0.20
```

含义：

```text
W4:
    AWQ 4-bit weight

GraphBFPA4to6:
    graph-aware BFPA4 base with optional BFPA6 refinement

B128:
    128 activation values share one exponent

deg:
    graph risk currently uses degree / propagation risk

t0.20:
    refinement threshold
```

对应脚本：

```text
GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py
GraphhopSimhash/scripts/simulate_dynamic_bfp_array_trace.py
GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh
```
