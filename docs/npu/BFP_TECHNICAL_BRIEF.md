# BFP Technical Brief

## 1. BFP 基本格式

BFP, Block Floating Point, 是介于 fixed-point 和 floating-point 之间的数据格式。它把一组 activation values 划成一个 block，block 内共享一个 exponent，每个 value 保留自己的 mantissa。

普通 FP16:

```text
x_i = sign_i * 2^{e_i} * mantissa_i

每个 value 有自己的 exponent。
```

BFP:

```text
x_i ~= sign_i * 2^{E_block} * mantissa_i

一个 block 共享 E_block。
```

好处：

```text
1. exponent 只存一份，metadata 更少。
2. mantissa 可以用低 bit 计算。
3. 相比普通 INT4/INT6，BFP 更能保留 block 的动态范围。
```

代价：

```text
block 内如果有 outlier，shared exponent 会被 outlier 拉高；
小值 mantissa 右移后可能损失有效位。
```

## 2. 一个例子

假设一个 block 里有两个数：

```text
A = 128 = 1.000 * 2^7
B =   1 = 1.000 * 2^0
```

为了表示最大值 `128`，BFP 通常选：

```text
E_block = 7
```

于是：

```text
A:
    1.000 * 2^7
    正常表示。

B:
    1.000 * 2^0
  = 0.0000001 * 2^7
```

如果 mantissa 只有 4 bit，`B` 右移后的有效信息可能被截断，甚至变成 0。

这说明 BFP 的核心取舍：

```text
shared exponent 保留 block 动态范围；
但 block 内小值可能被 outlier 牺牲。
```

## 3. Transformer Activation 中如何选 Scale

Transformer linear/GEMM 中，activation 通常按 block 量化。当前实现采用 rowwise `1 x 128` block：

```text
一个 token row 的连续 128 个 hidden values 组成一个 block。
```

例如 LLaMA hidden size = 4096，则一个 token row 被划成：

```text
4096 / 128 = 32 个 BFP blocks
```

每个 block 独立选 shared exponent：

```text
E_block = ceil(log2(max(abs(block))))
```

然后每个 activation value 用低 bit mantissa 表示：

```text
BFPA8 = shared exponent + 8-bit mantissa
BFPA6 = shared exponent + 6-bit mantissa
BFPA4 = shared exponent + 4-bit mantissa
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

## 4. 为什么选择 W4BFPA 而不是普通 W4A8 / W4A4

后端 miss-node encoder 的目标是在前端 reuse/residual 之后，用更低成本计算剩余必须进入 LLaMA encoder 的节点。

三种 activation path 的取舍是：

```text
W4A8:
    精度稳，但 activation 仍是 8-bit。

W4A4:
    成本低，但普通 4-bit 定点 activation 容易丢动态范围。

W4BFPA4:
    activation mantissa 只有 4-bit；
    shared exponent 保留 block 动态范围。
```

### 4.1 为什么不直接用 W4A8

W4A8 是保守 baseline。它通常精度稳，但 activation 侧仍按 8-bit 计算和搬运：

```text
1. activation-side MAC 位宽仍高。
2. activation buffer / RF / SRAM 仍按 8-bit 活动。
3. 图前端只减少 encoder 调用次数，
   没有继续降低剩余 miss-node encoder 内部成本。
```

因此 W4A8 更适合作为 reference / conservative path，而不是低成本默认路径。

### 4.2 为什么不直接用普通 W4A4

普通 W4A4 用定点 scale 表示 activation：

```text
a_int = round(a_fp / scale)
a_fp  ~= a_int * scale
```

4-bit 定点 activation 的可表示值是一组固定间隔的网格：

```text
..., -2*scale, -1*scale, 0, 1*scale, 2*scale, ...
```

如果 group 内最大值很大，为了避免溢出，scale 必须变大：

```text
scale >= max_abs / 7      # signed 4-bit 大致只能表示 -8..7
```

scale 一旦变大，网格间隔也变大，小值会被舍入到 0 或少数几个粗粒度刻度。

当 group 内同时有 outlier 和小值时：

```text
大值需要更大 scale 才不溢出；
小值需要更小 scale 才不被压扁。
```

举例：

```text
group = [128, 1]

为了表示 128:
    scale ~= 128 / 7 = 18.3

那么 1 会被量化为:
    round(1 / 18.3) = 0
```

此时普通 A4 保住了大值，但小值直接消失。同一个定点 scale 很难同时照顾大值和小值，因此普通 A4 成本低，但动态范围损失明显。

### 4.3 为什么 W4BFPA4 是合适的 base path

BFP 把动态范围和低 bit mantissa 拆开：

```text
shared exponent:
    表达 block 级动态范围。

4-bit mantissa:
    表达 block 内相对数值。
```

BFP 的 shared exponent 相当于先把整个 block 移到合适的数量级，再用 mantissa 表示相对大小。它不是用一个线性 scale 直接覆盖完整动态范围，而是用指数项表达数量级：

```text
a_fp ~= mantissa * 2^{E_block}
```

因此当 block 内数值跨越较大数量级时，BFP 至少能显式记录这个 block 的 exponent；普通 A4 则只能把所有数挤到一个固定 scale 网格里。

所以 `W4BFPA4` 的定位是：

```text
接近 W4A4 的 activation-side 低成本；
比普通 W4A4 更强的动态范围保持能力。
```

它不是为了替代 W4A8 的精度上限，而是补出一个低成本、动态范围友好的 miss-node encoder base path。

## 5. 图场景带来的新信号

普通 Transformer accelerator 只看到 token rows 和 activation block 数值分布。如果直接把 GFM 前端当普通 LLM batch 处理，会忽略这些图任务信息：

```text
degree / propagation risk:
    节点误差会影响多少邻居。

TSER / graph semantic risk:
    节点是否处在结构或语义边界。

reuse route:
    节点是 direct hit、fuzzy hit 还是 miss。
```

这些信号让后端 encoder 可以区分：

```text
低风险 miss node:
    BFPA4 误差更容易被接受。

高风险 miss node:
    embedding 误差更容易通过 GNN 传播，
    更值得追加 BFPA6 refinement。
```

这就是图场景给 BFP 阵列带来的增量：不是只看 activation 是否难量化，还要看这个 activation 所属节点在图任务中是否重要。

## 6. Activation Stress

BFP 的数值风险来自 shared exponent。当前实现用 activation stress 描述一个 block 是否容易受 outlier 影响：

```text
max_abs    = max(abs(block))
median_abs = median(abs(block))

stress = log2(max_abs / median_abs)
stress_norm = clamp(stress / stress_scale, 0, 1)
```

含义：

```text
stress 小:
    block 内数值比较均匀，BFPA4 通常足够。

stress 大:
    block 内有 outlier，shared exponent 可能牺牲小值，
    更需要 BFPA6 refinement。
```

Transformer accelerator 本身也可以观测 activation stress，因为 exponent selector 本来就需要看 block 最大值。我们的新增点是把它和 graph risk 结合：

```text
stress:
    这个 block 数值上是否危险。

graph risk:
    这个 block 所属节点在图任务上是否重要。
```

## 7. Graph-Aware Dynamic BFPA4 -> BFPA6 Refinement

当前主线不是离线固定某些节点永远 BFPA4 或 BFPA6，而是在 block 执行时动态判断：

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

硬件含义：

```text
BFPA4:
    compute mantissa bits m[3:0]

BFPA6 refinement:
    additionally compute mantissa bits m[5:4]

final:
    BFPA6 result = BFPA4 partial sum + extra 2-bit correction partial sum
```

因此不需要两套完整阵列。阵列默认跑 BFPA4；只有被选中的 block 才进入 refinement lane。

### 7.1 不是保存 BFPA4 embedding

`execute extra BFPA6 planes` 不是先生成 BFPA4 embedding，再决定是否重新生成 BFPA6 embedding。那样会重复运行 encoder，也需要保存大量中间 embedding。

正确实现是在同一个 GEMM tile 的 partial sum 层面追加低位贡献。

把 BFPA6 activation mantissa 拆成：

```text
A6 = A4_part + A2_refine_part
```

则 Linear/GEMM 可以写成：

```text
Y = A * W

BFPA4 base:
    Y4 = A4_part * W

BFPA6 refinement:
    ΔY = A2_refine_part * W

final:
    Y6 = Y4 + ΔY
```

硬件执行时维护的是 GEMM 本来就需要的 output partial sum：

```text
psum = A4_part * W

if priority >= threshold:
    psum += A2_refine_part * W

write output tile
```

因此需要保存的是 tile-level `psum buffer`，不是 node-level BFPA4 embedding。`psum buffer` 是 GEMM 的正常组成部分；dynamic refinement 只是让它支持可选追加低 2-bit mantissa plane 的贡献。

### 7.2 对执行周期的影响

如果 BFPA4 base 使用 4 个 mantissa bits，BFPA6 refinement 额外追加 2 个 mantissa bits：

```text
extra compute per refined block = 2 / 4 = 50% of BFPA4 block compute
```

如果只有 `r` 比例的 activation blocks 被 refine，则平均 mantissa bits 为：

```text
avg_bits = 4 + 2 * r
```

例如 `r = 20%`：

```text
avg_bits = 4 + 2 * 0.2 = 4.4
```

这意味着 dynamic path 不是在 BFPA4 和 BFPA6 之间频繁切换两套阵列，而是在同一条 BFP datapath 上对少量 block 多发射两个 mantissa-plane cycles。

### 7.3 控制与调度代价

该机制的额外硬件代价包括：

```text
1. stress 统计逻辑:
   在 exponent selection / activation scan 时得到 block stress。

2. priority compare:
   graph_risk(node) * stress(block) 与 threshold 比较。

3. refinement issue control:
   对 selected block 额外发射 m[5:4] mantissa-plane cycles。

4. psum merge:
   低 2-bit contribution 加到已有 partial sum。
```

主要风险是 selected blocks 分布过碎时，控制分支会降低调度效率。因此 array trace 需要统计：

```text
refined block ratio
effective mantissa bits
dynamic / BFPA4 cycles
dynamic / BFPA6 cycles
dynamic / BFPA8 cycles
```

当前 Cora dynamic pool 的 array trace：

```text
refined blocks:          20.79%
effective mantissa bits: 4.416
dynamic / BFPA4 cycles:  1.102x
dynamic / BFPA6 cycles:  0.735x
dynamic / BFPA8 cycles:  0.551x
```

含义：

```text
相比纯 BFPA4:
    多约 10.2% array cycles。

相比全 BFPA6:
    少约 26.5% array cycles。

相比全 BFPA8:
    少约 44.9% array cycles。
```

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
                         Load W4 tile
                                |
                         Load A block
                                |
                         Select exponent
                                |
                         Compute A[3:0] x W4
                                |
                             psum buffer
                                |
                         priority check
                         /            \
                      no               yes
                      |                |
                 write psum      compute A[5:4] x W4
                                      |
                                  psum += refine
                                      |
                                  write psum
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

```text
BFPA4 only:
    成本最低，但对 high-stress block 和高图风险节点不够稳。

BFPA6 everywhere:
    精度更稳，但所有 block 都多算 2 个 mantissa bits，
    没有利用图任务风险差异。

Graph-aware BFPA4-to-BFPA6 refinement:
    BFPA4 作为低成本底座；
    只对 graph-risk × activation-stress 高的 block 追加计算；
    refinement ratio 可控。
```

主要额外代价：

```text
1. stress 统计逻辑。
2. refinement lane 或额外 bit-plane execution cycles。
3. output merge / partial-sum accumulation 控制。
```

## 10. 当前实验结果

当前主要结果来自：

```text
docs/results/FINAL_BFP_VALIDATION_RESULT.md
```

### 10.1 BFPA safety boundary

该实验只比较 encoder target pool，不接入 reuse / residual 前端。reference 为 `W4BFPA8_B128`。

| Dataset | Runs | BFPA6 Drop | BFPA5 Drop | BFPA4 Drop | BFPA3 Drop |
|---|---:|---:|---:|---:|---:|
| Cora | 5 | 0.09% | 0.35% | 0.99% | 23.13% |
| PubMed | 3 | 0.02% | 0.25% | 1.16% | 27.43% |
| Arxiv | 1 | 0.03% | 0.13% | 0.04% | 35.31% |

结论：

```text
BFPA6 / BFPA5:
    基本接近 BFPA8。

BFPA4:
    Cora/PubMed 有约 1% 级别掉点，Arxiv 几乎无损。

BFPA3:
    三个数据集均明显崩塌，不作为默认路径。
```

### 10.2 BFPA4 -> BFPA6 refinement

该实验固定：

```text
base   = BFPA4
refine = BFPA6
```

在不同 refinement ratio 下比较 `Random / Stress / Degree / TSER / Graph×Stress` 等 selector。

| Dataset | BFPA4 Drop | BFPA6 Drop | Representative Best |
|---|---:|---:|---|
| Cora | 0.99% | 0.09% | 25% Degree: 0.59% |
| PubMed | 1.16% | 0.02% | 25% TSER: 0.55% |
| Arxiv | 0.06% | 0.03% | BFPA4 already safe |

结论：

```text
Cora / PubMed:
    BFPA4 有可恢复的精度损失；
    对部分 block 追加 BFPA6 能降低 drop。

PubMed:
    TSER selector 最稳定，说明图语义风险对 refinement 选择有价值。

Arxiv:
    BFPA4 已经足够安全，dynamic refinement 的收益很小。
```
