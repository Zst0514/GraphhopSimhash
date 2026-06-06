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

## 2. 为什么选择 W4BFPA 而不是普通 W4A8 / W4A4

后端 miss-node encoder 的目标不是单纯追求最高精度，而是在前端 reuse/residual 之后，用更低成本计算剩余必须进入 LLaMA encoder 的节点。

因此需要在三种 activation 路径之间取舍：

```text
W4A8:
    精度稳，但 activation 仍然是 8-bit。

W4A4:
    成本低，但普通 4-bit 定点 activation 容易丢动态范围。

W4BFPA4:
    activation mantissa 只有 4-bit，
    但 block shared exponent 保留动态范围。
```

### 2.1 为什么不直接用 W4A8

W4A8 是很强的保守 baseline。AWQ/QServe 一类 PTQ 路径已经证明 W4A8 对 LLaMA encoder 通常很稳。

问题是：

```text
W4A8 的 A 仍然是 8-bit。
```

在 miss-node encoder 中，如果所有 miss nodes 都走 W4A8：

```text
1. activation-side compute 仍按 8-bit 执行。
2. activation buffer / RF / SRAM 仍按 8-bit 搬运。
3. 图前端只减少 encoder 调用次数，
   没有继续降低剩余 miss-node encoder 内部成本。
```

因此 W4A8 更适合作为 reference / conservative path，而不是低成本默认路径。

### 2.2 为什么不直接用普通 W4A4

普通 W4A4 把 activation 量化成 4-bit 定点数，成本低，但它的 scale 表达能力有限。Transformer activation 经常有 outlier 和长尾分布：

```text
大值需要更大 scale 才不溢出；
小值需要更小 scale 才不被压扁。
```

同一个 group 里如果同时有大值和小值，普通 A4 很难同时照顾二者。

结果是：

```text
普通 W4A4:
    成本低，但动态范围损失更明显。
```

### 2.3 为什么 W4BFPA4 是更合适的 base path

BFP 把动态范围和低 bit mantissa 拆开：

```text
shared exponent:
    负责 block 级动态范围。

4-bit mantissa:
    负责 block 内相对数值。
```

所以 `W4BFPA4` 能同时提供两点：

```text
1. 接近 W4A4 的 activation-side 低成本。
2. 比普通 W4A4 更强的动态范围保持能力。
```

这就是当前后端选择 `BFPA4 base` 的原因。它不是为了替代 W4A8 的精度上限，而是把 W4A8 和 W4A4 中间缺失的低成本、动态范围友好的执行点补出来。

### 2.4 为什么还需要 BFPA6 refinement

BFPA4 仍然会受 shared exponent 的限制。若 block 内有强 outlier，小值 mantissa 仍可能被牺牲。因此当前设计不是全图固定 BFPA4，而是：

```text
默认:
    BFPA4 base compute

必要时:
    追加 BFPA6 refinement
```

BFPA6 在 BFPA4 基础上补 2 个 mantissa bits：

```text
BFPA4:
    m[3:0]

BFPA6:
    m[5:0]
```

关键是，不是所有 block 都补 BFPA6。图场景提供了节点风险，activation trace 提供了 block stress，因此 refinement 可以集中在：

```text
图任务重要节点里的高 stress activation block。
```

## 3. 一个简单例子

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

## 4. Transformer Activation 中常见的 BFP Scale 选择

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

## 5. 相对普通 W4A8 / W4A4 定点表示的区别

这里的 `W4A8` / `W4A4` 指普通定点 activation quantization：

```text
W4A8:
    W = 4-bit integer weight
    A = 8-bit integer activation

W4A4:
    W = 4-bit integer weight
    A = 4-bit integer activation
```

普通定点 activation 通常对一个 group / tensor 使用 scale：

```text
a_int = round(a_fp / scale)
a_fp  ~= a_int * scale
```

BFP activation 则是：

```text
a_fp ~= mantissa_int * 2^{E_block}
```

两者核心差异是：

```text
定点:
    用 scale 表示动态范围。

BFP:
    用 shared exponent 表示 block 动态范围。
```

### 5.1 BFP 相对 W4A8 的优势

W4A8 精度稳，但 activation 侧仍然是 8-bit 计算和存储。BFP 可以把 activation mantissa 降到 6-bit 或 4-bit：

```text
W4A8:
    A mantissa/effective integer width = 8

W4BFPA4:
    A mantissa width = 4

W4BFPA6:
    A mantissa width = 6
```

优势：

```text
1. activation-side MAC 位宽更低。
2. activation RF / SRAM / buffer 读写位宽更低。
3. PE 内乘法和累加活动更少。
4. shared exponent 保留 block 动态范围，比普通 A4 更稳。
```

因此，BFP 的目标不是比 W4A8 更准，而是在接近可接受精度时降低 activation-side 计算和片上数据活动。

### 5.2 BFP 相对 W4A4 的优势

普通 W4A4 的 activation 只有 4-bit 定点数。它的问题是：

```text
如果 group 内动态范围大，
一个 scale 很难同时照顾大值和小值。
```

BFP 的 shared exponent 能更自然地保留 block 级动态范围：

```text
block exponent:
    负责表达整体数量级。

4-bit mantissa:
    负责表达 block 内相对值。
```

所以在同样 4-bit activation mantissa 下，`BFPA4` 往往比普通 `A4` 更稳。它不是“普通 A4 变好了”，而是换了一种 activation 数值格式。

### 5.3 BFP 的劣势

BFP 的主要代价来自 shared exponent 和 block 管理：

```text
1. 每个 block 需要 exponent selection。
2. 每个 block 需要存储 / 搬运 shared exponent。
3. PE 输入前需要按 exponent 做 shift / align。
4. block 内 outlier 会拉高 exponent，牺牲小值 mantissa 精度。
```

和普通定点相比，BFP 控制逻辑更复杂：

```text
普通 W4A4:
    scale path 简单，整数 MAC 直接。

W4BFPA4:
    需要 exponent selector、mantissa alignment、block metadata。
```

因此 BFP 适合的前提是：

```text
dynamic range 保留带来的精度收益
    >
block exponent / shift / metadata 带来的额外开销
```

当前实验中，BFPA4 在 Cora/PubMed/Arxiv 上明显比 BFPA3 稳，并且比普通 A4 更能保留动态范围；这也是后端选择 BFPA4 作为 base path 的原因。

## 6. 为什么 BFP 适合 Miss-Node Encoder

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

## 7. 图场景带来的新信息

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

## 8. Activation Stress

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

## 9. Graph-Aware Dynamic BFP Refinement

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

## 10. NPU 通路

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

## 11. 设计取舍

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

## 12. 当前实验结果

当前主要结果来自：

```text
docs/results/FINAL_BFP_VALIDATION_RESULT.md
```

### 12.1 BFPA safety boundary

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

### 12.2 BFPA4 -> BFPA6 refinement

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

### 12.3 Full-stack dynamic BFP

该实验接入完整前端：

```text
SimHash + LRU/CAM
  -> TSER score gate
  -> direct reuse / residual-gate reuse / miss
  -> miss nodes run BFPA encoder
```

| Dataset | Runs | Reuse | Miss | FullP8 Drop | AllP4 Drop | Dynamic Drop |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 5 | 51.6% | 48.4% | 2.22% | 2.72% | 2.57% |
| PubMed | 3 | 41.9% | 58.1% | 1.86% | 2.42% | 2.31% |
| Arxiv | 1 | 44.9% | 55.1% | 2.29% | 2.36% | 2.34% |

其中：

```text
FullP8:
    miss nodes 使用 BFPA8 reference path。

AllP4:
    miss nodes 全部使用 BFPA4。

Dynamic:
    miss nodes 使用 graph-aware BFPA4->BFPA6 refinement。
```

整体结论：

```text
1. 前端 reuse/residual 决定有多少节点绕过 encoder。
2. BFPA4 是低成本 miss-node encoder base path。
3. Dynamic BFPA4->BFPA6 refinement 在 Cora/PubMed 上能回收部分 BFPA4 精度损失。
4. Arxiv 上 BFPA4 本身接近 FullP8，dynamic path 自然退化为低成本 BFPA4 为主。
```
