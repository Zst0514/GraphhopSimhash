# Dynamic BFP Block Refinement Datapath

本文档说明 `BFPA4 base + optional BFPA6 refinement` 在硬件阵列内部如何执行。重点是 block-level `refine_flag`、partial-sum 追加、refinement queue 和两阶段调度。

## 1. 目标

后端 miss-node encoder 不直接使用统一 W4A8，也不把所有节点固定成同一低位宽。当前主线采用：

```text
W:
    固定 AWQ W4

Activation:
    BFP shared exponent
    默认 BFPA4 mantissa
    只对 selected activation blocks 追加 BFPA6 extra mantissa bits
```

因此，一个 GEMM block 有两种执行状态：

```text
not refined:
    W4 x BFPA4

refined:
    W4 x BFPA4 + extra W4 x A_extra2
```

这里的 refinement 是 tile-level partial-sum correction，不是重跑完整 layer，也不是保存 BFPA4 embedding 后再改写 embedding。

## 2. Block Metadata

每个 activation block 对应一小段 metadata：

```text
BlockMeta:
    exponent      shared BFP exponent
    stress        activation stress from exponent selection
    graph_risk    node-level graph risk
    refine_flag   1 if this block needs BFPA6 refinement
```

`refine_flag` 由 block-level priority 产生：

```text
priority = graph_risk(node) * activation_stress(block)

if priority >= threshold:
    refine_flag = 1
else:
    refine_flag = 0
```

其中：

```text
graph_risk:
    来自图传播风险，目前主要使用 degree / propagation risk。

activation_stress:
    来自 BFP exponent selection 阶段，衡量一个 block 内 shared exponent 是否会压低小值精度。
```

最小必要 metadata 是：

```text
1-bit refine_flag per activation block
```

如果 block size 为 128 values，BFPA4 mantissa payload 是：

```text
128 values * 4 bits = 512 bits
```

则 `refine_flag` 本身的存储开销约为：

```text
1 / 512 ~= 0.2%
```

## 3. Mantissa Split

BFPA6 mantissa 可以拆成 base 4-bit 和 extra 2-bit：

```text
A6_mantissa = [m5 m4 m3 m2 m1 m0]

A4_base     = [      m3 m2 m1 m0]
A2_extra    = [m5 m4            ]
```

GEMM 可以写成：

```text
Y = A * W

Y4      = A4_base  * W4
DeltaY  = A2_extra * W4

Y_dyn =
    Y4                 if refine_flag = 0
    Y4 + DeltaY        if refine_flag = 1
```

所以 dynamic BFPA4-to-BFPA6 不需要两套完整阵列。阵列始终执行 W4 x mantissa 的乘加，只是 selected blocks 会额外发射 2-bit mantissa-plane cycles。

## 4. Naive Per-Block Branching

最直接的执行方式是：

```text
for each activation block:
    compute BFPA4 base
    if refine_flag:
        compute extra 2-bit refinement
```

例如：

```text
block0: refine
block1: no refine
block2: refine
block3: no refine
```

执行流为：

```text
block0: base4 + extra2
block1: base4
block2: base4 + extra2
block3: base4
```

这种方式逻辑简单，但 block 粒度的 `1/0/1/0` 分支会让主阵列控制流变碎，可能产生 issue bubble 和利用率下降。

## 5. Two-Phase Execution

更稳定的实现是两阶段执行：

```text
Phase 1: BFPA4 base phase
    所有 activation blocks 连续执行 BFPA4 base。

Phase 2: BFPA6 refinement phase
    只对 refine_flag = 1 的 blocks 追加 extra 2-bit contribution。
```

对上面的例子：

```text
base phase:
    block0 base4
    block1 base4
    block2 base4
    block3 base4

refine phase:
    block0 extra2
    block2 extra2
```

好处：

```text
1. BFPA4 base path 连续、规则。
2. selected blocks 被集中追加计算，减少主路径频繁切换。
3. refinement phase 的成本与 refined block ratio 成正比。
```

代价：

```text
1. 需要记录 selected block 的 psum address。
2. 需要小型 refinement queue。
3. psum buffer 需要支持后续追加更新。
```

## 6. Refinement Queue

两阶段执行可以用 `RefineQueue` 实现：

```text
RefineQueue entry:
    block_id
    psum_addr
    exponent
    A_extra2 pointer / packed bits
```

执行流程：

```text
for each W tile:
    load W4 tile into SRAM / RF

    for each activation block:
        compute BFPA4 base
        update psum buffer

        if refine_flag:
            push {block_id, psum_addr, exponent, A_extra2} into RefineQueue

    while RefineQueue not empty:
        pop selected block
        compute A_extra2 * W4
        add DeltaY into psum buffer

    write output tile
```

如果硬件资源允许，refinement lane 可以和 base lane 部分重叠：

```text
base lane:
    processes later BFPA4 blocks

refinement lane:
    processes earlier selected blocks from RefineQueue
```

这样 `refine_flag` 的不规则性被吸收到 queue 中，主 base array 不需要在每个 block 后停顿等待。

## 7. Partial-Sum Buffer

动态 refinement 保存的不是最终 embedding，而是 GEMM tile 的 partial sum。

```text
BFPA4 base:
    psum = A4_base * W4

optional refinement:
    psum += A2_extra * W4

output:
    write psum as output tile
```

因此需要：

```text
1. psum buffer:
   保存 output tile partial sums。

2. psum address table:
   让 RefineQueue entry 找回对应 block 的 psum location。

3. accumulate path:
   支持 extra2 contribution 加回已有 psum。
```

这个逻辑和普通 tiled GEMM 的 output-stationary / partial-sum accumulation 兼容，只是多了一次 optional update。

## 8. PE Datapath

PE 不需要切换成另一套 6-bit 阵列。它只需要支持两类 mantissa-plane issue：

```text
base issue:
    issue m[3:0] cycles

refine issue:
    issue m[5:4] cycles for selected blocks
```

简化 PE 数据流：

```text
                 W4 tile
                   |
                   v
        +----------------------+
A4 ---> | W4 x A_mantissa MAC  | ---> psum_base
        +----------------------+
                   ^
                   |
A2_extra ----------+  only if refine_flag = 1
```

计算模式：

```text
BFPA4:
    4 mantissa-bit work units

BFPA6 refined block:
    4 base work units + 2 extra work units
```

所以每个 refined block 的额外计算量相对 BFPA4 block 为：

```text
extra_compute_per_refined_block = 2 / 4 = 50%
```

但只对 selected blocks 生效。

## 9. Average Cost Model

若 refined block ratio 为 `r`：

```text
effective_bits = 4 + 2 * r
dynamic / BFPA4 compute ~= effective_bits / 4
dynamic / BFPA6 compute ~= effective_bits / 6
dynamic / BFPA8 compute ~= effective_bits / 8
```

例如 Cora 当前 dynamic pool：

```text
r = 20.79%

effective_bits = 4 + 2 * 0.2079 = 4.416

dynamic / BFPA4 ~= 4.416 / 4 = 1.104x
dynamic / BFPA6 ~= 4.416 / 6 = 0.736x
dynamic / BFPA8 ~= 4.416 / 8 = 0.552x
```

array trace 中观测到：

```text
refined blocks:          20.79%
effective mantissa bits: 4.416
dynamic / BFPA4 cycles:  1.102x
dynamic / BFPA6 cycles:  0.735x
dynamic / BFPA8 cycles:  0.551x
```

这说明当前 simulator 的 array activity 与 mantissa-bit cost model 基本一致。

## 10. Control Overhead

额外控制开销主要来自：

```text
1. refine_flag generation:
   priority compare。

2. queue push/pop:
   selected block 写入 / 读出 RefineQueue。

3. psum address lookup:
   找回 selected block 对应的 psum location。

4. optional psum update:
   extra2 contribution 加回 psum buffer。
```

其中 `refine_flag` 比较和 queue metadata 远小于 GEMM 本身。主要性能风险不是比较器面积，而是：

```text
selected blocks 分布过碎时，
refinement issue 可能造成调度不连续。
```

两阶段执行和 RefineQueue 的目的就是把这种不连续性从主 base path 中隔离出去。

## 11. Implementation Choices

### 11.1 Block-level refinement

```text
granularity:
    activation block

decision:
    graph_risk(node) * activation_stress(block)

effect:
    selected block BFPA4 -> BFPA6
```

优点：

```text
1. 只 refine 真正高风险 block。
2. 比 node-level 全 A6 更省。
3. 能体现 graph risk 和 BFP stress 的联合控制。
```

代价：

```text
1. block-level flag 和 queue。
2. psum buffer 需要支持 optional update。
```

### 11.2 Node-level refinement

```text
granularity:
    node

decision:
    graph_risk(node)

effect:
    selected node 全部 blocks BFPA4 -> BFPA6
```

优点是控制简单；缺点是会 refine 很多低 stress blocks，计算浪费更大。当前主线优先采用 block-level refinement。

## 12. End-to-End Placement

Dynamic BFP refinement 只处理 miss nodes：

```text
SimHash / CAM frontend:
    direct hit      -> embedding cache read
    fuzzy hit       -> residual-gate correction
    miss / reject   -> BFP encoder

BFP encoder:
    BFPA4 base for all miss-node activation blocks
    optional BFPA6 refinement for selected blocks
```

这使得前端 reuse 和后端 refinement 分工明确：

```text
reuse/residual:
    减少进入 encoder 的节点数量。

dynamic BFP:
    降低剩余 miss-node encoder 的执行成本。
```

## 13. Summary

Dynamic BFPA4-to-BFPA6 refinement 的具体实现是：

```text
1. 每个 activation block 产生 refine_flag。
2. 阵列连续执行 BFPA4 base。
3. selected blocks 被写入 RefineQueue。
4. refinement phase 对 selected blocks 追加 m[5:4] x W4。
5. extra contribution 加回 tile-level psum buffer。
6. 最终输出 tile，不保存或重写整层 embedding。
```

核心优势是：

```text
BFPA4 提供低成本 base path；
BFPA6 refinement 只对 graph-risk x activation-stress 高的 blocks 追加；
两阶段执行避免主阵列在 block 间频繁切换；
RefineQueue 把不规则 selected blocks 转化为可调度的 extra2 workload。
```
