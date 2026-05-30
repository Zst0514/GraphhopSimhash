# Graph-Bit End-to-End Algorithm, Theory, and Numeric Model

本文档把 Graph-Bit 的端到端链路放在一处说明，重点覆盖：

```text
1. 算法路径：SimHash/residual reuse + Graph-Bit miss-node encoder。
2. predictor-free early stop：风险分数如何进入 min_depth / tolerance。
3. 变 bit-depth 计算：activation bit-plane 如何少算、少取、少更新。
4. W tile 复用：risk bucket 如何减少 weight tile load。
5. 数值模型：如何从 reuse、AvgDepth、Wloads、Cycles 推导 speedup。
6. 与 HEAT-like static precision 的对比口径。
```

更底层的代码实现见：

```text
docs/npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
docs/archive/npu/GRAPH_BIT_TRACE_DRIVEN_SCHEDULER.md
docs/npu/GRAPH_BIT_NPU_DESIGN.md
```

## 1. 端到端执行路径

Graph-Bit 的主线不是单独做一个低精度 encoder，而是一个分层执行系统：

```text
graph node text + graph topology
        |
        v
HD-CAM / SimHash lookup
        |
        +-- hard hit
        |       -> direct embedding reuse
        |
        +-- soft hit
        |       -> residual-corrected reuse
        |
        +-- reject / miss
                -> Graph-Bit NPU encoder
```

其中：

```text
direct reuse:
    不跑 LLM encoder，直接读取 anchor embedding。

residual reuse:
    不跑完整 LLM encoder，用 anchor embedding + lightweight residual adapter。

Graph-Bit NPU:
    只处理 miss nodes。
    在 W4A8 encoder 内部使用 bit-serial / bit-grained activation execution。
```

所以端到端优化来自两层：

```text
1. 节点级减少 encoder 调用：
   reuse/residual 命中的节点不进入 LLM encoder。

2. NPU 内部减少 miss-node 计算和访存：
   miss nodes 根据 graph risk 做 predictor-free early stop 和 risk-bucket scheduling。
```

## 2. Graph Risk 如何进入 NPU

Graph-Bit 当前把图风险分成三类：

```text
high-risk:
    高 degree / 高 propagation risk。
    下游 GNN 对这些节点的误差更敏感。

mid-risk:
    中等传播风险。

low-risk:
    低传播风险。
```

当前主线使用 Degree / propagation risk 作为最稳的 deployable proxy：

```text
priority(v) = propagation_q(v)
```

也可以切换到：

```text
TSER / graph_context / low_unique / random
```

但实验目前更支持：

```text
Degree 是 Graph-Bit bit-depth 控制的主线；
TSER / Context / LowUnique 更适合作为消融。
```

Graph risk 不直接决定最终执行 P8/P6/P4，而是决定：

```text
min_depth:
    至少执行到多少 bit。

tolerance:
    剩余低位 bit-plane 的贡献上界小到多少才允许停止。
```

默认映射：

```text
high-risk:
    min_depth = 8
    tolerance = 0.00

mid-risk:
    min_depth = 6
    tolerance = 0.02

low-risk:
    min_depth = 4
    tolerance = 0.04
```

## 3. Predictor-Free Early Stop

Graph-Bit 的 early stop 不是 learned predictor，也不是用 FP-vs-quant oracle error。它是 predictor-free bound：

```text
for each bit-serial GEMM tile:
    execute activation bit-planes from MSB to LSB
    after reaching min_depth:
        estimate remaining low-bit contribution bound
        if bound <= tolerance:
            stop lower bit-planes
```

Python accuracy validation 中的简化 bound 是：

```text
bound(depth)
  = scale * (2^(ref_bit - depth) - 1) / (2^ref_bit - 1) * sqrt(tile_k / 128)
```

以 `ref_bit=8, scale=1, tile_k=128` 为例：

```text
depth=8:
    bound = 0

depth=6:
    bound = 3 / 255 = 0.01176

depth=5:
    bound = 7 / 255 = 0.02745

depth=4:
    bound = 15 / 255 = 0.05882
```

因此默认三档会变成：

```text
high:
    min=8, tau=0.00
    -> stop at D8

mid:
    min=6, tau=0.02
    -> D6 bound=0.01176 <= 0.02
    -> stop at D6

low:
    min=4, tau=0.04
    -> D4 bound=0.05882 > 0.04
    -> D5 bound=0.02745 <= 0.04
    -> stop at D5
```

ONNXim 侧还有 tile-aware bound：

```text
||A_low @ W|| <= max_abs(A_low) * sum_abs(W_tile)
```

并用 tile metadata 估计 normalized bound：

```text
normalized_bound =
    remaining_bound / (partial_norm + remaining_bound)
```

这对应 `GemmWS.cc` 里的：

```text
estimate_graphbit_remaining_bound(...)
select_graphbit_effective_depth(...)
```

## 4. 变 Bit-Depth GEMM 如何省计算

Graph-Bit 的 W 路径固定为 W4，activation 逻辑上是 A8，但用 bit-serial / bit-plane 方式执行：

```text
A8 bit-plane:
    b7 b6 b5 b4 b3 b2 b1 b0

D8:
    execute b7..b0

D6:
    execute b7..b2

D5:
    execute b7..b3

D4:
    execute b7..b4
```

如果只看 activation bit-plane arithmetic：

```text
compute_scale_node = stop_depth / 8
```

例如：

```text
D8 -> 1.000
D6 -> 0.750
D5 -> 0.625
D4 -> 0.500
```

但是真实 NPU 里不能只看 MAC 数，因为还有：

```text
weight tile load
activation fetch
partial sum read/update/write
output writeback
tile scheduling overhead
```

所以 Graph-Bit 需要同时实现四个 datapath gating：

```text
1. activation plane-group demand fetch
2. bit-plane issue gating
3. weight RF / broadcast gating
4. partial-sum update gating
```

### 4.1 Activation Demand Fetch

普通 byte-major activation：

```text
A_byte = [b7 b6 b5 b4 b3 b2 b1 b0]
```

一读就是完整 8 bit，early stop 后也已经把低位读进来了。

Graph-Bit 使用 plane-group activation buffer：

```text
Group 0: b7 b6
Group 1: b5 b4
Group 2: b3 b2
Group 3: b1 b0
```

如果 stop at D5，并且 group size 是 2：

```text
需要 b7..b3
实际 fetch 到 group boundary: b7..b2
fetch_depth = D6
```

因此 activation traffic 按：

```text
fetch_depth / 8
```

缩放，而不是按 effective depth 精确逐 bit 缩放。

### 4.2 Bit-Plane Issue Gating

PE array 不应该收到完整 activation 后再 mask，而是在 issue 阶段停止：

```text
if bound_satisfied:
    do not issue lower bit-plane cycles
```

这样才减少：

```text
PE cycles
weight RF/broadcast cycles
psum update cycles
```

### 4.3 Weight RF / Broadcast Gating

在 bit-plane cycles 中，每个 activation bit-plane 都要和同一 W tile 交互。停止低位后：

```text
低位 activation plane 不发射
对应的 W RF read / broadcast 也不发生
```

这主要减少片上访问和广播能耗。

注意：

```text
它不自动减少 HBM weight load。
```

因为 W tile 通常先从 HBM/上层缓存加载到片上，再服务多个 cycles。

### 4.4 Partial-Sum Gating

低位 bit-plane 不执行后，也不需要：

```text
psum read
add/update
psum writeback
```

这对片上 energy 很重要，但对 DRAM traffic 影响较小。

## 5. 为什么还需要 W Tile Reuse

如果只做 early stop，不改变 W tile 服务窗口，当前 trace 里收益很小：

```text
FullP8-miss:
    Cycles = 0.722

GraphBit-now:
    Cycles = 0.716
```

speedup 只有：

```text
0.722 / 0.716 = 1.01x
```

原因是：

```text
activation bit-plane 减少了，
但 weight tile load / output write / tile scheduling 仍然占大头。
```

所以 Graph-Bit 的硬件收益需要第二个机制：

```text
risk-bucket scheduler + weight-stationary tile window
```

## 6. Risk-Bucket Scheduler 如何省 W Tile

如果原始节点顺序混合了 D5/D6/D8，一个 micro-batch 的实际 depth 往往被最高风险节点拖到 D8：

```text
mixed batch depth = max(depth of nodes in batch)
```

Risk-bucket scheduler 做的是：

```text
1. 收集 miss node trace。
2. 按真实 stop_depth 分桶：D5 / D6 / D8。
3. 每个桶内组成 micro-batch。
4. 对同一个 W tile，连续服务多个同风险 node blocks。
5. W tile 留在片上，减少 HBM/上层缓存重复加载。
```

这不是凭空减少 W，而是改变执行顺序，让：

```text
load W tile once
serve more node blocks
then evict
```

当前 Cora trace：

```text
miss nodes = 1954
baseline tile batch = 16
baseline Wloads = ceil(1954 / 16) = 123
```

按照真实 stop-depth 分布：

```text
D5: about 30.0%
D6: about 50.0%
D8: about 20.0%
```

### 6.1 Bucket batch = 32

按真实 bucket 分桶后：

```text
Wloads = 63
Wscale = 63 / 123 = 0.512
```

含义：

```text
W tile load 次数减少 48.8%
```

### 6.2 Bucket batch = 64

按真实 bucket 分桶后：

```text
Wloads = 33
Wscale = 33 / 123 = 0.268
```

含义：

```text
W tile load 次数减少 73.2%
```

这里的 `Wscale` 不是最终 speedup。它只描述 weight-side tile load 缩放。

最终加速要看 cycles：

```text
speedup = baseline_cycles / method_cycles
```

## 7. 当前 Cora 数值结果

当前 Cora h8_54_T40 quick trace：

```text
nodes = 2708
reuse = 27.8%
miss  = 72.2%
```

trace replay 表：

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgD | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 27.8% | 72.2% | 0.722 | 0.722 | 0.722 | 0.77% | 8.00 | 123 | 1.000 |
| GraphBit-now | 27.8% | 72.2% | 0.716 | 0.719 | 0.717 | 2.13% | 6.10 | 123 | 1.000 |
| RiskBucket-b32 | 27.8% | 72.2% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | 63 | 0.512 |
| RiskBucket-b64 | 27.8% | 72.2% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | 33 | 0.268 |

### 7.1 相对 FullP8-miss 的 encoder 加速

```text
GraphBit-now:
    0.722 / 0.716 = 1.01x

RiskBucket-b32:
    0.722 / 0.384 = 1.88x

RiskBucket-b64:
    0.722 / 0.289 = 2.50x
```

### 7.2 相对所有节点都跑 FullP8/W4A8

表中 cycles 已经是相对 all-node FullP8/W4A8 归一化：

```text
FullP8-miss:
    speedup = 1 / 0.722 = 1.38x

RiskBucket-b32:
    speedup = 1 / 0.384 = 2.60x

RiskBucket-b64:
    speedup = 1 / 0.289 = 3.46x
```

### 7.3 AvgDepth 的意义

```text
AvgD = 6.10
```

表示 miss nodes 平均实际执行 activation depth：

```text
activation bit-plane arithmetic scale = 6.10 / 8 = 0.7625
```

也就是仅从 activation bit-plane arithmetic 看：

```text
节省约 23.75%
```

但最终 cycles 没有同比下降，因为 weight/output/scheduling 也占时间。

## 8. 如果 Reuse 达到 50%

假设：

```text
reuse/residual = 50%
miss = 50%
```

### 8.1 Miss 全部 FullP8

相对 all-node FullP8/W4A8：

```text
cost = 0.50
speedup = 2.0x
```

相对 W8A8，如果 W4A8 约等于 W8A8 的 0.5：

```text
cost_vs_W8A8 = 0.50 * 0.5 = 0.25
speedup_vs_W8A8 = 4.0x
```

### 8.2 Reuse + GraphBit-now

用当前 `GraphBit-now / FullP8-miss` 比例：

```text
0.716 / 0.722 = 0.992
```

则：

```text
cost_vs_W4A8 = 0.50 * 0.992 = 0.496
speedup_vs_W4A8 = 2.02x
```

说明只做 early stop，不做 W tile reuse，收益很有限。

### 8.3 Reuse + RiskBucket-b32

当前：

```text
RiskBucket-b32 / FullP8-miss = 0.384 / 0.722 = 0.532
```

如果 miss=50%：

```text
cost_vs_W4A8 = 0.50 * 0.532 = 0.266
speedup_vs_W4A8 = 3.76x
```

相对 W8A8：

```text
cost_vs_W8A8 = 0.266 * 0.5 = 0.133
speedup_vs_W8A8 = 7.52x
```

### 8.4 Reuse + RiskBucket-b64

当前：

```text
RiskBucket-b64 / FullP8-miss = 0.289 / 0.722 = 0.400
```

如果 miss=50%：

```text
cost_vs_W4A8 = 0.50 * 0.400 = 0.200
speedup_vs_W4A8 = 5.00x
```

相对 W8A8：

```text
cost_vs_W8A8 = 0.200 * 0.5 = 0.100
speedup_vs_W8A8 = 10.0x
```

## 9. 与 HEAT-like Static Precision 的理论对比

HEAT 的核心是：

```text
degree/topology -> static high/low precision assignment
```

其 variable-precision PE 将：

```text
m-bit weight x n-bit activation
```

拆成：

```text
m * n 个 1-bit x 1-bit bitmap GEMM
```

因此 HEAT-like static precision 的理论计算量：

```text
C_HEAT =
    alpha * A_high * W_high
  + (1 - alpha) * A_low * W_low
```

相对 W8A8：

```text
C_HEAT_norm =
    C_HEAT / (8 * 8)
```

以 HEAT 文中示意配置为例：

```text
alpha = 0.1
high = 10-bit activation x 8-bit weight
low  = 2-bit activation  x 4-bit weight
```

则：

```text
C_HEAT_norm =
    0.1 * (10 * 8) / 64
  + 0.9 * (2 * 4) / 64

= 0.1 * 1.25 + 0.9 * 0.125
= 0.2375
```

HEAT-like theoretical bit-op speedup：

```text
1 / 0.2375 = 4.21x
```

### 9.1 当前 Cora trace 下的 Graph-Bit 对比

当前 Graph-Bit cycles 是相对 all-node W4A8。若换成 W8A8 口径，近似乘 0.5：

```text
RiskBucket-b32:
    cost_vs_W8A8 = 0.384 * 0.5 = 0.192

RiskBucket-b64:
    cost_vs_W8A8 = 0.289 * 0.5 = 0.1445
```

相对 HEAT-like static precision：

```text
RiskBucket-b32 over HEAT:
    0.2375 / 0.192 = 1.24x

RiskBucket-b64 over HEAT:
    0.2375 / 0.1445 = 1.64x
```

所以当前 Cora trace 下可以得到一个保守理论判断：

```text
Graph-Bit full-stack 相比 HEAT-like aggressive static precision，
前端 encoder 预计有约 1.2x - 1.6x 额外加速空间。
```

### 9.2 50% reuse 情况下的对比

若 reuse/residual 达到 50%，则：

```text
RiskBucket-b32 cost_vs_W8A8 ~= 0.133
RiskBucket-b64 cost_vs_W8A8 ~= 0.100
```

相对 HEAT-like：

```text
b32:
    0.2375 / 0.133 = 1.79x

b64:
    0.2375 / 0.100 = 2.38x
```

因此：

```text
当 reuse/residual 约 50%，且 risk-bucket 能形成 32-64 规模的 W tile service window，
Graph-Bit 前端 encoder 相比 HEAT-like aggressive static precision，
理论上约 1.8x - 2.4x。
```

## 10. 端到端 Speedup 如何计算

如果只讨论 encoder NPU：

```text
encoder_speedup = baseline_encoder_cycles / optimized_encoder_cycles
```

如果讨论完整 TF-GNN pipeline，需要用 Amdahl 形式：

```text
E2E_speedup =
    1 / ((1 - transformer_ratio) + transformer_ratio / encoder_speedup)
```

例如 Transformer/LLM encoder 占 95%：

```text
encoder_speedup = 1.8x:
    E2E = 1 / (0.05 + 0.95 / 1.8)
        = 1.73x

encoder_speedup = 2.4x:
    E2E = 1 / (0.05 + 0.95 / 2.4)
        = 2.24x
```

所以如果相对 HEAT-like encoder 获得 1.8x - 2.4x，而 Transformer 仍占总时间 95%，端到端相对 HEAT-like 系统的理论空间大约是：

```text
1.7x - 2.2x
```

如果 HEAT 的系统瓶颈已经转移到 GNN/NDP，则端到端收益会小于这个值。这个需要用完整 pipeline profiling 再确认。

## 11. 论文中建议报告的指标

不要只报告 `Wscale`。建议同时报告以下四类：

```text
1. Accuracy:
   Acc / Drop

2. Reuse:
   Direct %
   Residual %
   Miss %

3. Graph-Bit arithmetic:
   AvgDepth
   DepthHist
   activation bit-plane saved ratio

4. Hardware replay:
   Wloads
   Wscale
   Cycles
   Traffic
   Energy
   SRAM feasibility
   Tail utilization
```

推荐主表：

| Method | Reuse | Miss | AvgD | Wloads | Wscale | Cycles | Traffic | Energy | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | ... | ... | 8.00 | ... | 1.000 | ... | ... | ... | ... |
| GraphBit-now | ... | ... | ... | ... | 1.000 | ... | ... | ... | ... |
| RiskBucket-b32 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| RiskBucket-b64 | ... | ... | ... | ... | ... | ... | ... | ... | ... |

对比 HEAT-like 时，建议增加：

```text
HEAT-like static precision:
    degree -> high/low precision
    no SimHash reuse
    no residual reuse
    no runtime bound
    no risk-bucket stop-depth scheduling
```

然后统一比较：

```text
normalized encoder cycles
normalized traffic
drop
```

## 12. 当前方案的关键结论

1. 单独的 bit-plane early stop 不够：

```text
FullP8-miss  -> 0.722 cycles
GraphBit-now -> 0.716 cycles
```

原因是 weight/output/scheduling 占比仍然大。

2. 真正的硬件收益来自：

```text
reuse/residual 减少 encoder nodes
    +
predictor-free stop-depth 减少 miss-node bit-plane effort
    +
risk-bucket scheduler 扩大 W tile service window
```

3. 当前 Cora trace 下：

```text
RiskBucket-b32:
    1.88x over FullP8-miss
    2.60x over all-node FullP8/W4A8

RiskBucket-b64:
    2.50x over FullP8-miss
    3.46x over all-node FullP8/W4A8
```

4. 与 HEAT-like aggressive static precision 相比：

```text
当前 Cora trace:
    about 1.2x - 1.6x encoder speedup potential

若 reuse/residual 达到 50%:
    about 1.8x - 2.4x encoder speedup potential
```

5. Graph-Bit 相对 HEAT 的核心差异不是“也用 degree 控制精度”，而是：

```text
HEAT:
    degree -> static precision assignment

Graph-Bit:
    SimHash/residual 决定是否进入 encoder；
    graph risk 决定 min_depth/tolerance；
    predictor-free bound 决定 runtime stop depth；
    risk bucket scheduler 决定 NPU batch order 和 W tile reuse。
```

也就是说，Graph-Bit 把图后端信息推进到了：

```text
encoder invocation
runtime arithmetic depth
NPU scheduling order
weight tile service window
```

这四个层次，而不是只做静态 mixed precision。
