# Graph-Bit NPU Design

本文档是 Graph-Bit NPU 的主设计入口。当前主线只维护五份 NPU 文档：

```text
GRAPH_BIT_NPU_DESIGN.md
    设计总览：datapath、scheduler、buffer、cost model、主结论。

GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
    实现细节：CLI、runner、ONNXim/GemmWS、trace replay 中 bit-plane early stop 的代码路径。

GRAPH_BIT_DEGREE_BOUND_POLICY.md
    风险策略：Degree 如何映射到 high/mid/low bucket、min_depth、tolerance 和 runtime stop depth。

GRAPH_BIT_PREDICTOR_FREE_WTILE.md
    机制细化：predictor-free bound、activation bit-plane demand fetch、W tile 省搬运和 b32/b64 tradeoff。

GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
    复现和调参：从 residual/Graph-Bit trace 导出到 ONNXim component lookup、scheduler replay、cycles 表的完整流程。
```

早期 proxy、理论拆分和旧 scheduler 文档已移到 `docs/archive/npu/`，不作为主入口。

## 1. 设计目标

Graph-Bit 解决的是 graph-text workload 下的 LLM encoder 执行问题：

```text
不是所有节点都应该完整执行 LLM encoder。
图后端风险决定节点是否复用、是否修正、以及 miss 后在 NPU 内部算多少 activation bit-plane。
```

完整系统分成四条路径：

```text
P0: exact hash reuse
    exact CAM hit，直接读 embedding cache，cost 近似 0。

P1: fuzzy hash reuse + residual correction
    fuzzy CAM 找 anchor，residual adapter 修正 embedding。

P2: Graph-Bit encoder
    miss node 必须跑 encoder，但根据 graph risk 控制 bit-serial activation 执行深度。

P3: Full W4A8 encoder
    高风险 miss node 兜底，完整执行 P8。
```

本文档只讨论 P2/P3 的 NPU 设计。P0/P1 见：

```text
docs/core/CAM设计.md
docs/core/RESIDUAL_CORRECTED_REUSE.md
```

## 2. 核心思想

Graph-Bit 不是静态地把节点分配成 W4A8 / W4A6 / W4A4，也不是只做 FFN channel gating。它的硬件主线是：

```text
Graph risk controls arithmetic effort inside the NPU datapath.
```

具体来说，W4 权重路径保持不变，activation 逻辑上是 A8，但 NPU 采用 bit-serial / bit-grained 执行：

```text
A8 bit-plane: b7 b6 b5 b4 b3 b2 b1 b0

P8: execute b7..b0
P6: execute b7..b2
P5: execute b7..b3
P4: execute b7..b4
```

在真正的 predictor-free early stop 中，P6/P5/P4 不是最终静态位宽，而是验证 anchor / safety floor：

```text
degree / graph risk -> min_depth + tolerance
runtime bound       -> actual stop depth
```

例子：

```text
high-risk:
    min_depth = 8
    tolerance = 0
    基本完整 P8

mid-risk:
    min_depth = 6
    tolerance = medium
    至少算到 P6，再看 bound 是否继续低位

low-risk:
    min_depth = 4
    tolerance = large
    至少算到 P4，再由 bound 决定是否停在 P5/P6/P8
```

当前 accuracy validation 使用离线 embedding pools 近似实际 depth：

```text
P8 = W4A8
P6 = W4A6
P5 = W4A5
P4 = W4A4
```

## 3. Predictor-Free Bound

Graph-Bit 不使用 learned predictor，也不使用 oracle error。它采用类似 PADE/BETA 思路的 predictor-free bound：

```text
for each node batch:
    for bit_plane from MSB to LSB:
        issue bit-plane MAC
        update partial sum
        estimate remaining low-bit bound
        if depth >= min_depth and bound <= tolerance:
            stop lower bit-planes
```

这里的区别是：PADE 的 bound 主要服务 sparse attention/top-k 正确性；Graph-Bit 的 bound 服务 graph downstream tolerance。

```text
PADE:
    bound 判断某个 attention candidate 是否不可能进入重要集合。

Graph-Bit:
    bound 判断继续执行低位 activation bit-plane 对图任务是否值得。
```

Graph risk 进入的是 `min_depth` 和 `tolerance`，而不是直接指定最终深度：

```text
Degree / TSER / Context / LowUnique
    -> risk bucket
    -> min_depth, tolerance
    -> runtime bound decides actual depth
```

当前主线更推荐 Degree 作为 Graph-Bit 控制信号，因为实验中它比 TSER 更稳定；TSER 保留为语义修正消融。

## 4. NPU Datapath

Graph-Bit 需要五个关键硬件组件。

### 4.1 Plane-Group Activation Buffer

普通 A8 byte-major layout 是：

```text
A_byte = [b7 b6 b5 b4 b3 b2 b1 b0]
```

一读就是完整 8 bit。即使 low-risk node 只需要高 5 bit，低 3 bit 也已经被读进来了。

Graph-Bit 使用 plane-group-major activation buffer：

```text
Group 0: b7 b6
Group 1: b5 b4
Group 2: b3 b2
Group 3: b1 b0
```

这样当 runtime bound 在 P5/P6 停止时，可以不再 demand-fetch 后续低位 plane group。

### 4.2 Bit-Plane Issue Scheduler

PE array 不应该只是拿到完整 activation 后 mask 结果，而是从 issue 阶段停止：

```text
if bound_satisfied:
    do not issue lower bit-plane cycles
```

这会减少：

```text
PE MAC cycles
activation plane reads
partial-sum update
weight RF/broadcast energy for skipped planes
```

### 4.3 Weight-Stationary Tile Window

W4 权重 tile 仍然要服务高位 activation plane，因此 Graph-Bit 不能简单说“低位停了所以 HBM weight 少读”。要减少 weight-side traffic，需要让一个 W tile 在片上服务更多 node blocks：

```text
load W tile once
serve many same-risk node blocks
evict W tile
```

这要求 scheduler 按 graph risk / stop-depth 把 miss nodes 分桶，形成更大的 same-risk micro-batch。

### 4.4 Partial-Sum Gating

跳过低位 bit-plane 后，partial sum 也不再做低位更新：

```text
skip psum read
skip add/update
skip psum writeback
```

这主要节省片上能耗，而不一定直接体现在 DRAM traffic。

### 4.5 Risk-Bucket Scheduler

如果 high-risk 和 low-risk 节点混在同一个 micro-batch，一个 high-risk node 会把整个 batch 拖到 P8：

```text
mixed batch depth = max(depth of nodes in batch)
```

因此 Graph-Bit 的 scheduler 需要：

```text
1. 收集 miss node trace
2. 按 stop-depth / risk bucket 分桶
3. 每个 bucket 内形成 micro-batch
4. 复用 W tile
5. 对每个 batch 执行 bit-plane early stop
```

这也是图场景相对于普通 LLM accelerator 多出来的机会：图风险不只是决定精度，还决定 NPU 的 batch order 和 W tile service window。

## 5. 当前在线策略

当前 residual + Graph-Bit 推荐先固定前端，再评估 miss-node NPU：

```text
8 heads x 16 bit
radius = 2
score gate on
score weights = 3 / 1 / 1
```

共享在线 residual reuse 推荐配置见：

```text
docs/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md
```

Graph-Bit hardware quick path 当前常用 Cora/LLaMA 配置：

```text
h8_54_T40
hard direct: support >= 5
residual: support == 4
compute / Graph-Bit: support < 4
```

注意：不同 backend 下要先检查 `FullP8-miss` 是否安全。如果 reuse 前端本身已经导致较大 drop，Graph-Bit 不应该背锅。

## 6. 仿真层次

当前有三层验证。

### 6.1 Embedding-Pool Proxy

用 P8/P6/P5/P4 embedding pools 验证不同 graph risk policy 对 accuracy/cost 的影响。

用途：

```text
验证 Degree / TSER / Context 是否能指导 precision depth。
```

主线结果汇总见 `docs/results/GRAPH_BIT_MAIN_RESULTS.md`。

### 6.2 ONNXim Component Simulation

用 ONNXim 跑 LLaMA-7B 关键 GEMM component：

```text
Q/K/V/O projection
FFN gate/up/down
```

输出：

```text
cycles
DRAM read/write requests
bit-plane depth
component energy proxy
```

### 6.3 Trace-Driven Scheduler Replay

这是当前最重要的硬件证据链：

```text
real miss node trace
    -> risk / stop-depth bucket
    -> micro-batch replay
    -> W tile load count
    -> ONNXim component cost lookup
```

复现流程见 `docs/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md`。

一键命令：

```bash
bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

## 7. Cora Trace-Driven Quick Result

当前 Cora/LLaMA h8_54_T40, seed42, Degree runtime-bound：

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgD | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 27.8% | 72.2% | 0.722 | 0.722 | 0.722 | 0.77% | 8.00 | 123 | 1.000 |
| GraphBit-now | 27.8% | 72.2% | 0.716 | 0.719 | 0.717 | 2.13% | 6.10 | 123 | 1.000 |
| RiskBucket-b32 | 27.8% | 72.2% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | 63 | 0.512 |
| RiskBucket-b64 | 27.8% | 72.2% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | 33 | 0.268 |

关键解释：

```text
GraphBit-now:
    AvgDepth 从 8.0 降到 6.1，但 Wloads 不变。
    说明只停低位 bit-plane 不足以显著降低总体 cycles。

RiskBucket-b32 / b64:
    Wloads 由 per-node trace replay 统计得到。
    b32: 63 / 123 = 0.512
    b64: 33 / 123 = 0.268
```

结论：

```text
Graph-Bit 的完整硬件收益来自两个协同：
1. predictor-free early stop 减少 activation bit-plane effort
2. risk-bucket scheduler 增大 W tile service window
```

## 8. 当前边界

当前仿真不是 full-system cycle-accurate。更准确地说：

```text
ONNXim component-cycle simulation
+ real workload trace
+ trace-driven bucket scheduler replay
```

当前仿真仍有边界：

```text
1. 没有完整模拟整个 LLaMA encoder graph 的所有 operator overlap。
2. on-chip SRAM/RF energy 仍是 proxy，不是电路级功耗。
3. per-MAC exact stop-depth trace 还没有完全下沉到每个 GEMM tile 的真实数值。
```

下一步如果要继续增强，应优先做：

```text
1. PubMed / Arxiv trace replay。
2. ONNXim per-tile event trace，把 replay 从 node-level 推到 tile-level。
3. SRAM capacity / NoC bandwidth sensitivity。
```

## 9. 论文表述建议

不要把贡献写成“Degree 指导量化”，这个点太弱，也容易和 HEAT/CATOR 混淆。

更稳的表述是：

```text
Graph-conditioned bit-serial encoder execution.
```

具体创新点：

```text
1. 图后端风险进入 NPU datapath，而不是只做外部调度。
2. predictor-free bound 决定 runtime stop-depth。
3. risk-bucket scheduler 把 graph risk 转化为 W tile reuse 机会。
4. 与 CAM reuse / residual reuse 组成 full-stack encoder execution hierarchy。
5. 优化目标是 graph downstream accuracy，不是单个 attention top-k。
```
