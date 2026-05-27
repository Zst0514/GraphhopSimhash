# Graph-Bit NPU Design

本文档把 Graph-Bit 从实验现象整理成正式的 NPU 设计。主线目标是：

```text
当 graph-text node 必须执行 LLM encoder 时，
图任务风险控制 NPU 内部 activation bit-plane 的执行深度。
```

Graph-Bit 不是普通的 W4A8/W4A4 路由，也不是 FFN channel gating。它深入到 GEMM datapath：同一个 W4 weight encoder path 下，activation 逻辑上仍是 A8，但硬件可以只执行 P8/P6/P5/P4 中的一种 precision depth。

## 1. 系统位置

完整系统可以分成四级：

```text
P0: exact hash reuse
    命中 exact anchor，直接读 embedding cache，cost ~= 0

P1: fuzzy hash reuse + residual correction
    读 anchor embedding，再用 low-rank residual adapter 修正，cost 很小

P2: Graph-Bit precision-depth encoder
    必须跑 encoder，但根据 graph risk 执行 P8/P6/P5/P4 bit-depth

P3: full W4A8 encoder
    高风险兜底路径，完整执行 P8
```

本文档只定义 P2/P3 的 NPU 内部设计。P0/P1 由 SimHash/CAM 和 residual reuse engine 负责。

## 2. 主线策略边界

主线 deployable policy 只能使用在线可得信息：

```text
Degree / propagation risk
TSER = propagation + graph context + low-degree uniqueness
Context-only
LowUnique-only
Random baseline
```

以下策略降级为 debug/oracle，不作为主策略：

```text
PredictorDepthBudget:
    需要 calibration nodes 拟合 damage predictor。
    可用于诊断 hand-crafted proxy 还有多少空间，但增加部署前校准成本。

OracleDamageBudget:
    需要全图 reference embedding 和 low-depth embedding 的真实误差。
    只能作为不可部署上界。
```

论文主表应主要报告 Random / Degree / TSER / Context / LowUnique；Predictor/Oracle 放在 debug 或 upper-bound 小节。

## 3. Datapath 定义

### 3.1 Precision Depth

Graph-Bit 使用 W4 weight path，activation 以 A8 逻辑格式进入。bit-serial / bit-grained datapath 从高位到低位执行：

```text
A8 activation bit planes:
    b7 b6 b5 b4 b3 b2 b1 b0

P8:
    execute b7..b0

P6:
    execute b7..b2

P5:
    execute b7..b3

P4:
    execute b7..b4
```

P6/P5/P4 可以理解成提前终止低位 activation bit-plane。当前实验用离线 embedding pools 近似这个过程：

```text
P8 = W4A8
P6 = W4A6
P5 = W4A5
P4 = W4A4
```

### 3.2 PE Array

每个 PE 支持 W4 x A-bit 的 bit-plane accumulation：

```text
for bit_plane in MSB_to_LSB:
    activation_slice = A[bit_plane]
    partial_sum += W4 * activation_slice * bit_weight
    if precision_depth reached:
        stop remaining lower bit planes
```

硬件需要的新增状态很少：

```text
mode register:
    2-bit precision mode: P8/P6/P5/P4

bit-plane sequencer:
    控制当前 batch 执行到哪一位

partial-sum buffer:
    保存每个 output tile 的累加值

early-stop mask:
    当前 batch 达到指定 precision depth 后停止低位计算
```

第一版不要求 per-element dynamic stopping。更稳的实现是 per-node-batch mode，即一个 micro-batch 内节点共享 P8/P6/P5/P4。这样调度简单，array utilization 更高。

## 4. Scheduler

### 4.1 输入

每个节点进入 encoder 前，scheduler 已经有：

```text
node id
reuse status: exact / fuzzy / miss
graph risk score: Degree / TSER / Context / LowUnique
target path: P0/P1/P2/P3
```

Graph-Bit 只处理 miss nodes，也就是仍然需要跑 encoder 的节点。

### 4.2 Risk-to-depth Mapping

固定 budget 映射：

```text
sort nodes by risk descending

top high_ratio:
    P8

next mid_ratio:
    P6

next low_ratio:
    P5

remaining:
    P4
```

典型 budget：

```text
10/20/30/40  -> aggressive
20/30/30/20  -> balanced-low-cost
30/40/20/10  -> balanced-safe
50/30/20/0   -> near-lossless
```

更硬件友好的执行顺序：

```text
1. 收集一批 miss nodes
2. 根据 risk 排序或桶化
3. 分成 P8/P6/P5/P4 mode queues
4. 同 mode nodes 组成 micro-batch
5. mode register 写入 NPU
6. 执行对应 bit-plane depth
```

### 4.3 为什么不主打 learned predictor

Predictor routing 会引入 calibration nodes 和训练成本，影响架构通用性。Graph-Bit 主线使用 Degree/TSER 这类无需训练的 graph proxy；Predictor 只作为 debug/profiling baseline。

## 5. Buffer 设计

Graph-Bit 需要四类 buffer：

```text
1. Node mode queue
   保存 P8/P6/P5/P4 四个队列的 node ids。

2. Activation bit-plane buffer
   将 A8 activation 按 bit-plane 或 packed group 读入。
   P4/P5/P6 batch 不读取或不送入低位 bit-plane。

3. Partial-sum buffer
   保存当前 GEMM tile 的累加结果。
   P4/P5/P6 仍输出同样 shape，只是低位贡献被省略。

4. Embedding output buffer
   保存 encoder output embedding，供后续 GNN 或 embedding cache 使用。
```

关键点：Graph-Bit 不改变 embedding shape，不改变 GNN 后端接口。它只改变 encoder 内部算术努力。

## 6. Cost Model

当前实验使用一个简单但可解释的 cost model：

```text
cost(bit) = cost_scale * (fixed_cost + (1 - fixed_cost) * bit / reference_bits)
```

本轮 LLaMA-7B 设置：

```text
reference_bits = 8
cost_scale     = 0.50
fixed_cost     = 0.15

P8 cost = 0.500
P6 cost = 0.394
P5 cost = 0.341
P4 cost = 0.287
```

含义：

```text
cost_scale:
    W4A8 encoder 相对 FP/full baseline 的归一化成本。

fixed_cost:
    不随 activation bit-plane 减少而消失的固定开销，
    包括 weight fetch、control、LayerNorm、softmax、pooling、scale/repack 等。

bit / reference_bits:
    可被 bit-plane early termination 缩减的 activation-side GEMM effort。
```

对于一个 batch：

```text
BatchCost =
    ratio_P8 * cost(P8)
  + ratio_P6 * cost(P6)
  + ratio_P5 * cost(P5)
  + ratio_P4 * cost(P4)
```

与 reuse 组合时：

```text
TotalCost =
    reuse_exact_ratio * 0
  + reuse_fuzzy_ratio * residual_adapter_cost
  + miss_ratio * GraphBitCost(miss nodes)
```

## 7. Step 3: Bit-Plane Early-Termination Simulation

Step 3 的目标是从 embedding-pool 近似推进到真正的 bit-plane 模型。

第一阶段已经完成：

```text
W4A8/W4A6/W4A5/W4A4 embedding pools
    ~= P8/P6/P5/P4 fixed precision depth
```

下一阶段软件仿真：

```text
for each GEMM tile:
    compute high bit-plane partial sums
    estimate remaining low-bit contribution bound
    compare with graph-conditioned tolerance
    stop low bit-planes when safe
```

Graph-conditioned tolerance：

```text
high-risk node:
    strict tolerance -> more bit-planes

low-risk node:
    loose tolerance -> early stop
```

这一步的验证指标：

```text
executed_bit_planes / full_bit_planes
partial-sum error
embedding error
downstream GNN accuracy drop
array utilization
mode-switch overhead
```

## 8. Step 4: Mode-Adaptive PE Array

Step 4 把已有 FFN gating 降级为一个 mode-adaptive PE array 的实例，而不是主贡献本身。

Graph-Bit PE array 应支持：

```text
P8 mode:
    full A8 bit-plane execution

P6/P5/P4 mode:
    early-stop lower bit-plane

optional FFN-gated mode:
    在低风险 batch 上减少部分 FFN channel
```

FFN gating 的定位：

```text
不是主线机制，
而是证明 mode-adaptive PE array 可以支持多种 low-effort execution mode。
```

最终硬件故事：

```text
Graph risk -> mode scheduler -> mode-adaptive PE array

mode can control:
    activation precision depth
    optional FFN channel budget
    optional outlier protection budget
```

其中最核心、最应该主打的是 activation precision depth，因为它直接作用于 GEMM bit-plane datapath，适用 QKV / attention projection / FFN 等主要线性层。

## 9. 当前验证结果

LLaMA-7B / Arxiv，10 runs：

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 10/20/30/40 | 0.346 | 0.82 | 0.59 | 0.66 | 0.75 | 0.87 | Degree |
| 20/30/30/20 | 0.378 | 0.52 | 0.36 | 0.38 | 0.45 | 0.57 | Degree |
| 30/40/20/10 | 0.404 | 0.36 | 0.22 | 0.26 | 0.34 | 0.43 | Degree |
| 50/30/20/0  | 0.436 | 0.21 | 0.12 | 0.13 | 0.18 | 0.21 | Degree |

这个结果说明：

```text
同样 precision-depth budget 下，
graph risk 尤其 Degree/propagation risk 能比 Random 更稳地保护精度。
```

更多 Cora/PubMed/Arxiv 结果见：

```text
docs/GRAPH_BIT_VALIDATION_SUMMARY.md
```

## 10. 论文表述建议

推荐主贡献表述：

```text
Graph-Bit is a graph-conditioned precision-depth NPU for graph-text LLM encoders.
It maps graph propagation and semantic risk to activation bit-plane execution depth,
allowing low-risk nodes to terminate bit-serial GEMM earlier while preserving high-risk nodes with full W4A8 execution.
```

不要写成：

```text
Degree-guided W4A8/W4A4 quantization routing.
```

更准确的是：

```text
Graph risk controls arithmetic effort inside the NPU datapath.
```

