# Graph-Bit Predictor-Free Early Stop and W-Tile Reuse

本文档说明 Graph-Bit 的两条硬件主线：

```text
1. predictor-free bit-plane early stop:
   runtime bound 决定 miss node 实际执行到几 bit。

2. risk-bucket W-stationary scheduling:
   同风险 miss nodes 聚成更大的 W tile service window，减少 W tile load 次数。
```

复现实验流程见：

```text
docs/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
```

Degree 到 `min_depth / tolerance / stop_depth` 的映射见：

```text
docs/archive/npu/GRAPH_BIT_DEGREE_BOUND_POLICY.md
```

## 1. Predictor-Free Early Stop

Graph-Bit 不训练额外 predictor，也不使用 FP/quant embedding 的 oracle 差值。硬件只使用：

```text
runtime partial sum
remaining low-bit upper bound
graph-risk tolerance
```

对一个 miss-node batch，NPU 按 activation bit significance 从高位到低位发射：

```text
A8 = b7 b6 b5 b4 b3 b2 b1 b0

P8: execute b7..b0
P6: execute b7..b2
P5: execute b7..b3
P4: execute b7..b4
```

运行时逻辑：

```text
for depth in min_depth..8:
    issue bit-plane work required by this depth
    bound = remaining_low_bit_bound(depth)

    if bound <= tolerance:
        stop at depth
        break
```

Degree / TSER / Context 不直接指定最终 P8/P6/P5/P4。它们只提供：

```text
risk bucket -> min_depth + tolerance
runtime bound -> actual stop depth
```

当前默认 Degree policy：

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

## 2. Bound 定义

低位 bit-plane 对 GEMM partial sum 的贡献有上界。执行到某个 depth 后，剩余低位贡献越小，越可以停止。

概念形式：

```text
remaining_low_bit_bound(depth)
    ~= remaining_activation_low_bits(depth)
       * weight_abs_bound
       * tile_scale
```

其中：

```text
remaining_activation_low_bits(depth):
    未执行低位的最大可能数值贡献。

weight_abs_bound:
    当前 W tile 的权重绝对值统计。

tile_scale:
    K tile 长度、量化 scale、累加范围带来的归一化因子。
```

这是一条保守上界路径，不依赖 learned predictor。

## 3. Early Stop 省什么

当 runtime bound 停止低位后，NPU 不再发射这些低位对应的工作：

```text
skip low-bit PE issue cycles
skip low-bit MAC activity
skip low-bit partial-sum update
```

因此 early stop 的直接收益主要体现在：

```text
PE bit-plane work
RF activity
partial-sum read/update/write
energy proxy
```

它不自动减少 W tile 的 HBM load。W tile 搬运需要由下一节的 bucket scheduler 处理。

## 4. W Tile Reuse

普通 Transformer/NPU dataflow 本来就会复用 W tile。Graph-Bit 的差异不是“发现 W 可以复用”，而是把 graph risk / stop-depth trace 变成调度信号：

```text
miss nodes
    -> runtime stop-depth trace
    -> D8 / D6 / D5 / D4 buckets
    -> same-risk micro-batch
    -> W-stationary execution
```

如果一个 batch 混有不同 stop-depth：

```text
node A: D8
node B: D5
node C: D6

batch effective depth = max(D8, D5, D6) = D8
```

低风险节点会被高风险节点拖到更深执行。Graph-Bit 按 depth/risk 分桶：

```text
D8 nodes -> D8 bucket
D6 nodes -> D6 bucket
D5 nodes -> D5 bucket
D4 nodes -> D4 bucket
```

这样一个 W tile 可以在同一 bucket 内服务更多同执行深度的 node blocks。

## 5. Wloads 统计

当前 replay 使用真实 per-node trace。每个 miss node 有：

```text
node_id
route: miss
stop_depth: 8 / 6 / 5 / 4
```

baseline：

```text
baseline_tile_batch = 16
miss_nodes = N
baseline_Wloads = ceil(N / 16)
```

risk-bucket scheduler：

```text
candidate_batch = B

bucket_Wloads =
    ceil(N8 / B)
  + ceil(N6 / B)
  + ceil(N5 / B)
  + ceil(N4 / B)

Wscale = bucket_Wloads / baseline_Wloads
```

Cora h8_54_T40 trace 例子：

```text
miss nodes = 1954
baseline_tile_batch = 16
baseline_Wloads = ceil(1954 / 16) = 123

RiskBucket-b32:
    Wloads = 63
    Wscale = 63 / 123 = 0.512

RiskBucket-b64:
    Wloads = 33
    Wscale = 33 / 123 = 0.268
```

## 6. b32 / b64 Tradeoff

`b32` 和 `b64` 表示 bucket 内 W-stationary tile batch 的候选大小：

```text
b32:
    每次 W tile load 最多服务 32 个同 depth/risk 节点。

b64:
    每次 W tile load 最多服务 64 个同 depth/risk 节点。
```

更大的 bucket batch 会减少 Wloads，但也增加：

```text
activation / psum / output buffer pressure
调度等待时间
tail utilization 损失
SRAM / NoC 压力
```

当前建议：

```text
b32:
    主线保守配置。

b64:
    sensitivity / aggressive point。
```

## 7. Trace Replay 到 Cycles

当前硬件表由四部分组成：

```text
1. GraphhopSimhash 导出真实 workload profile
   direct / residual / miss 比例和 stop-depth trace。

2. ONNXim 跑 LLaMA GEMM component
   Q/K/V/O projection、FFN gate/up、FFN down。

3. replay_graphbit_trace_scheduler.py 重放调度
   original order / risk-bucket order。

4. 按 stop-depth histogram、Wloads 和 ONNXim component cost
   汇总 cycles / traffic / energy。
```

关键字段：

```text
Cycles:
    相对所有节点跑 FullP8 encoder 的归一化 cycles。

Traffic:
    DRAM read/write requests 的归一化 proxy。

Energy:
    当前为 cycles 和 traffic 的 proxy 组合。

AvgDepth:
    miss nodes 的平均 stop depth。

Wloads:
    trace replay 统计出的 W tile load 次数。

Wscale:
    Wloads / baseline_Wloads。
```

## 8. 当前 Cora Trace Result

Cora h8_54_T40 trace-driven replay：

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgDepth | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 27.8% | 72.2% | 0.722 | 0.722 | 0.722 | 0.77% | 8.00 | 123 | 1.000 |
| GraphBit-now | 27.8% | 72.1% | 0.716 | 0.719 | 0.717 | 2.13% | 6.10 | 123 | 1.000 |
| FullP8-bucket-b32 | 27.8% | 72.2% | 0.385 | 0.368 | 0.377 | 0.77% | 8.00 | 62 | 0.504 |
| RiskBucket-b32 | 27.8% | 72.1% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | 63 | 0.512 |
| FullP8-bucket-b64 | 27.8% | 72.2% | 0.290 | 0.191 | 0.241 | 0.77% | 8.00 | 31 | 0.252 |
| RiskBucket-b64 | 27.8% | 72.1% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | 33 | 0.268 |

结论：

```text
W-stationary bucket scheduling:
    当前是 cycles / traffic 的主要收益来源。

predictor-free mixed depth:
    降低 AvgDepth 和片上 PE/RF/psum 活动。
```

bit-depth-sensitive activity breakdown：

| Compare | Activity-C Save | Activity-E Save | PE/Psum Save | Extra Drop |
|---|---:|---:|---:|---:|
| RiskBucket-b32 vs FullP8-bucket-b32 | 12.1% | 15.6% | 23.7% | +1.36% |
| RiskBucket-b64 vs FullP8-bucket-b64 | 13.9% | 16.8% | 23.7% | +1.36% |

## 9. 代码入口

```text
GraphhopSimhash/scripts/replay_graphbit_trace_scheduler.py
GraphhopSimhash/scripts/onnxim_graphbit_microbench.py
GraphhopSimhash/scripts/model_graphbit_internal_roofline.py
```

一键流程：

```bash
cd /home/zhangshangtong/Transformer/OFA
bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```
