# Graph-Bit Demand-Fetch Model

本文档定义 Graph-Bit 从“bit-depth proxy”走向 NPU 数据流建模的主模型。目标是回答一个核心问题：

```text
Graph risk 让节点少算 bit-plane 之后，
这些 bit-plane 是否真的少读、少占 PE cycles、少产生 traffic？
```

结论先说清楚：

```text
如果 activation 仍按普通 A8 byte layout 读取，Graph-Bit 只能省 bit-plane MAC，
总 cycles / traffic 几乎不降。

如果 activation 采用 bit-plane-major demand-fetch layout，
且 miss nodes 按 risk bucket 成批执行，
Graph-Bit 才能把少算 bit-plane 转化成真实 cycles / traffic 收益。
```

## 1. 建模范围

模型覆盖当前 full-stack 路径：

```text
direct reuse      -> cache read, 不跑 encoder
residual reuse    -> cache read + residual adapter, 不跑 full encoder
miss nodes        -> Graph-Bit NPU
```

Graph-Bit NPU 内部再分：

```text
P8 / high risk    -> 完整 activation bit-plane
P6 / mid risk     -> 至少执行到 P6，可 predictor-free early stop
P4 / low risk     -> 至少执行到 P4，可 predictor-free early stop
```

当前模型只改变 activation bit-plane 执行和读取，不假设 weight read / output write 自动下降。这是保守边界。

## 2. 输入

模型读取两个输入：

```text
1. predictor_free_workload.json
   来自 residual_precision_depth + summarize_graphbit_predictor_free_flow.py
   包含 reuse/direct/residual/P8/P6/P4 比例和 drop。

2. ONNXim aggregate.json
   来自 LLaMA GEMM microbenchmark
   包含 P8/P6/P4/early-stop 下的 cycles、read/write、bitcomp。
```

默认运行：

```bash
bash scripts/run_graphbit_demand_fetch_model.sh
```

默认读取：

```text
output/graphbit_predictor_free/cora_h8_53_T30/predictor_free_workload.json
output/onnxim_graphbit/microbench_s64_internal_*/aggregate.json
```

输出：

```text
output/graphbit_predictor_free/cora_h8_53_T30/demand_fetch_model/demand_fetch_model.txt
output/graphbit_predictor_free/cora_h8_53_T30/demand_fetch_model/demand_fetch_model.tsv
output/graphbit_predictor_free/cora_h8_53_T30/demand_fetch_model/demand_fetch_model.json
```

## 3. 核心公式

设全图节点比例为：

```text
r_direct
r_residual
r_p8
r_p6
r_p4

r_miss = r_p8 + r_p6 + r_p4
r_reuse = r_direct + r_residual
```

### 3.1 Miss-node useful depth

先只看 miss nodes，计算它们理论上想执行的平均深度：

```text
p_i = r_i / r_miss
d_p8 = 8
d_p6 = measured_or_bound_depth_p6
d_p4 = measured_or_bound_depth_p4

D_useful = p8 * d_p8 + p6 * d_p6 + p4 * d_p4
```

### 3.2 Random mixed batch 的执行深度

如果 high/mid/low risk 节点随机混在一个 bit-serial micro-batch，batch 必须执行到其中最深的节点：

```text
D_exec = E[max(depth in batch)]
```

例如 batch size = 64，只要 batch 里有一个 P8 节点，整个 batch 就要跑到 P8。这样低风险节点虽然被判成 P6/P4，实际也会被拖回 P8。

### 3.3 Risk-bucket batch 的执行深度

如果按 risk bucket 成批：

```text
high-risk batch -> P8 / strict tolerance
mid-risk batch  -> P6 / medium tolerance
low-risk batch  -> P4 / loose tolerance
```

则：

```text
D_exec = D_useful
bit_utilization = D_useful / D_exec = 1
```

这就是 Graph-Bit 为什么需要 graph-aware scheduler，而不只是 datapath。

### 3.4 Full-stack cycles / traffic

Miss-node NPU 指标来自 ONNXim：

```text
C_miss = weighted cycles over executed risk classes
T_miss = weighted traffic over executed risk classes
```

全图归一化：

```text
C_full =
    r_miss     * C_miss
  + r_direct   * C_cache
  + r_residual * C_residual

T_full =
    r_miss     * T_miss
  + r_direct   * T_cache
  + r_residual * (T_cache + T_residual)
```

默认小成本：

```text
C_cache = 0.001
C_residual = 0.005
T_cache = 0.003
T_residual = 0.005
```

这些值只用于避免把 reuse 读缓存写成绝对 0。

## 4. Dataflow 级别

模型报告五种路径：

### 4.1 FullP8-miss

```text
accepted reuse hits 不跑 encoder
所有 miss nodes 跑 P8
```

这是 Graph-Bit 的正确 baseline。

### 4.2 Degree compute-mask only

```text
低位 bit-plane MAC 被 mask
但 activation 仍按完整 A8 byte 读取
```

这对应“没有 bit-plane-major layout”的情况。它通常只省 bitcomp，不省 cycles/traffic。

### 4.3 Random demand-fetch

```text
有 bit-plane-major activation layout
但 risk assignment 是 random
```

它能省一些 traffic，但 drop 通常比 Degree 更差。

### 4.4 Degree random-mixed

```text
risk proxy 是 Degree
但 micro-batch 随机混合风险等级
```

如果 batch size 较大，几乎每个 batch 都含 high-risk 节点，实际执行深度会回到 P8。

### 4.5 Degree demand-fetch

```text
risk proxy = Degree
activation layout = bit-plane-major
scheduler = risk bucket batching
```

这是当前 Graph-Bit NPU 主线。

## 5. 当前结果

### 5.1 最新 Cora/LLaMA learned-gate 前端

配置：

```text
h8_53_T30
gate_accept_threshold = 0.60
BUDGET = p8heavy
P8/P6/P4 among all nodes = 56.5% / 14.1% / 0.0%
```

结果：

| Method | Reuse | Miss | UsefulD | ExecD | BitC | ActRd | WeightRd | FullC | FullT | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 29.4% | 70.6% | 8.00 | 8.00 | 1.000 | 1.000 | 1.000 | 0.707 | 0.707 | 1.60% |
| Degree compute-mask only | 29.4% | 70.6% | 7.60 | 7.60 | 0.950 | 1.000 | 1.000 | 0.707 | 0.707 | 2.18% |
| Degree random-mixed | 29.4% | 70.6% | 7.60 | 8.00 | 1.000 | 1.000 | 1.000 | 0.707 | 0.707 | 2.18% |
| Degree demand-fetch | 29.4% | 70.6% | 7.60 | 7.60 | 0.950 | 0.950 | 1.000 | 0.700 | 0.703 | 2.18% |

解释：

```text
p8heavy 很保守，只有 20% miss nodes 降到 P6，没有 P4。
因此 demand-fetch 只能带来约 0.9% full-stack cycles proxy 降低。
这不是机制失败，而是当前精度优先 budget 留给 Graph-Bit 的空间很小。
```

### 5.2 历史 balanced 前端，用于说明机制上限

配置：

```text
h8_54_T40
BUDGET = balanced
P8/P6/P4 among all nodes = 12.0% / 30.0% / 18.0%
```

结果：

| Method | Reuse | Miss | UsefulD | ExecD | BitC | ActRd | WeightRd | FullC | FullT | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 40.0% | 60.0% | 8.00 | 8.00 | 1.000 | 1.000 | 1.000 | 0.601 | 0.602 | 1.53% |
| Degree compute-mask only | 40.0% | 60.0% | 5.80 | 5.80 | 0.726 | 1.000 | 1.000 | 0.601 | 0.602 | 2.39% |
| Degree random-mixed | 40.0% | 60.0% | 6.10 | 8.00 | 1.000 | 1.000 | 1.000 | 0.601 | 0.602 | 2.39% |
| Degree demand-fetch | 40.0% | 60.0% | 6.10 | 6.10 | 0.764 | 0.762 | 1.000 | 0.576 | 0.583 | 2.39% |

解释：

```text
compute-mask only:
    bitcomp 已经降到 72.6%，但 cycles/traffic 不动。

random-mixed:
    有 demand-fetch 硬件也没用，因为 batch 被 high-risk 节点拖到 P8。

degree demand-fetch:
    bitcomp 和 activation read 都降到约 76%，
    但 weight/output 不变，所以 full-stack cycles 只降约 4.1%。
```

这张表是当前最重要的建模结论。

## 6. 可靠性边界

当前模型是保守的：

```text
1. 不假设 weight read 随 activation depth 下降；
2. 不假设 output write 下降；
3. 不把 graph-bit error 重新数值仿真，只继承 embedding proxy drop；
4. risk-bucket padding overhead 只报告，不直接乐观扣除；
5. cache/residual cost 以小常数计入。
```

所以它适合支撑：

```text
Graph-Bit can reduce bit-plane compute and activation-plane reads.
End-to-end gain requires bit-plane-major layout and risk-bucket scheduling.
Weight/output traffic is the remaining bottleneck.
```

不适合直接声称：

```text
真实芯片 speedup = 表里的 FullC-save
真实能耗 = energy proxy
```

## 7. 下一步建模

要继续把收益做大，需要新增两个不改变模型精度的硬件优化：

```text
1. weight-stationary / larger batch amortization
   减少每个节点看到的 weight traffic。

2. FFN intermediate on-chip bypass
   避免 ffn_up output write + ffn_down input read。
```

这两项和 Graph-Bit 不冲突：

```text
Graph-Bit:
    减 activation bit-plane compute/read。

weight-stationary / FFN bypass:
    减固定 weight/output/input traffic。
```

最终 NPU 叙事应是：

```text
reuse/residual reduces encoder invocations;
Graph-Bit demand-fetch reduces activation bit-plane work for remaining misses;
risk-bucket scheduling protects utilization;
weight-stationary and FFN bypass attack the remaining fixed memory cost.
```
