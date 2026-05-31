# Graph-Bit Main Results

本文档只保留当前主线结果。历史 sweep、proxy ablation 和过时探索已移到 `docs/archive/`。

## 1. Shared Online Residual Reuse

当前 residual reuse 的共享在线控制流：

```text
8 heads x 16 bits
radius = 2
score gate = on
score weights = 3 / 1 / 1
score threshold T = 30

support >= 5   -> direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute / Graph-Bit
gate_accept_threshold = 0.575
```

3-run 结果：

| Dataset | Baseline | Reuse | Acc | Drop | TrainPairs | Alpha |
|---|---:|---:|---:|---:|---:|---:|
| Cora/ST | 0.7200 | 46.5% | 0.7107 | 0.93% | 464.7 | 0.263 |
| PubMed/ST | 0.7587 | 42.3% | 0.7392 | 1.96% | 151.3 | 0.309 |

详细说明见：

```text
docs/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md
```

## 2. Cora Graph-Bit Trace-Driven Hardware Replay

当前 Graph-Bit full-stack 结果固定：

```text
Dataset: Cora
Front-end: h8_54_T40
Backend: LLaMA-7B W4A8/W4A6/W4A5/W4A4 pools
Graph-Bit policy: degree priority + predictor-free runtime bound
Trace replay source:
    output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/
```

结果相对“所有节点都跑 FullP8 encoder”归一化：

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgDepth | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 27.8% | 72.2% | 0.722 | 0.722 | 0.722 | 0.77% | 8.00 | 123 | 1.000 |
| GraphBit-now | 27.8% | 72.1% | 0.716 | 0.719 | 0.717 | 2.13% | 6.10 | 123 | 1.000 |
| FullP8-bucket-b32 | 27.8% | 72.2% | 0.385 | 0.368 | 0.377 | 0.77% | 8.00 | 62 | 0.504 |
| RiskBucket-b32 | 27.8% | 72.1% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | 63 | 0.512 |
| FullP8-bucket-b64 | 27.8% | 72.2% | 0.290 | 0.191 | 0.241 | 0.77% | 8.00 | 31 | 0.252 |
| RiskBucket-b64 | 27.8% | 72.1% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | 33 | 0.268 |

含义：

```text
FullP8-miss:
    reuse/residual 前端固定，所有 miss nodes 都完整执行 P8。

GraphBit-now:
    只启用 predictor-free stop-depth / bit-plane issue proxy。
    不额外假设 W tile batch amortization。

FullP8-bucket-b32 / b64:
    所有 miss nodes 仍完整执行 P8，但使用更大的 W-stationary service window。
    这行隔离出“只做 W tile batching、不做 mixed-depth early stop”的收益。

RiskBucket-b32 / b64:
    在 GraphBit-now 基础上，按真实 stop-depth trace 做 risk-bucket scheduler replay。
    Wloads/Wscale 来自 trace replay 统计。
```

这个消融暴露了当前最重要的边界：

```text
FullP8-bucket-b32: 0.385 cycles, 0.77% drop
RiskBucket-b32:    0.384 cycles, 2.13% drop

FullP8-bucket-b64: 0.290 cycles, 0.77% drop
RiskBucket-b64:    0.289 cycles, 2.13% drop
```

也就是说，在当前 ONNXim component model 和 Cora trace 下，大头收益几乎全部来自 W-stationary bucket batching。predictor-free mixed-depth 把 AvgDepth 从 8.00 降到 6.10，但没有带来明显额外 cycles 下降，反而引入了额外 accuracy drop。因此当前硬件主线应优先强调：

```text
graph-risk bucket scheduler + W-stationary W tile reuse
```

mixed-depth / early-stop 应作为第二层片上算术与能耗优化，后续需要用更细粒度 RF / psum / PE 活动模型证明它相对 FullP8-bucket 的额外收益。

### 2.1 Bit-depth-sensitive Activity Breakdown

为避免 ONNXim component cycles 掩盖 mixed-depth 的片上收益，新增 activity breakdown：

```text
script:
    scripts/model_graphbit_activity_breakdown.py

output:
    output/graphbit_trace_replay/cora_h8_54_T40_fullp8_bucket_ablation/activity_breakdown/
```

该模型把每行拆成：

```text
W_HBM, A_HBM, A_RF, PE, W_RF, Psum, Out, Scheduler
```

关键结果：

| Compare | ONNX Cycles Save | Activity Cycles Save | Activity Energy Save | PE/W_RF/Psum Save | Extra Drop |
|---|---:|---:|---:|---:|---:|
| RiskBucket-b32 vs FullP8-bucket-b32 | 0.1% | 12.1% | 15.6% | 23.7% | +1.36% |
| RiskBucket-b64 vs FullP8-bucket-b64 | 0.3% | 13.9% | 16.8% | 23.7% | +1.36% |

解释：

```text
ONNX component cycles:
    当前对 P8/P6/P5 的区别不敏感，因此看不到 mixed-depth cycles 收益。

activity model:
    A_RF / PE / W_RF / Psum 随 AvgDepth/8 缩放。
    AvgDepth 从 8.00 降到 6.10 后，这些片上活动下降约 23.7%。
```

因此当前更准确的定位是：

```text
W-stationary bucket batching:
    主要 latency / traffic 收益来源。

mixed-depth predictor-free early stop:
    主要片上 RF / PE / psum activity 和能耗优化。
```

复现流程见：

```text
docs/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
```

## 3. 当前结论

主线结论分两层：

```text
Residual reuse:
    用 hash/CAM 找 anchor，用 residual gate 控制 fuzzy hit 质量。
    Cora/PubMed 在共享在线配置下可以达到 40%+ reuse，drop < 2%。

Graph-Bit NPU:
    对 miss nodes 使用 graph-risk-controlled bit-serial execution。
    predictor-free bound 决定 runtime stop depth；
    risk-bucket scheduler 将同 stop-depth miss nodes 组织成更大的 W-tile reuse batch。
```

论文表述时应避免把贡献写成“degree 指导量化”。更准确的边界是：

```text
Graph structure controls encoder execution hierarchy:
    reuse whether to run encoder,
    bound how many activation bit-planes to execute,
    scheduler how to group miss nodes for W tile reuse.
```
