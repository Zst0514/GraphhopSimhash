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
| RiskBucket-b32 | 27.8% | 72.1% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | 63 | 0.512 |
| RiskBucket-b64 | 27.8% | 72.1% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | 33 | 0.268 |

含义：

```text
FullP8-miss:
    reuse/residual 前端固定，所有 miss nodes 都完整执行 P8。

GraphBit-now:
    只启用 predictor-free early stop 和 activation demand fetch。
    不额外假设 W tile batch amortization。

RiskBucket-b32 / b64:
    在 GraphBit-now 基础上，按真实 stop-depth trace 做 risk-bucket scheduler replay。
    Wloads/Wscale 来自 trace replay 统计。
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
