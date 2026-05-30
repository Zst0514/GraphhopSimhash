# GraphHopSimhash TODO

本文档记录当前版本还缺什么、下一步实验怎么补，以及哪些方向暂时不建议继续投入。

## 1. 当前已经足够支撑主线的部分

### 1.1 Residual reuse front-end

已经完成：

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

当前 ST 结果：

| Dataset | Reuse | Drop |
|---|---:|---:|
| Cora/ST | 46.5% | 0.93% |
| PubMed/ST | 42.3% | 1.96% |

这部分说明 hash/CAM + residual gate 在 ST front-end 下已经足够强。

### 1.2 Graph-Bit NPU proof chain

当前已经形成两条证据链：

```text
Accuracy validation:
    W4A8 / W4A6 / W4A5 / W4A4 embedding pools
    -> 验证 effective bit-depth 对 GNN accuracy/drop 的影响

Hardware validation:
    ONNXim component simulation
    + per-node stop-depth trace
    + trace-driven scheduler replay
    -> 验证 bit-plane early stop / risk-bucket batching 对 cycles/traffic/energy 的影响
```

这比“只做 embedding proxy”更强，也避免了直接手写一个理想 speedup。

### 1.3 当前 Cora Graph-Bit hardware table

当前 Cora h8_54_T40 trace-driven replay：

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgDepth | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 27.8% | 72.2% | 0.722 | 0.722 | 0.722 | 0.77% | 8.00 | 123 | 1.000 |
| GraphBit-now | 27.8% | 72.1% | 0.716 | 0.719 | 0.717 | 2.13% | 6.10 | 123 | 1.000 |
| RiskBucket-b32 | 27.8% | 72.1% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | 63 | 0.512 |
| RiskBucket-b64 | 27.8% | 72.1% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | 33 | 0.268 |

结论：

```text
单独 early stop 对 full-stack cycles 收益较小；
risk-bucket scheduler + W tile reuse 才是把收益放大的关键。
```

## 2. 必须补的实验

### TODO-1: Cora/PubMed 统一 full-stack 主表

目标：

```text
把 residual front-end + Graph-Bit NPU 放到同一个结果表里。
```

需要报告：

| Dataset | Reuse | Miss | Method | Cycles | Traffic | Energy | Drop | AvgDepth | Wloads |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Cora | ... | ... | FullP8-miss | ... | ... | ... | ... | ... | ... |
| Cora | ... | ... | GraphBit-now | ... | ... | ... | ... | ... | ... |
| Cora | ... | ... | RiskBucket-b32 | ... | ... | ... | ... | ... | ... |
| PubMed | ... | ... | FullP8-miss | ... | ... | ... | ... | ... | ... |

当前缺口：

```text
Cora 已经有 trace-driven hardware table。
PubMed 还需要用相同 replay pipeline 跑一遍。
```

优先级：最高。

### TODO-2: HEAT-like baseline 对比

需要构造一个清晰 baseline：

```text
HEAT-like static degree precision:
    degree 直接决定 P8/P6/P4
    不做 predictor-free runtime bound
    不做 residual/reuse hierarchy
    不做 risk-bucket stop-depth scheduler
```

对比表建议：

| Method | Reuse | Bound | Bucket | Cycles | Traffic | Energy | Drop |
|---|---:|---|---|---:|---:|---:|---:|
| HEAT-like Degree Precision | 0% | no | no | ... | ... | ... | ... |
| Reuse + FullP8-miss | yes | no | no | ... | ... | ... | ... |
| Reuse + Static Graph-Bit | yes | no | maybe | ... | ... | ... | ... |
| Reuse + Predictor-Free Graph-Bit | yes | yes | yes | ... | ... | ... | ... |

这张表要回答：

```text
我们的收益来自哪里？
    reuse/residual
    predictor-free runtime bound
    risk-bucket scheduler
```

优先级：最高。

### TODO-3: PubMed trace-driven scheduler replay

PubMed 当前 residual reuse 已经有好结果，但 Graph-Bit hardware replay 还需要补。

建议先做：

```text
runs = 1
dataset = pubmed
front-end = shared online residual reuse
Graph-Bit = degree + predictor-free bound
candidate batch = 32 / 64
```

如果耗时可接受，再补：

```text
runs = 3
```

优先级：高。

### TODO-4: Small-sample bit-plane proxy sanity check

不建议全量逐节点、逐层、逐 GEMM 重跑完整 LLaMA encoder。

建议只做小样本 sanity check：

```text
sample nodes = 64 / 128
layers = 1-2 个 representative layer
GEMM = projection + FFN up/down
compare:
    A8 full
    static A6/A5 proxy
    runtime high-bit-plane truncation
```

目的：

```text
验证 W4A6/W4A5 pools 作为 runtime bit-depth proxy 是否方向一致。
```

这不是主实验，只是用于回答 reviewer 对 proxy 合理性的质疑。

优先级：中高。

## 3. 应该补的实验

### TODO-5: Risk-bucket batch size / SRAM feasibility sweep

当前 b32/b64 结果需要进一步给出硬件可行性边界。

需要 sweep：

```text
candidate batch = 16 / 32 / 64 / 128
SRAM budget = small / medium / large
tile_K = 64 / 128 / 256
```

报告：

```text
Wloads
Wscale
tail waste
activation buffer size
psum buffer size
SRAM feasible or not
cycles / traffic / energy
```

这能回答：

```text
b64 是否过于理想？
b32 是否是更实际的主线点？
```

优先级：高。

### TODO-6: Arxiv feasibility-only run

Arxiv 很耗时，不建议一上来跑完整多 seed accuracy。

先跑 feasibility-only：

```text
reuse / miss profile
stop-depth histogram
risk bucket size
Wloads / Wscale
SRAM feasibility
```

目标不是立刻拿最终 accuracy，而是证明：

```text
大图上 risk bucket 更大，W tile reuse 机会更充分。
```

优先级：中。

### TODO-7: Degree / TSER / Context 在 Graph-Bit 里的边界

当前很多结果显示 Degree 是最稳的 deployable policy。

后续文档和实验应明确：

```text
Degree:
    主线 graph risk proxy，简单、稳定、硬件友好。

TSER / Context:
    作为语义修正或 ablation，不作为主结论。
```

还需要补一张小表：

| Dataset | Policy | Drop | AvgDepth | Wloads |
|---|---|---:|---:|---:|
| Cora | Random | ... | ... | ... |
| Cora | Degree | ... | ... | ... |
| Cora | TSER | ... | ... | ... |
| PubMed | Degree | ... | ... | ... |

优先级：中。

## 4. 可选增强

### Optional-1: 更细粒度 ONNXim per-tile event trace

当前 replay 是：

```text
real node trace
    + ONNXim component cost lookup
    + scheduler replay
```

可选增强是把 replay 进一步下沉：

```text
per-tile issue event
per-plane fetch event
per-tile W load event
psum update event
```

这会让 simulator 更像 cycle-level trace，但工程量较大。

优先级：可选。

### Optional-2: 更完整的 energy model

当前 energy 是 proxy。

可以补：

```text
HBM read/write energy
SRAM read/write energy
RF/broadcast energy
MAC energy per bit-plane
psum update energy
```

优先级：可选。

### Optional-3: LLaMA residual adapter 再调

ST 下 residual reuse 很强，但 LLaMA embedding 维度更高、空间不同，当前 LLaMA residual 需要更谨慎。

可以补：

```text
rank 64 / 128
train pairs 4096 / 8192
separate/shared accept gate
support split sweep
```

但不要让它拖慢 Graph-Bit NPU 主线。

优先级：可选。

## 5. 暂时不建议继续做的方向

### Not recommended-1: 全量 bit-plane LLaMA encoder 重跑

不建议做：

```text
逐节点、逐层、逐 GEMM 真正按 bit-plane early stop
重新跑完整 LLaMA encoder
再生成全图 embedding
```

原因：

```text
工程量极大
运行时间极长
不一定比当前两阶段验证多回答核心问题
容易把论文主线拖入 kernel 实现细节
```

当前更合理的做法是：

```text
embedding-pool accuracy validation
    +
ONNXim / trace-driven hardware validation
```

最多补 small-sample sanity check。

### Not recommended-2: 继续主打 FFN channel gating

FFN gating 可以作为历史探索，但不建议作为主线。

原因：

```text
精度控制更难
容易像普通 channel pruning
不如 Graph-Bit 的 bit-serial datapath + graph scheduler 有体系结构新意
```

### Not recommended-3: 把 TSER 写成一定优于 Degree

实验更支持：

```text
Degree 是更稳定的 deployable graph risk proxy。
TSER / Context 是语义修正消融。
```

不要把主结论写成 TSER 一定优于 Degree。

## 6. 推荐执行顺序

建议按以下顺序推进：

```text
1. 补 PubMed trace-driven scheduler replay
2. 做 Cora/PubMed unified full-stack main table
3. 做 HEAT-like baseline 对比表
4. 做 risk-bucket batch size / SRAM feasibility sweep
5. 做 small-sample bit-plane proxy sanity check
6. 做 Arxiv feasibility-only run
7. 最后再考虑更细粒度 ONNXim per-tile event trace
```

## 7. 最小论文主表

如果只保留一张主表，建议是：

| Dataset | Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgDepth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | FullP8-miss | ... | ... | ... | ... | ... | ... | 8.00 |
| Cora | HEAT-like Degree | ... | ... | ... | ... | ... | ... | ... |
| Cora | GraphBit-now | ... | ... | ... | ... | ... | ... | ... |
| Cora | GraphBit-b32 | ... | ... | ... | ... | ... | ... | ... |
| PubMed | FullP8-miss | ... | ... | ... | ... | ... | ... | 8.00 |
| PubMed | HEAT-like Degree | ... | ... | ... | ... | ... | ... | ... |
| PubMed | GraphBit-b32 | ... | ... | ... | ... | ... | ... | ... |

其中：

```text
FullP8-miss:
    reuse 前端固定，miss 全部 P8。

HEAT-like Degree:
    degree 直接控制静态 bit-depth。

GraphBit-now:
    predictor-free bound + activation demand fetch。

GraphBit-b32:
    GraphBit-now + risk-bucket scheduler replay。
```

这张表能完整回答：

```text
reuse 省了多少节点；
Graph-Bit 让 miss nodes 平均算到几 bit；
risk-bucket scheduler 把 W tile reuse 放大了多少；
精度 drop 是否可接受。
```
