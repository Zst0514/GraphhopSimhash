# Project TODO

本文档只记录当前仍需补齐的实验和技术缺口。历史探索保留在 `docs/archive/`。

## 1. Immediate

### 1.1 PubMed Full-Stack Trace Replay

输入：

```text
shared online residual-gate front-end
degree / propagation risk tolerance
predictor-free runtime bound
risk-bucket W-stationary replay
```

输出：

```text
FullP8-miss
GraphBit-now
FullP8-bucket-b32 / b64
RiskBucket-b32 / b64
```

指标：

```text
Reuse / Miss / Cycles / Traffic / Energy / Drop / AvgDepth / Wloads / Wscale
```

### 1.2 LLaMA Residual-Gate Sanity

ST residual MLP 不能直接迁移到 LLaMA embedding 空间。LLaMA front-end 需要：

```text
LLaMA W4A8 target embeddings
LLaMA-specific residual/gate training
FullP8-miss sanity check
```

只有 `FullP8-miss` 的 drop 足够低，Graph-Bit 才能接在后面评估 miss-node NPU cost。

### 1.3 HEAT-Like Baseline

需要一个清晰的 baseline：

```text
static degree-guided precision
no residual/reuse hierarchy
no predictor-free runtime bound
no risk-bucket scheduler
```

对比目标：

```text
reuse/residual contributes how much
runtime bound contributes how much
risk-bucket scheduling contributes how much
```

## 2. Hardware Modeling

### 2.1 Large-M GEMM Roofline

真实 encoder GEMM 的 row dimension 是：

```text
M = node_batch * sequence_length
```

下一轮 profile 使用：

```text
M = 2048 / 4096 / 8192 / 16384 / 65536
```

报告每类 GEMM：

```text
Q/K/V/O projection
FFN gate/up
FFN down
```

并区分：

```text
memory-bound
compute-bound
```

### 2.2 Activity Model

当前 ONNXim component cycles 对 P8/P6/P5 的 latency 差异不够敏感。需要继续保留 activity-level 拆分：

```text
W_HBM
A_HBM
A_RF
PE
Psum
Output
Scheduler
```

Graph-Bit mixed-depth 的主要收益先按：

```text
PE / Psum / activation-side activity
```

报告；W-stationary bucket scheduling 单独报告 W tile service window 的收益。

### 2.3 Bucket Feasibility

对 b16 / b32 / b64 / b128 做 feasibility sweep：

```text
bucket size
Wloads
tail waste
SRAM requirement
cycles / traffic / energy
```

b32 / b64 表示 W tile service window，不是 bit-width。

## 3. Accuracy Validation

### 3.1 Tolerance Sweep

第一版不引入 operator sensitivity。只扫：

```text
degree risk -> tolerance
A_low_bound(depth) * W_tile_abs_bound -> stop depth
```

输出：

```text
AvgDepth
Drop
Depth histogram
```

### 3.2 Small-Sample Runtime Check

不全量重跑逐节点、逐层、逐 GEMM 的 LLaMA encoder。只做小样本 sanity：

```text
sample nodes = 64 / 128
representative layers = 1-2
GEMM = projection + FFN up/down
compare:
    A8 full
    W4A6/W4A5 proxy
    runtime truncated activation depth
```

目的：

```text
确认 W4A6/W4A5 embedding pool 和 runtime high-bit truncation 方向一致。
```

## 4. Documentation

Keep these as primary entry points:

```text
README.md
docs/README.md
docs/results/GRAPH_BIT_MAIN_RESULTS.md
docs/results/RESIDUAL_GATE_GRAPHBIT_NPU_PROGRESS.md
docs/npu/GRAPH_BIT_NPU_DESIGN.md
docs/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
```

Historical sweeps and superseded proposals belong in:

```text
docs/archive/
```

## 5. Low-Priority / Historical

These are not current mainline:

```text
partial-depth encoder L4/L8/L16
token compaction / prefix truncation
FFN channel gating as main contribution
oracle error routing
predictor/calibration-node learned damage model
```

They can be mentioned as explored alternatives, but not used as the main contribution.
