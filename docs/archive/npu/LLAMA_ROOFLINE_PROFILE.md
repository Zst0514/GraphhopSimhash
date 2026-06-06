# LLaMA ONNXim Roofline Profile

本文档记录当前用 ONNXim 对 LLaMA-7B 主要 GEMM 组件做 roofline profiling 的方法和结论。这个实验的目的不是验证 Graph-Bit 本身，而是先回答一个前置问题：

```text
LLaMA encoder 的主要计算部分到底是 memory-bound 还是 compute-bound？
```

如果某部分明显 memory-bound，那么只减少 activation bit-plane compute 不一定会降低 latency；如果某部分已经 compute-exposed，那么 mixed-depth / early-stop 才更可能转化为 cycles 收益。

## 1. Profiling Scope

当前 profile 覆盖 LLaMA-7B 每层最主要的 linear GEMM：

```text
Projection:
    Q / K / V / O projection
    shape: [M, 4096] x [4096, 4096]
    count_per_layer = 4

FFN up/gate:
    gate_proj + up_proj
    shape: [M, 4096] x [4096, 11008]
    count_per_layer = 2

FFN down:
    down_proj
    shape: [M, 11008] x [11008, 4096]
    count_per_layer = 1
```

当前还没有把 attention score/value GEMM 和 softmax 加进 profile。原因是当前 Graph-Bit NPU 主线主要作用于 projection / FFN GEMM，且这几类 linear GEMM 已经覆盖 LLaMA encoder 的主要 W traffic。

## 2. Reproduction

使用已有 ONNXim microbench 输出：

```bash
cd /home/zhangshangtong/Transformer/OFA

/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/profile_onnxim_llama_roofline.py \
  --summary \
    output/onnxim_graphbit/microbench_s16_internal_p8/summary.tsv \
    output/onnxim_graphbit/microbench_s32_internal_p8/summary.tsv \
    output/onnxim_graphbit/microbench_s64/summary.tsv \
    output/onnxim_graphbit/microbench_s128_internal_p8/summary.tsv \
  --output-dir output/onnxim_graphbit/llama_roofline_p8_m16_m128
```

输出：

```text
output/onnxim_graphbit/llama_roofline_p8_m16_m128/llama_roofline_profile.txt
output/onnxim_graphbit/llama_roofline_p8_m16_m128/llama_roofline_components.tsv
output/onnxim_graphbit/llama_roofline_p8_m16_m128/llama_roofline_layers.tsv
```

绘图命令：

```bash
cd /home/zhangshangtong/Transformer/OFA/GraphhopSimhash

/home/zhangshangtong/.conda/envs/OFA/bin/python \
  scripts/plot_llama_roofline.py \
  --profile-dir /home/zhangshangtong/Transformer/OFA/output/onnxim_graphbit/llama_roofline_p8_m16_m128 \
  --output docs/figures/llama_roofline_profile.png
```

生成图：

```text
docs/figures/llama_roofline_profile.png
docs/figures/llama_roofline_profile.pdf
docs/figures/llama_roofline_profile.svg
```

![LLaMA ONNXim roofline profile](../figures/llama_roofline_profile.png)

默认 peak 参数：

```text
peak_compute = 131.1 TFLOP/s
peak_mem     = 614.4 GB/s
ridge point  = 213.3 FLOP/Byte
```

这些参数来自当前 ONNXim config 的 4-core 128x128 systolic array 和 HBM 配置。后续如果换 NPU 配置，需要同步调整脚本参数：

```bash
--peak-gflops ...
--peak-gbps ...
```

## 3. Key Results

Layer-level profile：

| M | Layer Cycles | OI | Achieved TFLOP/s | Achieved GB/s | W Request Share | A Request Share | Bound |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 1,192,410 | 15.9 | 5.43 | 341.5 | 96.0% | 3.7% | memory / W-memory |
| 32 | 1,252,017 | 31.6 | 10.34 | 327.3 | 92.3% | 7.0% | memory / W-memory |
| 64 | 1,348,735 | 62.5 | 19.21 | 307.5 | 85.8% | 13.1% | memory, compute starts to expose |
| 128 | 1,504,912 | 122.0 | 34.43 | 282.2 | 75.1% | 22.9% | memory, closer to compute-exposed |

Per-layer cycle share is stable:

| Part | Approx. Layer Cycle Share |
|---|---:|
| Q/K/V/O projection | 32% - 33% |
| FFN gate/up | 40% - 41% |
| FFN down | 26% |

Thus FFN total is about:

```text
FFN gate/up + FFN down ~= 66% - 67% layer cycles
```

## 4. Interpretation

### 4.1 Small M is strongly W-memory-bound

For M=16/32:

```text
W request share = 92% - 96%
OI              = 16 - 32 FLOP/Byte
ridge point     = 213 FLOP/Byte
```

This means:

```text
The layer is far below the ridge point.
Most traffic is weight traffic.
Reducing activation bit-depth alone cannot strongly reduce wall cycles.
```

This explains why earlier P8/P6/P5 ONNXim component cycles were nearly identical: bit-plane compute decreased, but the exposed critical path was still dominated by W memory / fixed path.

### 4.2 Larger M exposes more compute

As M grows:

```text
M=16  -> OI 15.9, Wshare 96.0%
M=32  -> OI 31.6, Wshare 92.3%
M=64  -> OI 62.5, Wshare 85.8%
M=128 -> OI 122.0, Wshare 75.1%
```

The workload is still theoretically memory-bound at M=128, but compute is much more visible. This is why risk-bucket batching matters:

```text
larger same-risk bucket
    -> larger effective M
    -> W tile amortized across more rows
    -> OI increases
    -> bit-depth reduction has a better chance to affect latency
```

### 4.3 Mixed-depth is not the first-order latency knob at small M

For small M, the correct priority is:

```text
1. increase W tile reuse / bucket batch size
2. reduce W reloads
3. then use mixed-depth early stop to reduce exposed PE/RF/psum activity
```

So the current Graph-Bit claim should be:

```text
Graph risk first improves NPU scheduling locality and W reuse;
predictor-free mixed-depth then reduces arithmetic/on-chip activity for the remaining miss nodes.
```

Do not claim:

```text
A8 -> A6/P5 alone gives large end-to-end latency reduction.
```

## 5. Design Consequences

### 5.1 What to optimize first

The roofline says the first-order target is W movement:

```text
Graph-risk bucket scheduler
    group same-risk miss nodes
    increase effective GEMM M
    amortize W tile loads
```

This is the mechanism that can move the workload from strongly W-memory-bound toward compute-exposed.

### 5.2 Where mixed-depth fits

Mixed-depth / predictor-free early stop is still useful, but its role should be stated carefully:

```text
small M / W-memory-bound:
    mainly energy / activity saving

larger M / W-reuse strong:
    can become latency-visible
```

Thus mixed-depth should be coupled with W-stationary bucket scheduling, not presented as an isolated latency optimization.

### 5.3 What to measure next

The next measurement should be:

```text
M sweep:
    M = 16 / 32 / 64 / 128 / 256 / 512

For each M:
    P8 / P6 / P5 wall cycles
    W request share
    OI
    achieved TFLOP/s
    achieved GB/s
```

This will directly show the transition point where bit-depth begins to affect latency.

## 6. Current Takeaway

The current ONNXim roofline supports the following conclusion:

```text
For small graph miss buckets, LLaMA GEMM is W-memory-bound.
Graph-Bit's most important hardware contribution is graph-risk bucket scheduling for W tile reuse.
Predictor-free mixed-depth is a second-stage optimization that reduces bit-op and on-chip activity,
and becomes latency-relevant only after W movement is sufficiently amortized.
```
The generated table uses `PEwork` rather than strict utilization:

```text
PEwork = ONNXim ideal matmul work / wall cycles
```

It can exceed 100% when work is spread across cores or pipeline overlap. It is used only as a compute-exposure proxy; the theoretical roofline bound is still determined by OI and the peak compute/memory ratio.
