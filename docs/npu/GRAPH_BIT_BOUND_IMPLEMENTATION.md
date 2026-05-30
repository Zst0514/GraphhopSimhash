# Graph-Bit Predictor-Free Bound Implementation

本文档记录 Graph-Bit 逐 bit-plane early stop 的实现逻辑。它是当前 NPU 主线里最关键的一步：Degree / TSER 不再直接指定 `P8/P6/P4`，而是给每个 risk bucket 配置 `min_depth` 和 `tolerance`，真正执行到几 bit 由运行时 bound 决定。

## 1. 设计目标

Graph-Bit 的硬件目标不是“离线生成一个 6-bit embedding pool”，而是：

```text
所有 miss nodes 进入 W4A8 bit-serial GEMM。
GEMM 从高位 activation bit-plane 开始执行。
每执行到一个候选深度 d，就估计剩余低位最多还能改变输出多少。
如果剩余影响小于图风险允许的 tolerance，就停止低位 bit-plane。
```

这个机制不使用：

```text
FP reference embedding
quant embedding error
calibration nodes
learned predictor
oracle damage
```

只使用：

```text
当前 tile 的 activation 剩余低位范围
当前 W tile 的 abs-sum / abs-mean metadata
当前 partial output norm 的硬件代理
graph risk bucket 给出的 min_depth / tolerance
```

## 2. Bound 公式

对一个 GEMM tile：

```text
Y = A @ W
A = A_high(d) + A_low(d)
Y_partial = A_high(d) @ W
DeltaY = A_low(d) @ W
```

early-stop 要判断：

```text
||DeltaY|| 是否已经小到可以忽略？
```

硬件友好的上界：

```text
||DeltaY|| <= max_abs(A_low_remaining(d)) * sum_abs(W_tile)
```

对 A8 activation：

```text
max_abs(A_low_remaining(d)) = 2^(8-d) - 1
```

例如：

```text
P4 后剩余低位最大幅度 = 15
P5 后剩余低位最大幅度 = 7
P6 后剩余低位最大幅度 = 3
```

为了跨 layer / tile 比较，需要归一化：

```text
norm_bound =
    remaining_bound / (partial_norm_proxy + remaining_bound)
```

其中：

```text
remaining_bound =
    normalized_low_range(d) * tile_k * W_abs_stat

partial_norm_proxy =
    normalized_high_range(d) * tile_k * W_abs_mean * partial_norm_scale
```

## 3. Bound Mode

当前实现支持三种模式：

```text
range:
    旧版范围上界。
    只看 remaining low-bit range 和 sqrt(tile_k)。
    用于 ablation。

tile_mean:
    主线模式。
    用 W tile 的 mean abs 统计估计 remaining_bound 和 partial_norm。
    更接近真实 tile-level predictor-free bound。

tile_max:
    保守模式。
    remaining_bound 使用 W_abs_max，partial_norm 仍使用 W_abs_mean。
    会更晚停止，用于 stress test。
```

## 4. Graph Risk 如何进入

Degree / TSER 不直接决定最终位宽。它们只决定 risk bucket：

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

停止条件：

```text
if depth >= min_depth[bucket] and norm_bound <= tolerance[bucket]:
    stop
else:
    continue next bit-plane
```

这就是 runtime-bound 的核心：low-risk 节点可以从 P4 开始尝试停止，但如果 bound 判断 P4 不安全，就继续补到 P5/P6。

## 5. ONNXim 实现位置

核心实现文件：

```text
ONNXim/src/SimulationConfig.h
ONNXim/src/Common.cc
ONNXim/src/operations/GemmWS.cc
scripts/onnxim_graphbit_microbench.py
scripts/graphbit_bound_sanity.py
```

新增配置项：

```json
{
  "graphbit_bound_mode": "tile_mean",
  "graphbit_bound_weight_abs_mean": 0.50,
  "graphbit_bound_weight_abs_max": 1.00,
  "graphbit_bound_partial_norm_scale": 1.0,
  "graphbit_bound_partial_norm_floor": 1e-6,
  "graphbit_bound_safety_factor": 1.0
}
```

`GemmWS.cc` 的流程：

```text
1. annotate_graphbit() 为每条 MOVIN / GEMM_PRELOAD 指令计算 effective depth。
2. select_graphbit_effective_depth() 从 min_depth 扫到 config_depth。
3. estimate_graphbit_remaining_bound() 根据 bound mode 计算 norm_bound。
4. bound <= tolerance 时停止。
5. fetch_depth / issue_depth / weight_depth / psum_depth 写进 Instruction。
6. SystolicWS 根据这些 depth 缩放 issue cycles 并统计 GraphBitDataflow。
```

## 6. 快速检查命令

不跑 ONNXim，只看不同 bound mode 的停止深度：

```bash
python GraphhopSimhash/scripts/graphbit_bound_sanity.py \
  --mode tile_mean \
  --min-depth 4 \
  --config-depth 8 \
  --tolerance 0.04 \
  --tile-k 32 64 128 256
```

对比保守模式：

```bash
python GraphhopSimhash/scripts/graphbit_bound_sanity.py \
  --mode tile_max \
  --min-depth 4 \
  --config-depth 8 \
  --tolerance 0.04 \
  --tile-k 32 64 128 256
```

## 7. ONNXim Microbenchmark

运行单个 LLaMA-7B GEMM microbenchmark：

```bash
python GraphhopSimhash/scripts/onnxim_graphbit_microbench.py \
  --seq-len 64 \
  --workspace output/onnxim_graphbit/microbench_s64_tilemean \
  --graphbit-depth 8 \
  --graphbit-min-depth 4 \
  --graphbit-bound-enable \
  --graphbit-bound-mode tile_mean \
  --graphbit-bound-tolerance 0.04 \
  --graphbit-activation-layout plane_group \
  --graphbit-plane-group-bits 2 \
  --graphbit-weight-rf-gate \
  --graphbit-psum-gate \
  --action all
```

运行完整 datapath suite：

```bash
SEQ_LEN=16 bash GraphhopSimhash/scripts/run_onnxim_graphbit_datapath_suite.sh
```

输出：

```text
output/onnxim_graphbit/datapath_suite_s16/datapath_summary.txt
output/onnxim_graphbit/datapath_suite_s16/datapath_summary.tsv
```

## 8. 结果如何解读

关注这些字段：

```text
AvgDepth:
    runtime-bound 最终平均执行深度。

fetch:
    activation plane-group fetch 深度。

issue:
    PE bit-plane issue 深度。

wrf:
    W RF / broadcast 深度。

psum:
    partial-sum update 深度。

act/orig:
    activation demand-fetch 是否真的减少。
```

判断标准：

```text
byte_major:
    即使 AvgDepth 下降，act/orig 仍接近 1.0。

plane_group:
    AvgDepth / fetch 下降时，act/orig 应随之下降。

tile_mean vs range:
    tile_mean 是主线，更接近 tile-aware predictor-free bound。

tile_max:
    更保守，若收益仍存在，说明机制更稳。
```

## 9. 快速验证结果

当前用 `SEQ_LEN=8` 跑通了 ONNXim datapath suite：

```bash
SEQ_LEN=8 LOG_LEVEL=info \
OUT_ROOT=output/onnxim_graphbit/datapath_bound_modes_s8 \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_datapath_suite.sh
```

核心行如下：

```text
case                         cycles      act/orig  w/orig  fetch issue wrf psum
full_p8                      37645344    1.000     1.000   0.00  0.00  0.00 0.00
byte_major_mask_only_p6      37645344    1.000     1.000   8.00  8.00  8.00 8.00
plane_group2_issue_rf_psum_p6 37230336    0.750     1.000   6.00  6.00  6.00 6.00
plane_group2_bound_low       37222912    0.750     1.000   6.00  5.00  5.00 5.00
plane_group2_tilemean_low    37222912    0.750     1.000   6.00  5.00  5.00 5.00
plane_group2_tilemax_low     37230336    0.750     1.000   6.00  6.00  6.00 6.00
```

这说明：

```text
1. byte_major_mask_only 几乎没有收益：
   只在 PE 内部 mask 低位，而 activation 仍按完整 byte 读入。

2. plane_group2_issue_rf_psum_p6 开始减少 activation fetch：
   activation fetch 从 8 plane 降到 6 plane，act/orig=0.75。

3. tile_mean bound 在 tolerance=0.04 下进一步把 issue/wrf/psum 降到 5：
   最终执行深度由 runtime bound 决定，不是静态写死 P5。

4. tile_max 更保守，停在 6：
   这是 stress-test bound，用来证明收益和安全性之间可以调节。
```

这个验证的关键意义是：Graph-Bit 现在已经不是“Degree 直接指定 P8/P6/P4”，而是：

```text
graph risk -> min_depth / tolerance
runtime tile bound -> actual stop depth
```

## 10. 当前边界

这仍然是 simulator-level 实现，不是 RTL：

```text
已实现:
    tile-level bound
    bit-depth selection
    activation demand fetch
    PE issue gating
    W RF / psum gating statistics

未实现:
    真实 Verilog PE array
    真实 activation value distribution
    per-channel exact partial sum
```

因此论文表述应是：

```text
Graph-Bit implements a predictor-free tile-level bit-plane bound in ONNXim
and estimates cycles / traffic / energy proxy under a bit-plane-major NPU dataflow.
```

不要写成已经完成 RTL 或硅级能耗测量。
