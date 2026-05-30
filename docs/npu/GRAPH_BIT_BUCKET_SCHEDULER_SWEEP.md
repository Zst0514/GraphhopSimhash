# Graph-Bit Bucket Scheduler Sweep

本文档记录 Graph-Bit NPU 里最关键的 memory-side 验证：当 miss nodes 按图风险分桶后，NPU 是否能让同一个 weight tile 服务更多同风险节点，从而把 predictor-free bit-plane early stop 的收益从“只省 activation”放大到“cycles / traffic 都下降”。

## 1. 为什么需要这个实验

前面的 predictor-free early stop 已经证明：

```text
activation bit-plane 可以少 fetch
PE issue / W RF / psum update 可以随 stop depth 下降
```

但在保守数据流下：

```text
weight HBM traffic 仍然约等于 FullP8
w/orig = 1.0
```

因此总 cycles 只下降约 1%。这不是 early stop 无效，而是说明如果每个小 batch 都重新读巨大 W tile，LLaMA GEMM 会被 weight-side traffic 主导。

所以需要验证：

```text
graph risk bucket scheduler
    -> 把同风险 miss nodes 聚成更大的 micro-batch
    -> 同一个 W tile 驻留片上服务更多节点
    -> 摊薄 W HBM traffic
```

## 2. 实验设计

脚本：

```bash
bash GraphhopSimhash/scripts/run_onnxim_graphbit_bucket_sweep.sh
```

两个维度分开建模：

```text
seq_len:
    真实 GEMM M 维，表示同风险 bucket 的 micro-batch size。

stationary_tile_batch:
    显式 weight-stationary 调度假设。
    表示一个 W tile 被加载后，可以服务多少个同风险 node blocks。
```

这样可以避免把真实 batch-size 效应和额外的 W tile reuse 假设混在一起。

## 3. 默认命令

快速验证：

```bash
SEQ_LENS="8" \
STATIONARY_TILE_BATCHES="16 32" \
OUT_ROOT=output/onnxim_graphbit/bucket_sweep_smoke \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_bucket_sweep.sh
```

主 sweep：

```bash
SEQ_LENS="8 16" \
STATIONARY_TILE_BATCHES="16 32 64" \
OUT_ROOT=output/onnxim_graphbit/bucket_sweep \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_bucket_sweep.sh
```

更完整但更慢：

```bash
SEQ_LENS="8 16 32" \
STATIONARY_TILE_BATCHES="16 32 64 128" \
OUT_ROOT=output/onnxim_graphbit/bucket_sweep_full \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_bucket_sweep.sh
```

## 4. Case 定义

```text
full_p8:
    所有 miss nodes 完整 W4A8/P8 执行。

gb_now:
    当前保守 Graph-Bit datapath。
    使用 tile_mean bound + plane-group activation fetch + issue/RF/psum gating。
    不假设额外 W HBM reuse。

gb_ws_b16:
    与 gb_now 相同，但显式打开 weight-stationary 模型。
    stationary_tile_batch = baseline_tile_batch = 16，因此等价于无额外 W HBM 收益。

gb_ws_b32 / gb_ws_b64 / gb_ws_b128:
    同一个 W tile 服务更多同风险 node blocks。
    这是 scheduler/capacity sensitivity，不是默认免费收益。
```

## 5. 输出字段

结果文件：

```text
output/onnxim_graphbit/bucket_sweep/bucket_sweep_summary.txt
output/onnxim_graphbit/bucket_sweep/bucket_sweep_summary.tsv
```

关键字段：

```text
CycRed:
    相对同 seq_len 的 FullP8 cycles 下降。

TrafRed:
    DRAM read + write request 下降。

EnerRed:
    简单 energy proxy:
        0.5 * cycle_norm + 0.5 * traffic_norm

act/orig:
    activation bit-plane demand fetch 是否下降。

w/orig:
    weight HBM traffic 是否被摊薄。

fetch / issue:
    runtime bound 最终导致的 activation fetch depth / PE issue depth。
```

## 6. 当前 smoke test 结果

`seq_len=8`，`stationary_tile_batch=16/32` 的 smoke test：

```text
seq  case       tileB  CycRed  TrafRed  EnerRed  act/orig  w/orig  fetch issue
--------------------------------------------------------------------------------
8    gb_now     -       1.1%     0.5%     0.8%     0.750    1.000   6.00  5.00
8    gb_ws_b16  16      1.1%     0.5%     0.8%     0.750    1.000   6.00  5.00
8    gb_ws_b32  32     46.8%    49.5%    48.1%     0.750    0.500   6.00  5.00
```

解释：

```text
gb_now:
    predictor-free early stop 有效，但只减少 activation / issue。
    weight traffic 不变，所以端到端收益小。

gb_ws_b16:
    baseline tile batch 与 stationary tile batch 相等。
    这是 sanity check，结果应与 gb_now 一致。

gb_ws_b32:
    W tile 被 2x 更多同风险 node blocks 复用。
    w/orig 从 1.0 变成 0.5，cycles / traffic 明显下降。
```

这说明 Graph-Bit 的硬件收益关键不只是 bit-plane early stop，而是：

```text
graph risk bucket scheduler
    + bit-plane-major activation fetch
    + weight-stationary W tile reuse
```

## 7. 当前主 sweep 结果

主 sweep 使用：

```bash
SEQ_LENS="8 16" \
STATIONARY_TILE_BATCHES="16 32 64" \
OUT_ROOT=output/onnxim_graphbit/bucket_sweep \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_bucket_sweep.sh
```

结果：

```text
seq  case       tileB  CycRed  TrafRed  EnerRed  act/orig  w/orig  fetch issue
--------------------------------------------------------------------------------
8    gb_now     -       1.1%     0.5%     0.8%     0.750    1.000   6.00  5.00
8    gb_ws_b16  16      1.1%     0.5%     0.8%     0.750    1.000   6.00  5.00
8    gb_ws_b32  32     46.8%    49.5%    48.1%     0.750    0.500   6.00  5.00
8    gb_ws_b64  64     59.9%    73.9%    66.9%     0.750    0.250   6.00  5.00
16   gb_now     -       0.6%     0.9%     0.8%     0.750    1.000   6.00  5.00
16   gb_ws_b16  16      0.6%     0.9%     0.8%     0.750    1.000   6.00  5.00
16   gb_ws_b32  32     46.4%    48.9%    47.7%     0.750    0.500   6.00  5.00
16   gb_ws_b64  64     60.2%    72.9%    66.6%     0.750    0.250   6.00  5.00
```

结论：

```text
1. gb_now 证明 predictor-free early stop 已经减少 activation fetch：
   act/orig = 0.75，fetch=6，issue=5。

2. 但 gb_now 的 cycles/traffic 收益仍小：
   因为 w/orig = 1.0，weight-side traffic 没被摊薄。

3. gb_ws_b32 / gb_ws_b64 说明如果 risk-bucket scheduler 能让 W tile 服务更多同风险 node blocks：
   w/orig = 0.5 / 0.25，
   cycles / traffic / energy proxy 都大幅下降。

4. 因此 Graph-Bit 的完整硬件主张应该是：
   predictor-free bit-plane early stop + risk-bucket scheduler + weight-stationary tile reuse。
```

## 8. 论文中应该如何表述

保守主张：

```text
Graph-Bit predictor-free early stop reduces activation bit-plane fetch and PE issue depth.
```

完整 NPU 设计主张：

```text
Graph-conditioned risk buckets expose larger same-risk micro-batches.
The NPU schedules these buckets with a weight-stationary dataflow, allowing
one W tile to serve more miss nodes before eviction.
```

需要小心：

```text
不要直接说“Graph-Bit 免费省 50% weight HBM”。
应该说：当 scheduler/capacity 支持 W tile 跨 X 个同风险 node blocks 复用时，
ONNXim sensitivity 显示 weight-side bottleneck 可以被显著摊薄。
```
