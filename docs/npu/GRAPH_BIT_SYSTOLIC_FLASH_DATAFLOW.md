# Graph-Bit Systolic + Flash-Style Dataflow

本文档说明 Graph-Bit 在 NPU 内部如何落到脉动阵列和类 FlashAttention 的 IO-aware tiling 上。

核心结论：

```text
Graph-Bit 不是假设整层权重常驻片上。
它假设一个 W tile 可以驻留在片上 SRAM / RF，
并通过 graph-risk bucket 调度，让这个 W tile 在被换出前服务更多 token rows。
```

## 1. 为什么需要这条数据流

LLM encoder 的主要计算是 Linear / FFN GEMM：

```text
Y = X @ W

X: [M, K]
W: [K, N]
Y: [M, N]
```

其中：

```text
M = token rows = node_batch * padded_sequence_length
K = hidden dimension
N = output channels / FFN channels
```

例如 LLaMA-7B：

```text
Q/K/V/O projection:
    X: [M, 4096]
    W: [4096, 4096]

FFN up/gate:
    X: [M, 4096]
    W: [4096, 11008]

FFN down:
    X: [M, 11008]
    W: [11008, 4096]
```

权重矩阵很大，无法整体放到片上：

```text
4096 x 4096 W4      ≈ 8 MB
4096 x 11008 W4     ≈ 22 MB
单层所有 Linear W4 约百 MB 级别
```

但一个小 W tile 可以放进片上 buffer。例如：

```text
128 x 128 W4 tile = 128 * 128 * 4 bit = 8 KB
```

因此硬件目标不是“整层 W 常驻片上”，而是：

```text
每次只让一个 W tile 驻留片上；
在 evict 前让它服务尽量多的 token rows。
```

## 2. 脉动阵列上的 Weight-Stationary GEMM

Graph-Bit 使用的基础 GEMM 数据流是 weight-stationary。

简化图：

```text
            X token rows stream in
                    |
                    v
          +-------------------+
W tile    |  PE  PE  PE  PE   |
stays --> |  PE  PE  PE  PE   | --> partial sums / output tile
          |  PE  PE  PE  PE   |
          +-------------------+
```

执行过程：

```text
1. 从 HBM 读取 W[K_tile, N_tile]。
2. W tile 放入片上 SRAM / weight buffer / PE RF。
3. X[M_tile, K_tile] token rows 流入阵列。
4. PE array 计算 X tile @ W tile。
5. partial sums 在 PE / accumulator / output buffer 中累加。
6. 当前 W tile 服务完足够多 token rows 后再被换出。
```

注意：

```text
W tile 驻留片上 != 整个 W matrix 驻留片上。
```

当前 ONNXim 统计的 `dram_read_requests` 仍然很大，说明仿真没有把 W 全部视作片上常驻。Graph-Bit 改变的是每个 W tile 的 service window。

## 3. 类 FlashAttention 的相似点和差异

Graph-Bit 借鉴的是 FlashAttention 的 IO-aware 思想。

```text
FlashAttention:
    处理 attention。
    Q/K/V tile 进入 SRAM。
    在线 softmax。
    避免完整 attention matrix 写回 HBM。

Graph-Bit:
    处理 encoder GEMM。
    W tile 进入 SRAM/RF。
    同风险 miss nodes 的 token rows 连续消费该 W tile。
    减少同一 W tile 的 HBM reload。
```

共同点：

```text
1. 不让可复用 tile 反复进出 HBM。
2. 用 tile-level dataflow 暴露片上复用。
3. 把执行顺序服务于片上 locality。
```

差异：

```text
FlashAttention 的 locality 来自 sequence 内 attention tiling。
Graph-Bit 的 locality 来自 graph-risk bucket 调度。
```

Graph-Bit 新增的图信息：

```text
SimHash/reuse:
    先把可复用节点从 encoder workload 中移除。

Graph risk:
    对剩余 miss nodes 分桶。

Risk bucket:
    同桶节点使用相似 stop-depth / tolerance。
    同桶 token rows 连续进入同一组 GEMM tile。

W-stationary scheduler:
    W tile 在片上停留更久，服务更多同风险 token rows。
```

## 4. Graph-Bit 的完整数据流

Graph-Bit 不是直接对全图节点跑 encoder，而是先做前端分流：

```text
graph nodes
    |
    |-- exact CAM hit -----------------> direct embedding cache
    |
    |-- fuzzy CAM hit -----------------> residual-corrected reuse
    |
    |-- reject / miss -----------------> Graph-Bit encoder queue
```

只有 miss nodes 进入 NPU encoder。

对 miss nodes：

```text
Step 1. risk tagging
    计算 Degree / TSER 等 graph risk。
    当前主线优先使用 Degree，TSER 用作拓扑感知语义风险消融。

Step 2. risk bucket formation
    high-risk bucket:
        更保守，接近 P8。
    mid-risk bucket:
        中等 min_depth / tolerance。
    low-risk bucket:
        更激进 early stop。

Step 3. W tile scheduling
    对每个 Linear/FFN GEMM:
        load W tile once
        keep W tile stationary
        stream high/mid/low bucket token rows
        evict W tile

Step 4. bit-serial / bounded execution
    对每个 token-row block:
        先执行高位 activation effort
        更新 partial sum
        检查 remaining low-bit bound
        若 bound <= tolerance，停止低位 effort

Step 5. output merge
    Graph-Bit encoder 输出和 direct/residual reuse 输出合并成最终 embedding pool。
```

## 5. 为什么 graph bucket 能帮助 W tile reuse

普通 batch 顺序：

```text
node order: v1, v2, v3, v4, ...
risk:      high, low, mid, high, ...
```

这种顺序会造成两个问题：

```text
1. 同一个 batch 内 risk 混杂，执行深度被高风险节点拉高。
2. W tile service window 较短，后续 batch 可能再次 reload 同一 W tile。
```

Graph-Bit 重排 miss nodes：

```text
high bucket: v1, v4, ...
mid bucket:  v3, ...
low bucket:  v2, ...
```

然后按 bucket 形成更大的 token-row batch：

```text
load W tile once
    stream high-risk token rows
    stream mid-risk token rows
    stream low-risk token rows
evict W tile
```

这不会改变模型权重，也不会让不同 bucket 使用不同 W。所有 bucket 都乘同一个 W tile。区别只是：

```text
同一个 W tile 被加载一次后，服务更多 token rows。
```

## 6. b32 / b64 是什么

`b32` 和 `b64` 不是 bit-width。它们表示 W tile 的 service window / bucket batch size。

```text
baseline b16:
    一个 W tile 服务 16 个 token-row blocks 后换出。

b32:
    一个 W tile 服务 32 个 token-row blocks 后换出。

b64:
    一个 W tile 服务 64 个 token-row blocks 后换出。
```

更大的 service window 可能减少 W tile reload，但也带来约束：

```text
1. 片上 SRAM/RF 需要容纳 W tile、activation tile、psum tile。
2. bucket 内要有足够 token rows，否则 tail padding 增加。
3. 调度等待可能增加，尤其小图或 miss nodes 很少时。
4. batch 太大时可能影响 latency-oriented execution。
```

因此 b32 / b64 是硬件调度参数，需要结合：

```text
SRAM capacity
NoC bandwidth
token-row M
bucket size
latency / throughput 目标
```

一起评估。

## 7. 与普通 Transformer accelerator 的区别

普通 Transformer accelerator 也会做 weight-stationary 或 output-stationary GEMM，不是没有 W tile reuse。

Graph-Bit 的区别不是“首次复用 W tile”，而是：

```text
普通 accelerator:
    按请求 / batch / sequence 顺序执行。
    不知道哪些 graph nodes 可以跳过 encoder。
    不知道哪些 miss nodes 有相似传播风险。

Graph-Bit:
    SimHash/residual 前端先缩小 encoder workload。
    graph risk 决定 miss-node batch order。
    同风险 bucket 暴露更稳定的 W tile service window。
    同桶节点采用相似 bit-serial stop-depth policy。
```

所以 Graph-Bit 的硬件新意在于：

```text
graph backend information controls NPU scheduling and arithmetic effort.
```

## 8. 当前仿真如何对应这条数据流

当前代码中有三类证据：

```text
1. residual / reuse workload profile
    记录 direct / residual / miss 比例。

2. Graph-Bit stop-depth trace
    对 miss nodes 记录 high/mid/low bucket 和 stop-depth。

3. ONNXim component lookup
    对 LLaMA projection / FFN GEMM 统计 cycles、read/write、energy proxy。
```

trace replay 将它们合起来：

```text
真实 miss node trace
    -> risk bucket replay
    -> Wloads / Wscale
    -> ONNXim component cost lookup
    -> full-stack cycles / traffic / energy table
```

真实 token-row 大 M 的 component sweep 使用：

```bash
M_VALUES=2048 bash scripts/run_onnxim_graphbit_tokenrow_components.sh
```

输出目录：

```text
output/onnxim_graphbit/tokenrow_components/
```

已经完成的 M=2048 P8 对照显示：

```text
FullP8:
    cycles   = 269,336,864
    read_req = 3,254,779,904

P8-ws-b32:
    cycles   = 211,486,144
    read_req = 2,445,279,232
```

即在真实 token-row 大 M 下，W-stationary service window 仍能降低约：

```text
cycles:   21.5%
read_req: 24.9%
```

这说明 graph-risk bucket scheduling 在大 M 场景下仍然有价值，但收益不应直接沿用小 M microbench 的比例。

## 9. 当前边界

当前模型已经显式区分：

```text
W tile 驻留片上
整层 W 不驻留片上
```

仍需继续增强的部分：

```text
1. 用 M=2048/4096/8192 重跑 P6/P5/P4 component lookup。
2. 将 token-row-scale component lookup 接入 full-stack trace replay。
3. 做 SRAM capacity / bucket tail padding / scheduling latency sensitivity。
4. 如果要进一步提高硬度，可把 trace replay 下沉到 ONNXim per-tile event trace。
```
