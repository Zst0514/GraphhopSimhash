# Graph-Bit Predictor-Free Variable Bit-Depth Execution

本文档记录 Graph-Bit NPU 中 predictor-free variable bit-depth execution 的定义、输入输出、执行流程、建模项、当前结果和复现实验命令。

## 1. 范围

Graph-Bit full-stack 的在线路径分为三类：

```text
direct hit      -> cache reuse
fuzzy hit       -> residual correction
miss node       -> LLM encoder compute
```

本文只讨论第三类：`miss node -> LLM encoder compute`。

目标是在 miss node 必须执行 Transformer encoder 的情况下，让 NPU 内部根据图风险和 runtime bound 决定 activation bit-plane 的执行深度。

## 2. 输入与输出

### 2.1 输入

对每个 miss node，需要以下信息：

```text
node_id
degree / graph risk score
text sequence length or padded sequence length
Transformer layer GEMM shape
```

在 NPU 内部，每个 Linear 层按 GEMM 建模：

```text
X: [M, K]
W: [K, N]
Y: [M, N]

M = batch_nodes * padded_sequence_length
```

当前 LLaMA-7B encoder-style GEMM 配置：

```text
hidden = 4096
intermediate = 11008

Q/K/V/O projection:
    [M, 4096] x [4096, 4096], count = 4

FFN gate/up:
    [M, 4096] x [4096, 11008], count = 2

FFN down:
    [M, 11008] x [11008, 4096], count = 1
```

### 2.2 输出

模型输出以下指标：

```text
cycles
cycles_compute
cycles_memory
bit_compute_cycles
bound_cycles
HBM bytes
PE / RF / psum activity
cycle saving vs P8
activity saving vs P8
```

结果文件：

```text
graphbit_internal_roofline.txt
graphbit_internal_roofline.tsv
graphbit_internal_roofline.json
```

## 3. Risk 到 Runtime Policy

Graph risk 不直接指定最终 P8/P6/P5/P4，而是指定：

```text
min_depth
tolerance
```

当前默认 policy：

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

运行时流程：

```text
for depth in min_depth..8:
    compute remaining_low_bit_bound(depth)
    if remaining_low_bit_bound(depth) <= tolerance:
        stop at depth
        break
```

最终执行深度由 runtime bound 判断得到。

## 4. Predictor-Free Bound

A8 activation 拆成 8 个 bit-plane：

```text
A8 = b7 b6 b5 b4 b3 b2 b1 b0
```

执行到某个 depth 后，剩余低位 bit-plane 的最大贡献用上界估计：

```text
remaining_low_bit_bound(depth)
  ~= remaining_activation_low_bits(depth)
    * weight_abs_bound
    * tile_scale
    * quant_scale
```

停止条件：

```text
remaining_low_bit_bound(depth) <= tolerance
```

该判断不使用 learned predictor，不需要 FP reference embedding，也不读取全图 quantization error。

## 5. NPU Datapath

Graph-Bit predictor-free execution 涉及以下 NPU 组件：

```text
1. Risk-bucket scheduler
   将 miss nodes 按风险桶组织执行。

2. Weight-stationary tile buffer
   同一个 W tile 服务同桶中的多行 token rows。

3. Bit-plane issue controller
   从高位到低位发射 activation bit-plane。

4. Bound estimator
   计算 remaining_low_bit_bound 并比较 tolerance。

5. RF / broadcast / psum gating
   stop 后不再执行低位 bit-plane 对应的片上访问和更新。
```

当前主线保守采用 `byte_major` activation：

```text
activation HBM read 仍按 A8 byte 读取
early stop 主要减少 PE issue、RF access、W_RF broadcast、psum update
```

`plane_group` activation 是增强路径：

```text
activation 按 bit-plane group 组织
low-bit group 可 demand fetch
可进一步减少 activation fetch
```

## 6. Bound Overhead 建模

完整 compute path：

```text
T_compute =
    T_bitserial(depth)
  + T_bound_estimator_visible
  + T_bound_control

T_total = max(T_compute, T_memory)
```

脚本中显式建模以下开销：

```text
bound_ops:
    remaining-bit bound estimation and compare

bound_control_cycles:
    per-tile stop-check control and issue decision

bound_overlap:
    bound estimator 与 bit-plane GEMM 的重叠比例
```

默认参数：

```text
bound_ops_per_output = 8
bound_tops = 16 TOPS
bound_overlap = 0.5
bound_control_cycles_per_check = 4
m_tile = 128
n_tile = 128
```

悲观参数：

```text
bound_ops_per_output = 16
bound_tops = 4 TOPS
bound_overlap = 0
bound_control_cycles_per_check = 8
```

## 7. M 的取值

Transformer Linear 的 GEMM row 数：

```text
M = batch_nodes * padded_sequence_length
```

TAPE-like 设置：

```text
batch_nodes ≈ 9..36
sequence length ≈ 285..512
M ≈ 2k..18k
```

DyLGNN-like 设置：

```text
sub-batch = 32 nodes
sequence length = 2048
M = 65536
```

当前主表使用：

```text
M = 2048, 4096, 8192, 16384, 32768, 65536
```

这些 M 覆盖了常见 LLM+GNN encoder 前端 batch 规模。

## 8. 当前结果

默认 bound overhead：

```text
output/graphbit_internal_roofline/tape_dylgnn_bound_overhead/
```

Layer-level 结果：

| M | P6 Net Save | P5 Net Save | P4 Net Save | Bound |
|---:|---:|---:|---:|---|
| 2048 | 23.6% | 34.8% | 48.6% | compute |
| 4096 | 23.6% | 34.8% | 48.6% | compute |
| 8192 | 23.6% | 34.8% | 48.6% | compute |
| 16384 | 23.6% | 34.8% | 48.6% | compute |
| 32768 | 23.6% | 34.8% | 48.6% | compute |
| 65536 | 23.6% | 34.8% | 48.6% | compute |

Per-stage P6 net saving：

| Stage | Cycle Share | P6 Net Save |
|---|---:|---:|
| Q/K/V/O projection | about 33% | about 23.4% |
| FFN gate/up | about 45% | about 23.4% |
| FFN down | about 22% | about 24.4% |

悲观 bound overhead：

```text
output/graphbit_internal_roofline/tape_dylgnn_bound_pessimistic/
```

Layer-level 结果：

| Stop Depth | Net Cycle Save |
|---|---:|
| A6 | about 12.6% |
| A5 | about 12.8% |
| A4 | about 37.6% |

## 9. 复现命令

默认 bound overhead：

```bash
cd /home/zhangshangtong/Transformer/OFA

python GraphhopSimhash/scripts/model_graphbit_internal_roofline.py \
  --batch-nodes 4 8 16 32 \
  --seq-lens 512 2048 \
  --activation-hbm-mode byte_major \
  --output-dir output/graphbit_internal_roofline/tape_dylgnn_bound_overhead
```

悲观 bound overhead：

```bash
python GraphhopSimhash/scripts/model_graphbit_internal_roofline.py \
  --batch-nodes 4 8 16 32 \
  --seq-lens 512 2048 \
  --activation-hbm-mode byte_major \
  --bound-ops-per-output 16 \
  --bound-tops 4 \
  --bound-overlap 0 \
  --bound-control-cycles-per-check 8 \
  --output-dir output/graphbit_internal_roofline/tape_dylgnn_bound_pessimistic
```

## 10. 代码入口

主脚本：

```text
GraphhopSimhash/scripts/model_graphbit_internal_roofline.py
```

主要参数：

```text
--batch-nodes
--seq-lens
--activation-hbm-mode {byte_major, plane_group}
--peak-tops
--mem-gbps
--bound-enable / --no-bound-enable
--bound-ops-per-output
--bound-tops
--bound-overlap
--bound-control-cycles-per-check
--m-tile
--n-tile
```

相关文档：

```text
docs/npu/GRAPH_BIT_INTERNAL_ROOFLINE.md
docs/npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
docs/npu/GRAPH_BIT_DEGREE_BOUND_POLICY.md
```

## 11. 后续统计项

下一步需要把该 NPU-internal model 和真实 workload trace 连接：

```text
1. 读取 Cora/PubMed/Arxiv miss-node trace
2. 统计真实 stop-depth histogram
3. 按真实 histogram 加权 P8/P6/P5/P4 cycles
4. 加入 risk-bucket scheduler 的 W tile reuse 统计
5. 输出 full-stack cycles / traffic / energy / drop
```
