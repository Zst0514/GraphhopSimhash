# Graph-Bit Predictor-Free Variable Bit-Depth Execution

本文档说明 Graph-Bit NPU 中的 predictor-free variable bit-depth execution：在 miss node 必须执行 LLM encoder 时，NPU 按 bit-plane 从高位到低位执行 activation，并用运行时上界判断是否提前停止低位计算。

## 1. Full-Stack 位置

Graph-Bit full-stack 在线路径：

```text
node text + graph
        |
        v
SimHash / CAM retrieval
        |
        +-- direct hit  -> cache reuse
        |
        +-- fuzzy hit   -> residual correction
        |
        +-- miss node   -> LLM encoder on Graph-Bit NPU
```

本文只讨论最后一条路径：

```text
miss node -> LLM encoder compute
```

miss node 进入 encoder 后，Graph-Bit NPU 不固定执行完整 A8 activation，而是：

```text
graph risk -> min_depth + tolerance
runtime bound -> actual stop depth
```

其中 `actual stop depth` 是运行时真正执行到的 activation bit-depth。

## 2. Bit-Plane 是什么

一个 A8 activation 可以写成 8 个 bit：

```text
A8 = b7 b6 b5 b4 b3 b2 b1 b0
```

bit-serial GEMM 按 bit-plane 顺序执行：

```text
先执行高位: b7, b6, ...
再按需执行低位: ..., b2, b1, b0
```

Graph-Bit 的 early stop 作用在 bit-plane 维度：

```text
如果 A6 已经足够，则不再执行 b1,b0。
如果 A5 已经足够，则不再执行 b2,b1,b0。
```

本文中的 bit-plane 指执行粒度：PE 先处理高位贡献，再由 runtime bound 决定是否继续发射低位贡献。

## 3. GEMM 形状

Transformer Linear 层按 GEMM 建模：

```text
X: [M, K]
W: [K, N]
Y: [M, N]
```

其中：

```text
M = batch_nodes * padded_sequence_length
```

LLaMA-7B encoder-style GEMM：

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

TAPE / DyLGNN 对应的 M 量级：

```text
TAPE-like:
    batch_nodes ≈ 9..36
    sequence length ≈ 285..512
    M ≈ 2k..18k

DyLGNN-like:
    sub-batch = 32
    sequence length = 2048
    M = 65536
```

因此 Graph-Bit 的 NPU 评估使用：

```text
M = 2048, 4096, 8192, 16384, 32768, 65536
```

## 4. Risk 到 Min-Depth / Tolerance

Graph risk 不直接决定最终执行 P8/P6/P5/P4。它只给 runtime bound 设置两个参数：

```text
min_depth:
    至少执行到几 bit

tolerance:
    剩余低位贡献上界的可接受阈值
```

当前默认策略：

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

Degree / TSER / Context 可以作为 risk source。当前主线优先使用 Degree。

## 5. Predictor-Free Bound

执行到 depth 后，未执行低位 bit-plane 的最大贡献可以用上界估计：

```text
remaining_low_bit_bound(depth)
  = bound_scale
    * remaining_activation_low_bits(depth)
    * tile_scale
```

当前代码中的 proxy：

```python
omitted = (2 ** (ref_bit - depth)) - 1
denom = (2 ** ref_bit) - 1
tile_scale = sqrt(tile_k / 128)

bound = bound_scale * (omitted / denom) * tile_scale
```

对应代码：

```text
runner.py:
    precision_depth_remaining_bit_bound
    select_runtime_bound_depth
```

该 bound 是 predictor-free 的：

```text
不训练 predictor
不读取 FP reference embedding
不读取全图 quantization error
只使用剩余低位 bit-plane 的理论最大贡献
```

停止条件：

```text
if remaining_low_bit_bound(depth) <= tolerance:
    stop at depth
else:
    continue lower bit-plane
```

## 6. Stop Depth 示例

假设：

```text
ref_bit = 8
tile_k = 128
bound_scale = 1.0
```

各 depth 的低位剩余上界：

| Depth | 已执行 bit | 剩余 bit | Bound |
|---:|---|---|---:|
| A8 | b7..b0 | none | 0.0000 |
| A7 | b7..b1 | b0 | 1/255 = 0.0039 |
| A6 | b7..b2 | b1..b0 | 3/255 = 0.0118 |
| A5 | b7..b3 | b2..b0 | 7/255 = 0.0275 |
| A4 | b7..b4 | b3..b0 | 15/255 = 0.0588 |

### 6.1 High-Risk

```text
min_depth = 8
tolerance = 0.00
```

执行：

```text
start at A8
bound(A8) = 0.0000 <= 0.00
stop at A8
```

结果：

```text
actual_depth = A8
```

### 6.2 Mid-Risk

```text
min_depth = 6
tolerance = 0.02
```

执行：

```text
start at A6
bound(A6) = 0.0118 <= 0.02
stop at A6
```

结果：

```text
actual_depth = A6
```

### 6.3 Low-Risk

```text
min_depth = 4
tolerance = 0.04
```

执行：

```text
start at A4
bound(A4) = 0.0588 > 0.04
continue

depth = A5
bound(A5) = 0.0275 <= 0.04
stop at A5
```

结果：

```text
actual_depth = A5
```

这也是当前很多 low-risk 节点最终落到 A5 的原因。

## 7. 硬件执行流程

NPU 执行一个 risk bucket 时：

```text
1. load / keep W tile in weight-stationary buffer

2. fetch activation high bit-plane

3. issue bit-plane GEMM:
       partial_sum += A_bitplane * W4

4. reach min_depth

5. bound estimator computes remaining_low_bit_bound

6. compare bound <= tolerance

7. if true:
       stop issuing lower bit-planes
       gate A_RF / W_RF / PE / psum update
   else:
       continue next lower bit-plane
```

Datapath modules：

```text
Risk-bucket scheduler:
    group miss nodes by risk bucket

Weight-stationary tile buffer:
    reuse W tile across token rows in the same bucket

Bit-plane issue controller:
    controls b7 -> b0 execution order

Bound estimator:
    computes remaining low-bit upper bound

Comparator:
    bound <= tolerance

Gating logic:
    disables low-bit PE issue, RF access, W broadcast, psum update
```

## 8. Bound Estimator 流水重叠

Bound estimator 可以串行执行，也可以和 bit-plane GEMM 重叠。

### 8.1 不重叠

```text
execute A6
pause PE array
compute bound
compare tolerance
resume / stop
```

时间模型：

```text
T_compute = T_bitserial + T_bound
```

### 8.2 重叠

```text
while PE executes high bit-planes:
    bound estimator prepares remaining-bit bound terms

when min_depth finishes:
    comparator checks bound <= tolerance
```

时间模型：

```text
T_compute = max(T_bitserial, T_bound)
```

当前 roofline 脚本用 `bound_overlap` 表示可重叠比例：

```text
bound_overlap = 0.5:
    50% bound cycles can be hidden behind bit-plane GEMM

bound_overlap = 0:
    bound cycles are fully visible
```

## 9. 建模项

完整 compute path：

```text
T_compute =
    T_bitserial(depth)
  + T_bound_estimator_visible
  + T_bound_control

T_total = max(T_compute, T_memory)
```

脚本显式统计：

```text
bit_compute_cycles:
    bit-serial GEMM cycles after depth reduction

bound_cycles:
    visible bound estimator + control cycles

cycles_memory:
    W/A/output HBM bytes / bandwidth

PE / RF / psum activity:
    on-chip activity proxy
```

Bound overhead 参数：

```text
bound_ops_per_output:
    每个 output element 每次 bound check 的估计操作数

bound_tops:
    bound estimator 的有效吞吐

bound_overlap:
    bound estimator 与 bit-plane GEMM 的重叠比例

bound_control_cycles_per_check:
    每次 tile-level stop check 的控制开销

m_tile, n_tile:
    tile-level check 粒度
```

## 10. Accuracy Validation 与硬件模型的关系

当前有两条验证线：

### 10.1 Accuracy Validation

使用已生成的 embedding pools：

```text
P8 = W4A8
P6 = W4A6
P5 = W4A5
P4 = W4A4
```

runtime depth 会映射到最近的不低于该 depth 的 embedding pool，用于估计 GNN 分类精度。

例如：

```text
actual_depth = A5 -> use W4A5 embedding pool
```

### 10.2 Hardware Model

硬件模型不重新生成 embedding，而是统计：

```text
bit-depth
bound overhead
cycles
traffic
PE/RF/psum activity
```

该模型用于评估 NPU 内部执行成本。

## 11. 当前结果

默认 bound overhead：

```text
output/graphbit_internal_roofline/tape_dylgnn_bound_overhead/
```

配置：

```text
bound_ops_per_output = 8
bound_tops = 16 TOPS
bound_overlap = 0.5
bound_control_cycles_per_check = 4
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

配置：

```text
bound_ops_per_output = 16
bound_tops = 4 TOPS
bound_overlap = 0
bound_control_cycles_per_check = 8
```

Layer-level 结果：

| Stop Depth | Net Cycle Save |
|---|---:|
| A6 | about 12.6% |
| A5 | about 12.8% |
| A4 | about 37.6% |

## 12. 复现命令

默认 bound overhead：

```bash
cd /home/zhangshangtong/Transformer/OFA

python GraphhopSimhash/scripts/model_graphbit_internal_roofline.py \
  --batch-nodes 4 8 16 32 \
  --seq-lens 512 2048 \
  --output-dir output/graphbit_internal_roofline/tape_dylgnn_bound_overhead
```

悲观 bound overhead：

```bash
python GraphhopSimhash/scripts/model_graphbit_internal_roofline.py \
  --batch-nodes 4 8 16 32 \
  --seq-lens 512 2048 \
  --bound-ops-per-output 16 \
  --bound-tops 4 \
  --bound-overlap 0 \
  --bound-control-cycles-per-check 8 \
  --output-dir output/graphbit_internal_roofline/tape_dylgnn_bound_pessimistic
```

## 13. 代码入口

Accuracy-side runtime bound：

```text
runner.py
    precision_depth_remaining_bit_bound
    select_runtime_bound_depth
    bound_policy_bucket_bits
```

NPU-side roofline/activity model：

```text
scripts/model_graphbit_internal_roofline.py
```

主要参数：

```text
--batch-nodes
--seq-lens
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

## 14. 后续统计项

下一步需要把 NPU-internal model 和真实 workload trace 连接：

```text
1. 读取 Cora/PubMed/Arxiv miss-node trace
2. 统计真实 stop-depth histogram
3. 按真实 histogram 加权 P8/P6/P5/P4 cycles
4. 加入 risk-bucket scheduler 的 W tile reuse 统计
5. 输出 full-stack cycles / traffic / energy / drop
```
