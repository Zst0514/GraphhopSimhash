# Graph-Aware BFP Block Packing

本文档记录当前 BFP 相关实验已经验证的内容，以及后续需要逐步补齐的实验路径。这里的主线不是提出新的 BFP 数值格式，而是研究在 text-attributed graph encoder 场景下，如何利用图前端信息组织 BFP exponent-sharing block。

## 1. 背景

BFP activation 的核心是：

```text
一组 activation 共享一个 exponent
每个 activation 只保留 mantissa
```

这种格式能降低 activation 精度和缩放开销，但它有一个天然风险：如果同一个 block 内有 outlier，大值会拉高共享 exponent，导致其他小值的 mantissa 精度浪费。

普通 Transformer accelerator 通常按输入顺序、token 顺序或 GEMM tile 顺序形成 BFP block。Graph encoder workload 多了额外信息：

```text
SimHash bucket
graph context
degree / propagation risk
reuse / miss route
```

这些信号可以用于组织进入同一个 BFP block 的 node/token rows，使共享 exponent 的 block 内部动态范围更一致。

## 2. 两种 BFP Block 方式

把 Linear 输入 activation 看成：

```text
X shape = [M, K]
M = token rows = node_batch * seq_len
K = hidden dim
```

### 2.1 Rowwise BFP

当前已有 `W4BFPA*_B128` pool 使用的是 rowwise block：

```text
block = 1 row * 128 hidden dims
```

特点：

```text
每个 token row 内部共享 exponent
节点顺序不影响 block
实现简单，和普通 hidden-dim block quantization 兼容
```

### 2.2 Cross-Row Tile BFP

cross-row tile BFP 让一个 exponent 覆盖多个 token/node rows 和少量 hidden dims，例如：

```text
tile_16x8:
    block = 16 rows * 8 hidden dims = 128 values

tile_32x4:
    block = 32 rows * 4 hidden dims = 128 values
```

特点：

```text
节点/token row 顺序会影响 block 组成
图信息可以用于 row grouping
更接近 NPU tile 内部的 row-block execution
```

## 3. 已完成实验

### 3.1 Shared Exponent Ordering Sanity Check

脚本：

```text
GraphhopSimhash/scripts/bfp_shared_exponent_order_validation.py
```

输出：

```text
output/graphbfp_shared_exponent/cora_a4/summary.txt
output/graphbfp_shared_exponent/cora_a3/summary.txt
output/graphbfp_shared_exponent/cora_a3_extreme/summary.txt
```

结论：

```text
rowwise_1x128:
    Original / Random / Activation-norm / SimHash / Graph-context 完全一致。
    说明当前已有 BFP pool 的 layout 下，节点排序不会改变 BFP 误差。

cross-row tile:
    Random 通常更差。
    SimHash / Graph-context 在若干设置下略好。
    说明只有当 exponent-sharing block 横跨多个 rows 时，graph-aware grouping 才有作用空间。
```

### 3.2 Embedding-Level Block Layout Accuracy

脚本：

```text
GraphhopSimhash/scripts/evaluate_bfp_block_layout_accuracy.py
```

输出：

```text
output/graphbfp_block_layout_accuracy/cora_a4_runs3/summary.txt
output/graphbfp_block_layout_accuracy/cora_a3_runs3/summary.txt
```

这一步从已有 `cora_llama2_7b_oracle_W4A8.pt` 出发，在最终 embedding 矩阵上模拟不同 BFP block layout，然后喂给 GNN 后端评估。

Cora / A4 / 3 runs：

```text
Baseline Acc: 0.7289

rowwise_1x128:
    Drop = 0.02%

tile_16x8:
    Drop = 0.08% ~ 0.18%

tile_32x4:
    Drop = -0.08% ~ 0.24%

tile_128x1:
    Drop = -0.06% ~ 0.21%
```

Cora / A3 / 3 runs：

```text
rowwise_1x128:
    Drop = 1.21%

tile_16x8:
    Drop = 0.06% ~ 0.31%

tile_32x4:
    Drop = 0.00% ~ 0.19%

tile_128x1:
    Drop = -0.03% ~ 0.05%
```

解释：

```text
A4 下两类 block 都安全。
A3 下 cross-row tile 在 embedding-level proxy 中比 rowwise 更稳。
这说明 cross-row BFP block 不一定更危险，值得继续做 activation-level 验证。
```

### 3.3 Real LLaMA Activation Hook Validation

脚本：

```text
GraphhopSimhash/scripts/bfp_activation_order_validation.py
```

输出：

```text
output/graphbfp_activation_order/cora_a4_b4/summary.txt
output/graphbfp_activation_order/cora_a3_b4/summary.txt
```

设置：

```text
dataset = cora
sample_nodes = 32
batch_size = 4
max_length = 128
layers = 0 / 15 / 31
modules = q_proj / o_proj / up_proj / down_proj
```

Cora / A4 / real activation：

```text
rowwise_1x128:
    rel_err ~= 0.243
    cos ~= 0.9688

tile_16x8:
    rel_err ~= 0.197 ~ 0.200
    cos ~= 0.978

tile_32x4:
    rel_err ~= 0.190 ~ 0.196
    cos ~= 0.978 ~ 0.979
```

Cora / A3 / real activation：

```text
rowwise_1x128:
    rel_err ~= 0.441
    cos ~= 0.9025

tile_16x8:
    rel_err ~= 0.387 ~ 0.391
    cos ~= 0.921 ~ 0.922

tile_32x4:
    rel_err ~= 0.384 ~ 0.392
    cos ~= 0.920 ~ 0.923
```

结论：

```text
真实 LLaMA activation 上，cross-row tile 的 BFP error 明显低于 rowwise。
目前 Original / Random / SimHash / Graph-context 的排序差异还不稳定。
这说明 cross-row tile 本身已经有价值；graph-aware ordering 需要更真实的 scheduler 和更大 batch 继续验证。
```

### 3.4 Graph-Aware BFP Lift

输出：

```text
output/graphbfp_lift/cora/*.log
```

该实验以 BFPA4 为默认低成本路径，用 graph score 选择一部分节点 lift 到 BFPA6 或 BFPA8。

Cora / 5 runs 代表结果：

```text
All BFPA4:
    Cost = 0.287
    Drop ~= 0.98% ~ 1.00%

All BFPA6:
    Cost = 0.394
    Drop ~= 0.08% ~ 0.09%

P6 lift 30%:
    Cost = 0.319
    Random Drop = 0.72%
    Degree Drop = 0.66%
    TSER Drop = 0.54%
    Context Drop = 0.48%

P8 lift 30%:
    Cost = 0.351
    Random Drop = 0.74%
    Degree Drop = 0.59%
    TSER Drop = 0.44%
    Context Drop = 0.57%
```

解释：

```text
BFPA4 本身已经相当强。
Graph score 的作用不是让 BFPA4 从不可用变可用，
而是在低成本 BFPA4 baseline 上保护更敏感节点。
```

## 4. 当前结论

当前证据可以支持以下几点：

```text
1. BFP shared exponent 问题真实存在。
2. 当前 rowwise BFP pool 下，节点排序不会起作用。
3. cross-row tile BFP 给 graph-aware grouping 留出了作用空间。
4. Cora 上 cross-row tile 的 embedding-level 和 real-activation proxy 都没有表现出明显精度风险。
5. graph-aware ordering 的收益还没有完全坐实，需要更真实的 scheduler / batch 组织验证。
```

因此当前最稳的表述是：

```text
BFP format 本身不是新点。
新点应放在 graph-aware BFP block packing:
    利用 SimHash / graph context / risk 信息组织 exponent-sharing group。
```

## 5. 下一步验证路径

### Step 1: 更真实的 Activation-Level Scheduler

当前 hook 已比较：

```text
sequence_major
token_position_major
```

下一步需要把 scheduler 做得更接近硬件：

```text
按 SimHash / graph-context bucket 形成 node batch
同一 batch 内按 token position 重排 token rows
再形成 tile_16x8 / tile_32x4 BFP block
```

目标：

```text
验证 graph-aware order 是否能稳定优于 random / original。
```

### Step 2: Cora Full Activation-Level Pool

在 hook 结果稳定后，生成 Cora 的真实 encoder pool：

```text
rowwise BFPA4
cross-row tile16x8 BFPA4
cross-row tile32x4 BFPA4
cross-row tile16x8 BFPA3
cross-row tile32x4 BFPA3
```

然后跑 3~5 runs GNN accuracy：

```text
Acc / Drop / AvgErr
```

目标：

```text
把 proxy 结论升级为真实 encoder pool 结论。
```

### Step 3: PubMed 验证

Cora 成立后，扩展 PubMed：

```text
BFPA4 rowwise vs cross-row
graph-context / SimHash grouping
3 runs accuracy
```

PubMed 文本更长、节点更多，更适合验证 scheduler grouping 是否稳定。

### Step 4: Hardware Cost Model

如果 cross-row graph-aware packing 在精度上成立，需要补硬件收益：

```text
exponent metadata 数量
shift / scale control
tile buffer 组织
row reorder / bucket scheduler overhead
有效 activation mantissa bit
```

需要比较：

```text
rowwise block:
    每 row 每 128 hidden dims 一个 exponent

cross-row block:
    每 16x8 或 32x4 tile 一个 exponent
```

目标：

```text
证明 graph-aware packing 不只是精度技巧，
也能形成合理的 NPU dataflow / metadata / tile scheduling 设计。
```

## 6. 复现命令

Shared exponent ordering：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/bfp_shared_exponent_order_validation.py \
  --dataset cora \
  --mantissa_bits 4 \
  --layouts rowwise_1x128 tile_16x8 tile_32x4 \
  --output_dir output/graphbfp_shared_exponent/cora_a4
```

Embedding-level downstream accuracy：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/evaluate_bfp_block_layout_accuracy.py \
  --dataset cora \
  --mantissa_bits 4 \
  --runs 3 \
  --layouts rowwise_1x128 tile_16x8 tile_32x4 tile_128x1 \
  --output_dir output/graphbfp_block_layout_accuracy/cora_a4_runs3
```

Real activation hook：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/bfp_activation_order_validation.py \
  --dataset cora \
  --mantissa_bits 4 \
  --sample_nodes 32 \
  --batch_size 4 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj o_proj up_proj down_proj \
  --layouts rowwise_1x128 tile_16x8 tile_32x4 \
  --output_dir output/graphbfp_activation_order/cora_a4_b4
```
