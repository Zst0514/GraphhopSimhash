# Graph-Bit Tile-Aware Score V2 Validation

本文档记录 `node risk + W tile risk + low-bit budget` 这一版 Graph-Bit stop policy 的 Cora 快速验证。

## 1. Goal

上一版 hard validation 发现：

```text
omitted_low_bits / 255
```

不能作为可靠 stop bound，因为真正影响 GEMM tile 输出的是：

```text
A_low @ W_tile
```

因此这一版改为 tile-aware score：

```text
node_norm(v) = degree_q(v) / 15

w_norm(tile) =
    clamp(W_tile_strength / reference_strength, 0, w_cap)

low_norm(depth) =
    (2^(8 - depth) - 1) / 255

risk_score(v, tile, depth) =
    node_norm(v)^alpha
    * w_norm(tile)^beta
    * low_norm(depth)
```

stop rule：

```text
choose the lowest depth where risk_score <= tau
```

当前验证使用：

```text
alpha = 1
beta = 1
w_cap = 2
reference_strength = p90(W_tile_strength)
```

## 2. Command

基础命令如下。不同 tile size 只改 `--tile_k / --tile_n / --output_dir`。

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python GraphhopSimhash/scripts/tile_bound_numeric_validation.py \
  --dataset cora \
  --sample_nodes 8 \
  --batch_size 2 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --tile_k 128 \
  --tile_n 128 \
  --tiles_per_module 2 \
  --rows_per_module 32 \
  --score_taus 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 \
  --output_dir output/graphbit_tile_score_v2/cora_layers0_15_31_n8_k128n128
```

结果文件：

```text
output/graphbit_tile_score_v2/cora_layers0_15_31_n8_k64n64/tile_score_v2_summary.tsv
output/graphbit_tile_score_v2/cora_layers0_15_31_n8_k128n128/tile_score_v2_summary.tsv
output/graphbit_tile_score_v2/cora_layers0_15_31_n8_k256n128/tile_score_v2_summary.tsv
```

## 3. Main Result: 128x128 Tile

| Tau | Runtime AvgDepth | Oracle AvgDepth | Aggressive | Actual@Stop P90 | Depth Hist |
|---:|---:|---:|---:|---:|---|
| 0.0001 | 8.00 | 7.91 | 0.0% | 0.000 | P8 100.0% |
| 0.0002 | 8.00 | 7.91 | 0.0% | 0.000 | P8 100.0% |
| 0.0005 | 7.99 | 7.91 | 1.3% | 0.000 | P7 1.4%, P8 98.6% |
| 0.0010 | 7.32 | 7.91 | 59.8% | 0.242 | P7 67.7%, P8 32.3% |
| 0.0020 | 6.81 | 7.91 | 92.6% | 0.443 | P6 20.7%, P7 77.5%, P8 1.7% |
| 0.0050 | 5.77 | 7.91 | 99.8% | 1.130 | P5 27.9%, P6 67.3%, P7 4.7% |
| 0.0100 | 4.94 | 7.91 | 100.0% | 2.181 | P4 20.7%, P5 64.6%, P6 14.7% |
| 0.0200 | 4.20 | 7.91 | 100.0% | 3.519 | P4 80.0%, P5 20.0% |

结论：

```text
1. tau 很小时，策略安全，但几乎全 P8，没有收益。
2. tau 放到 0.001 后，AvgDepth 明显下降，但相对 hard oracle 已经明显偏激进。
3. 这说明 v2 score 能单调控制 depth，但还不能作为严格 tile-level stop bound。
```

## 4. Tile Size Sensitivity

同样设置下，额外验证了 `64x64` 和 `256x128`。

### 4.1 64x64

| Tau | Runtime AvgDepth | Oracle AvgDepth | Aggressive | Actual@Stop P90 | Depth Hist |
|---:|---:|---:|---:|---:|---|
| 0.0005 | 7.98 | 7.91 | 2.1% | 0.000 | P7 2.4%, P8 97.6% |
| 0.0010 | 7.33 | 7.91 | 60.3% | 0.265 | P7 67.5%, P8 32.5% |
| 0.0020 | 6.79 | 7.91 | 92.2% | 0.531 | P6 23.0%, P7 75.0%, P8 2.1% |
| 0.0050 | 5.74 | 7.91 | 99.6% | 1.365 | P5 31.2%, P6 63.6%, P7 5.2% |

`64x64` 下 actual ratio 会更容易出现极端值，因为单个 tile 的 `||A8 @ W_tile||` 更容易接近 0，分母 cancellation 会放大 `actual_delta_ratio`。

### 4.2 256x128

| Tau | Runtime AvgDepth | Oracle AvgDepth | Aggressive | Actual@Stop P90 | Depth Hist |
|---:|---:|---:|---:|---:|---|
| 0.0005 | 7.98 | 7.89 | 2.1% | 0.000 | P7 2.4%, P8 97.6% |
| 0.0010 | 7.31 | 7.89 | 59.9% | 0.203 | P7 69.2%, P8 30.8% |
| 0.0020 | 6.77 | 7.89 | 91.5% | 0.388 | P6 25.0%, P7 73.0%, P8 2.1% |
| 0.0050 | 5.64 | 7.89 | 99.6% | 1.101 | P5 41.4%, P6 53.6%, P7 5.0% |

`256x128` 趋势和 `128x128` 一致：更大的 tile 会稍微降低 actual ratio 的极端波动，但仍然没有形成严格安全的低 AvgDepth 区间。

## 5. Interpretation

当前 v2 scoring 的作用是：

```text
把 node risk、W tile strength、low-bit budget 放到同一个 stop score 中，
并且能随 tau 单调调节 AvgDepth。
```

但在当前 hard oracle 下，它还不能满足：

```text
AvgDepth 显著低于 P8
同时 aggressive rate 很低
```

这说明：

```text
1. 作为严格 tile-level predictor-free bound，当前 v2 score 仍然不够。
2. 如果坚持严格局部数值安全，策略会退回接近 P8。
3. 后续需要重新定义更贴近任务目标的验证标准，或者引入更强的 tile-level normalization。
```

## 6. Next Design Choices

后续有两条路线。

### Route A: Strict Numeric Bound

继续追求 tile-level conservative stop：

```text
actual_delta_ratio <= tolerance
```

需要加入：

```text
denominator floor
module-wise normalization
tile output norm smoothing
outlier-tile force P8
```

优点是数值解释更强；缺点是可能长期接近 P8，收益有限。

### Route B: Task-Budgeted Statistical Stop

把 stop score 看成任务误差预算，而不是每个 tile 的严格上界：

```text
node risk + W tile risk + low-bit budget
    -> statistical precision schedule
    -> embedding / GNN accuracy validation
```

这条路线更接近当前 Graph-Bit 的实际目标：

```text
不是保证每个 tile 的局部 delta 都小，
而是在图任务误差预算内减少 encoder GEMM effort。
```

当前结果显示，Route B 更有希望；Route A 需要很强的 normalization 才可能有足够收益。

## 7. Runtime Policy Integration

当前代码已经把 `tile_score_v2` 接入 `residual_precision_depth` 的 miss-node Graph-Bit 路径。

开启方式：

```bash
--precision_depth_bound_enable
--precision_depth_bound_assignment nodewise
--precision_depth_bound_rule tile_score
--precision_depth_bound_priorities degree
--precision_depth_score_tau 0.001
--precision_depth_score_alpha 1.0
--precision_depth_score_beta 1.0
--precision_depth_score_w_cap 2.0
--precision_depth_score_w_reference 1.0
--precision_depth_bound_w_strength 1.0
```

运行时每个 miss node 的 depth 由下面的分数决定：

```text
score(v, tile, depth)
    = node_risk(v)^alpha
    * W_strength(tile)^beta
    * low_bit_budget(depth)

stop if score <= tau
```

其中当前主路径先用常数 `W_strength` 做可控 sweep；真实 per-tile W metadata 由
`tile_bound_numeric_validation.py` 侧验证。这样可以把实验拆成两层：

```text
tile-level validation:
    检查 score / tau 和真实 A_low @ W_tile 的关系。

residual_precision_depth:
    用同一套 score / tau 做 Cora/PubMed 的 accuracy-cost 验证。
```

快速 sweep 命令：

```bash
cd /home/zhangshangtong/Transformer/OFA/GraphhopSimhash

RUNS=1 \
TAUS="0.0005 0.001 0.002 0.005" \
DATASET=cora \
scripts/run_graphbit_tile_score_tau_sweep.sh
```

核心输出：

```text
output/graphbit_tile_score_tau_sweep/cora/tau_sweep_summary.tsv
```
