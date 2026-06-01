# Graph-Bit Tile-Level Bound Numeric Validation

本文档记录 Graph-Bit predictor-free early stop 的 tile-level 数值验证。目标是回答一个核心问题：

```text
如果 runtime bound 判断某个 GEMM tile 可以停在 P5/P6，
那么被跳过的 activation low bits 与真实 W tile 相乘后，
实际输出增量到底有多大？
```

这个验证不是重新跑完整 bit-serial LLaMA encoder，而是在真实 LLaMA activation 和真实 Linear weight tile 上采样，直接计算被省略低位的数值贡献。

## 1. Validation Setup

本次先在 Cora / LLaMA-7B 上做快速硬验证。

```text
Dataset: Cora
Model: LLaMA-2-7B fp16 weights
Sample nodes: 8
Max length: 128
Layers: 0 / 15 / 31
Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Tile shape: K=128, N=128
Tiles per module: 2
Token rows per module: 32
Total samples per depth: 5376
```

运行命令：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python GraphhopSimhash/scripts/tile_bound_numeric_validation.py \
  --dataset cora \
  --sample_nodes 8 \
  --batch_size 2 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --tiles_per_module 2 \
  --rows_per_module 32 \
  --output_dir output/graphbit_tile_bound_numeric/cora_layers0_15_31_n8_exact
```

输出位置：

```text
output/graphbit_tile_bound_numeric/cora_layers0_15_31_n8_exact/global_depth_summary.tsv
output/graphbit_tile_bound_numeric/cora_layers0_15_31_n8_exact/global_decision_summary.tsv
```

## 2. What Is Measured

对每个采样 activation tile，先做 per-row A8 affine quantization：

```text
A8 = high_bits(depth) + low_bits(depth)
```

对每个 depth，显式计算低位真实贡献：

```text
delta(depth) = A_low(depth) @ W_tile.T

actual_delta_ratio =
    ||delta(depth)|| / ||A8 @ W_tile.T||
```

这里的 `actual_delta_ratio` 直接衡量：

```text
如果停在 P5/P6，被跳过的低位 bit-plane 对当前 tile 输出有多大影响。
```

然后比较不同 predictor-free bound：

```text
range:
    只看 activation omitted range。

tile_p95:
    range bound 乘以 W tile p95 强度。

ratio_mean / ratio_max:
    用 W tile mean/max 估计低位贡献比例。

exact_l1:
    用真实 scale_row * sum_abs(W_tile) 做三角不等式上界。
```

## 3. Low-Bit Contribution

全局统计如下：

| Bound | Depth | Actual Mean | Actual P90 | Bound Mean | Coverage |
|---|---:|---:|---:|---:|---:|
| range | P7 | 0.147 | 0.227 | 0.0039 | 0.0% |
| range | P6 | 0.398 | 0.597 | 0.0118 | 0.0% |
| range | P5 | 0.907 | 1.429 | 0.0275 | 0.0% |
| tile_p95 | P7 | 0.147 | 0.227 | 0.0051 | 0.0% |
| tile_p95 | P6 | 0.398 | 0.597 | 0.0152 | 0.0% |
| tile_p95 | P5 | 0.907 | 1.429 | 0.0354 | 0.0% |
| ratio_max | P7 | 0.147 | 0.227 | 0.0335 | 13.8% |
| ratio_max | P6 | 0.398 | 0.597 | 0.0931 | 15.0% |
| ratio_max | P5 | 0.907 | 1.429 | 0.1908 | 12.6% |
| exact_l1 | P7 | 0.147 | 0.227 | 1.936 | 100.0% |
| exact_l1 | P6 | 0.398 | 0.597 | 5.809 | 100.0% |
| exact_l1 | P5 | 0.907 | 1.429 | 13.554 | 100.0% |

结论很明确：

```text
1. 只用 activation omitted range 的 bound 明显偏乐观。
2. 乘 W tile p95 后仍然不足以覆盖真实 low-bit contribution。
3. exact_l1 是严格安全上界，但非常松，会让 early stop 变得很保守。
4. ratio_max 介于二者之间，但仍不是严格安全 bound。
```

## 4. Stop-Depth Decision Check

用真实低位贡献定义 oracle minimal depth：

```text
oracle_min_depth =
    最小 depth，使 actual_delta_ratio(depth) <= tolerance(node)
```

再比较 runtime bound 给出的 stop depth。

| Bound | Runtime Avg Depth | Oracle Avg Depth | Exact Match | Conservative | Aggressive |
|---|---:|---:|---:|---:|---:|
| range | 5.36 | 7.91 | 0.0% | 0.0% | 100.0% |
| tile_p95 | 5.88 | 7.91 | 0.1% | 0.0% | 99.9% |
| ratio_mean | 5.36 | 7.91 | 0.0% | 0.0% | 100.0% |
| ratio_max | 7.46 | 7.91 | 48.0% | 3.7% | 48.3% |
| exact_l1 | 8.00 | 7.91 | 91.1% | 8.9% | 0.0% |

含义：

```text
range / tile_p95:
    经常让节点过早停止，不能作为严格 predictor-free 判据。

exact_l1:
    不会低估真实 delta，但几乎退化成 P8。

ratio_max:
    比 range 更接近 oracle，但仍有接近一半样本偏激进。
```

## 5. Current Takeaway

这组 hard validation 给出两个重要结论。

第一，当前 embedding-pool sweep 仍然有价值：

```text
它证明了 P8/P6/P5/P4 depth choice 对下游精度和 cost 的影响。
```

第二，严格的 runtime predictor-free bound 还需要继续改：

```text
旧的 range / tile_p95 bound 不能直接声称是数值安全停止判据。
安全的 exact_l1 bound 太松，需要设计更紧的 tile-aware bound。
```

因此当前 Graph-Bit 状态应表述为：

```text
system-level accuracy/cost proxy 已经跑通；
tile-level numeric validation 已经建立；
下一步是把 runtime bound 从简单 range bound 升级成更可靠的 tile-aware bound。
```

## 6. Next Bound Direction

后续优先验证三类改进：

```text
1. normalized ratio bound
   使用 W tile abs statistics 和 partial norm floor，避免分母 cancellation 造成误判。

2. module-wise safety factor
   不同 GEMM 模块的 low-bit delta 分布不同，可用固定模块级安全系数。

3. conservative hybrid policy
   high-risk node 使用 exact_l1 / P8；
   mid/low-risk node 使用 ratio_max + safety factor；
   仍然保持 predictor-free，不引入 learned predictor。
```

对应的评估标准：

```text
coverage rate:
    bound >= actual_delta_ratio 的比例。

tightness:
    bound / actual_delta_ratio 的倍率。

depth agreement:
    runtime_bound_depth 与 oracle_min_depth 的差距。
```

这一步完成后，Graph-Bit 的 runtime stop-depth 才能从“accuracy proxy”推进到“数值可解释的 predictor-free bound”。
