# BFP Activation Format For W4 LLaMA Encoder

本文档说明当前代码中的 BFP activation 格式、`B64/B128/B256` 参数含义，以及它和传统 `W4A8` activation quantization 在计算与访存上的区别。

## 1. BFP 是什么

BFP 是 Block Floating Point。它把一组数放进同一个 block 中，让这个 block 共享一个 exponent，每个元素只保存自己的 signed mantissa。

概念上：

```text
普通 FP:
    每个元素都有自己的 sign / exponent / mantissa

BFP:
    一个 block 共享 exponent
    每个元素保存 signed mantissa
```

当前实现用于 activation fake quantization：

```text
activation block x
    -> block_abs_max
    -> shared exponent = ceil(log2(block_abs_max / mantissa_max))
    -> scale = 2^exponent
    -> mantissa = round(x / scale)
    -> reconstructed x_hat = mantissa * scale
```

代码位置：

```text
generate_real_quant_pools.py
    bfp_fake_quantize(...)
    ActivationQuantLinear(...)
```

当前 BFP 路径仍然使用官方 AWQ 的 W4 权重量化，只替换 activation 表示：

```text
W4BFPA8_B128:
    weight: official AWQ W4
    activation: BFP mantissa 8-bit, block size 128
```

## 2. B64 / B128 / B256 是什么

`B64/B128/B256` 表示 activation 在 hidden dimension 上多少个元素共享一个 exponent。

以 LLaMA-7B hidden size 4096 为例：

```text
B64:
    4096 / 64  = 64 个 exponent block

B128:
    4096 / 128 = 32 个 exponent block

B256:
    4096 / 256 = 16 个 exponent block
```

block size 越小：

```text
优点:
    每个 block 的动态范围更局部
    遇到 outlier 时，对其它 channel 的拖累更小

代价:
    exponent metadata 更多
    block 边界更多
    不同 block 的 scale jitter 更多
```

block size 越大：

```text
优点:
    exponent metadata 更少
    scale 更统一
    group-level 扰动更少

代价:
    如果 block 内有大 outlier，小值精度更容易受影响
```

因此 BFP block size 没有固定的单调结论，需要按模型、数据集、文本长度和 activation outlier 分布实测。

## 3. 和传统 W4A8 的区别

当前传统 `W4A8` 路径：

```text
weight:
    official AWQ W4

activation:
    affine A8 fake quantization
    scale / zero-point 来自 activation row 的动态范围
```

当前 `W4BFPA8_B*` 路径：

```text
weight:
    official AWQ W4

activation:
    BFP A8
    每个 block 共享 power-of-two exponent
    每个元素保存 signed 8-bit mantissa
```

核心区别：

```text
W4A8:
    affine scale + zero-point
    需要处理 zero-point correction / affine dequant

W4BFPA8:
    block shared exponent
    scale 是 2^exponent
    硬件上更接近 shift-based scaling
    不需要 affine zero-point
```

也就是说，BFP 的主要意义不是简单把 `A8` 改成另一个名字，而是把 activation 表示变成更硬件友好的 block-shared exponent 格式。

## 4. 计算和访存收益

### 4.1 W4BFPA8_B* 本身

`W4BFPA8_B64/B128/B256` 的 mantissa 仍然是 8-bit，所以 activation payload 位宽和传统 A8 接近。

因此单独看 `W4BFPA8`：

```text
不会显著减少 activation HBM payload
不会减少 W4 weight payload
主要减少 affine zero-point / scale 处理复杂度
提供 power-of-two shift-friendly scaling
```

还需要额外保存 exponent metadata：

```text
每个 activation block 一个 exponent
B64  metadata 多
B256 metadata 少
```

但相对 4096 维 activation mantissa 本身，exponent metadata 较小。

### 4.2 BFP mixed-depth 的潜在收益

BFP 更适合和 mixed mantissa / Graph-Bit variable depth 结合：

```text
W4BFPA8:
    full activation mantissa

W4BFPA6 / W4BFPA5 / W4BFPA4:
    更短 mantissa
    更少 bit-serial MAC activity
    更少 activation/RF/psum activity
```

这条线的硬件收益主要来自：

```text
1. mantissa bit 数减少
   -> bit-serial PE 执行周期减少
   -> partial-sum update 活动减少

2. block shared exponent
   -> 缩放逻辑更简单
   -> 避免 per-value affine zero-point correction

3. 与 graph-risk routing 结合
   -> 高风险节点保留更高 mantissa
   -> 低风险节点使用更低 mantissa
```

因此当前 `W4BFPA8_B*` 实验首先回答：

```text
BFP-A8 格式本身是否接近 W4A8？
```

只有 BFP-A8 自身足够稳，后续 BFP mixed-depth 才值得继续做。

## 5. Cora/LLaMA-7B BFP-A8 Block Sweep

当前在 Cora/LLaMA-7B 下生成了：

```text
cache_data/cora_llama2_7b_oracle_W4BFPA8_B64.pt
cache_data/cora_llama2_7b_oracle_W4BFPA8_B128.pt
cache_data/cora_llama2_7b_oracle_W4BFPA8_B256.pt
```

5-run `real_quant_ablation` 结果如下。这里 baseline 是 `W4A8` pool，表中只看 uniform BFP-A8 自身误差：

| Config | Baseline Acc | Acc | Drop | AvgErr |
|---|---:|---:|---:|---:|
| `W4BFPA8_B64` | 0.7193 | 0.7165 | 0.28% | 0.00224 |
| `W4BFPA8_B128` | 0.7193 | 0.7163 | 0.30% | 0.00176 |
| `W4BFPA8_B256` | 0.7197 | 0.7190 | 0.08% | 0.00122 |

Embedding-level distance to `W4A8`:

| Config | mean L2 | max L2 | mean 1-cos |
|---|---:|---:|---:|
| `W4BFPA8_B64` | 0.08277 | 0.51948 | 0.00523 |
| `W4BFPA8_B128` | 0.07784 | 0.41360 | 0.00433 |
| `W4BFPA8_B256` | 0.06708 | 0.34437 | 0.00325 |

当前 Cora 结果显示 `B256` 最稳。这个现象说明 Cora/LLaMA-7B 的 activation 分布在当前配置下没有体现出强 outlier 压力；较大的 block 减少了 block-wise scale jitter，最终 embedding 方向更稳定。

## 6. 实验命令

生成 Cora/LLaMA-7B BFP-A8 pools：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4BFPA8_B64 W4BFPA8_B128 W4BFPA8_B256 \
  --batch_size 4 \
  --w4a_backend awq \
  --w4a_calib_samples 128 \
  --overwrite
```

5-run block sweep：

```bash
mkdir -p output/graphbfp_block_sweep/cora

for tag in W4BFPA8_B64 W4BFPA8_B128 W4BFPA8_B256; do
  python -m GraphhopSimhash \
    --datasets cora \
    --runs 5 \
    --experiment_suite real_quant_ablation \
    --real_quant_policy_suite w4a8_budget \
    --real_quant_model_name llama2_7b \
    --real_quant_fp_tag W4A8 \
    --real_quant_int8_tag "$tag" \
    --real_quant_int4_tag W4A4 \
    --real_quant_fp_ratio 0.0 \
    --real_quant_int8_ratio 0.20 \
    --real_quant_error_norm 1.0 \
    2>&1 | tee "output/graphbfp_block_sweep/cora/${tag}_runs5.log"
done
```

结果日志：

```text
output/graphbfp_block_sweep/cora/W4BFPA8_B64_runs5.log
output/graphbfp_block_sweep/cora/W4BFPA8_B128_runs5.log
output/graphbfp_block_sweep/cora/W4BFPA8_B256_runs5.log
```

## 7. 当前 Takeaway

当前 Cora/LLaMA-7B 上：

```text
BFP-A8 格式本身可行。
B256 当前最稳，drop 约 0.08%。
B64/B128 也可用，但 drop 约 0.28%-0.30%。
```

下一步应在 PubMed / Arxiv 上复核 block size 趋势，因为长文本和更复杂 activation 分布可能让较小 block size 重新占优。
