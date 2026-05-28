# FFN Channel Gating for Graph-Text Encoder NPU

本文档记录当前新增的 FFN channel gating 原型：它不是 hash reuse，也不是 token truncation，而是面向必须执行 LLM/ST encoder 的节点，在 Transformer FFN 中间维度上做硬件友好的 channel-group gating。

## 1. 动机

对 encoder-only 的图文本 workload，Transformer 计算仍然主要集中在每层 Attention 和 FFN。相比 attention tile skipping，第一版选择 FFN channel gating 更稳：

- FFN 计算和权重/activation traffic 占比高；
- FFN 中间维度天然可以按 channel group 划分；
- 硬件上对应规则的 grouped GEMM / grouped SRAM fetch mask；
- 不改变 attention 的精确语义，精度更容易控制。

核心思想不是全图统一少算 FFN，而是：

```text
高风险节点: Full W4A8 encoder
低风险节点: W4A8 + FFN channel-gated encoder
```

也就是说，真正的机制是 graph-aware scheduler + FFN-gated execution path。

## 2. 当前实现

生成 embedding pool 时，先基于少量 calibration texts 统计每层 FFN 中间激活能量：

```text
energy_l[c] = sum |FFN_activation_l[..., c]|
```

然后按连续 channel group 选 top groups：

```text
group_size = 64
keep_ratio = 0.75 / 0.50 / 0.25
```

推理时，在 FFN activation 后、第二个 FFN linear 前把未选中的 channel group 置零：

```text
h = FFN_up(x)
h = activation(h)
h = h * channel_group_mask
out = FFN_down(h)
```

当前支持：

- DistilBERT/ST FFN: `lin1 -> activation -> gate -> lin2`
- LLaMA-style MLP: `act(gate_proj) * up_proj -> gate -> down_proj`

## 3. 成本模型

当前评估用相对 full FP encoder 的归一化成本。Full W4A8 cost 设为 0.5：

```text
Cost(FFN keep r) =
    W4A8_cost * (attn_weight + ffn_weight * r)
              / (attn_weight + ffn_weight)
```

默认：

```text
W4A8_cost  = 0.50
attn_weight = 0.35
ffn_weight  = 0.65
```

因此：

```text
FullW4A8: 0.500
FFN75:   0.419
FFN50:   0.338
FFN25:   0.256
```

## 4. Cora/ST 初步结果

### 4.1 Uniform gating 不成立

全图统一使用 FFN-gated encoder 会明显伤精度：

```text
Config     Gate    FFN Keep  Cost   Drop    AvgErr
FullW4A8   0%      100%      0.500  0.08%   0.00063
FFN75      100%    75%       0.419  6.06%   0.10008
FFN50      100%    50%       0.338  9.75%   0.16750
FFN25      100%    25%       0.256  22.32%  0.29318
```

结论：FFN channel 不能静态全图关闭。它必须作为可选执行路径，由 scheduler 选择低风险节点使用。

### 4.2 Graph-aware routed gating 有效

只让部分低风险节点走 FFN-gated path，其余节点保持 FullW4A8：

```text
Config           Gate   Keep  Cost   Drop   AvgErr
TSER20_FFN75     20%    75%   0.484  0.44%  0.01994
Degree20_FFN75   20%    75%   0.484  0.52%  0.02069
TSER40_FFN75     40%    75%   0.468  1.08%  0.04029
Degree40_FFN75   40%    75%   0.468  1.10%  0.04106
TSER60_FFN75     60%    75%   0.451  2.19%  0.06018
Degree60_FFN75   60%    75%   0.451  1.82%  0.06207
```

Random routing 明显更差，例如：

```text
Random60_FFN75: Drop 3.16%
Degree60_FFN75: Drop 1.82%
TSER60_FFN75:   Drop 2.19%
```

这说明图风险分数确实能帮助 scheduler 判断哪些节点可以安全走 FFN-gated path。

## 5. 复现实验命令

生成 Cora/ST gated pools：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs W4A8 \
  --batch_size 128 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --ffn_channel_gating \
  --ffn_gate_keep_ratio 0.75 \
  --ffn_gate_group_size 64 \
  --ffn_gate_calib_samples 256 \
  --ffn_gate_calibration_strategy random \
  --tag_suffix FFN75 \
  --overwrite
```

把 `--ffn_gate_keep_ratio` / `--tag_suffix` 改为 `0.50/FFN50` 和 `0.25/FFN25` 可生成另外两组。

评估 uniform + routed gating：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite ffn_channel_gating \
  --real_quant_model_name ST \
  --ffn_gating_reference_tag W4A16 \
  --ffn_gating_full_tag W4A8 \
  --ffn_gating_tags W4A8_FFN75 W4A8_FFN50 \
  --ffn_gating_names FFN75 FFN50 \
  --ffn_gating_keep_ratios 0.75 0.50 \
  --ffn_gating_route_ratios 0.20 0.40 0.60
```

## 6. 体系结构含义

这条路径对应一个可落地的 NPU 机制：

```text
Graph/TSER scheduler
        |
        |-- high-risk nodes -> Full W4A8 FFN
        |
        |-- low-risk nodes  -> FFN channel-gated W4A8 FFN
```

硬件需要支持：

- 每层 FFN channel-group mask buffer；
- grouped FFN weight/activation fetch；
- 对低风险 batch 启用 reduced-channel GEMM；
- 对高风险 batch 走 full-channel GEMM；
- scheduler 以 degree / TSER / confidence score 选择 gated node set。

当前结论：FFN channel gating 不是单独替代 FullW4A8 的精度方案，也不再作为 P2 主线。它更适合作为 Graph-Bit NPU 中 mode-adaptive PE array 的辅助执行模式；主线硬件点是 graph-conditioned activation precision-depth / bit-plane execution。
