# W4A8/W4A4 Embedding PTQ 说明

本文档说明当前 `GraphhopSimhash` 中真实 embedding pool 的 PTQ 生成方式，以及这些 pool 应该如何配合固定预算量化路由实验使用。

当前主线不是把全图都压到 W4A4，而是：

```text
W4A8:
    safe low-precision path

W4A4:
    aggressive low-precision path

GraphHop / Degree / TSER policy:
    决定哪些节点可以走 W4A4，哪些节点应该留在 W4A8。
```

因此，W4A4 的意义不是证明“全图 W4A4 也能很好”，而是提供一个更便宜但更危险的执行路径，让图相关路由策略去选择低风险节点。

## 1. 为什么需要 calibration-aware PTQ

最早的 fake quant 路径只是做：

```text
W -> round / clip / dequant
X -> round / clip / dequant
Y = Linear(X_q, W_q)
```

这个路径适合作为 stress test，但对 LLaMA/ST 这种 Transformer 前端不够稳，尤其是 A4 激活量化。

核心问题是：

```text
A4 最容易被 activation outlier 破坏。
```

如果 activation scale 直接由 raw max 决定，少数极端大值会占掉 4-bit 的大部分表示范围，密集区间反而表示得很差。对于 LLaMA-7B，这个问题更严重，沿用轻量 encoder 上的朴素 W4A4 策略可能直接产生 NaN 或极大 embedding damage。

当前 PTQ 路径使用：

```text
group-wise W4 weight quantization
+ activation-aware scale search
+ SmoothQuant/AWQ-style scale migration
+ percentile activation clipping
+ dynamic per-token activation quantization
+ optional activation outlier protection
+ output embedding affine alignment
+ damage check against FP16 pool
```

## 2. 高层流程

对每个 dataset / model / config：

```text
1. 加载 FP16 Transformer encoder。
2. 将 nn.Linear 替换成 CalibratedPTQLinear。
3. 在 calibration texts 上跑一次 FP forward，收集每层 activation。
4. 每层搜索 scale mode、alpha、activation clipping threshold。
5. 选择 layer output MSE 最小的 PTQ 配置。
6. 用 finalized PTQ Linear 提取全图 embedding。
7. 可选：在 calibration embeddings 上拟合 output affine alignment。
8. 保存 embedding pool:
      cache_data/{dataset}_{model}_oracle_{tag}.pt
9. 如果 FP16 pool 存在，打印 cosine damage statistics。
```

主要实现位置：

```text
generate_real_quant_pools.py
    CalibratedPTQLinear
    groupwise_weight_quantize
    dynamic_activation_quantize
    fit_output_alignment
```

## 3. Calibration Set

推荐使用随机 calibration nodes：

```bash
--w4a_calib_samples 256
--calibration_strategy random
--seed 42
```

每个 Linear 记录有限数量的 activation rows：

```bash
--ptq_sample_rows 128
```

注意：calibration forward 期间，层本身仍然执行原始 FP Linear，只记录 activation samples。这样可以避免校准阶段就引入级联量化噪声。

## 4. Weight Quantization

Linear 层形式为：

```text
Y = X W^T + b
W shape = [out_features, in_features]
```

权重量化沿 input-channel 维度 group-wise 进行：

```bash
--ptq_group_size 64
```

对每个 output channel 和每个 input-channel group：

```text
s_w = max(abs(W_group)) / q_max
Q_w = clamp(round(W_group / s_w), q_min, q_max)
W_hat = Q_w * s_w
```

4-bit signed quantizer：

```text
q_min = -8
q_max =  7
```

8-bit signed quantizer：

```text
q_min = -128
q_max =  127
```

group-wise scale 比整层一个 scale 更稳，尤其适合 LLaMA 这种通道分布差异很大的模型。

## 5. Activation Quantization

Activation 使用 per-token / per-row dynamic quantization：

```text
s_x = quantile(abs(x), p) / q_max
Q_x = clamp(round(x / s_x), q_min, q_max)
x_hat = Q_x * s_x
```

推荐搜索：

```bash
--ptq_clip_grid 1.0 0.999
```

解释：

```text
1.0:
    使用 max absolute value。

0.999:
    clip 极端 activation outlier，把更多 4-bit bucket 留给密集区间。
```

这是 A4 激活量化最关键的保护之一。

## 6. Scale Migration

量化前，代码会搜索 channel-wise scale vector `s`：

```text
X' = X / s
W' = W * s
```

数学上，未量化时它保持等价：

```text
X W^T = (X / s) (W * s)^T
```

量化后，scale migration 可以把 activation outlier 的压力迁移到 weight side，让 group-wise W4 更容易表示。

当前搜索的模式包括：

```text
identity:
    s = 1

smooth:
    s = act_stat^alpha / weight_stat^(1 - alpha)

awq:
    s = act_stat^alpha

balanced:
    s = act_stat^alpha * weight_stat^(1 - alpha)
```

推荐 alpha grid：

```bash
--ptq_smooth_grid 0.0 0.25 0.5
```

所有 scale 都会做 non-finite 清理、clamp、median normalize，避免少数异常 channel 产生极端 scale。

## 7. LLaMA W4A4 的 outlier backend

LLaMA-7B 的 W4A4 比 ST 更脆弱。当前 LLaMA W4A4 推荐使用：

```bash
--w4a_backend ptq_outlier
--ptq_outlier_ratio 0.02
--ptq_outlier_a_bit 8
```

含义是：

```text
每层统计 activation channel 的离群程度；
top 2% outlier channels 使用 A8 side path；
剩余 activation channels 使用 A4。
```

这不是图相关策略，而是为了让 W4A4 backend 本身先变成“可用的 aggressive path”。如果 W4A4 pool 本身 NaN 或 damage 过大，后面的 TSER/Degree/GraphHop routing 没有实验意义。

## 8. Output Affine Alignment

即使每层 local MSE 被压低，最终 Transformer embedding 仍可能和 FP16 embedding 发生分布错位。为了降低 GFM 到 GNN 接口处的数值分布偏移，可以打开：

```bash
--ptq_align_output
```

在 calibration embeddings 上：

```text
E_fp = FP16 embeddings
E_q  = quantized embeddings
```

拟合 per-dimension affine：

```text
E_fp ~= gamma * E_q + beta
```

闭式解：

```text
gamma = sum((E_q - mean(E_q)) * (E_fp - mean(E_fp)))
        / sum((E_q - mean(E_q))^2)

beta = mean(E_fp) - gamma * mean(E_q)
```

然后对全图 quantized embeddings 做：

```text
E_aligned = normalize(gamma * E_q + beta)
```

这一步不是“恢复原始语义”，而是缓解量化前端输出分布和下游 GNN 期望输入分布之间的错位。

## 9. Damage Check

保存 quantized pool 后，如果 FP16 pool 存在，脚本会打印：

```text
cos_err(v) = 1 - cosine(E_fp(v), E_quant(v))
```

统计项：

```text
mean, p50, p90, p95, p99, max
```

这是必须看的 sanity check：

```text
W4A8 damage 很低:
    可以作为 safe low-precision path。

W4A4 damage 明显更高:
    只能作为 aggressive path，不能默认全图使用。
```

如果 W4A4 全图掉点很大，这首先说明 W4A4 backend 本身损伤重，不应该直接归因于 TSER/Degree 路由策略。

## 10. 推荐生成命令

### 10.1 ST / Arxiv FP16

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets arxiv \
  --llm_name ST \
  --configs fp16 \
  --batch_size 128 \
  --overwrite
```

### 10.2 ST / PubMed + Arxiv W4A4

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets pubmed arxiv \
  --llm_name ST \
  --configs W4A4 \
  --batch_size 128 \
  --w4a_backend ptq \
  --w4a_calib_samples 256 \
  --calibration_strategy random \
  --seed 42 \
  --ptq_group_size 64 \
  --ptq_sample_rows 128 \
  --ptq_smooth_grid 0.0 0.25 0.5 \
  --ptq_clip_grid 1.0 0.999 \
  --ptq_output_clip_percentile 0.999 \
  --ptq_output_clip_multiplier 4.0 \
  --ptq_align_output \
  --tag_suffix PTQ_TEST2 \
  --overwrite
```

### 10.3 LLaMA-7B / PubMed W4A8

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets pubmed \
  --llm_name llama2_7b \
  --configs W4A8 \
  --batch_size 4 \
  --w4a_backend ptq \
  --w4a_calib_samples 256 \
  --calibration_strategy random \
  --seed 42 \
  --ptq_group_size 64 \
  --ptq_sample_rows 128 \
  --ptq_smooth_grid 0.0 0.25 0.5 \
  --ptq_clip_grid 1.0 0.999 \
  --ptq_output_clip_percentile 0.999 \
  --ptq_output_clip_multiplier 4.0 \
  --ptq_align_output \
  --tag_suffix LLAMA7B_PTQ_TEST \
  --overwrite
```

### 10.4 LLaMA-7B / PubMed W4A4 outlier backend

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets pubmed \
  --llm_name llama2_7b \
  --configs W4A4 \
  --batch_size 4 \
  --w4a_backend ptq_outlier \
  --w4a_calib_samples 256 \
  --calibration_strategy random \
  --seed 42 \
  --ptq_group_size 64 \
  --ptq_sample_rows 128 \
  --ptq_smooth_grid 0.0 0.25 0.5 \
  --ptq_clip_grid 1.0 0.999 \
  --ptq_outlier_ratio 0.02 \
  --ptq_outlier_a_bit 8 \
  --ptq_output_clip_percentile 0.999 \
  --ptq_output_clip_multiplier 4.0 \
  --ptq_align_output \
  --tag_suffix LLAMA7B_W4A4O_R2 \
  --overwrite
```

### 10.5 LLaMA-7B / Arxiv FP16

Arxiv + LLaMA-7B 很慢，建议先只生成 FP16：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets arxiv \
  --llm_name llama2_7b \
  --configs fp16 \
  --batch_size 4 \
  --overwrite
```

## 11. 推荐评估命令

当前固定预算主线使用：

```text
fixed_aggressive_budget
```

例如 Cora / LLaMA-7B，20% W4A4 + 80% W4A8：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite fixed_aggressive_budget \
  --real_quant_model_name llama2_7b \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_LLAMA7B_PTQ_TEST \
  --real_quant_int4_tag W4A4_LLAMA7B_W4A4O_R2 \
  --real_quant_error_norm 1.0 \
  --real_quant_int8_ratio 0.80 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

注意：

```text
--real_quant_int8_ratio 0.80
```

表示：

```text
80% W4A8
20% W4A4
```

如果要复现旧表里的：

```text
80% W4A4
20% W4A8
```

则应该设置：

```bash
--real_quant_int8_ratio 0.20
```

## 12. 当前实验解读

### 12.1 W4A8

W4A8 通常非常稳，适合作为 safe low-precision baseline。

例如 PubMed / LLaMA-7B W4A8 pool：

```text
DamageCheck mean ~= 0.0047
```

在下游 GNN 上通常只造成很小掉点。

### 12.2 W4A4

W4A4 全图使用通常掉点明显：

```text
ST / Cora:
    AllW4A4 drop ~= 14 points

LLaMA-7B / Cora:
    AllW4A4 drop ~= 20+ points

LLaMA-7B / PubMed:
    AllW4A4 drop ~= 15 points
```

这不是异常，而是说明 W4A4 aggressive path 本身有明显损伤。合理用法是只给低风险节点使用。

### 12.3 固定预算评估

推荐主表不是看全图 W4A4，而是看：

```text
0% W4A4 + 100% W4A8
10% W4A4 + 90% W4A8
20% W4A4 + 80% W4A8
30% W4A4 + 70% W4A8
```

每个固定预算下比较：

```text
RandomBudget
DegreeBudget
TSERBudget
GraphHopSafeBudget
```

当前经验是：

```text
Cora / LLaMA-7B:
    TSER 3/1/1 可能略优于 Degree/Random。

Cora / ST:
    Degree 通常更稳，low-degree unique 项可能伤精度。

PubMed / ST 和 PubMed / LLaMA-7B:
    Degree 往往最好，说明 PubMed 更传播主导。
```

因此论文里不要写成“TSER 总是优于 Degree”。更稳的说法是：

```text
图相关路由是必要的；
不同数据集/backend 下，传播风险和图语义修正的重要性不同；
GraphHopSafeBudget 是可部署的 graph/hash stability routing；
Degree 是强 baseline，尤其在传播主导数据集上。
```

## 13. 和 TSER 文档的关系

PTQ pool generation 和 routing policy 是两层：

```text
PTQ generator:
    生成 FP16 / W4A8 / W4A4 embedding pools。

Routing policy:
    决定每个节点读取哪个 pool。
```

这两层必须分开看：

```text
如果 W4A4 pool 本身 damage 很大，
    任何 routing 策略都会受到上限约束。

如果固定 W4A4 budget 相同，
    Random / Degree / TSER / GraphHopSafe 的差异才表示路由策略差异。
```

对应的分数定义见：

```text
SCORE_DEFINITIONS.md
```

## 14. 限制

当前实现是 calibration-aware PTQ embedding generator，不是生产级 INT4 kernel：

```text
1. 它通过 quantize-dequantize tensor 模拟低 bit 数值效果。
2. 它没有把 INT4 weight 真正 pack 到自定义 CUDA/NPU kernel。
3. 它不是完整工业级 AWQ/OmniQuant 实现。
4. 它没有 Hessian / second-order reconstruction。
5. W4A4 对 sentence/node embedding 仍然脆弱。
```

对 architecture / routing 研究来说，它的价值是提供真实 embedding damage trend，并支持公平的 policy comparison。最终系统论文中，软件 PTQ 结果应该和硬件侧 W4A8/W4A4 execution path、NDP/NPU/CAM routing pipeline 一起解释。
