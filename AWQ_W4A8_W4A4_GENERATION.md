# AWQ W4A16 / W4A8 / W4A4 Embedding Pool 说明

本文档说明当前版本如何生成低精度 embedding pool，以及为什么它比旧版本的 W4A4 明显稳定。

## 1. 当前三种配置

```text
W4A16:
    官方 AWQ W4 weight + FP16 activation。
    当前 AWQ-family reference。

W4A8:
    官方 AWQ W4 weight + dynamic affine A8 activation fake quant。

W4A4:
    官方 AWQ W4 weight + dynamic affine A4 activation fake quant。
```

旧的本地 fake quant 仍保留为：

```text
W4A16_FAKE
W4A8_FAKE
W4A4_FAKE
```

但主线实验优先使用 `W4A16 / W4A8 / W4A4`。

## 2. 生成流程

入口：

```bash
python -m GraphhopSimhash.generate_real_quant_pools
```

核心流程：

```text
1. 加载原始节点文本。
2. 加载 ST / LLaMA 模型。
3. 对 W4A16/W4A8/W4A4 执行官方 AWQ W4 weight search。
4. 对 W4A8/W4A4 额外包一层 activation quant。
5. mean-pool 最后一层 hidden states 得到节点 embedding。
6. 保存到 cache_data/{dataset}_{model}_oracle_{tag}.pt。
```

AWQ search 结果会单独缓存：

```text
cache_data/awq/{dataset}_{model}_w4_g{group}_n{samples}_s{seqlen}.pt
```

这样之后重复生成 W4A16/W4A8/W4A4 时不用重新搜索 AWQ scale。

## 3. 为什么当前 W4A4 效果更好

根本原因不是 output affine alignment，也不是 error-aware routing，而是 **W4A4 backend 本身换了**。

旧方案主要问题：

```text
1. W4A4 fake quant 过于粗糙；
2. activation 使用对称量化，极易被 outlier 拉坏 scale；
3. calibration 少且不贴近图节点文本；
4. LLaMA / ST 的 embedding 方向对 activation 扰动很敏感。
```

当前方案的改进：

```text
1. 权重量化使用官方 AWQ。
2. calibration 文本来自真实图节点文本，而不是无关 prompt。
3. activation quant 使用 dynamic affine quant，而不是简单对称量化。
4. W4A4 可选 activation outlier channel 保护。
5. W4A16 / W4A8 / W4A4 属于同一 AWQ-family，评估口径一致。
```

因此当前 W4A4 的掉点下降，主要说明：

```text
低精度 backend 先变得可用了；
之后 Degree/TSER routing 的比较才有意义。
```

## 4. 它和 output affine alignment 的区别

output affine alignment 是后处理：

```text
E_q_aligned = gamma * E_q + beta
```

它需要用校准节点的 FP embedding 拟合 `gamma/beta`。这可以作为 debug 或补救手段，但不是当前主线原因。

当前 AWQ 路径是在模型内部改善：

```text
weight scale
activation quant
outlier handling
```

因此更适合作为量化 backend 的主线。

## 5. ST 生成命令

Cora + PubMed：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora pubmed \
  --llm_name ST \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

单独 Cora：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

输出：

```text
cache_data/cora_ST_oracle_W4A16.pt
cache_data/cora_ST_oracle_W4A8.pt
cache_data/cora_ST_oracle_W4A4.pt

cache_data/pubmed_ST_oracle_W4A16.pt
cache_data/pubmed_ST_oracle_W4A8.pt
cache_data/pubmed_ST_oracle_W4A4.pt
```

## 6. LLaMA-7B 生成命令

Cora：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --overwrite
```

PubMed：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets pubmed \
  --llm_name llama2_7b \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --overwrite
```

LLaMA 比 ST 慢很多，这是正常的：模型更大、hidden dim 更大、节点数也可能更多。

## 7. 可选：activation outlier 保护

如果 W4A4 仍然损伤很重，可以先统计 activation outlier：

```bash
python -m GraphhopSimhash.activation_outlier_calibration \
  --dataset cora \
  --llm_name llama2_7b \
  --calib_samples 128 \
  --seed 42
```

然后生成 W4A4 时打开：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --activation_outlier_clip \
  --overwrite
```

含义：

```text
普通 channel:
    A4 dynamic affine quant。

outlier channel:
    可保护到 A8，降低极端激活对 W4A4 的破坏。
```

## 8. 评测命令

ST / Cora：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag W4A16 \
  --real_quant_int8_tag W4A8 \
  --real_quant_int4_tag W4A4 \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0
```

LLaMA-7B / Cora：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name llama2_7b \
  --real_quant_fp_tag W4A16 \
  --real_quant_int8_tag W4A8 \
  --real_quant_int4_tag W4A4 \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0
```

## 9. 结果怎么看

重点看：

```text
UniformW4A8:
    W4A8 backend 是否接近 reference。

UniformW4A4:
    W4A4 backend 本身是否可用。

RandomTopK_W4A8:
    固定预算随机 baseline。

DegreeTopK_W4A8:
    保护高传播节点。

TSERTopK_W4A8:
    保护高传播 + graph context + low-degree unique 节点。
```

如果 `UniformW4A4` 已经掉点很大，先不要责怪 TSER/Degree routing；应该先优化 W4A4 backend。

如果 `RandomTopK` 比 Degree/TSER 好，说明当前分数和该 dataset/backend 的真实敏感性没有对齐，需要调权重或调 budget。

## 10. 当前边界

```text
ST:
    AWQ W4A4 通常比较可用，适合作为主线低精度实验。

LLaMA-7B:
    W4A4 更敏感，可能需要 outlier 保护、更大 calibration、更保守 W4A8 ratio。

Arxiv:
    节点数大，生成 LLaMA embedding 很慢；建议睡前跑。
```
