# 官方 AWQ 版 W4A8 / W4A4 Embedding Pool 生成说明

本文档说明 `GraphhopSimhash_reuse_real_quant_latest/GraphhopSimhash` 当前同步自 `test-access` 分支后的 W4A8 / W4A4 embedding pool 是如何生成的，以及它和旧版 fake quant / PTQ 路线的区别。

对应实现文件：

```text
GraphhopSimhash/generate_real_quant_pools.py
GraphhopSimhash/real_quant.py
GraphhopSimhash/runner.py
GraphhopSimhash/third_party/llm-awq/
```

当前核心结论：

```text
W4A16 = 官方 llm-awq W4 weight-only + FP16 activation
W4A8  = 官方 llm-awq W4 weight path + 本地 A8 dynamic activation fake quant
W4A4  = 官方 llm-awq W4 weight path + 本地 A4 dynamic activation fake quant
```

也就是说，`W4A8` / `W4A4` 不是纯官方 AWQ 原生算子。官方 `llm-awq` 负责 W4 权重量化；activation 的 A8 / A4 是 GraphhopSimhash 在 Linear 输入处额外加的动态仿真量化。

## 1. 配置名和含义

代码中的配置表在 `generate_real_quant_pools.py`：

```python
CONFIG_SPECS = {
    "fp16": {"tag": "FP16", "kind": "bnb", "w_bit": 16, "a_bit": 16},
    "int8": {"tag": "INT8", "kind": "bnb", "w_bit": 8, "a_bit": 8},
    "int4": {"tag": "INT4", "kind": "bnb", "w_bit": 4, "a_bit": 16},
    "W4A16": {"tag": "W4A16", "kind": "awq", "w_bit": 4, "a_bit": 16},
    "W4A8": {"tag": "W4A8", "kind": "awq_act", "w_bit": 4, "a_bit": 8},
    "W4A4": {"tag": "W4A4", "kind": "awq_act", "w_bit": 4, "a_bit": 4},
    "W4A16_FAKE": {"tag": "W4A16_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 16},
    "W4A8_FAKE": {"tag": "W4A8_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 8},
    "W4A4_FAKE": {"tag": "W4A4_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 4},
}
```

推荐主线看这三类：

```text
FP16:
    原始 reference embedding pool。

W4A8:
    中等精度/中等成本路径。
    使用官方 AWQ 搜索出的 W4 weight scale/clip；
    再对每个 Linear 输入 activation 做 A8 动态仿射 fake quant。

W4A4:
    激进低成本路径。
    使用同一套官方 AWQ W4 weight path；
    再对每个 Linear 输入 activation 做 A4 动态仿射 fake quant。
```

旧配置：

```text
W4A8_FAKE / W4A4_FAKE:
    旧版本地 FakeQuantLinear 路径。
    不是当前官方 AWQ 主线，只适合复现旧实验或 debug。
```

## 2. 生成流程总览

生成一个 W4A8 / W4A4 pool 时，代码做的是：

```text
1. 加载 FP16 模型和 tokenizer
2. 读取图节点文本
3. 用图节点文本构造 AWQ calibration blocks
4. 调用 third_party/llm-awq 官方 AWQ search
5. 保存或复用 AWQ search 结果
6. 对模型权重做 W4 pseudo quantization
7. 如果是 W4A8 / W4A4，则给 Linear 套 activation fake quant wrapper
8. 对全图节点文本重新 encode
9. mean pooling 得到 node embedding
10. torch.save 保存 embedding pool 到 cache_data
```

需要注意：保存下来的不是量化模型 checkpoint，而是每个节点的 embedding tensor。

默认保存路径：

```text
cache_data/{dataset}_{llm_name}_oracle_{tag}.pt
```

例如：

```text
cache_data/cora_ST_oracle_W4A8.pt
cache_data/cora_ST_oracle_W4A4.pt
cache_data/cora_llama2_7b_oracle_W4A8.pt
cache_data/cora_llama2_7b_oracle_W4A4.pt
```

如果命令显式传入 `--output_path`，则保存到指定路径。我们测试时为了不覆盖主 cache，用过：

```text
cache_data/cora_ST_oracle_W4A8_OFFICIAL_AWQ_BRANCH.pt
cache_data/cora_ST_oracle_W4A4_OFFICIAL_AWQ_BRANCH.pt
```

## 3. 官方 AWQ 权重量化部分

官方 AWQ 源码放在：

```text
GraphhopSimhash/third_party/llm-awq/
```

生成时会先把这个目录加入 `sys.path`：

```python
ensure_awq_project_on_path()
```

核心调用在 `apply_official_awq_w4`：

```python
from awq.quantize.pre_quant import apply_awq, run_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight
```

AWQ search 使用：

```python
awq_results = run_awq(
    model,
    tokenizer,
    w_bit=4,
    q_config={
        "zero_point": not args.awq_no_zero_point,
        "q_group_size": args.awq_q_group_size,
    },
    n_samples=args.awq_calib_samples,
    seqlen=args.awq_seqlen,
    auto_scale=not args.awq_disable_auto_scale,
    mse_range=mse_range,
    calib_data="graph_text",
)
```

默认 AWQ 参数：

```text
--awq_calib_samples 128
--awq_seqlen 512
--awq_q_group_size 128
zero_point = True
auto_scale = True
mse_clip = True, 但 ST/DistilBERT 默认关闭
```

ST/DistilBERT 默认使用更小的自动生成参数：

```text
awq_calib_samples = 16
awq_seqlen = 128
```

这是 `real_quant.py::regenerate_real_quant_pools` 里写死的自动生成设置：

```python
is_st_model = str(args.real_quant_model_name).upper() == "ST"
awq_calib_samples = 16 if is_st_model else 128
awq_seqlen = 128 if is_st_model else 512
```

AWQ search 结果会缓存到：

```text
cache_data/awq/{dataset}_{llm_name}_w4_g{group_size}_n{calib_samples}_s{seqlen}.pt
```

例如：

```text
cache_data/awq/cora_ST_w4_g128_n16_s128.pt
```

后续再次生成 W4A8 / W4A4 时，如果不加：

```text
--awq_overwrite_results
```

代码会复用这个 AWQ search 结果，不重新跑 AWQ scale/clip 搜索。

## 4. Calibration 数据怎么来

官方 AWQ 默认常用语言模型 calibration 数据；这里改成图节点文本。

实现位置：

```python
build_local_awq_calib_getter(texts)
patch_awq_calibration_data(texts)
```

它会遍历当前 dataset 的原始节点文本：

```text
Cora / PubMed / Arxiv node text
```

然后 tokenizer encode，过滤空文本和超过 block size 的文本，把若干文本拼成 token block：

```text
block_size = --awq_seqlen
n_samples  = --awq_calib_samples
```

所以这版 AWQ 是 graph-text calibration-aware 的，不是用通用语料做校准。

## 5. W4 权重量化公式和含义

官方 AWQ 的核心思想是 activation-aware weight quantization。

直觉上，它不是直接对权重做普通 int4，而是先根据 calibration activation 估计哪些 channel 更重要，然后对权重做 scale migration，使重要 channel 的量化误差更小。

代码层面：

```text
run_awq:
    搜索每层 scale / clip。

apply_awq:
    把搜索出的 scale / clip 应用到模型。

pseudo_quantize_model_weight:
    把 Linear 权重伪量化成 W4。
```

这里的 pseudo quantization 仍然在 PyTorch 中产生浮点张量形式的“量化-反量化后权重”，用于生成 embedding pool。它不是最终硬件 packed int4 kernel。

## 6. W4A8 / W4A4 的 activation 量化怎么做

`W4A8` / `W4A4` 的区别只在 activation bit：

```text
W4A8: activation_bit = 8
W4A4: activation_bit = 4
```

在官方 AWQ W4 权重量化之后，代码会执行：

```python
replace_linear_with_activation_quant(model, a_bit=8 or 4)
```

它会把模型中的 `nn.Linear` 包成：

```python
ActivationQuantLinear(original_linear, a_bit)
```

forward 逻辑：

```python
qx = affine_fake_quantize(x, a_bit, mode="per_channel", dim=-1)
return original_linear(qx)
```

因此每个 Linear 执行的是：

```text
Y = Linear(QA(X), W_awq4, b)
```

其中：

```text
W_awq4:
    官方 AWQ 处理后的 W4 权重。

QA(X):
    对 Linear 输入 activation 做动态仿射 fake quant 后的结果。
```

activation fake quant 使用非对称 min-max 仿射量化：

```text
q_min = 0
q_max = 2^a_bit - 1

scale = (max(X) - min(X)) / q_max
zero_point = round(q_min - min(X) / scale)

X_q = clamp(round(X / scale + zero_point), q_min, q_max)
X_dq = (X_q - zero_point) * scale
```

代码中是：

```python
affine_fake_quantize(x, bit_width, mode="per_channel", dim=-1)
```

这里的 `mode="per_channel", dim=-1` 对 activation 最后一维单独取 min/max。对于输入形状 `[batch, seq_len, hidden_dim]`，效果接近：

```text
per token / per row dynamic activation quantization
```

这比旧版对称 activation quant 更适合 LayerNorm/GELU 后的非对称 activation 分布。

## 7. ST / DistilBERT adapter

官方 `llm-awq` 主要支持 causal LM，例如：

```text
LLaMA / Qwen2 / OPT / Bloom / MPT / Falcon / BigCode / NeoX
```

这个 branch 额外给 ST 使用的 DistilBERT 加了本地 adapter。

支持判断中包含：

```python
AWQ_SUPPORTED_CLASS_NAMES = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "OPTForCausalLM",
    "BloomForCausalLM",
    "DistilBertModel",
}
```

对应补丁在：

```text
third_party/llm-awq/awq/quantize/pre_quant.py
third_party/llm-awq/awq/quantize/auto_scale.py
```

ST/DistilBERT 默认关闭 AWQ MSE clip：

```text
[AWQ] DistilBERT adapter disables MSE clip by default to avoid large activation-clip memory spikes.
```

原因是 DistilBERT 的 activation clip 搜索在当前环境里显存/内存峰值较大；关闭 MSE clip 后，先保证 W4 weight scale search 和 pool generation 可用。

如果一定要打开，可以传：

```text
--awq_force_mse_clip
```

但不建议作为默认主线。

## 8. Output affine alignment 是否参与

这点很重要：

```text
官方 AWQ 路径的 W4A16 / W4A8 / W4A4 不走 output affine alignment。
```

代码中：

```python
def maybe_align_output_embeddings(...):
    if config_spec["kind"] != "fake_wa":
        return embs
```

因此：

```text
W4A8 / W4A4:
    kind = awq_act
    不做 output affine alignment。

W4A8_FAKE / W4A4_FAKE:
    kind = fake_wa
    默认可以做 output affine alignment。
```

这和之前 qiumingzhi 的 `AFF512` 思路不同。当前官方 AWQ branch 版的 `W4A8/W4A4` 质量提升主要来自官方 AWQ W4 weight search + 动态仿射 activation quant，而不是最后的 embedding affine alignment。

## 9. Embedding pool 如何保存

生成模型输出时使用：

```python
hidden = _forward_hidden_states(model, tokens)
embs = mean_pool(hidden, tokens["attention_mask"])
```

mean pooling：

```text
emb(v) = normalize(mean(last_hidden_state over valid tokens))
```

最后保存：

```python
torch.save(embs, out_path)
```

shape：

```text
ST:       (num_nodes, 768)
LLaMA-7B: (num_nodes, 4096)
```

例如 Cora：

```text
cache_data/cora_ST_oracle_W4A8_OFFICIAL_AWQ_BRANCH.pt | shape=(2708, 768)
cache_data/cora_ST_oracle_W4A4_OFFICIAL_AWQ_BRANCH.pt | shape=(2708, 768)
```

## 10. 生成命令示例

因为这个目录是一个单独副本，推荐用下面这种方式确保加载的是：

```text
/home/zhangshangtong/Transformer/OFA/GraphhopSimhash_reuse_real_quant_latest/GraphhopSimhash
```

而不是主目录下另一个 `GraphhopSimhash`。

### 10.1 Cora / ST / W4A8

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -u -c "
import sys, runpy
sys.path.insert(0, '/home/zhangshangtong/Transformer/OFA/GraphhopSimhash_reuse_real_quant_latest')
sys.path.insert(1, '/home/zhangshangtong/Transformer/OFA')
runpy.run_module('GraphhopSimhash.generate_real_quant_pools', run_name='__main__')
" \
  --datasets cora \
  --llm_name ST \
  --configs W4A8 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --output_path cache_data/cora_ST_oracle_W4A8_OFFICIAL_AWQ_BRANCH.pt \
  --overwrite
```

### 10.2 Cora / ST / W4A4

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -u -c "
import sys, runpy
sys.path.insert(0, '/home/zhangshangtong/Transformer/OFA/GraphhopSimhash_reuse_real_quant_latest')
sys.path.insert(1, '/home/zhangshangtong/Transformer/OFA')
runpy.run_module('GraphhopSimhash.generate_real_quant_pools', run_name='__main__')
" \
  --datasets cora \
  --llm_name ST \
  --configs W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --output_path cache_data/cora_ST_oracle_W4A4_OFFICIAL_AWQ_BRANCH.pt \
  --overwrite
```

### 10.3 同时生成默认命名的 FP16 / W4A8 / W4A4

如果允许使用默认文件名：

```bash
PYTHONPATH=/home/zhangshangtong/Transformer/OFA/GraphhopSimhash_reuse_real_quant_latest:/home/zhangshangtong/Transformer/OFA \
/home/zhangshangtong/.conda/envs/OFA/bin/python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs fp16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

会生成：

```text
cache_data/cora_ST_oracle_FP16.pt
cache_data/cora_ST_oracle_W4A8.pt
cache_data/cora_ST_oracle_W4A4.pt
```

### 10.4 LLaMA-7B 生成

LLaMA 会更慢，batch size 大于 4 时会自动降到 4：

```bash
PYTHONPATH=/home/zhangshangtong/Transformer/OFA/GraphhopSimhash_reuse_real_quant_latest:/home/zhangshangtong/Transformer/OFA \
/home/zhangshangtong/.conda/envs/OFA/bin/python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A8 W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --awq_q_group_size 128 \
  --overwrite
```

默认输出：

```text
cache_data/cora_llama2_7b_oracle_W4A8.pt
cache_data/cora_llama2_7b_oracle_W4A4.pt
```

## 11. 评测时如何引用

### 11.1 只评估真实量化 pool

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -u -c "
import sys, runpy
sys.path.insert(0, '/home/zhangshangtong/Transformer/OFA/GraphhopSimhash_reuse_real_quant_latest')
sys.path.insert(1, '/home/zhangshangtong/Transformer/OFA')
runpy.run_module('GraphhopSimhash', run_name='__main__')
" \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_OFFICIAL_AWQ_BRANCH \
  --real_quant_int4_tag W4A4_OFFICIAL_AWQ_BRANCH \
  --real_quant_fp_path cache_data/cora_ST_oracle_FP16.pt \
  --real_quant_int8_path cache_data/cora_ST_oracle_W4A8_OFFICIAL_AWQ_BRANCH.pt \
  --real_quant_int4_path cache_data/cora_ST_oracle_W4A4_OFFICIAL_AWQ_BRANCH.pt \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0
```

这张表里：

```text
UniformW4A8:
    全图都用 W4A8 embedding pool。

UniformW4A4:
    全图都用 W4A4 embedding pool。

RandomTopK / DegreeTopK / TSERTopK:
    20% 节点走 W4A8，其余走 W4A4。
```

### 11.2 Hash reuse + real quant 联合实验

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -u -c "
import sys, runpy
sys.path.insert(0, '/home/zhangshangtong/Transformer/OFA/GraphhopSimhash_reuse_real_quant_latest')
sys.path.insert(1, '/home/zhangshangtong/Transformer/OFA')
runpy.run_module('GraphhopSimhash', run_name='__main__')
" \
  --datasets cora \
  --runs 1 \
  --experiment_suite reuse_real_quant \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8 \
  --real_quant_int4_tag W4A4 \
  --real_quant_fp_path cache_data/cora_ST_oracle_FP16.pt \
  --real_quant_int8_path cache_data/cora_ST_oracle_W4A8_OFFICIAL_AWQ_BRANCH.pt \
  --real_quant_int4_path cache_data/cora_ST_oracle_W4A4_OFFICIAL_AWQ_BRANCH.pt \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --route_min_support_hits 3
```

这张表里：

```text
Reuse %:
    hash reuse 命中比例。

W4A4 / W4A8 / FP %:
    全图最终实际计算比例。
    注意：reuse hit 节点 cost=0，不会计入 FP/W4A8/W4A4。

AllFP:
    不是原始 baseline。
    它表示 reuse hit 节点复用缓存，hash miss 节点走 FP。
    因此 AllFP 也可能有 Drop，这个 Drop 是 reuse 本身带来的误差。
```

## 12. 和旧版本的区别

旧版 W4A8 / W4A4 有两条容易混淆的路线：

```text
1. W4A8_FAKE / W4A4_FAKE:
   本地 FakeQuantLinear。
   使用简单 AWQ-style scaling search。
   可配合 output affine alignment。

2. 之前主目录里的 PTQ / outlier PTQ:
   更复杂的 calibration-aware PTQ backend。
   支持 percentile clipping、outlier activation protection、output alignment 等。
```

当前这个 `test-access` branch 副本中的主线：

```text
W4A8 / W4A4:
    官方 llm-awq W4 weight search
    + pseudo_quantize_model_weight
    + dynamic affine activation fake quant
    + 保存 embedding pool
```

它的好处：

```text
1. 权重量化部分更接近官方 AWQ。
2. W4A4 的 ST embedding damage 明显比旧 naive fake quant 小。
3. 可以和 reuse_real_quant 联合，研究 hash reuse + mixed precision 的系统效果。
```

它的边界：

```text
1. A8/A4 activation 不是官方 AWQ 原生 kernel，而是 PyTorch fake quant 仿真。
2. 保存的是 embedding pool，不是可部署的 packed quantized model。
3. W4A4 对 LLaMA-7B 仍然可能比较脆弱，需要单独看 DamageCheck / AllW4A4 drop。
4. 当前官方 AWQ 路径不做 output affine alignment；如果看到 AFF512 风格结果，要确认是否来自旧 fake_wa 或其他 branch。
5. 对 BERT/e5 等 encoder，如果没有 adapter，不能默认认为官方 AWQ 路径可用。
```

## 13. 已验证的 Cora / ST 结果

使用本文件上面的 `OFFICIAL_AWQ_BRANCH` pool 做 10-run real quant ablation，得到：

```text
Baseline Acc: 0.6819

UniformW4A8_OFFICIAL_AWQ_BRANCH:
    Acc=0.6651
    Drop=1.67%
    AvgErr=0.02347

UniformW4A4_OFFICIAL_AWQ_BRANCH:
    Acc=0.6258
    Drop=5.61%
    AvgErr=0.12204

TSERErrorTopK_W4A8_OFFICIAL_AWQ_BRANCH:
    Acc=0.6487
    Drop=3.31%
    AvgErr=0.08286
```

这说明：

```text
1. 当前 official-AWQ-branch 的 ST W4A4 pool 已经比旧 naive W4A4 更可用。
2. W4A8 仍然是更安全的低精度路径。
3. W4A4 仍然是 aggressive path，适合配合路由策略，而不是无脑全图使用。
```

