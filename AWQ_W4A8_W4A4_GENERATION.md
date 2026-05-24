# AWQ W4A8 / W4A4 Embedding Pool 生成与效果解释

本文档说明当前 `GraphhopSimhash` 主目录里的 AWQ-based W4A8 / W4A4 embedding pool 是如何生成的，以及为什么这一版的 W4A4 在 ST/Cora 上比旧方案明显更稳。

核心结论先放前面：

```text
当前 W4A4 效果变好，不是因为最后做了 FP embedding affine 对齐，
也不是因为路由时偷看了每个节点的真实量化误差。

根本原因是量化路径换了：

1. 权重量化从本地 fake/PTQ wrapper 换成官方 llm-awq 的 activation-aware W4 search。
2. calibration 数据从通用/随意样本变成当前图数据集的 node text。
3. activation 量化从旧的粗糙对称量化换成每次 forward 动态仿射量化。
4. A4 路径可以额外使用 activation outlier channel protection，避免少数异常通道把 A4 scale 拉坏。
```

因此当前版本的 `W4A4` 更像是：

```text
AWQ-W4 weights + dynamic affine A4 activations + graph-text calibration
```

而不是过去的：

```text
普通 fake W4/A4 wrapper
或
后处理 embedding affine alignment
或
逐节点真实误差 oracle routing
```

## 1. 当前配置含义

实现位置：

```text
GraphhopSimhash/generate_real_quant_pools.py
GraphhopSimhash/activation_outlier_calibration.py
GraphhopSimhash/real_quant.py
GraphhopSimhash/runner.py
GraphhopSimhash/third_party/llm-awq/
```

当前主线配置在 `generate_real_quant_pools.py` 中：

```python
CONFIG_SPECS = {
    "fp16": {"tag": "FP16", "kind": "bnb", "w_bit": 16, "a_bit": 16},
    "W4A16": {"tag": "W4A16", "kind": "awq", "w_bit": 4, "a_bit": 16},
    "W4A8": {"tag": "W4A8", "kind": "awq_act", "w_bit": 4, "a_bit": 8},
    "W4A4": {"tag": "W4A4", "kind": "awq_act", "w_bit": 4, "a_bit": 4},
    "W4A16_FAKE": {"tag": "W4A16_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 16},
    "W4A8_FAKE": {"tag": "W4A8_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 8},
    "W4A4_FAKE": {"tag": "W4A4_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 4},
}
```

推荐主线只看：

```text
FP16:
    全精度 reference embedding pool。

W4A16:
    官方 llm-awq W4 weight-only，activation 保持 FP16。
    它用于单独观察 AWQ 权重量化本身的损伤。

W4A8:
    官方 llm-awq W4 weight + 动态 A8 activation fake quant。
    它是较安全的低成本路径。

W4A4:
    官方 llm-awq W4 weight + 动态 A4 activation fake quant。
    它是最激进的低成本路径。
```

旧配置：

```text
W4A16_FAKE / W4A8_FAKE / W4A4_FAKE:
    旧版本地 FakeQuantLinear 路径。
    只适合历史对照或 debug，不建议作为当前论文主线。
```

## 2. 当前生成流程

生成 `W4A8` / `W4A4` embedding pool 时，流程是：

```text
1. 加载 FP16 模型和 tokenizer。
2. 读取当前图数据集的 node text。
3. 用 node text 构造 AWQ calibration blocks。
4. 调用 third_party/llm-awq 官方 AWQ search。
5. 保存或复用 AWQ search 结果。
6. 应用 AWQ scale/clip，并把 Linear 权重 pseudo-quantize 到 W4。
7. 如果是 W4A8/W4A4，再给每个 Linear 输入套 activation fake quant wrapper。
8. 对全图 node text encode。
9. mean pooling + L2 normalize 得到 node embedding。
10. 保存 embedding pool 到 cache_data。
```

默认保存路径：

```text
cache_data/{dataset}_{llm_name}_oracle_{tag}.pt
```

例如：

```text
cache_data/cora_ST_oracle_W4A16.pt
cache_data/cora_ST_oracle_W4A8.pt
cache_data/cora_ST_oracle_W4A4.pt

cache_data/pubmed_llama2_7b_oracle_W4A16.pt
cache_data/pubmed_llama2_7b_oracle_W4A8.pt
cache_data/pubmed_llama2_7b_oracle_W4A4.pt
```

保存的是全图 embedding tensor，不是 packed int4 模型 checkpoint。

## 3. 为什么当前 W4A4 效果明显更好

### 3.1 权重不是普通 int4，而是 activation-aware AWQ W4

旧方案里，W4 权重量化主要是本地 wrapper 或自写 PTQ：

```text
weight -> 直接量化/反量化
activation -> 同一套粗糙 fake quant
```

这种做法的问题是：每一层里不同输入通道的重要性不一样，直接 int4 会把重要通道和不重要通道一视同仁。对于 embedding 任务，少数关键通道的方向误差会被 mean pooling 和 normalize 继续传播，最后表现为 cosine drift。

当前版本调用官方 `llm-awq`：

```python
from awq.quantize.pre_quant import apply_awq, run_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight
```

核心调用：

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

AWQ 的关键不是“把权重压成 4bit”这么简单，而是先看 calibration activation，估计哪些通道更敏感，再搜索 scale/clip，把重要通道的量化误差压低。

这就是当前 W4A4 比旧 fake W4A4 稳定的第一层原因。

### 3.2 Calibration 数据变成图节点文本

当前版本不是拿通用语料或随便的样本做 calibration，而是把当前数据集的 node text 注入官方 AWQ calibration pipeline。

实现位置：

```python
build_local_awq_calib_getter(texts)
patch_awq_calibration_data(texts)
```

也就是说，Cora 的 pool 用 Cora node text 校准，PubMed 的 pool 用 PubMed node text 校准。这样 AWQ search 看到的 activation 分布更接近最终生成 embedding 时的真实分布。

这点对图文本 embedding 很重要，因为 node text 通常短、领域集中、模板性强。用图节点文本做 calibration，比用通用 LM 语料更贴近后续 workload。

### 3.3 Activation 量化从对称量化换成动态仿射量化

旧 fake 路径里，activation 更接近对称量化：

```text
scale = absmax / qmax
zero_point = 0
```

这对 LayerNorm / GELU / attention block 后的 activation 分布不友好，因为 activation 往往不是严格零中心、也不是正负范围对称。

当前 W4A8/W4A4 使用：

```python
ActivationQuantLinear(original_linear, a_bit)
```

其 forward 逻辑是：

```python
qx = affine_fake_quantize(x, a_bit, mode="per_channel", dim=-1)
return original_linear(qx)
```

实际公式是非对称 min-max 仿射量化：

```text
q_min = 0
q_max = 2^a_bit - 1

scale = (max(x) - min(x)) / q_max
zero_point = round(q_min - min(x) / scale)

x_q  = clamp(round(x / scale + zero_point), q_min, q_max)
x_dq = (x_q - zero_point) * scale
```

对输入形状 `[batch, seq_len, hidden_dim]` 来说，`dim=-1` 会对每个 token row 动态估计 min/max。直觉上，它比全局固定 scale 更能适应每个节点文本、每个 token 的 activation 范围。

这就是当前 A4 还能保住 embedding 方向的第二层原因。

### 3.4 A4 可以保护 activation outlier channels

当前代码还支持一个专门给 W4A4 用的 outlier channel protection：

```text
GraphhopSimhash/activation_outlier_calibration.py
```

先跑 calibration，收集每个 Linear 输入 activation 的异常通道：

```bash
python -m GraphhopSimhash.activation_outlier_calibration \
  --dataset cora \
  --llm_name ST \
  --calib_samples 128 \
  --batch_size 64 \
  --max_length 128 \
  --seed 42
```

生成 W4A4 时打开：

```bash
--activation_outlier_clip
```

默认模式是：

```text
activation_outlier_mode = channel_protect
activation_outlier_channel_a_bit = 8
activation_outlier_apply_max_a_bit = 4
```

含义是：

```text
普通 activation 通道:
    A4 动态仿射量化。

少数 outlier activation 通道:
    不用 A4，改用 A8 保护。
```

它不是把所有 activation 都升到 A8，而是只保护每层少数最危险的通道。这样对成本影响小，但能显著避免 A4 scale 被异常值拉坏。

这一项对 LLaMA / PubMed / Arxiv 这种更大模型或更复杂文本分布通常更重要；Cora/ST 上即使不打开，也可能已经比旧方案好很多。

### 3.5 Embedding 任务本身对方向误差更敏感，也更容易被 AWQ 救回来

当前保存的是 normalized embedding：

```python
embs = mean_pool(hidden, attention_mask)
```

mean pooling 内部会做：

```text
mean(last_hidden_state over valid tokens)
L2 normalize
```

所以最终评测更看重 embedding 方向是否保持，而不是每一维数值完全一致。

如果 AWQ + dynamic activation quant 能保住主要语义方向，GNN 下游 accuracy 就可能只掉几个点。Cora/ST 当前看到的 W4A4 drop 明显降低，主要就是 embedding cosine direction 被保住了。

## 4. 和旧方案的根本区别

| 方案 | 做法 | 根本问题 | 是否推荐主线 |
|---|---|---|---|
| `W4A4_FAKE` 旧 fake wrapper | 本地 `FakeQuantLinear`，近似 AWQ，但不是官方 AWQ search | calibration 弱，activation 量化粗糙，容易放大 embedding drift | 不推荐 |
| 旧 custom PTQ / outlier PTQ | 自写 smooth/clip/grid/outlier ratio | 能调，但稳定性和可解释性弱，LLaMA W4A4 仍容易崩 | 可做历史对照 |
| `AFF512` embedding affine alignment | 用少量 FP embedding 拟合逐维 `gamma/beta` 修正输出 | 需要 FP reference calibration embedding；它是输出后处理，不是模型内部量化变好 | 只做消融 |
| 当前 `W4A4` | 官方 AWQ W4 + graph-text calibration + dynamic affine A4 activation | 仍是 fake quant embedding pool，不是 packed kernel | 推荐作为当前 AWQ 主线 |

最关键的分界线是：

```text
旧方案主要在“量化后补救”或“粗粒度假量化”。
当前方案是在“量化过程中”用 activation-aware search 和动态 activation scale 降低误差。
```

所以当前 W4A4 的改善不是偶然调参，而是量化机制本身换了。

## 5. 为什么不是 output affine alignment 的功劳

当前官方 AWQ 路径不会做 output affine alignment。

代码中：

```python
def maybe_align_output_embeddings(...):
    if config_spec["kind"] != "fake_wa":
        return embs
```

因此：

```text
W4A16:
    kind = awq
    不做 output affine alignment。

W4A8 / W4A4:
    kind = awq_act
    不做 output affine alignment。

W4A8_FAKE / W4A4_FAKE:
    kind = fake_wa
    才可能做 output affine alignment。
```

这点很重要。当前 `W4A4` 结果好，不是因为它偷看全图 FP embedding 做了输出修正。它的提升来自模型内部的 AWQ 权重量化和 activation 处理。

## 6. 为什么不是 error-aware oracle routing 的功劳

当前主表已经去掉了这几类策略：

```text
QuantTSERTopK_W4A8
DegreeErrorTopK_W4A8
TSERErrorTopK_W4A8
```

尤其 `DegreeErrorTopK` / `TSERErrorTopK` 会用：

```python
compute_real_quant_errors(fp_embs, int8_embs, int4_embs, args)
```

这等价于已经知道每个节点的 FP embedding 和 quant embedding 的差距，只能作为 oracle/debug，不能作为 deployable 量化路由。

当前推荐主表只保留：

```text
AllFP
UniformW4A8
UniformW4A4
RandomTopK_W4A8
DegreeTopK_W4A8
TSERTopK_W4A8
```

这样表里的路由策略不依赖逐节点真实量化误差。

## 7. 生成命令

### 7.1 Cora / ST

如果只需要默认 AWQ W4A16/W4A8/W4A4：

```bash
cd /home/zhangshangtong/Transformer/OFA

python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

如果要启用 A4 outlier channel protection，先生成 outlier report：

```bash
python -m GraphhopSimhash.activation_outlier_calibration \
  --dataset cora \
  --llm_name ST \
  --calib_samples 128 \
  --batch_size 64 \
  --max_length 128 \
  --seed 42
```

再生成：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --activation_outlier_clip \
  --overwrite
```

### 7.2 PubMed / ST

```bash
cd /home/zhangshangtong/Transformer/OFA

python -m GraphhopSimhash.activation_outlier_calibration \
  --dataset pubmed \
  --llm_name ST \
  --calib_samples 128 \
  --batch_size 64 \
  --max_length 128 \
  --seed 42

python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets pubmed \
  --llm_name ST \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --activation_outlier_clip \
  --overwrite
```

### 7.3 Cora / LLaMA-7B

LLaMA 生成会慢很多，batch size 建议保持 4。

```bash
cd /home/zhangshangtong/Transformer/OFA

python -m GraphhopSimhash.activation_outlier_calibration \
  --dataset cora \
  --llm_name llama2_7b \
  --calib_samples 128 \
  --batch_size 4 \
  --max_length 128 \
  --seed 42

python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --activation_outlier_clip \
  --overwrite
```

### 7.4 PubMed / LLaMA-7B

```bash
cd /home/zhangshangtong/Transformer/OFA

python -m GraphhopSimhash.activation_outlier_calibration \
  --dataset pubmed \
  --llm_name llama2_7b \
  --calib_samples 128 \
  --batch_size 4 \
  --max_length 128 \
  --seed 42

python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets pubmed \
  --llm_name llama2_7b \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --activation_outlier_clip \
  --overwrite
```

## 8. 评测命令

以 Cora / ST 为例：

```bash
cd /home/zhangshangtong/Transformer/OFA

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

当前主表含义：

```text
AllFP:
    全部节点使用 W4A16/FP-like reference 路径。

UniformW4A8:
    全图节点都使用 W4A8 embedding。

UniformW4A4:
    全图节点都使用 W4A4 embedding。

RandomTopK_W4A8:
    20% 节点随机走 W4A8，其余走 W4A4。

DegreeTopK_W4A8:
    20% 高传播风险节点走 W4A8，其余走 W4A4。

TSERTopK_W4A8:
    20% 高 TSER 图风险节点走 W4A8，其余走 W4A4。
```

## 9. 如何解读当前好结果

如果看到类似：

```text
UniformW4A8  Drop 很小
UniformW4A4  Drop 从旧版十几个点降到几个点
```

合理解释是：

```text
1. W4 权重已经由 AWQ 搜索保护了关键通道。
2. A4 activation 使用动态非对称 scale，避免旧对称量化的大范围失配。
3. calibration 来自图节点文本，activation 分布和真实 workload 对齐。
4. mean-pooled normalized embedding 对小幅数值误差有一定鲁棒性，只要方向保持即可。
```

但也要注意：

```text
Cora/ST 上 W4A4 好，不代表 LLaMA/PubMed/Arxiv 一定同样好。
LLaMA-7B 的 hidden dim 更大、层数更多、activation outlier 更强，W4A4 仍可能明显掉点。
```

所以论文里更稳的表述应该是：

```text
官方 AWQ + graph-text calibration 显著改善了 W4A4 embedding pool 的可用性；
对于小型 sentence encoder，它可以让全图 W4A4 从不可用变成可作为激进低成本路径；
对于 LLaMA 级模型，W4A8 仍是更稳主线，W4A4 更适合作为激进或局部路径。
```

## 10. 当前方案的边界

当前 pool generation 仍然是仿真实验路径：

```text
1. 保存的是 embedding pool，不是硬件 packed int4/A4 kernel 输出。
2. AWQ 权重是 pseudo quantize 后的浮点权重，用于生成 embedding。
3. activation quant 是 fake quant wrapper，用来模拟 A8/A4 对 embedding 的影响。
4. activation outlier protection 会让少数通道使用更高 activation bit，相当于 mixed-channel activation precision。
```

因此它适合回答论文中的问题：

```text
如果 embedding generator 支持 AWQ-style W4 weights 和 A8/A4 activation，
图节点 embedding 的精度/成本 trade-off 能做到什么程度？
```

但它还不是最终硬件实现。硬件章节里需要把它映射成：

```text
W4 weight storage
A8/A4 activation datapath
少数 outlier channel A8 bypass/protection
embedding cache / reuse path
```

## 11. 一句话总结

当前 W4A4 效果好的根本原因是：

```text
量化误差被前移到模型内部处理：
AWQ 保护权重关键通道，动态仿射量化适配每次 activation 分布，
graph-text calibration 让搜索目标贴近真实节点文本。
```

而旧方案更多是在：

```text
粗糙假量化之后再补救，或者依赖 FP reference 做输出对齐。
```

这就是当前版本和先前方案的本质差别。
