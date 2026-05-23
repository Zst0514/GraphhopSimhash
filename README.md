# GraphHop SimHash

`GraphhopSimhash` 是一个围绕图节点 embedding 前端计算优化的实验框架。当前主线包含两层：

```text
1. GraphHop SimHash reuse:
   用图感知 SimHash / CAM 检索跳过冗余节点的前端 embedding 计算。

2. W4A8/W4A4 fixed-budget routing:
   在已生成的 FP16 / W4A8 / W4A4 embedding pools 上，
   比较 Random / Degree / TSER / GraphHopSafe 等节点路径选择策略。
```

核心目标不是证明“全图 W4A4 可用”，而是利用图结构和 cheap hash/statistics 找到低风险节点，让它们走更便宜的 W4A4 aggressive path，高风险节点保留在 W4A8 safe path。

## 1. 当前主线

### 1.1 GraphHop SimHash Reuse

GraphHop reuse 用 cheap feature 构造 self/neighbor/hash context，再通过多 head SimHash 找候选复用节点。

常见配置：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --radius 2 \
  --hash_heads_per_route 4 \
  --main_hash_head_bits 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor
```

打开 TSER score gate：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --radius 2 \
  --hash_heads_per_route 4 \
  --main_hash_head_bits 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate
```

注意：score gate 默认关闭，需要显式加 `--enable_score_gate`。

### 1.2 Fixed-Budget W4A8/W4A4 Routing

当前真实量化主表使用：

```text
--experiment_suite real_quant_ablation
--real_quant_policy_suite fixed_aggressive_budget
```

其中：

```text
W4A8 = safe low-precision path
W4A4 = aggressive low-precision path
```

`--real_quant_int8_ratio` 表示 W4A8 节点比例。例如：

```text
--real_quant_int8_ratio 0.80
```

表示：

```text
80% W4A8
20% W4A4
```

主表策略包括：

```text
AllW4A8
AllW4A4
RandomBudget
DegreeBudget
TSERBudget
GraphHopSafeBudget
```

其中 `GraphHopSafeBudget` 是当前推荐的 deployable graph/hash stability routing。它不依赖逐节点真实量化误差，而是用 hash bucket density、multi-head agreement、self/context consistency、low-propagation safety、non-rare-tail safety 判断哪些节点适合走 W4A4。

## 2. TSER 分数

TSER 的综合风险分数为：

```text
sensitivity_q =
    3 * propagation_q
  + 1 * graph_context_q
  + 1 * low_degree_unique_q
```

含义：

```text
propagation_q:
    degree / 传播影响风险。

graph_context_q:
    图上下文或语义边界风险。

low_degree_unique_q:
    低度但语义/hash 稀有节点保护。
```

当前默认权重是 `3/1/1`，比旧的 `3/2/2` 更简洁。已有实验显示：

```text
Cora / LLaMA-7B:
    3/1/1 的 TSERBudget 可能略优于 Degree/Random。

Cora / ST:
    Degree 往往更稳，low-degree unique 可能伤精度。

PubMed / ST 和 PubMed / LLaMA-7B:
    DegreeBudget 通常最好，说明 PubMed 更传播主导。
```

因此论文表述应避免写成“TSER 总是优于 Degree”。更稳的说法是：图相关路由是必要的，但不同数据集/backend 下，传播风险和图语义修正的重要性不同。

## 3. 项目结构

```text
cli.py
    命令行参数与参数校验。

runner.py
    实验调度、baseline 训练、reuse/quant ablation 评估。

controller.py
    SimHash cache、多 route 检索、structure check、score gate。

scoring.py
    TSER reuse score：degree/context/rare-tail sensitivity。

real_quant.py
    真实 FP/W4A8/W4A4 embedding pool 的固定预算路由评估。

generate_real_quant_pools.py
    生成 FP16 / W4A8 / W4A4 embedding pools；包含 PTQ / outlier backend。

features.py
    self/1-hop/2-hop cheap feature 和 neighbor mean 构造。

projections.py
    raw / learned multi-head hash projections。

data.py
    OFA 数据加载与 cheap feature 加载。

models.py
    下游 GNN wrapper。

SCORE_DEFINITIONS.md
    TSER reuse score 与 TSER-Q routing score 的详细定义。

PTQ_EMBEDDING_QUANT.md
    W4A8/W4A4 embedding PTQ 生成流程、命令和实验解释。
```

## 4. 生成 Embedding Pools

生成的 pool 默认保存到：

```text
cache_data/{dataset}_{model}_oracle_{tag}.pt
```

### 4.1 ST / Arxiv FP16

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets arxiv \
  --llm_name ST \
  --configs fp16 \
  --batch_size 128 \
  --overwrite
```

### 4.2 ST / PubMed + Arxiv W4A4

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

### 4.3 LLaMA-7B / PubMed W4A8

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

### 4.4 LLaMA-7B / PubMed W4A4 Outlier Backend

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

### 4.5 LLaMA-7B / Arxiv FP16

Arxiv + LLaMA-7B 很慢，建议先只生成 FP16：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets arxiv \
  --llm_name llama2_7b \
  --configs fp16 \
  --batch_size 4 \
  --overwrite
```

## 5. 评估命令

### 5.1 Cora / LLaMA-7B Fixed Budget

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

### 5.2 Cora / ST Fixed Budget

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite fixed_aggressive_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_PTQ_TEST \
  --real_quant_int4_tag W4A4_PTQ_TEST2 \
  --real_quant_error_norm 1.0 \
  --real_quant_int8_ratio 0.80 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

### 5.3 PubMed / LLaMA-7B Fixed Budget

```bash
python -m GraphhopSimhash \
  --datasets pubmed \
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

### 5.4 Arxiv / ST Fixed Budget

```bash
python -m GraphhopSimhash \
  --datasets arxiv \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite fixed_aggressive_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_PTQ_TEST \
  --real_quant_int4_tag W4A4_PTQ_TEST2 \
  --real_quant_error_norm 1.0 \
  --real_quant_int8_ratio 0.80 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

## 6. 结果解读原则

### 6.1 AllW4A4 掉点大是合理现象

W4A4 是 aggressive path。当前观察到：

```text
ST / Cora:
    AllW4A4 drop ~= 14 points

LLaMA-7B / Cora:
    AllW4A4 drop ~= 20+ points

LLaMA-7B / PubMed:
    AllW4A4 drop ~= 15 points
```

这说明 W4A4 不能作为全图统一精度使用，应该只给少量低风险节点。

### 6.2 固定预算才是主表

建议报告：

```text
0% W4A4 + 100% W4A8
10% W4A4 + 90% W4A8
20% W4A4 + 80% W4A8
30% W4A4 + 70% W4A8
```

并在相同 W4A4 budget 下比较：

```text
RandomBudget
DegreeBudget
TSERBudget
GraphHopSafeBudget
```

### 6.3 不使用逐节点真实量化误差作为主策略

早期的 `ErrorBudget / TSERQBudget / Calib*Budget` 可以作为 oracle 或历史消融，但当前主线不把逐节点真实量化误差作为核心路由依据。

原因是：全图精确 error 需要先生成 FP/W4A8/W4A4 三套 embedding，再逐节点比较，部署成本过高。

## 7. 文档索引

```text
SCORE_DEFINITIONS.md
    TSER reuse score、TSER-Q routing score、3/1/1 权重和 GraphHopSafeBudget 定义。

PTQ_EMBEDDING_QUANT.md
    PTQ embedding pool 生成流程、LLaMA outlier backend、推荐命令和实验解释。

REAL_QUANT.md
    较早的真实量化 pool 消融说明，部分内容偏历史版本。
```

## 8. 限制

当前实现是实验框架，不是生产级 INT4/NPU kernel：

```text
1. W4A8/W4A4 通过 quantize-dequantize tensor 模拟低 bit 数值效果。
2. INT4 weight 尚未 pack 到定制 CUDA/NPU kernel。
3. PTQ backend 不是完整工业级 AWQ/OmniQuant。
4. GraphHopSafeBudget 是 deployable routing proxy，但仍需硬件 pipeline 建模。
```

对论文来说，当前最稳的贡献链条是：

```text
GraphHop SimHash reuse
+ TSER reuse gate
+ fixed outlier-preserved low-precision backend
+ graph/hash stability routing
+ NDP/NPU/CAM routing pipeline
```
