# Graph-Aware Adaptive Encoder NPU 实验设计

本文档设计一套用于验证 **graph-aware adaptive Transformer encoding** 的实验。目标不是再证明 GNN 后端加速，而是验证：

```text
graph 后端风险 proxy 是否能预测 LLM encoder 少算带来的 downstream damage，
并据此驱动 NPU 在不同低成本执行模式之间切换。
```

核心结论要服务硬件设计：

```text
Graph risk -> choose how much LLM encoder compute to spend
```

而不是：

```text
Graph risk -> accelerate GNN aggregation
```

---

## 1. 总体假设

不同低成本执行模式破坏的信息类型不同，因此不能用一个统一 TSER 总分控制所有模式。

建议拆成 mode-specific proxy：

| Encoder 减算模式 | 主要破坏 | 主要 proxy | 原因 |
|---|---|---|---|
| Partial-depth | 整体 embedding 成熟度 | propagation / degree | embedding 偏差会被 GNN 沿边传播，高度节点影响范围更大 |
| Token pruning | 文本信息完整性 | graph_context | graph context 可靠时，少看 token 更容易被邻居上下文补偿 |
| FFN channel gating | 细粒度语义通道 | semantic rarity + confidence | 稀有/低置信节点更依赖细粒度语义特征 |

最终希望证明：

```text
mode-specific proxy > single TSER score > degree-only/random
```

---

## 2. 实验阶段划分

### Phase A: Partial-Depth Encoder

这是第一优先级，因为它最直接减少 Transformer 层数，也是当前代码最接近可跑的路径。

模式：

```text
Full W4A8
L4
L8
L16
```

对 ST 可替换成：

```text
Full
L1
L2
L3
```

要回答的问题：

```text
degree / propagation_q 是否能预测 L4/L8/L16 相对 Full 的 downstream damage？
```

### Phase B: Token-Pruned Encoder

模式：

```text
S128
S256
S512
```

其中 `S` 是 max sequence length。要验证：

```text
graph_context_q 是否比 degree 更能预测 token pruning damage？
```

### Phase C: FFN-Channel Gated Encoder

模式：

```text
r25
r50
r75
r100
```

其中 `r` 是保留 FFN hidden channels 的比例。要验证：

```text
low_degree_unique_q / cheap margin 是否能预测 channel gating damage？
```

---

## 3. Damage 定义

对每个节点 `v` 和低成本模式 `m`，先生成：

```text
E_full(v)
E_m(v)
```

然后计算三类 damage。

### 3.1 Embedding Damage

```text
emb_damage(v, m) =
    1 - cosine(E_m(v), E_full(v))
```

它衡量 embedding 本身偏差。

### 3.2 Logit / Margin Damage

先把 embedding 喂给同一个 GNN classifier：

```text
logits_full(v) = GNN(E_full)
logits_m(v)    = GNN(E_m)
```

定义：

```text
margin_drop(v, m) =
    margin_full(v) - margin_m(v)
```

其中：

```text
margin(v) = top1_logit(v) - top2_logit(v)
```

它比 embedding cosine 更接近任务目标。

### 3.3 Prediction Flip

```text
flip(v, m) =
    argmax(logits_m(v)) != argmax(logits_full(v))
```

这是最直接的 downstream damage。

---

## 4. Proxy 验证

对每个模式分别计算 proxy 与 damage 的关系。

候选 proxy：

```text
propagation_q
graph_context_q
low_degree_unique_q
sensitivity_q = a*prop + b*ctx + c*low_unique
cheap_margin_risk
```

验证指标：

```text
Spearman correlation(proxy, emb_damage)
Spearman correlation(proxy, margin_drop)
AUC(proxy predicts flip)
Top-risk bucket mean damage
```

期望看到：

```text
Partial-depth:
    propagation_q / degree 最强

Token pruning:
    graph_context_q 更强

FFN gating:
    low_degree_unique_q + margin_risk 更强
```

如果这个结论不成立，就不能直接把该 proxy 写成硬件路由依据。

---

## 5. Router 设计

### 5.1 Baselines

必须比较：

```text
Full W4A8
AllLite
RandomBudget
DegreeBudget
TSERBudget
OracleDamageBudget
```

其中 `OracleDamageBudget` 只能作为 upper bound，不能作为可部署策略。

### 5.2 Mode-Specific Router

不要一个阈值控制所有模式，而是按模式定义风险：

```text
depth_risk(v) =
    a1 * propagation_q(v)
  + a2 * margin_risk(v)

token_risk(v) =
    b1 * graph_context_q(v)
  + b2 * low_degree_unique_q(v)
  + b3 * margin_risk(v)

ffn_risk(v) =
    c1 * low_degree_unique_q(v)
  + c2 * margin_risk(v)
  + c3 * propagation_q(v)
```

然后选择最低成本且 risk 合格的模式：

```text
try cheapest mode
if risk too high -> next deeper / longer / wider mode
if still high    -> Full W4A8
```

---

## 6. Cost Model

对硬件 NPU，用统一 cost model 估算每个模式成本。

```text
depth_ratio = K / L
token_ratio = S' / S
ffn_ratio   = r
```

基础公式：

```text
Cost(mode) =
    depth_ratio *
    [ attn_weight * token_ratio^2
    + ffn_weight  * token_ratio * ffn_ratio ] *
    W4A8_cost_scale
```

初始可用：

```text
W4A8_cost_scale = 0.5
attn_weight = 0.35
ffn_weight  = 0.65
```

后续可以用 profiling 替换这两个权重。

硬件统计还应报告：

```text
MAC reduction
weight traffic
activation traffic
padding ratio
array utilization
```

---

## 7. 最小可行实验

先只做 Phase A：Partial-depth。

### 7.1 生成 Partial Pools

Cora / LLaMA-7B：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A8 \
  --batch_size 4 \
  --w4a_backend awq \
  --w4a_calib_samples 128 \
  --calibration_strategy random \
  --seed 42 \
  --awq_group_size 128 \
  --awq_parallel_calib_samples 128 \
  --partial_layers 4 8 16 \
  --overwrite
```

预期生成：

```text
cache_data/cora_llama2_7b_oracle_W4A8_L4.pt
cache_data/cora_llama2_7b_oracle_W4A8_L8.pt
cache_data/cora_llama2_7b_oracle_W4A8_L16.pt
```

### 7.2 跑 Partial Encoder Routing

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite partial_encoder \
  --real_quant_model_name llama2_7b \
  --partial_encoder_reference_tag W4A16 \
  --partial_encoder_full_tag W4A8 \
  --partial_encoder_partial_tag W4A8 \
  --partial_encoder_layers 4 8 16 \
  --partial_encoder_full_ratio 0.20 \
  --partial_encoder_deep_ratio 0.30 \
  --partial_encoder_mid_ratio 0.30
```

当前代码会比较：

```text
FullW4A8
AllL4 / AllL8 / AllL16
RandomCascade
DegreeCascade
TSERCascade
EarlyExitBudget
```

### 7.3 需要补的统计

为了完成 proxy 验证，还需要增加一个 `partial_damage_analysis` 输出：

```text
每个节点:
    propagation_q
    graph_context_q
    low_degree_unique_q
    margin_full
    damage_L4 / damage_L8 / damage_L16
    flip_L4 / flip_L8 / flip_L16
```

保存为：

```text
output/partial_encoder_damage/{dataset}_{model}_damage.tsv
```

再汇总：

```text
output/partial_encoder_damage/{dataset}_{model}_correlation.tsv
```

---

## 8. 论文图表设计

### Table 1: Proxy-Damage Correlation

| Mode | Best Proxy | Spearman EmbErr | Spearman MarginDrop | Flip AUC |
|---|---|---:|---:|---:|
| L4 | propagation_q | | | |
| L8 | propagation_q | | | |
| S128 | graph_context_q | | | |
| r50 | low_unique + margin | | | |

### Figure 1: Cost-Accuracy Curve

横轴：

```text
normalized encoder cost
```

纵轴：

```text
accuracy / drop
```

曲线：

```text
Random
Degree
TSER total
Mode-specific router
Oracle
```

### Figure 2: Hardware Execution Mix

显示每个策略下节点进入各路径的比例：

```text
S128 / S256 / Full
或
L4 / L8 / L16 / Full
```

### Figure 3: NPU Cost Breakdown

```text
Attention MAC
FFN MAC
Weight traffic
Activation traffic
```

---

## 9. 当前最小验证结果

### 9.0 Token/chunk compaction 的第一轮结论

当前已经实现 `token_compaction` 实验，用来回答：

```text
如果每个节点都只给 128-token budget，应该选哪些 token/chunks?
```

运行 Cora/LLaMA-7B：

```bash
bash GraphhopSimhash/run_cora_llama_token_compaction.sh
```

运行 PubMed/ST：

```bash
bash GraphhopSimhash/run_pubmed_st_token_compaction.sh
```

Cora/LLaMA-7B 结果：

```text
Baseline Acc: 0.7308

Config            Cost   Acc     Drop    AvgErr
FullW4A8          0.500  0.7340 -0.32%  0.00265
Prefix128         0.092  0.7260  0.48%  0.01451
Random128         0.092  0.7102  2.06%  0.04323
TFIDF128          0.092  0.7063  2.45%  0.04924
GraphContext128   0.092  0.7237  0.71%  0.04064
```

PubMed/ST 结果：

```text
Baseline Acc: 0.7710

Config                  Cost   Acc     Drop    AvgErr
FullW4A8                0.500  0.7711 -0.02%  0.00034
Prefix128               0.092  0.7729 -0.19%  0.02407
Random128               0.092  0.7129  5.81%  0.07392
TFIDF128                0.092  0.6757  9.53%  0.09203
GraphContext128         0.092  0.7182  5.28%  0.06113
HeadTail128             0.092  0.7639  0.71%  0.02793
PrefixTFIDF128          0.092  0.7513  1.97%  0.02655
PrefixGraphContext128   0.092  0.7588  1.22%  0.02583
```

结论：

```text
1. 当前数据格式下，title / abstract 前部信息非常强，Prefix128 是强 baseline。
2. 朴素 TF-IDF 或 graph-context chunk 重排会破坏 title/front-loaded 信息，效果明显变差。
3. 即使强制保留 prefix，再用 TF-IDF/GraphContext 补充尾部 chunk，也没有超过 Prefix128。
4. 因此 Step 1 暂时不支持“手写 graph-guided chunk selection 优于 prefix”。
```

这不否定 token-budget lite encoder。它说明更有希望的方向是：

```text
node-level budget routing:
    哪些节点用 S128/S256/Full

learned damage predictor:
    用少量 calibration 学短序列损伤，而不是手写 chunk scorer
```

### 9.1 Naive partial-depth 不够

Cora/LLaMA-7B 上直接取中间层 hidden state 做 mean-pooling，效果很差：

```text
FullW4A8        Cost=0.500 | Drop=-0.32%
AllL4           Cost=0.062 | Drop=58.06%
AllL8           Cost=0.125 | Drop=56.61%
AllL16          Cost=0.250 | Drop=38.88%
DegreeCascade   Cost=0.225 | Drop=27.56%
TSERCascade     Cost=0.225 | Drop=30.53%
EarlyExitBudget Cost=0.225 | Drop=24.11%
```

结论：early-depth path 不能直接把第 K 层当 final embedding，需要 projection/alignment 或重新训练 early-exit head。

### 9.2 Graph-Eager token budget 更有希望

当前已实现 `graph_eager_token` 实验：预生成不同 token 长度的 W4A8 embedding pool，然后用图风险 proxy 决定哪些节点走短序列、哪些节点走完整序列。

生成 Cora/LLaMA-7B token pool：

```bash
bash GraphhopSimhash/run_cora_llama_graph_eager_token.sh
```

该脚本会生成：

```text
cache_data/cora_llama2_7b_oracle_W4A8_S128.pt
cache_data/cora_llama2_7b_oracle_W4A8_S256.pt
```

然后运行：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite graph_eager_token \
  --real_quant_model_name llama2_7b \
  --graph_eager_reference_tag W4A16 \
  --graph_eager_full_tag W4A8 \
  --graph_eager_token_tag_prefix W4A8_S \
  --graph_eager_token_lengths 128 256 \
  --graph_eager_full_length 512 \
  --graph_eager_full_ratio 0.20 \
  --graph_eager_mid_ratio 0.30
```

Cora/LLaMA-7B 初步结果：

```text
Baseline Acc: 0.7310

Config              Full   S128   S256   Cost   Acc     Drop    AvgErr
FullW4A8            100%   0%     0%     0.500  0.7339 -0.29%  0.00265
AllS128             0%     100%   0%     0.092  0.7258  0.52%  0.01451
AllS256             0%     0%     100%   0.206  0.7315 -0.05%  0.00420
RandomTokenBudget   20%    50%    30%    0.208  0.7278  0.32%  0.00907
DegreeTokenBudget   20%    50%    30%    0.208  0.7290  0.19%  0.00874
TSERTokenBudget     20%    50%    30%    0.208  0.7292  0.18%  0.00893
ContextTokenBudget  20%    50%    30%    0.208  0.7282  0.27%  0.00909
PredictorTokenBudget 20%    50%    30%    0.208  0.7308  0.02%  0.00455
OracleDamageBudget  20%    50%    30%    0.208  0.7328 -0.18%  0.00349
```

这说明两个点：

```text
1. Token-budget lite encoder 比 naive partial-depth 更可行。
2. 单个 hand-crafted proxy 只小幅优于 random，但少量校准的 PredictorTokenBudget 明显更接近 oracle。
```

当前实现的 PredictorTokenBudget 用少量 calibration nodes 拟合一个线性 damage predictor：

```text
input:
    degree / graph_context / low_unique / rarity / similar_count / text length

target:
    S128 相对 reference 的 embedding damage

output:
    node -> S128 / S256 / Full
```

当前 Cora/LLaMA-7B 上，512 个校准点即可得到：

```text
rho_all ~= 0.42 - 0.52
PredictorTokenBudget Drop=0.02% at Cost=0.208
```

这就是把 FACT-style eager prediction 迁移到 graph-text encoder 场景的核心验证路径。

---

## 10. 推荐执行顺序

```text
Step 1:
    Cora/LLaMA token budget S128/S256/Full

Step 2:
    graph-eager damage analysis:
    proxy correlation + flip AUC

Step 3:
    Random/Degree/TSER/Context/Oracle token routing

Step 4:
    训练小型 Graph-Eager predictor，替换固定 proxy

Step 5:
    PubMed/LLaMA 或 Arxiv/LLaMA 扩展验证

Step 6:
    partial-depth 加 projection/alignment 后再复测

Step 7:
    FFN gating r25/r50/r75
```

优先保证 Step 1-4 成立。它们更适合支撑：

```text
graph-aware eager execution on W4A8 encoder array
```

如果 partial-depth / FFN gating 后续结果一般，也可以作为扩展或 future work。
