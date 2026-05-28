# Graph-Aware Adaptive Encoder NPU 实验设计

本文档记录当前用于验证 **graph-aware adaptive Transformer encoding** 的实验线。目标不是加速 GNN 聚合，而是让图后端风险信息指导 LLM encoder 本体少算：

```text
Graph/text risk -> choose token budget / channel budget / full W4A8 encoder
```

当前已经验证：直接把中间层 hidden state 当作最终 embedding 的少层数路径无效，因此主线不再保留这条路径。后续实验集中在更适合 NPU 落地的两类机制：

```text
1. Token-budget lite encoder
2. FFN / channel compaction
```

---

## 1. 总体假设

不同减算模式破坏的信息不同，因此不能用一个统一 TSER 总分控制所有路径。

| Encoder 减算模式 | 主要破坏 | 推荐 proxy | 原因 |
|---|---|---|---|
| Token budget | 文本信息完整性 | graph_context / text length / calibration predictor | 短序列会丢 token，风险和文本长度、上下文边界、局部语义偏移相关 |
| Channel gating | 细粒度语义通道 | low_degree_unique / confidence / calibration predictor | 稀有、低置信节点更依赖细粒度语义特征 |

更稳的路线是：

```text
mode-specific learned predictor > hand-crafted TSER/Degree proxy > random
```

---

## 2. Damage 定义

对每个节点 `v` 和低成本模式 `m`，先生成：

```text
E_full(v)
E_m(v)
```

然后计算：

```text
emb_damage(v, m) = 1 - cosine(E_m(v), E_full(v))
```

任务侧还可以记录：

```text
margin_drop(v, m) = margin_full(v) - margin_m(v)
flip(v, m) = argmax(logits_m(v)) != argmax(logits_full(v))
```

其中 `logits` 来自同一个 GNN classifier。

---

## 3. Proxy 验证

候选 proxy：

```text
propagation_q
graph_context_q
low_degree_unique_q
sensitivity_q
text_length / token_count
cheap margin / confidence
calibration predictor
```

验证指标：

```text
Spearman correlation(proxy, emb_damage)
Spearman correlation(proxy, margin_drop)
AUC(proxy predicts flip)
Top-risk bucket mean damage
```

如果手写 proxy 相关性弱，就用少量 calibration nodes 学一个 damage predictor，而不是硬套固定阈值。

---

## 4. Router 设计

### 4.1 Baselines

必须比较：

```text
FullW4A8
AllLite
RandomBudget
DegreeBudget
TSERBudget
ContextBudget
```

主线策略只使用在线可得的图/文本 proxy。`PredictorBudget` 和 `OracleDamageBudget` 降级为 debug/oracle：

```text
PredictorBudget:
    需要额外 calibration nodes 学 damage predictor。
    用于检查 proxy 上限，不作为主策略。

OracleDamageBudget:
    需要全图已知低成本路径相对 reference 的真实误差。
    只能作为不可部署上界。
```

### 4.2 Token Budget Router

以 `S128 / S256 / Full` 为例：

```text
低风险节点 -> S128
中风险节点 -> S256
高风险节点 -> Full W4A8
```

可部署路由可以来自：

```text
Degree
TSER
GraphContext
PredictorTokenBudget
```

当前结果显示，单个 hand-crafted proxy 只小幅优于 random；少量 calibration 学出的 predictor 更接近 oracle。

---

## 5. Cost Model

对 encoder NPU 使用统一 cost model：

```text
Cost(mode) =
    [ attn_weight * token_ratio^2
    + ffn_weight  * token_ratio * channel_ratio ] *
    W4A8_cost_scale
```

初始参数：

```text
W4A8_cost_scale = 0.5
attn_weight = 0.35
ffn_weight  = 0.65
```

后续可以用 profiling 替换权重。硬件统计还应报告：

```text
MAC reduction
weight traffic
activation traffic
padding ratio
array utilization
```

---

## 6. Token Compaction 实验

这组实验回答：

```text
如果每个节点只给固定 128-token budget，应该保留哪些 token/chunks？
```

运行 Cora/LLaMA-7B：

```bash
bash GraphhopSimhash/scripts/run_cora_llama_token_compaction.sh
```

运行 PubMed/ST：

```bash
bash GraphhopSimhash/scripts/run_pubmed_st_token_compaction.sh
```

### 6.1 Cora/LLaMA-7B

```text
Baseline Acc: 0.7308

Config            Cost   Acc     Drop    AvgErr
FullW4A8          0.500  0.7340 -0.32%  0.00265
Prefix128         0.092  0.7260  0.48%  0.01451
Random128         0.092  0.7102  2.06%  0.04323
TFIDF128          0.092  0.7063  2.45%  0.04924
GraphContext128   0.092  0.7237  0.71%  0.04064
```

### 6.2 PubMed/ST

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
2. 朴素 TF-IDF 或 graph-context chunk 重排会破坏 front-loaded 信息。
3. 即使保留 prefix，再补充 TF-IDF/GraphContext chunk，也没有超过 Prefix128。
4. 因此 token compaction 的主线不应是手写 chunk scorer，而应转向 node-level budget routing。
```

---

## 7. Graph-Eager Token Budget

`graph_eager_token` 实验预生成不同 token 长度的 W4A8 embedding pool，然后用图/文本 proxy 决定每个节点走短序列还是完整序列。

生成 Cora/LLaMA-7B token pools 并运行评估：

```bash
bash GraphhopSimhash/scripts/run_cora_llama_graph_eager_token.sh
```

该脚本会生成：

```text
cache_data/cora_llama2_7b_oracle_W4A8_S128.pt
cache_data/cora_llama2_7b_oracle_W4A8_S256.pt
```

核心评估命令：

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

Config                Full   S128   S256   Cost   Acc     Drop    AvgErr
FullW4A8              100%   0%     0%     0.500  0.7339 -0.29%  0.00265
AllS128               0%     100%   0%     0.092  0.7258  0.52%  0.01451
AllS256               0%     0%     100%   0.206  0.7315 -0.05%  0.00420
RandomTokenBudget     20%    50%    30%    0.208  0.7278  0.32%  0.00907
DegreeTokenBudget     20%    50%    30%    0.208  0.7290  0.19%  0.00874
TSERTokenBudget       20%    50%    30%    0.208  0.7292  0.18%  0.00893
ContextTokenBudget    20%    50%    30%    0.208  0.7282  0.27%  0.00909
PredictorTokenBudget  20%    50%    30%    0.208  0.7308  0.02%  0.00455
OracleDamageBudget    20%    50%    30%    0.208  0.7328 -0.18%  0.00349
```

`PredictorTokenBudget` 使用少量 calibration nodes 拟合线性 damage predictor。它是 debug/profiling baseline，不作为主线 deployable 策略：

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

## 8. 推荐执行顺序

```text
Step 1:
    固定 token budget 的 token/chunk compaction

Step 2:
    S128/S256/Full token budget routing

Step 3:
    proxy correlation + flip AUC

Step 4:
    用少量 calibration nodes 训练 Graph-Eager predictor

Step 5:
    PubMed/LLaMA 或 Arxiv/LLaMA 扩展验证

Step 6:
    FFN channel gating r25/r50/r75
```

优先保证 Step 1-4 成立。它们更适合支撑：

```text
graph-aware eager execution on W4A8 encoder array
```
