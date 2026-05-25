# Residual-Corrected Hash Reuse

本文档记录当前 GraphhopSimhash 中新增的 residual-corrected reuse 思路、实现方式、参数设计、开销和初步实验结果。

核心结论先说清楚：这个机制不是从 SimHash bits 还原 embedding。SimHash/CAM 只负责快速找到一个相似锚点节点 `u`；residual adapter 在已缓存的锚点 embedding `E_u` 基础上，用 cheap feature 和图上下文差异预测一个小残差，修正 fuzzy reuse 的误差。

## 1. Motivation

原始 hash reuse 有两个极端路径：

1. Direct reuse:
   `E_v_hat = E_u`
2. Recompute:
   对节点 `v` 重新跑完整 encoder，得到 `E_v`

Direct reuse 便宜，但 fuzzy hit 时误差较大；recompute 准确，但成本高。Residual-corrected reuse 在二者之间加入一条轻量中间路径：

```text
exact hit: 直接复用
fuzzy hit: 复用锚点 embedding + 低秩残差修正
reject/miss: 重新计算
```

因此完整角色分工是：

```text
CAM/SimHash: 找候选锚点 u
TSER gate: 过滤高风险复用
Residual adapter: 修正被接受的 fuzzy hit
Full encoder: 处理 miss / reject 节点
```

这条路径的目标不是替代 full encoder 的精度，而是在相同 reuse 率下降低掉点，或者在相近掉点下允许更高 reuse。

## 2. Online Decision Flow

对每个节点 `v`：

1. CAM/SimHash 查询候选锚点 `u`
2. 计算候选的 Hamming distance、route support、cosine proxy 等信息
3. TSER gate 判断是否允许复用
4. 如果被拒绝，走 full encoder
5. 如果 `dist == 0`，走 direct reuse
6. 如果 `dist > 0`，走 residual-corrected reuse

当前默认策略是：

```text
E_v_hat = E_u                              if exact hit
E_v_hat = normalize(E_u + alpha * R(z_vu)) if fuzzy hit
E_v_hat = E_v                              if rejected / miss
```

其中：

```text
R(.)     = low-rank residual adapter
z_vu    = pair feature between target node v and source node u
alpha   = residual strength, default from validation set auto-selected
```

## 3. Adapter Input

Adapter 不直接使用 hash bits 生成 embedding，而是使用“目标节点和锚点节点之间的差异特征”。

当前 `z_vu` 包含三类信息：

```text
cheap_delta   = cheap_feature(v) - cheap_feature(u)
context_delta = context_signature(v) - context_signature(u)
scalar_stats  = [
    hamming_dist,
    route_hit_count,
    base_route_hit_count,
    winning_table_hit_count,
    best_candidate_cosine,
    cheap_feature_cosine,
    context_cosine,
    log_degree_ratio,
    sensitivity_q,
]
```

其中 cheap feature 通常是 DistilBERT layer-1 特征，context signature 是：

```text
context_signature(v) =
    normalize(0.5 * cheap_feature(v) + 0.5 * neighbor_mean(v))
```

这些输入的含义是：

- `cheap_delta` 描述文本语义代理差异；
- `context_delta` 描述局部图上下文差异；
- Hamming / support / cosine 信息描述 CAM 命中的可信度；
- `log_degree_ratio` 和 `sensitivity_q` 描述复用误差传播风险。

## 4. Low-Rank Adapter

当前实现使用一个两层低秩 MLP：

```text
R(z) = W_up * GELU(W_down * z)
```

对应代码：

```python
class LowRankResidualAdapter(nn.Module):
    down: input_dim -> rank
    up:   rank -> emb_dim
```

训练目标是让：

```text
normalize(E_u + R(z_vu)) ≈ E_v
```

loss 为 cosine loss 加一个很小的 residual L2 正则：

```text
loss =
    mean(1 - cosine(normalize(E_u + R(z_vu)), E_v))
  + lambda * ||R(z_vu)||_2^2
```

默认训练样本来自 train/val 中已被 CAM 命中的节点对。它需要少量 full embedding 作为 calibration target，这一点类似 AWQ calibration：校准阶段需要少量 `E_v`，但在线阶段不需要对全图每个节点跑 full encoder。

## 5. Key Parameters

当前命令行参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--experiment_suite residual_reuse` | - | 启用 residual reuse 实验 |
| `--residual_rank` | `32` | adapter bottleneck rank |
| `--residual_epochs` | `200` | adapter 训练轮数 |
| `--residual_lr` | `1e-3` | adapter 学习率 |
| `--residual_weight_decay` | `1e-4` | AdamW weight decay |
| `--residual_l2` | `1e-4` | 残差幅度正则 |
| `--residual_alpha` | `-1` | 小于 0 表示在 val 上自动选择 alpha |
| `--residual_alpha_grid` | `0,0.125,0.25,0.5,0.75,1.0` | alpha 搜索集合 |
| `--residual_min_dist` | `1.0` | 只修正 Hamming distance >= 1 的 fuzzy hit |
| `--residual_direct_threshold` | `-1` | 可选：低风险节点 direct，高风险 fuzzy 节点 residual |
| `--residual_anchor_mode` | `cam` | `cam` 使用 CAM 锚点；`random` 用于消融 |
| `--residual_max_train_pairs` | `4096` | 最大 residual calibration pair 数 |
| `--residual_train_split` | `train_val` | adapter 使用 train / train_val / all_hits |

目前最稳的版本是：

```text
residual_min_dist = 1.0
residual_direct_threshold = -1
anchor_mode = cam
```

也就是 exact hit 直接复用，所有被 TSER gate 接受的 fuzzy hit 都用 residual 修正。

## 6. Cost Analysis

以 Cora/ST 为例：

```text
cheap feature dim = 768
context dim       = 768
scalar dim        = 9
adapter input dim = 1545
embedding dim     = 768
rank              = 32
```

参数量约为：

```text
input_dim * rank + rank * emb_dim
= 1545 * 32 + 32 * 768
≈ 74K parameters
```

FP16 存储约 150 KB，远小于 full encoder 权重。每个被修正的 fuzzy hit 需要约：

```text
1545 * 32 + 32 * 768 ≈ 74K MACs
```

如果 `T=30` 时复用率约 44%，其中 exact hit 约 7%，需要 residual 的 fuzzy hit 约 37%，则平均到全图约：

```text
0.37 * 74K ≈ 27K MACs / node
```

这比重新运行 ST / LLaMA encoder 低很多。对 LLaMA-7B 这类大模型，adapter 本身开销仍然很小，但 embedding 维度变为 4096，adapter 输出维度和参数量会增大，需要单独评估 rank 和 calibration size。

在线额外状态包括：

- CAM 查询得到的 source id；
- Hamming distance / support counters；
- cheap feature 和 context signature；
- adapter 权重。

不需要在线访问目标节点的 full embedding `E_v`。

## 7. Experimental Evidence

当前主要在 Cora/ST 上做了初步验证。

### 7.1 Fixed TSER 3/1/1, T=45

```text
Baseline Acc: 0.7200

Config             Reuse    Acc     Drop    HitErr
DirectReuse        48.3%    0.6817  3.84%   0.55376
ResidualReuse      48.3%    0.6884  3.16%   0.43761
```

结论：相同 reuse 率下，residual 将 drop 从 3.84% 降到 3.16%，hit embedding error 也明显下降。

### 7.2 Fixed TSER 3/1/1, T=30

```text
Baseline Acc: 0.7200

Config             Reuse    Acc     Drop    HitErr
DirectReuse        44.1%    0.6925  2.76%   0.54999
ResidualReuse      44.1%    0.6957  2.43%   0.45402
```

T=30 本身已经比较保守，所以 residual 的收益较小，但仍然稳定降低 drop。

### 7.3 CAM Anchor Ablation

```text
Ablation                         Reuse    Acc     Drop    HitErr
Exact-only CAM                   7.6%     0.7178  0.23%   0.38644
Fuzzy CAM + DirectReuse          48.3%    0.6817  3.84%   0.55376
Fuzzy CAM + ResidualReuse        48.3%    0.6884  3.16%   0.43761
Random anchor + ResidualReuse    48.3%    0.6765  4.35%   0.48008
```

这个消融说明：

- Exact-only 很安全，但 reuse 太低；
- fuzzy CAM 提供了主要 reuse 率；
- residual adapter 能修正 fuzzy hit；
- random anchor 明显更差，说明 CAM 找到的锚点仍然关键。

因此不能说 adapter 替代了 CAM。更准确的说法是：

```text
CAM provides the anchor;
TSER filters unsafe reuse;
Residual adapter corrects fuzzy-hit embeddings.
```

### 7.4 T Sweep: Direct vs Residual

TSER 3/1/1 下扫描 `T = 20 / 30 / 45 / 60 / 90`：

| T | Direct Reuse | Direct Drop | Residual Reuse | Residual Drop |
|---:|---:|---:|---:|---:|
| 20 | 4.5% | 0.24% | 4.5% | 0.21% |
| 30 | 44.1% | 2.76% | 44.1% | 2.43% |
| 45 | 48.3% | 3.84% | 48.3% | 3.16% |
| 60 | 49.2% | 3.92% | 49.2% | 3.27% |
| 90 | 59.8% | 5.82% | 59.8% | 4.16% |

结论：residual curve 整体低于 direct curve。它的价值不是在单点上“救回很多精度”，而是把 reuse-drop Pareto 曲线整体向更优方向移动。

### 7.5 How to Reproduce

下面命令都在仓库根目录运行：

```bash
cd /home/zhangshangtong/Transformer/OFA
```

固定使用 Cora/ST、4 个 16-bit hash head、TSER `3/1/1`：

```bash
BASE=(python -m GraphhopSimhash
  --datasets cora
  --runs 3
  --experiment_suite residual_reuse
  --radius 2
  --hash_heads_per_route 4
  --main_hash_head_bits 16 16 16 16
  --learned_hash_epochs 10
  --learned_hash_dim 128
  --hamming_only_acceptor
  --enable_score_gate
  --allow_rare_fuzzy
  --score_propagation_weight 3
  --score_graph_context_weight 1
  --score_low_unique_weight 1
  --residual_rank 32
  --residual_epochs 100
  --residual_max_train_pairs 1024
  --residual_min_dist 1.0)
```

复现 `TSER 3/1/1, T=45`：

```bash
"${BASE[@]}" --score_reuse_threshold 45
```

复现 `TSER 3/1/1, T=30`：

```bash
"${BASE[@]}" --score_reuse_threshold 30
```

复现 CAM anchor 消融：

```bash
# Exact-only CAM: only exact hash reuse, no residual correction.
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite residual_reuse \
  --radius 0 \
  --hash_heads_per_route 4 \
  --main_hash_head_bits 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 45 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --residual_rank 32 \
  --residual_epochs 0 \
  --residual_max_train_pairs 1024

# Fuzzy CAM + residual.
"${BASE[@]}" --score_reuse_threshold 45

# Random anchor + residual: keep the same hit set, replace CAM source with random source.
"${BASE[@]}" \
  --score_reuse_threshold 45 \
  --residual_anchor_mode random
```

复现 `T = 20 / 30 / 45 / 60 / 90` 曲线：

```bash
for T in 20 30 45 60 90; do
  "${BASE[@]}" --score_reuse_threshold "$T"
done
```

复现 `residual_min_dist` 阈值扫描：

```bash
for MD in 0.0 1.0 2.0 3.0; do
  "${BASE[@]}" \
    --score_reuse_threshold 30 \
    --residual_min_dist "$MD"
done
```

其中：

```text
residual_min_dist=0.0: exact + fuzzy 都修正
residual_min_dist=1.0: 只修正 fuzzy hit，当前最稳
residual_min_dist=2.0: 只修正 dist>=2 的候选
residual_min_dist=3.0: 基本退化为 direct reuse
```

当前 Cora/ST、TSER `3/1/1`、`T=30` 下的扫描结果支持：

```text
dist=0 exact hit -> direct reuse
dist=1 fuzzy hit -> residual correction
dist>=2 hit      -> 当前 gate 下样本极少，通常不进入 residual 主路径
```


<!-- PUBMED_ST_RESIDUAL_BIAS_SWEEP_START -->
## 7.6 PubMed/ST 3/1/1 Confidence-Bias Sweep

这组实验固定 PubMed/ST、TSER `3/1/1`、`residual_min_dist=1.0`，扫描复用阈值 `T` 和 `score_pair_confidence_discount`。

这里的 `confidence_bias` 指的是在计算 `reuse_risk = sensitivity_q * reuse_error_q` 之前，对高置信候选的 `reuse_error_q` 做折扣：

```text
reuse_error_q <- max(1, reuse_error_q - confidence_bias)
```

因此它不是 oracle error，而是基于 route support / base support / cosine margin 的在线置信修正。

| Bias | T | Direct Reuse | Direct Drop | Residual Reuse | Residual Drop | Residual HitErr | Alpha | TrainPairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 28.6% | 2.45% | 28.6% | 2.72% | 0.40267 | 0.500 | 160.0 |
| 0 | 30 | 70.0% | 5.83% | 70.0% | 7.32% | 0.37287 | 0.917 | 403.3 |
| 0 | 45 | 78.7% | 7.38% | 78.7% | 6.53% | 0.36921 | 0.500 | 455.7 |
| 0 | 60 | 80.3% | 7.56% | 80.3% | 7.13% | 0.37755 | 0.500 | 466.3 |
| 1 | 20 | 28.6% | 2.45% | 28.6% | 2.72% | 0.40267 | 0.500 | 160.0 |
| 1 | 30 | 70.0% | 5.83% | 70.0% | 7.32% | 0.37287 | 0.917 | 403.3 |
| 1 | 45 | 78.7% | 7.38% | 78.7% | 6.53% | 0.36921 | 0.500 | 455.7 |
| 1 | 60 | 80.3% | 7.56% | 80.3% | 7.13% | 0.37755 | 0.500 | 466.3 |
| 2 | 20 | 28.6% | 2.45% | 28.6% | 2.72% | 0.40267 | 0.500 | 160.0 |
| 2 | 30 | 70.0% | 5.83% | 70.0% | 7.32% | 0.37287 | 0.917 | 403.3 |
| 2 | 45 | 78.7% | 7.38% | 78.7% | 6.53% | 0.36921 | 0.500 | 455.7 |
| 2 | 60 | 80.3% | 7.56% | 80.3% | 7.14% | 0.37755 | 0.500 | 466.3 |

结果日志：

```text
/home/zhangshangtong/Transformer/OFA/output/residual_reuse/pubmed_st_bias_sweep/summary.tsv
/home/zhangshangtong/Transformer/OFA/output/residual_reuse/pubmed_st_bias_sweep/summary.md
output/residual_reuse/pubmed_st_bias_sweep/*.log
```

<!-- PUBMED_ST_RESIDUAL_BIAS_SWEEP_END -->

## 8. Current Limitations

当前版本仍是第一版机制验证，有几个边界需要注意：

1. calibration pair 数量会影响 adapter 稳定性；
2. rank 太小可能欠拟合，rank 太大会增加在线 MAC 和过拟合风险；
3. LLaMA-7B 的 embedding 维度为 4096，不能直接假设 ST 上的 rank=32 最优；
4. `residual_direct_threshold` 的简单风险分段目前效果不如“exact direct + fuzzy residual”；
5. 当前 adapter 是离线校准后固定使用，还没有做 dataset-adaptive 或 class-aware 版本。

## 9. Suggested Paper Framing

建议把它作为第三条执行路径，而不是简单附属 trick：

```text
Reuse-only path:      zero encoder compute, higher error
Residual reuse path:  tiny adapter compute, corrected fuzzy reuse
Full encoder path:    highest compute, exact embedding
```

这样系统形成三段式计算层级：

```text
Hash lookup < residual correction < full Transformer encoder
```

对应论文主线可以表述为：

1. SimHash/CAM 负责低成本候选检索；
2. TSER gate 负责图语义风险控制；
3. low-rank residual adapter 在 accepted fuzzy reuse 上提供轻量纠错；
4. 整体目标是在精度约束下最大化 encoder computation saving。
