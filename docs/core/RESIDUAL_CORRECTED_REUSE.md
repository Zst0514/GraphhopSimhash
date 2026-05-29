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
| `--residual_fit_profile` | `manual` | adapter 容量配置；`llama` 会按 4096d LLaMA embedding 放大 rank/epoch/pair |
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
| `--residual_hard_min_support_hits` | `-1` | 可选：支持头数达到该阈值的命中走 hard direct reuse |
| `--residual_soft_min_support_hits` | `-1` | 可选：支持头数在 soft/hard 阈值之间的命中走 residual correction |
| `--residual_max_train_pairs` | `4096` | 最大 residual calibration pair 数 |
| `--residual_train_split` | `train_val` | adapter 使用 train / train_val / all_hits |

注意：ST/768d 和 LLaMA-7B/4096d 不应共用同一组 residual 容量。LLaMA 路径建议使用：

```text
residual_fit_profile = llama
residual_rank >= 64
residual_epochs >= 120
residual_max_train_pairs >= 4096
residual_alpha_grid <= 0.5
```

这样做不是额外 oracle 信息，只是让同一个 residual correction engine 的容量匹配 4096d embedding 空间。LLaMA/Cora 这类小校准集上不宜盲目把 rank/alpha 放太大；`llama` profile 会限制 alpha 搜索上限，避免 residual 过强扰动分类边界。

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

<!-- PUBMED_ST_SUPPORT_SPLIT_START -->
## 7.7 PubMed/ST Support-Split Residual Tuning

这组实验专门验证“三段式中间态复用”：

```text
support heads >= hard_min:
    hard direct reuse

soft_min <= support heads < hard_min:
    residual-corrected reuse

support heads < soft_min:
    compute / full embedding
```

这里 `support heads` 对应日志里的 `winning_base_table_hit_count`，即同一个 base route 下有多少个 hash head 支持当前候选。实验固定 PubMed/ST、8 个 16-bit heads、`radius=2`、`--hamming_only_acceptor`、`residual_min_dist=1.0`，因此不依赖 `cosine_tau` 做额外向量过滤。

新增的三行输出含义：

```text
DirectReuse:
    只评估 hard direct 节点复用，其余节点 compute

SoftDirectReuse:
    hard + soft 节点都直接复用 anchor，不做 residual

ResidualReuse:
    hard 节点直接复用，soft 节点使用 anchor + residual correction
```

因此判断 residual 是否有用，主要看 `ResidualReuse` 是否优于 `SoftDirectReuse`。

### 7.7.1 Seed-42 Quick Sweep

下面表格是 seed=42 的快速调参结果，主要用于比较不同 support split 和 score threshold 的相对趋势。除特别说明外，使用：

```text
rank=64, epochs=200, max_train_pairs=2048,
alpha_grid=0/0.0625/0.125/0.25
```

| Config | Score T | Hard / Soft | Direct Reuse | Direct Drop | SoftDirect Reuse | SoftDirect Drop | Residual Reuse | Residual Drop | Residual Gain vs Soft | Alpha | TrainPairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conservative support | 35 | `>=5 / ==4` | 16.0% | 0.59% | 28.1% | 1.26% | 28.1% | 1.18% | +0.08% | 0.250 | 57 |
| Balanced support | 38 | `>=5 / ==4` | 23.3% | 0.72% | 42.2% | 1.92% | 42.2% | 1.72% | +0.20% | 0.250 | 102 |
| Higher reuse | 40 | `>=5 / ==4` | 25.7% | 0.90% | 47.8% | 2.25% | 47.8% | 2.10% | +0.15% | 0.125 | 115 |
| Too aggressive soft set | 40 | `>=5 / 3..4` | 22.5% | 0.86% | 62.3% | 3.65% | 62.3% | 3.54% | +0.11% | 0.062 | 199 |
| Stricter hard support | 45 | `>=6 / ==5` | 11.5% | 0.35% | 32.5% | 1.09% | 32.5% | 1.08% | +0.01% | 0.125 | 134 |
| Stricter hard, looser T | 60 | `>=6 / ==5` | 11.6% | 0.34% | 33.6% | 1.24% | 33.6% | 1.38% | -0.14% | 0.250 | 136 |

结论：

1. `>=5 / ==4` 是当前最合理的 support split。它保留了明确的中间态 residual 路径，同时不把 3-head 这种低置信候选放进 residual。
2. `>=5 / 3..4` 复用率能到 62.3%，但 drop 到 3.54%，说明 3-head 候选仍然太脏。
3. `>=6 / ==5` 很稳，但 hard direct 过少，整体 reuse 只有 32% 左右。
4. `T=38, >=5 / ==4` 是 seed42 上最平衡的点：`Reuse=42.2%`，`Drop=1.72%`，并且 residual 相比 soft direct 救回约 `0.20%`。

### 7.7.2 Selected 3-Seed Result

对 seed42 上较平衡的 `T=38, >=5 / ==4` 做 3 runs：

```bash
python -m GraphhopSimhash \
  --datasets pubmed \
  --runs 3 \
  --experiment_suite residual_reuse \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --score_reuse_threshold 38 \
  --radius 2 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 4 \
  --residual_rank 64 \
  --residual_epochs 200 \
  --residual_max_train_pairs 2048 \
  --residual_min_dist 1.0 \
  --residual_alpha_grid 0 0.0625 0.125 0.25
```

结果：

```text
Baseline Acc: 0.7587

Config             Reuse    Acc     Drop    AvgErr   HitErr   Alpha
DirectReuse        25.8%    0.7486  1.02%   0.10655  0.41216  -
SoftDirectReuse    44.6%    0.7354  2.34%   0.19136  0.42909  -
ResidualReuse      44.6%    0.7357  2.30%   0.18575  0.41647  0.167
```

这个 3-seed 结果比 seed42 单点更保守：`ResidualReuse` 的 drop 从单 seed 的 `1.72%` 回到平均 `2.30%`。它说明 support split 的方向可行，但当前 low-rank residual adapter 仍然偏弱，分类精度收益小于 embedding error 收益。

当前推荐结论：

```text
如果目标是稳健精度：
    用 T=35, >=5 / ==4，reuse 约 28%，drop 约 1.2%（seed42）

如果目标是中等复用率：
    用 T=38, >=5 / ==4，3-run reuse 约 44.6%，drop 约 2.30%

如果目标是更高 reuse：
    T=40, >=5 / ==4 可到约 47.8% reuse，但 seed42 drop 已到 2.10%，需要更多 seed 验证

不建议：
    把 3-head 大量放入 residual，因为 3-head soft set 会把 drop 推到 3.5% 以上
```

日志位置：

```text
output/residual_reuse/pubmed_support_split_sweep/pubmed_h5_s4_t35.log
output/residual_reuse/pubmed_support_split_sweep/pubmed_h5_s4_t38.log
output/residual_reuse/pubmed_support_split_sweep/pubmed_h5_s4_t40.log
output/residual_reuse/pubmed_support_split_sweep/pubmed_h5_s3_t40.log
output/residual_reuse/pubmed_support_split_sweep/pubmed_h6_s5_t45.log
output/residual_reuse/pubmed_support_split_sweep/pubmed_h6_s5_t60.log
output/residual_reuse/pubmed_support_split_sweep/pubmed_h5_s4_t38_runs3.log
```

### 7.7.3 Cross-Dataset Common Parameter Sweep

前面的 Cora 和 PubMed 单独调参能找到各自更优点，但硬件上不应该为每个数据集重新设计 CAM/head/support 规则。因此又补了一组共同参数 sweep，目标是：

```text
同一套 R / head 数 / hard-soft support split / score threshold
同时适用于 Cora 和 PubMed；
在 drop < 3% 的约束下，尽量提高 reuse。
```

实验设置：

```text
datasets = Cora, PubMed
runs = 3
radius = 2
target = ST/data.x
hash bits = 4x16 或 8x16
residual_rank = 64
residual_epochs = 200
residual_max_train_pairs = 2048
alpha_grid = 0 / 0.0625 / 0.125 / 0.25
```

完整结果在：

```text
output/residual_reuse/common_param_sweep_20260528/summary_compact.txt
output/residual_reuse/common_param_sweep_20260528/summary.tsv
```

复现当前主线共同参数 `h8_54_T40` 的命令如下。这一命令只评估 residual reuse 前端，后续 Graph-Bit / full-stack 实验会固定沿用这组前端参数。

```bash
python -m GraphhopSimhash \
  --datasets cora pubmed \
  --runs 3 \
  --experiment_suite residual_reuse \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hash_heads_per_route 8 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 40 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --radius 2 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --residual_embedding_source data_x \
  --residual_fit_profile st \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 4 \
  --residual_rank 64 \
  --residual_epochs 200 \
  --residual_max_train_pairs 2048 \
  --residual_train_split train_val \
  --residual_min_dist 1.0 \
  --residual_alpha_grid 0 0.0625 0.125 0.25
```

对应日志：

```text
output/residual_reuse/common_param_sweep_20260528/logs/cora_h8_54_T40_runs3.log
output/residual_reuse/common_param_sweep_20260528/logs/pubmed_h8_54_T40_runs3.log
```

核心表如下。`minReuse` 是 Cora/PubMed 两者中较低的 reuse，`maxDrop` 是两者中较高的 drop；因此它们更适合判断“共同参数”是否稳健。

| Config | Heads | Hard / Soft | T | Cora Reuse | Cora Drop | PubMed Reuse | PubMed Drop | minReuse | maxDrop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `h4_43_T35` | 4 | `>=4 / ==3` | 35 | 6.0% | 0.15% | 23.0% | 1.08% | 6.0% | 1.08% |
| `h4_43_T40` | 4 | `>=4 / ==3` | 40 | 17.7% | 0.32% | 38.6% | 2.25% | 17.7% | 2.25% |
| `h4_43_T45` | 4 | `>=4 / ==3` | 45 | 24.3% | 0.61% | 45.3% | 2.78% | 24.3% | 2.78% |
| `h8_54_T35` | 8 | `>=5 / ==4` | 35 | 8.2% | 0.18% | 29.9% | 1.45% | 8.2% | 1.45% |
| `h8_54_T40` | 8 | `>=5 / ==4` | 40 | 25.7% | 0.45% | 50.3% | 2.52% | 25.7% | 2.52% |
| `h8_64_T35` | 8 | `>=6 / ==4` | 35 | 8.2% | 0.19% | 29.9% | 1.44% | 8.2% | 1.44% |
| `h8_64_T40` | 8 | `>=6 / ==4` | 40 | 25.7% | 0.42% | 50.3% | 2.70% | 25.7% | 2.70% |
| `h8_64_T45` | 8 | `>=6 / ==4` | 45 | 35.7% | 0.61% | 58.3% | 3.28% | 35.7% | 3.28% |

结论：

1. 如果只看 Cora，`8 heads, hard>=6, soft=4, T=45` 很漂亮：`35.7%` reuse，`0.61%` drop。
2. 但同一配置放到 PubMed 后，drop 到 `3.28%`，已经超过当前希望的 `3%` 线，因此不适合作为跨数据集固定硬件参数。
3. 当前最推荐的共同参数是：

```text
radius = 2
heads = 8 x 16-bit
score threshold T = 40
hard direct reuse: support heads >= 5
residual reuse: support heads == 4
compute: support heads < 4
```

对应结果：

```text
Cora:   reuse = 25.7%, drop = 0.45%
PubMed: reuse = 50.3%, drop = 2.52%
```

4. 如果希望 hard direct 更保守，可以使用 `hard>=6, soft=4, T=40`：

```text
Cora:   reuse = 25.7%, drop = 0.42%
PubMed: reuse = 50.3%, drop = 2.70%
```

这组和 `hard>=5` 的总 reuse 相同，因为 soft threshold 都是 `4`，差别主要在 hard direct 与 residual correction 的分配。当前结果里 `hard>=5` 在 PubMed 上略好，因此作为默认共同参数更合适。

5. `4 heads, hard>=4, soft=3, T=45` 也是可用的简单版本：

```text
Cora:   reuse = 24.3%, drop = 0.61%
PubMed: reuse = 45.3%, drop = 2.78%
```

但它的 `minReuse` 略低于 `8 heads, hard>=5, soft=4, T=40`，因此更适合作为低硬件开销 baseline，而不是主推配置。

最终建议：

```text
主线固定配置:
    R = 2
    heads = 8 x 16-bit
    T = 40
    hard >= 5
    soft = 4

稳健备选:
    R = 2
    heads = 8 x 16-bit
    T = 40
    hard >= 6
    soft = 4

低开销 baseline:
    R = 2
    heads = 4 x 16-bit
    T = 45
    hard >= 4
    soft = 3
```

这说明 residual reuse 的硬件策略可以固定，不需要为 Cora/PubMed 分别调规则；只要把目标设为 `drop < 3%`，`8x16, T=40, hard>=5, soft=4` 是当前最平衡的共同点。

选择 `h8_54_T40` 而不是 `h8_64_T40` 的关键原因是：两者总 reuse 相同，因为二者的 soft threshold 都是 `support>=4`；差别只在 `support=5` 节点走哪条路径。

```text
h8_54_T40:
    support >= 5 -> direct reuse
    support == 4 -> residual correction

h8_64_T40:
    support >= 6 -> direct reuse
    support == 4 or 5 -> residual correction
```

实验显示 `support=5` 的节点已经足够可靠，直接复用比强制 residual correction 更稳。因此 `h8_54_T40` 在相同 reuse 下取得更低的跨数据集最大掉点：

```text
h8_54_T40 maxDrop = 2.52%
h8_64_T40 maxDrop = 2.70%
```

后续 Graph-Bit / full-stack 实验默认沿用这一组统一复用前端参数。Graph-Bit 只负责 accepted miss nodes 的 P8/P6/P5/P4 precision-depth 路由，不再重新调 reuse gate。

<!-- PUBMED_ST_SUPPORT_SPLIT_END -->

## 8. Residual Reuse + Graph-Bit Full Stack

Residual reuse now has a full-stack evaluation entry with Graph-Bit:

```text
exact hit      -> direct cache reuse
fuzzy hit      -> residual correction
reject / miss  -> Graph-Bit P8/P6/P5/P4
```

后续 Graph-Bit full-stack 默认使用上一节确定的统一复用前端：

```text
R = 2
heads = 8 x 16-bit
score threshold T = 40
hard direct reuse: support heads >= 5
residual correction: support heads == 4
compute / Graph-Bit: support heads < 4
```

推荐复现实验命令：

```bash
python -m GraphhopSimhash \
  --datasets cora pubmed \
  --runs 3 \
  --experiment_suite residual_precision_depth \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4A8 \
  --precision_depth_tags W4A6 W4A5 W4A4 \
  --precision_depth_bits 6 5 4 \
  --precision_depth_reference_bits 8 \
  --precision_depth_high_ratio 0.20 \
  --precision_depth_mid_ratio 0.30 \
  --precision_depth_low_ratio 0.30 \
  --radius 2 \
  --hash_heads_per_route 8 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 40 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --residual_fit_profile llama \
  --residual_rank 64 \
  --residual_epochs 120 \
  --residual_max_train_pairs 4096 \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 4 \
  --residual_alpha_grid 0 0.125 0.25 0.5 \
  --residual_min_dist 1.0
```

下面是早期 full-stack smoke run，未使用当前 `h8_54_T40` support split，仅保留为历史对照：

| Dataset | T | Reuse | FullP8-miss Drop | Degree Graph-Bit Drop |
|---|---:|---:|---:|---:|
| Cora | 20 | 4.5% | 0.27% | 2.22% |
| Cora | 30 | 46.9% | 3.68% | 5.25% |
| PubMed | 20 | 31.3% | 2.79% | 3.90% |
| PubMed | 30 | 77.2% | 6.02% | 6.60% |

Interpretation:

1. `FullP8-miss` is the right reuse baseline: hits use direct/residual reuse, misses use P8.
2. Graph-Bit only changes miss-node bit depth; it cannot fix bad fuzzy reuse hits.
3. `T=20` is the safer full-stack point for LLaMA-7B. `T=30` is too aggressive on PubMed because reuse hit error already dominates.
4. ST results should not be directly extrapolated to LLaMA. Cora/ST at `T=30` is easier because the residual target is 768d ST. LLaMA uses 4096d targets and needs the `llama` residual fit profile.
5. Earlier LLaMA full-stack runs used a ST-sized adapter (`rank=32`, `epochs=60`, `max_pairs=1024`). After switching to `residual_fit_profile=llama` (`rank=64`, `epochs=120`, `max_pairs=4096`, `alpha<=0.5`), the Cora/LLaMA T30 `FullP8-miss` 3-run smoke result improved from about `3.68%` drop to about `3.24%` drop. This confirms that the previous setting was biased toward ST, although LLaMA reuse is still intrinsically harder than ST reuse.

Logs:

```text
output/residual_precision_depth_manual/cora_pubmed_llama7b_T20_fullstack.log
output/residual_precision_depth_manual/cora_llama7b_T30_fullstack.log
output/residual_precision_depth_manual/pubmed_llama7b_T30_fullstack.log
output/residual_precision_depth_manual/cora_ST_T30_fullstack.log
```

## 9. Current Limitations

当前版本仍是第一版机制验证，有几个边界需要注意：

1. calibration pair 数量会影响 adapter 稳定性；
2. rank 太小可能欠拟合，rank 太大会增加在线 MAC 和过拟合风险；
3. LLaMA-7B 的 embedding 维度为 4096，不能直接假设 ST 上的 rank=32 最优；
4. `residual_direct_threshold` 的简单风险分段目前效果不如“exact direct + fuzzy residual”；
5. 当前 adapter 是离线校准后固定使用，还没有做 dataset-adaptive 或 class-aware 版本。

## 10. Suggested Paper Framing

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
