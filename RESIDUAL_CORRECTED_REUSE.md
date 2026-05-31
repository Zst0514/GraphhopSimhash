# Residual-Corrected Hash Reuse

这个功能来自 `GraphhopSimhash-main`，用于在普通 hash reuse 和重新计算之间增加一条轻量中间路径。

核心思想不是从 SimHash bits 直接反解 embedding。SimHash/CAM 仍然只负责找到一个相似锚点节点 `u`；residual adapter 在已经缓存的锚点 embedding `E_u` 基础上，根据 cheap feature 和图上下文差异预测一个小残差，用来修正 fuzzy reuse 的误差。

## 执行路径

普通 hash reuse 只有两种结果：

```text
hit: 直接复用候选 embedding
miss/reject: 重新计算
```

Residual-corrected reuse 变成三种结果：

```text
exact hit: 直接复用
fuzzy hit: 复用锚点 embedding + residual correction
miss/reject: 重新计算
```

在线公式是：

```text
E_v_hat = E_u                              if exact hit
E_v_hat = normalize(E_u + alpha * R(z_vu)) if fuzzy hit
E_v_hat = E_v                              if rejected / miss
```

其中：

```text
E_v     = 目标节点真实 raw embedding
E_u     = CAM/SimHash 找到的锚点节点 embedding
R(.)    = low-rank residual adapter
z_vu    = 目标节点 v 和锚点节点 u 的 pair feature
alpha   = residual 强度，默认在 validation set 上自动选择
```

## Adapter 输入

`z_vu` 不是 hash bits 本身，而是目标节点和锚点节点的差异特征：

```text
cheap_delta   = cheap_feature(v) - cheap_feature(u)
context_delta = context_signature(v) - context_signature(u)
scalar_stats  = [
    hamming_dist,
    route_hit_count,
    base_route_hit_count,
    winning_base_table_hit_count,
    best_candidate_cosine,
    cheap_feature_cosine,
    context_cosine,
    log_degree_ratio,
    sensitivity_q,
]
```

`context_signature` 使用 cheap feature 和 1-hop 邻居均值构造：

```text
context_signature(v) = normalize(0.5 * cheap_feature(v) + 0.5 * neighbor_mean(v))
```

## Adapter 结构

当前实现是一个两层低秩 MLP：

```text
R(z) = W_up * GELU(W_down * z)
```

训练目标：

```text
normalize(E_u + R(z_vu)) ≈ E_v
```

loss：

```text
mean(1 - cosine(normalize(E_u + R(z_vu)), E_v))
+ residual_l2 * ||R(z_vu)||_2^2
```

它需要少量 train/val reuse-hit 节点作为 calibration pair。在线推理时不需要目标节点的 full embedding。

## 关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--experiment_suite residual_reuse` | - | 启用 residual reuse 实验 |
| `--residual_rank` | `32` | adapter bottleneck rank |
| `--residual_epochs` | `200` | adapter 训练轮数 |
| `--residual_lr` | `1e-3` | adapter 学习率 |
| `--residual_weight_decay` | `1e-4` | AdamW weight decay |
| `--residual_l2` | `1e-4` | 残差幅度正则 |
| `--residual_alpha` | `-1` | 小于 0 表示在 val 上自动选择 alpha |
| `--residual_alpha_grid` | `0 0.125 0.25 0.5 0.75 1.0` | alpha 搜索集合 |
| `--residual_min_dist` | `1.0` | 只修正 Hamming distance >= 1 的 fuzzy hit |
| `--residual_direct_threshold` | `-1` | 可选：低风险直接复用，高风险 fuzzy 节点 residual |
| `--residual_anchor_mode` | `cam` | `cam` 使用 CAM 锚点；`random` 用于消融 |
| `--residual_max_train_pairs` | `4096` | 最大 residual calibration pair 数 |
| `--residual_train_split` | `train_val` | adapter 使用 train / train_val / all_hits |

## 推荐命令

```bash
cd /home/qiumingzhi/Simhash-S/OneForAll

python -m GraphhopSimhash \
  --datasets cora \
  --runs 1 \
  --experiment_suite residual_reuse \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --score_reuse_threshold 45 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --route_min_support_hits 3 \
  --residual_rank 32 \
  --residual_epochs 100 \
  --residual_max_train_pairs 1024 \
  --residual_min_dist 1.0
```

输出表会同时给出：

```text
DirectReuse:    原始 hash reuse
ResidualReuse:  fuzzy hit 上做 residual correction 后的结果
```

重点看：

```text
Reuse %
Acc / Drop %
AvgErr
HitErr
Corrected
TrainPairs
Alpha
```

## 当前分支接入点

代码位置：

```text
residual_reuse.py   low-rank adapter、训练、应用、embedding error
controller.py       query_full_batch 记录 last_query_trace
runner.py           residual_reuse 实验流程
cli.py              residual 参数和 experiment_suite 入口
routing.py          residual 实验日志 tag
```

其中 `last_query_trace` 会记录：

```text
hit_mask
source_ids
hit_kinds
best_dists
best_cosines
route_hit_counts
base_route_hit_counts
winning_base_table_hit_counts
```

这些字段用于构造 residual pair feature。

## 注意

`ResidualReuse` 修正的是已被 CAM/SimHash 和 score gate 接受的 reuse hit，默认只修正 fuzzy hit。它不是替代 CAM，也不是从 hash bits 直接生成 embedding。更准确地说：

```text
CAM provides the anchor;
score gate filters unsafe reuse;
residual adapter corrects accepted fuzzy-hit embeddings.
```
