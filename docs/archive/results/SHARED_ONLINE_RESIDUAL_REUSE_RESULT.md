# 共享在线配置的残差复用结果

日期：2026-05-29

本文档记录当前在 `cora` 和 `pubmed` 上得到的一组**共享在线复用策略**结果。

目标是：

```text
同一套在线配置
复用率 >= 40%
掉点 <= 2%
```

其中，**离线 residual 训练方式允许按数据集不同**，但**在线控制流保持一致**。

## 结果总览

最终 3-run 结果如下：

| 数据集 | Baseline Acc | ResidualReuse | Acc | Drop | TrainPairs | Alpha |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.7200 | 46.5% | 0.7107 | 0.93% | 464.7 | 0.263 |
| PubMed | 0.7587 | 42.3% | 0.7392 | 1.96% | 151.3 | 0.309 |

这是一组目前已经满足目标的配置。

## 共享在线配置

两套数据集共用的在线策略如下：

```text
8 个 head
每个 head 16 bit
radius = 2
score gate 打开
score 权重 = 3 / 1 / 1
score reuse threshold T = 30

support >= 5   -> hard direct reuse
support = 3..4 -> residual 路径
support < 3    -> compute

gate_accept_threshold = 0.575
```

对应的关键参数是：

```bash
--hash_heads_per_route 8
--main_hash_head_bits 16 16 16 16 16 16 16 16
--radius 2
--enable_score_gate
--allow_rare_fuzzy
--score_reuse_threshold 30
--score_propagation_weight 3
--score_graph_context_weight 1
--score_low_unique_weight 1
--residual_hard_min_support_hits 5
--residual_soft_min_support_hits 3
--residual_gate_accept_threshold 0.575
```

## 离线 Residual 训练设置

在线策略固定不变，Residual 的离线训练按数据集分别配置。

### Cora

`cora` 使用 **separate accept gate**。  
也就是 accept gate 和 correction gate 分开学习：

- accept gate 尽量保持宽松，让 soft hit 尽量保留
- correction gate 只决定“修多少”

关键参数：

```bash
--residual_adapter_type mlp
--residual_accept_mode separate
--residual_rank 64
--residual_epochs 200
--residual_max_train_pairs 4096
--residual_alpha_grid 0 0.03125 0.0625 0.125 0.25 0.5
--residual_support_aware_alpha
--residual_bucket_mode support_dist
--residual_offline_extra_anchors_per_node 8
--residual_positive_error_max -1
--residual_offline_extra_query_nodes 4096
--residual_offline_negative_anchors_per_node 0
--residual_negative_gate_weight 0.0
--residual_gate_loss_weight 0.5
--residual_accept_loss_weight 1.0
--residual_gate_error_scale 0.25
--residual_gate_error_max 0.45
--residual_gate_sparsity_weight 0.0
```

对应 3-run 结果：

| 配置 | Reuse % | Acc | Drop % |
|---|---:|---:|---:|
| DirectReuse | 16.0% | 0.7160 | 0.40% |
| SoftDirectReuse | 46.5% | 0.7023 | 1.77% |
| ResidualReuse | 46.5% | 0.7107 | 0.93% |

解释：

- `SoftDirectReuse` 本身已经满足复用率要求
- residual 的主要作用是**在不降低复用率的前提下把精度拉回来**
- separate accept gate 避免了对 `cora` 的 soft hit 过度拒绝

### PubMed

`pubmed` 使用 **shared accept gate**。  
也就是 accept/reject 和 correction strength 使用同一个 learned signal：

- 同一个 gate 既控制“修多少”
- 也控制“是否继续复用”

这样会让 `pubmed` 的 soft bucket 更保守。

关键参数：

```bash
--residual_adapter_type mlp
--residual_accept_mode shared
--residual_rank 64
--residual_epochs 200
--residual_max_train_pairs 4096
--residual_alpha_grid 0 0.03125 0.0625 0.125 0.25 0.5
--residual_support_aware_alpha
--residual_bucket_mode support_dist
--residual_offline_extra_anchors_per_node 8
--residual_positive_error_max 0.40
--residual_offline_extra_query_nodes 4096
--residual_offline_negative_anchors_per_node 4
--residual_negative_error_min 0.45
--residual_negative_gate_weight 1.0
--residual_gate_loss_weight 0.5
--residual_accept_loss_weight 0.0
--residual_gate_error_scale 0.25
--residual_gate_error_max 0.45
--residual_gate_sparsity_weight 0.02
```

对应 3-run 结果：

| 配置 | Reuse % | Acc | Drop % |
|---|---:|---:|---:|
| DirectReuse | 25.7% | 0.7483 | 1.04% |
| SoftDirectReuse | 69.5% | 0.7140 | 4.48% |
| ResidualReuse | 42.3% | 0.7392 | 1.96% |

解释：

- `SoftDirectReuse` 在 `pubmed` 上过于激进
- shared accept gate 会把相当一部分 soft hit 打回 `compute`
- 这样复用率从 `69.5%` 降到 `42.3%`
- 同时掉点从 `4.48%` 压到 `1.96%`

## 为什么这组配置有效

这组结果不是单纯靠“更强的 residual correction”得到的，而是以下几部分共同作用：

1. 共享的 hash / score / support split 在线逻辑
2. support-aware residual correction
3. residual 路径内部的 learned accept/reject
4. 针对数据集差异的离线训练方式

核心区别可以概括为：

```text
Cora:
    尽量保留 soft hit，只做轻量修正

PubMed:
    拒绝更多 soft hit，只保留更可靠的一部分做 residual 修正
```

这个差异是通过**离线 residual 训练方式**实现的，不是通过改变在线阈值实现的。

## 复现实验命令

下面命令假设仓库可以作为 `GraphhopSimhash` 包直接运行。

### Cora

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite residual_reuse \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hash_heads_per_route 8 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 30 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --radius 2 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 3 \
  --residual_rank 64 \
  --residual_epochs 200 \
  --residual_max_train_pairs 4096 \
  --residual_min_dist 1.0 \
  --residual_alpha_grid 0 0.03125 0.0625 0.125 0.25 0.5 \
  --residual_support_aware_alpha \
  --residual_adapter_type mlp \
  --residual_accept_mode separate \
  --residual_dropout 0.05 \
  --residual_loss_cosine_weight 1.0 \
  --residual_loss_mse_weight 0.5 \
  --residual_loss_delta_weight 0.75 \
  --residual_bucket_mode support_dist \
  --residual_offline_extra_anchors_per_node 8 \
  --residual_positive_error_max -1 \
  --residual_offline_extra_query_nodes 4096 \
  --residual_offline_negative_anchors_per_node 0 \
  --residual_negative_gate_weight 0.0 \
  --residual_train_split train_val \
  --residual_gate_loss_weight 0.5 \
  --residual_accept_loss_weight 1.0 \
  --residual_gate_error_scale 0.25 \
  --residual_gate_error_max 0.45 \
  --residual_gate_sparsity_weight 0.0 \
  --residual_gate_accept_threshold 0.575
```

### PubMed

```bash
python -m GraphhopSimhash \
  --datasets pubmed \
  --runs 3 \
  --experiment_suite residual_reuse \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hash_heads_per_route 8 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 30 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --radius 2 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 3 \
  --residual_rank 64 \
  --residual_epochs 200 \
  --residual_max_train_pairs 4096 \
  --residual_min_dist 1.0 \
  --residual_alpha_grid 0 0.03125 0.0625 0.125 0.25 0.5 \
  --residual_support_aware_alpha \
  --residual_adapter_type mlp \
  --residual_accept_mode shared \
  --residual_dropout 0.05 \
  --residual_loss_cosine_weight 1.0 \
  --residual_loss_mse_weight 0.5 \
  --residual_loss_delta_weight 0.75 \
  --residual_bucket_mode support_dist \
  --residual_offline_extra_anchors_per_node 8 \
  --residual_positive_error_max 0.40 \
  --residual_offline_extra_query_nodes 4096 \
  --residual_offline_negative_anchors_per_node 4 \
  --residual_negative_error_min 0.45 \
  --residual_negative_gate_weight 1.0 \
  --residual_train_split train_val \
  --residual_gate_loss_weight 0.5 \
  --residual_accept_loss_weight 0.0 \
  --residual_gate_error_scale 0.25 \
  --residual_gate_error_max 0.45 \
  --residual_gate_sparsity_weight 0.02 \
  --residual_gate_accept_threshold 0.575
```

## 总结

当前推荐的**共享在线配置**是：

```text
8 heads x 16 bits
score gate on
T = 30
hard >= 5
soft = 3..4
tau = 0.575
```

其中：

- `cora` 通过更宽松的 separate accept gate 保留 soft hit
- `pubmed` 通过更保守的 shared accept gate 控制 soft hit 质量

在这组设置下，已经可以满足：

```text
cora:   reuse 46.5%, drop 0.93%
pubmed: reuse 42.3%, drop 1.96%
```

这是当前共享在线 residual reuse 的推荐结果。
