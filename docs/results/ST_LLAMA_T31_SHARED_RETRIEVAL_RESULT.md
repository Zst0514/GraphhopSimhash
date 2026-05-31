# ST 与 Llama 共享检索骨架的残差复用结果

日期：2026-05-31

本文档是对 2026-05-30 版本的纠错重跑记录。旧版本中的 ST 行写成了 `ST full/HQ cache (data.x)`，但实际日志是：

```text
[ResidualTarget] source=data_x | path=<data.x> | shape=(2708, 384)
[ResidualTarget] source=data_x | path=<data.x> | shape=(19717, 384)
```

这不是我们原意里的 ST 全量 embedding。按照残差复用实验的原意，ST 目标 embedding 应该来自独立的 ST oracle pool。本次重跑改为：

```text
Cora:   cache_data/cora_ST_oracle_W4A16.pt    shape=(2708, 768)
PubMed: cache_data/pubmed_ST_oracle_W4A16.pt  shape=(19717, 768)
```

因此，旧文档中的 ST 两行结果作废，不能再作为“ST full embedding”结论使用。

## 当前结论

纠错后，原来的结论需要收回：

```text
同一套 T31 在线检索骨架
不能证明 ST 与 Llama2-7B W4A16
同时在 Cora/PubMed 上达到 40%+ 复用和 2% 内掉点
```

本次只重跑了 ST 的真实 768 维 oracle embedding。Llama2-7B W4A16 旧行不受 `data_x` 误用影响，因为它原本就是通过 `real_quant_fp` 读取独立 Llama embedding pool；但如果要写入最终论文或主结果表，Llama 也建议按同一脚本重新复核一次。

## 共享在线检索骨架

原始共享在线骨架为：

```text
8 个 head
每个 head 16 bit
radius = 2
关闭结构检查
score gate 打开
score 权重 = 3 / 1 / 1
score reuse threshold T = 31

support >= 5   -> hard direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute
```

对应关键参数：

```bash
--hash_heads_per_route 8
--main_hash_head_bits 16 16 16 16 16 16 16 16
--radius 2
--disable_structure_check
--enable_score_gate
--allow_rare_fuzzy
--score_reuse_threshold 31
--score_propagation_weight 3
--score_graph_context_weight 1
--score_low_unique_weight 1
--score_pair_confidence_discount 1
--residual_hard_min_support_hits 5
--residual_soft_min_support_hits 3
```

## 纠错重跑结果

### 严格 T31 复核

| Embedding 源 | 数据集 | Baseline Acc | ResidualReuse | Acc | Drop | gate 设置 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| ST:W4A16 768d | Cora | 0.6789 | 39.4% | 0.6562 | 2.27% | separate, tau=0.575 | 未达标：复用低于 40%，掉点高于 2% |
| ST:W4A16 768d | PubMed | 0.7710 | 35.7% | 0.7520 | 1.90% | shared, tau=0.65 | 未达标：复用低于 40% |

### 小幅在线调参复核

这两组不是严格 T31 共享配置，只用于判断纠错后是否能通过小调参恢复旧结论。

| Embedding 源 | 数据集 | 调整 | Baseline Acc | ResidualReuse | Acc | Drop | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| ST:W4A16 768d | Cora | score T=32, tau=0.575 | 0.6789 | 40.2% | 0.6575 | 2.14% | 复用达标，但掉点仍高于 2% |
| ST:W4A16 768d | PubMed | score T=31, tau=0.60 | 0.7710 | 37.3% | 0.7510 | 2.00% | 掉点到边界，但复用仍低于 40% |

### 追加 sweep 结果

本轮继续修正了残差在线应用逻辑：残差修正后的向量不再强制做单位归一化，而是保留 ST oracle embedding 的原始尺度。这样 Cora 的残差修正明显改善，但 PubMed 仍没有在同一套在线配置下同时满足目标。

| 配置 | 数据集 | Baseline Acc | ResidualReuse | Acc | Drop | 结论 |
|---|---|---:|---:|---:|---:|---|
| 8 heads, hard>=5, soft=3..4, T=45, separate tau=0.575 | Cora | 0.6838 | 41.1% | 0.6639 | 1.98% | 单跑达标，是当前 Cora 最好点 |
| 8 heads, hard>=5, soft=4, T=23 | PubMed | 0.7751 | 39.6% | 0.7539 | 2.11% | 接近目标，但复用略低且掉点略高 |
| 8 heads, hard>=5, soft=4, T=24 | PubMed | 0.7751 | 42.3% | 0.7507 | 2.43% | 复用达标，但掉点超出 |
| 4 heads, hard>=4, soft=3, T=31 | PubMed | 0.7751 | 40.3% | 0.7550 | 2.01% | PubMed 几乎踩线，但不是 8 头主线 |
| 4 heads, hard>=4, soft=3, T=31 | Cora | 0.6838 | 15.7% | 0.6731 | 1.06% | Cora 复用过低，不能作为共享配置 |
| 8 heads, hard>=5, soft=4, T=45 | Cora | 0.6838 | 27.8% | 0.6668 | 1.69% | PubMed 较稳的 4-head 中间态逻辑迁移到 Cora 后复用不足 |

本轮也测试了两类更强的 accept gate：

1. `--residual_class_aware_accept`：离线阶段用 train/val 标签构造“同类候选更可接受”的监督。
2. `--residual_classifier_accept_gate`：离线阶段用冻结 GNN 检查候选复用是否保持预测和 logits KL，在线阶段仍然只用 learned accept score。

分类保持 gate 的 PubMed 单跑结果如下：

| 配置 | 数据集 | Baseline Acc | ResidualReuse | Acc | Drop | 说明 |
|---|---|---:|---:|---:|---:|---|
| 8 heads, hard>=5, soft=4, T=24, classifier gate, tau=0.50 | PubMed | 0.7751 | 37.8% | 0.7550 | 2.01% | 掉点接近 2%，但复用低于 40% |
| 8 heads, hard>=5, soft=4, T=25, classifier gate, tau=0.50 | PubMed | 0.7751 | 40.9% | 0.7515 | 2.36% | 复用达标，但掉点仍高 |
| 8 heads, hard>=5, soft=4, T=25, local train/val classifier gate, tau=0.40 | PubMed | 0.7751 | 41.4% | 0.7512 | 2.39% | 加入局部邻居监督后仍未改善 |
| 8 heads, hard>=5, soft=4, T=25, two-stage classifier gate, probe_alpha=0.25, tau=0.50 | PubMed | 0.7751 | 42.3% | 0.7494 | 2.57% | 用残差修正后候选生成 target，未改善 |
| 8 heads, hard>=5, soft=4, T=25, two-stage classifier gate, probe_alpha=0.125, tau=0.50 | PubMed | 0.7751 | 41.9% | 0.7512 | 2.39% | 与直接 classifier gate 基本持平，仍未达标 |

这说明当前问题不是简单加一个 accept gate 就能解决。PubMed 的 4-head 边界非常窄：门控稍微严格，复用跌到 40% 以下；门控稍微宽松，掉点就超过 2%。

阶段性结论：

```text
Cora 需要 3-head soft 区才能达到 40%+ 复用；
PubMed 的 3-head soft 区污染较重，必须收紧到 4-head 附近才接近 2% 掉点。

因此，在真实 ST:W4A16 768d 上，
暂时没有找到一套完全相同的 8-head 在线配置，
同时让 Cora/PubMed 都达到 40%+ 复用和 2% 内掉点。
```

### 旧 Llama 行

以下 Llama 行来自 2026-05-30 的旧日志，未在本次纠错中重跑。它们没有使用 `data_x`，而是通过 `real_quant_fp` 加载 Llama2-7B W4A16 embedding pool。

| Embedding 源 | 数据集 | Baseline Acc | ResidualReuse | Acc | Drop | gate 设置 | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| Llama2-7B W4A16 | Cora | 0.7308 | 40.8% | 0.7132 | 1.76% | classifier-aware separate, tau=0.40 | 旧有效日志，建议重跑复核 |
| Llama2-7B W4A16 | PubMed | 0.7000 | 40.8% | 0.6819 | 1.81% | shared, tau=0.91 | 旧有效日志，建议重跑复核 |

## 为什么旧 ST 结果会偏乐观

旧 ST 行实际复用的是 `data.x` 中的 384 维缓存特征，而不是 768 维 ST oracle embedding。`data.x` 的维度更小，分布也不同，残差修正和哈希检索难度都会变化。因此旧结果不能外推到真实 ST 全量 embedding。

纠错后可以看到：

```text
Cora:
    T31 下复用率从旧表 48.4% 下降到 39.4%
    掉点从旧表 1.21% 上升到 2.27%

PubMed:
    T31 下复用率从旧表 40.4% 下降到 35.7%
    掉点仍在 2% 内，但复用率不达标
```

这说明问题不是单纯的文档命名错误，而是实验目标 embedding 选错后导致的主结论错误。

## 后续处理

目前应把 ST 这条线从“已达标”改成“真实 ST oracle 下尚未恢复原目标”。

优先级建议如下：

1. 论文或主结果表不能继续引用旧 ST(data.x) 结果。
2. 如果坚持一套 8-head 在线配置，需要重新设计 PubMed 的中间态 accept 规则，而不是只调现有 gate 阈值。
3. 如果允许按数据集选择 support split，Cora 可以走 `soft=3..4`，PubMed 更适合 `soft=4`，但这不再是“完全相同在线配置”。
4. Llama2-7B W4A16 旧结果需要用当前代码重新复核，避免 ST 纠错后文档中混用不同日期和不同代码状态的结果。

## 本轮日志

纠错重跑日志：

```text
/tmp/st_full_w4a16_T31_rerun_20260531/cora.log
/tmp/st_full_w4a16_T31_rerun_20260531/pubmed_tau065.log
/tmp/st_full_w4a16_T31_rerun_20260531/cora_T32.log
/tmp/st_full_w4a16_T31_rerun_20260531/pubmed_tau060.log
/tmp/st_w4a16_nonorm_sweep_20260531/cora_T45_sep_fulltrain_nonorm.log
/tmp/st_w4a16_pubmed_h5s4_lowT_20260531/pubmed_h5s4_T23.log
/tmp/st_w4a16_pubmed_h5s4_lowT_20260531/pubmed_h5s4_T24.log
/tmp/st_w4a16_pubmed_4head_sweep_20260531/pubmed_4h_T31.log
/tmp/st_w4a16_4head_candidate_20260531/cora_4h_T31.log
/tmp/st_w4a16_classgate_hightau_20260531/pubmed_T45_tau097.log
/tmp/st_w4a16_cora_h5s4_pubmedlogic_20260531/cora_h5s4_T45.log
/tmp/st_w4a16_classifier_gate_20260531/pubmed_h5s4_T24_kl020_tau050.log
/tmp/st_w4a16_classifier_gate_refine_20260531/pubmed_T25_tau050.log
/tmp/st_w4a16_classifier_local_20260531/pubmed_T25_tau040.log
/tmp/st_w4a16_classifier_after_residual_20260531/pubmed_T25_alpha025_tau050.log
/tmp/st_w4a16_classifier_after_residual_20260531/pubmed_T25_alpha0125_tau050.log
```

旧 Llama 日志：

```text
/tmp/llama_cora_T31_tau_relax_3run_20260530/tau040.log
/tmp/llama_T31_final_3run_20260530/pubmed.log
```

## 当前 main 分支复现入口

已提供统一脚本：

```bash
bash GraphhopSimhash/scripts/run_t31_shared_frontend_reuse.sh
```

只跑其中一组：

```bash
CASES="llama_cora" RUNS=3 \
bash GraphhopSimhash/scripts/run_t31_shared_frontend_reuse.sh
```

四组 case 对应关系：

| Case | Embedding 源 | 数据集 | gate 设置 |
|---|---|---|---|
| `st_cora` | `data.x` | Cora | `separate`, tau=0.575 |
| `st_pubmed` | `data.x` | PubMed | `shared`, tau=0.65 |
| `llama_cora` | `llama2_7b:W4A16` | Cora | classifier-aware `separate`, tau=0.40 |
| `llama_pubmed` | `llama2_7b:W4A16` | PubMed | `shared`, tau=0.91 |

输出目录：

```text
output/t31_shared_frontend_reuse/logs/
```

## 接入 Graph-Bit 全栈实验

后续 Graph-Bit full-stack 默认使用同一套在线前端：

```text
8 heads x 16 bit
radius = 2
T = 31
hard direct: support >= 5
residual candidate: support = 3..4
compute / Graph-Bit miss: support < 3 或 residual accept reject
```

入口脚本：

```bash
RUNS=10 DATASET=cora \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```

其中 Cora 默认启用 classifier-aware accept gate；PubMed 默认使用 shared accept gate 高阈值：

```bash
RUNS=3 DATASET=pubmed \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```
