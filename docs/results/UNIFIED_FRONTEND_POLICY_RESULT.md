# Unified Front-End Policy Register

本文档归档当前 SimHash + residual-gate + progressive BFP full-stack 的前端参数收敛结果。核心结论是：硬件和在线控制流保持固定，`T` 作为离线 profiling 后写入的 dataset-level policy register。

## 1. 固定在线配置

当前主线固定以下在线配置：

```text
SimHash:
    8 heads x 16 bit
    radius = 2

Score gate:
    TSER weights = 3 / 1 / 1
    score_reuse_threshold = T

Support split:
    support >= 5   -> direct reuse
    support = 3..4 -> residual-gate candidate
    support < 3    -> encoder path

Residual gate:
    MLP adapter
    rank = 64
    epochs = 200
    max_train_pairs = 4096
    accept threshold = 0.575

Encoder miss path:
    reference = W4BFPA8_B128
    base path = W4BFPA4_B128
    refinement path = W4BFPA6_B128
```

这里的 `T` 不改变硬件结构，也不改变在线执行状态机。它只决定 score gate 的接收宽松程度，因此适合作为离线 profiling 后写入的 dataset-level policy register。

## 2. 当前结果

### 2.1 Cora

当前 Cora 使用 `T=31`、`refine_ratio=0.30`，10 runs：

```text
log: output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.30/logs/cora_runs10.log
```

| Config | Reuse | Direct | Residual | Cost | Acc | Drop |
|---|---:|---:|---:|---:|---:|---:|
| FullP8 | 39.5% | 18.6% | 20.9% | 0.304 | 0.6964 | 1.64% |
| Rand | 39.5% | 18.6% | 20.9% | 0.194 | 0.6917 | 2.12% |
| Deg | 39.5% | 18.6% | 20.9% | 0.194 | 0.6906 | 2.22% |
| TSER | 39.5% | 18.6% | 20.9% | 0.194 | 0.6905 | 2.24% |

Cora 上 `T=31` 已经接近目标区间：FullP8 miss baseline 掉点约 1.6%，progressive BFP 后端进一步降低 cost，但 drop 会到约 2.1-2.2%。

### 2.2 PubMed

当前 PubMed 使用 `T=31`、`refine_ratio=0.30`，3 runs：

```text
log: output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.30/logs/pubmed_runs3.log
```

| Config | Reuse | Direct | Residual | Cost | Acc | Drop |
|---|---:|---:|---:|---:|---:|---:|
| FullP8 | 59.5% | 42.7% | 16.9% | 0.203 | 0.7172 | 3.50% |
| Rand | 59.5% | 42.7% | 16.9% | 0.130 | 0.7104 | 4.18% |
| Deg | 59.5% | 42.7% | 16.9% | 0.130 | 0.7126 | 3.96% |
| TSER | 59.5% | 42.7% | 16.9% | 0.130 | 0.7124 | 3.98% |

PubMed 上 `T=31` 明显偏松。FullP8 miss baseline 本身已经掉 3.50%，说明主要误差来自前端 reuse/residual 接收过宽，而不是 BFPA6/BFPA4 后端。

已有 T sweep 结果显示 PubMed 需要收紧到 `T≈24` 一带再进入 BFP 后端主表。

### 2.3 Arxiv

Arxiv 当前用单 run 做 T sweep，`refine_ratio=0.25`：

| T | FullP8 Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | Deg Cost | Deg Drop | Log |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 51.5% | 23.2% | 28.3% | 0.244 | 2.66% | 0.154 | 2.73% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T24_r0.25_runs1/logs/arxiv_runs1.log` |
| 26 | 57.3% | 25.8% | 31.5% | 0.215 | 2.90% | 0.136 | 2.94% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T26_r0.25_runs1/logs/arxiv_runs1.log` |
| 28 | 58.9% | 26.5% | 32.4% | 0.207 | 3.20% | 0.131 | 3.31% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T28_r0.25_runs1/logs/arxiv_runs1.log` |

Arxiv 的 T24 仍然略宽。下一步需要确认 `T=20/22/23`，目标是在 FullP8 miss baseline drop 约 2% 左右时尽量保持 reuse 在 45% 以上。

## 3. 当前推荐策略

当前更合理的论文表述是：

```text
硬件配置固定。
在线控制流固定。
T 是 dataset-level policy register。
T 由离线 profiling 写入，不需要改硬件。
```

当前暂定策略：

| Dataset | T | 状态 |
|---|---:|---|
| Cora | 31 | 已有 10-run 结果 |
| PubMed | 24 | 需要补 progressive BFP full-stack 复核 |
| Arxiv | 22 或 23 | 正在补低 T sweep |

如果强行要求三个数据集使用完全相同的 `T`，当前 `T=24` 对 Cora 会过保守，对 Arxiv 仍略宽。因此更稳的主线是固定硬件，用离线 profiling 写入每个数据集的 `T`。

## 4. 复现实验

Cora:

```bash
DATASET=cora RUNS=10 THRESHOLD=31 REFINE_RATIO=0.30 \
  OUT_DIR=output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.30 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

PubMed T31:

```bash
DATASET=pubmed RUNS=3 THRESHOLD=31 REFINE_RATIO=0.30 \
  OUT_DIR=output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.30 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

Arxiv single-T:

```bash
DATASET=arxiv RUNS=1 THRESHOLD=24 REFINE_RATIO=0.25 FORCE=1 \
  OUT_DIR=output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T24_r0.25_runs1 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

Arxiv lower-T queue:

```bash
bash GraphhopSimhash/scripts/run_arxiv_lower_t_policy_sweep.sh
```

