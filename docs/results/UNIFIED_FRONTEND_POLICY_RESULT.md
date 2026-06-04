# Unified Front-End Policy Register

本文档归档当前 SimHash + residual-gate + progressive BFP full-stack 的前端参数检索结果。

核心设定是：硬件结构和在线状态机固定，`T` 作为离线 profiling 后写入的 dataset-level policy register。`T` 只改变 score gate 的接收宽松程度，不改变 SimHash/CAM、Residual-Gate 或 Progressive-BFP NPU 的硬件结构。

## 1. 固定在线配置

当前主线固定以下在线控制流：

```text
SimHash:
    8 heads x 16 bit
    radius = 2

TSER score gate:
    weights = 3 / 1 / 1
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

在 full-stack 表中：

```text
FullP8:
    reuse/residual 前端保持不变；
    所有 miss nodes 走 W4BFPA8_B128。

Rand / Deg / TSER:
    reuse/residual 前端保持不变；
    miss nodes 中一部分走 W4BFPA6_B128 refinement，
    其余走 W4BFPA4_B128 base。
```

因此 `FullP8 Drop` 主要反映前端 reuse/residual 的误差；`Deg/TSER Drop` 反映前端误差再叠加 progressive BFP 后端误差。

## 2. 当前推荐点

当前最稳的三数据集 policy register 如下：

| Dataset | T | Refine Ratio | Runs | Reuse | Direct | Residual | FullP8 Drop | Rand Drop | Deg Drop | TSER Drop | BFP Cost | 推荐用途 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cora | 31 | 0.25 | 10 | 39.0% | 18.5% | 20.5% | 1.56% | 2.10% | 2.15% | 2.21% | 0.193 | 稳健主点 |
| PubMed | 31 | 0.25 | 3 | 42.3% | 42.2% | 0.0% | 1.95% | 2.69% | 2.52% | 2.53% | 0.181 | 当前可用点 |
| Arxiv | 22 | 0.25 | 1 | 46.2% | 20.4% | 25.8% | 2.02% | 2.06% | 2.11% | 2.10% | 0.170 | 当前最佳折中 |

解释：

```text
Cora:
    T31 比 T30 更稳，10-run 下 FullP8 drop 约 1.56%。
    progressive BFP 后端把 cost 从 0.306 降到 0.193，Rand/Deg/TSER drop 分别约 2.10%/2.15%/2.21%。

PubMed:
    T31 + refine_ratio=0.25 时前端 reuse 约 42.3%，FullP8 drop 约 1.95%。
    继续提高 refine_ratio 到 0.30 会引入 residual bucket，FullP8 drop 升到 3.50%，过宽。

Arxiv:
    T22 是当前低 T sweep 中最合适的点。
    T20 更稳但 reuse 只有 36.8%；T23 reuse 接近 50%，但 drop 已到 2.46%。
```

## 3. Cora 结果

### 3.1 T31 refine-ratio sweep

Cora 使用 `T=31`，10 runs：

| Refine Ratio | Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | BFP Cost | Rand Drop | Deg Drop | TSER Drop | Log |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 39.1% | 18.4% | 20.7% | 0.306 | 1.74% | 0.179 | 2.43% | 2.41% | 2.36% | `output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.05/logs/cora_runs10.log` |
| 0.10 | 39.1% | 18.4% | 20.7% | 0.305 | 1.74% | 0.183 | 2.40% | 2.40% | 2.40% | `output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.10/logs/cora_runs10.log` |
| 0.15 | 38.9% | 18.5% | 20.4% | 0.307 | 1.59% | 0.186 | 2.20% | 2.19% | 2.23% | `output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.15/logs/cora_runs10.log` |
| 0.20 | 39.1% | 18.4% | 20.7% | 0.305 | 1.70% | 0.189 | 2.24% | 2.27% | 2.29% | `output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.20/logs/cora_runs10.log` |
| 0.25 | 39.0% | 18.5% | 20.5% | 0.306 | 1.56% | 0.193 | 2.10% | 2.15% | 2.21% | `output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.25/logs/cora_runs10.log` |
| 0.30 | 39.5% | 18.6% | 20.9% | 0.304 | 1.64% | 0.194 | 2.12% | 2.22% | 2.24% | `output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.30/logs/cora_runs10.log` |

当前 Cora 推荐用 `T=31, refine_ratio=0.25`。它比 `0.30` 略稳，Deg drop 为 `2.15%`。

### 3.2 Cora T30 高复用参考

`T=30, refine_ratio=0.25`，3 runs：

| Config | Reuse | Direct | Residual | Cost | Acc | Drop |
|---|---:|---:|---:|---:|---:|---:|
| FullP8 | 49.7% | 17.7% | 32.0% | 0.253 | 0.6797 | 2.10% |
| Deg | 49.7% | 17.7% | 32.0% | 0.160 | 0.6779 | 2.27% |
| TSER | 49.7% | 17.7% | 32.0% | 0.160 | 0.6778 | 2.29% |

日志：

```text
output/progressive_bfp_fullstack/cora_h8_53_T30_bfpa6_r0.25/logs/cora_runs3.log
```

这个点 reuse 更高，但目前只有 3 runs，且 FullP8 drop 已经到 2.10%。主表暂时使用更稳的 `T31`。

## 4. PubMed 结果

PubMed 使用 `T=31`，3 runs。这里 refine ratio 对 front-end 的 accepted reuse 也会产生影响，因为 residual gate / alpha 选择会随训练和验证目标变化。

| Refine Ratio | Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | BFP Cost | Rand Drop | Deg Drop | TSER Drop | Log |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 42.7% | 42.6% | 0.0% | 0.287 | 2.01% | 0.168 | 2.96% | 2.80% | 2.82% | `output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.05/logs/pubmed_runs3.log` |
| 0.10 | 42.7% | 42.7% | 0.0% | 0.287 | 2.01% | 0.171 | 2.92% | 2.77% | 2.75% | `output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.10/logs/pubmed_runs3.log` |
| 0.15 | 42.7% | 42.6% | 0.0% | 0.287 | 2.01% | 0.174 | 2.90% | 2.72% | 2.71% | `output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.15/logs/pubmed_runs3.log` |
| 0.20 | 42.7% | 42.7% | 0.0% | 0.287 | 2.03% | 0.177 | 2.87% | 2.65% | 2.64% | `output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.20/logs/pubmed_runs3.log` |
| 0.25 | 42.3% | 42.2% | 0.0% | 0.289 | 1.95% | 0.181 | 2.69% | 2.52% | 2.53% | `output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.25/logs/pubmed_runs3.log` |
| 0.30 | 59.5% | 42.7% | 16.9% | 0.203 | 3.50% | 0.130 | 4.18% | 3.96% | 3.98% | `output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.30/logs/pubmed_runs3.log` |

当前 PubMed 推荐用 `T=31, refine_ratio=0.25`。`0.30` 虽然 reuse 提高到 `59.5%`，但前端过宽，FullP8 drop 已经达到 `3.50%`。

## 5. Arxiv 结果

Arxiv 当前为 single-run T sweep，`refine_ratio=0.25`。

| T | Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | BFP Cost | Rand Drop | Deg Drop | TSER Drop | Log |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 36.8% | 15.6% | 21.2% | 0.317 | 1.77% | 0.199 | 1.84% | 1.89% | 1.86% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T20_r0.25_runs1/logs/arxiv_runs1.log` |
| 22 | 46.2% | 20.4% | 25.8% | 0.270 | 2.02% | 0.170 | 2.06% | 2.11% | 2.10% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T22_r0.25_runs1/logs/arxiv_runs1.log` |
| 23 | 49.8% | 22.0% | 27.8% | 0.252 | 2.32% | 0.159 | 2.47% | 2.46% | 2.48% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T23_r0.25_runs1/logs/arxiv_runs1.log` |
| 24 | 51.5% | 23.2% | 28.3% | 0.244 | 2.66% | 0.154 | 2.79% | 2.73% | 2.77% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T24_r0.25_runs1/logs/arxiv_runs1.log` |
| 26 | 57.3% | 25.8% | 31.5% | 0.215 | 2.90% | 0.136 | 2.96% | 2.94% | 2.96% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T26_r0.25_runs1/logs/arxiv_runs1.log` |
| 28 | 58.9% | 26.5% | 32.4% | 0.207 | 3.20% | 0.131 | 3.29% | 3.31% | 3.32% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T28_r0.25_runs1/logs/arxiv_runs1.log` |
| 30 | 61.3% | 26.5% | 34.8% | 0.195 | 3.34% | 0.123 | 3.47% | 3.46% | 3.47% | `output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T30_r0.25_runs1/logs/arxiv_runs1.log` |

当前 Arxiv 推荐用 `T=22`。它在 single-run 下达到 `46.2%` reuse，FullP8 drop 为 `2.02%`，Deg/TSER progressive BFP drop 约 `2.1%`。

## 6. 当前结论

当前三数据集的统一硬件路径已经稳定：

```text
SimHash/LRU-CAM
    -> TSER score gate
    -> Residual-Gate fuzzy reuse
    -> Progressive BFP encoder for miss nodes
```

在线状态机固定，数据集差异主要由 `T` 和离线 residual training 处理：

| Dataset | 当前 policy | 结论 |
|---|---|---|
| Cora | `T=31, refine_ratio=0.25` | 稳健，Deg drop 约 2.15%，reuse 约 39.0% |
| PubMed | `T=31, refine_ratio=0.25` | 可用，FullP8 drop 约 1.95%，Deg drop 约 2.52% |
| Arxiv | `T=22, refine_ratio=0.25` | 当前最佳折中，reuse 约 46.2%，Deg/TSER drop 约 2.1% |

如果论文主表要严格控制在 `2%` 内，可以使用：

```text
Cora:
    FullP8 row 或更低 refine_ratio 的 conservative row。

PubMed:
    T31, refine_ratio=0.25 的 FullP8 row 在 2% 内；
    progressive BFP row 需要接受约 2.5% drop，或者降低 BFPA4 比例。

Arxiv:
    T20 更稳，T22 更平衡。
```

如果目标是“reuse 尽可能高，drop 约 2% 上下”，当前推荐：

```text
Cora:   T31, refine_ratio=0.25
PubMed: T31, refine_ratio=0.25
Arxiv:  T22, refine_ratio=0.25
```

## 7. 复现实验

Cora:

```bash
DATASET=cora RUNS=10 THRESHOLD=31 REFINE_RATIO=0.25 \
  OUT_DIR=output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.25 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

PubMed:

```bash
DATASET=pubmed RUNS=3 THRESHOLD=31 REFINE_RATIO=0.25 \
  OUT_DIR=output/progressive_bfp_fullstack/pubmed_h8_53_T31_bfpa6_r0.25 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

Arxiv:

```bash
DATASET=arxiv RUNS=1 THRESHOLD=22 REFINE_RATIO=0.25 FORCE=1 \
  OUT_DIR=output/progressive_bfp_fullstack_unified_t_sweep/arxiv_T22_r0.25_runs1 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

Arxiv T sweep:

```bash
T_VALUES="20 22 23" RUNS=1 REFINE_RATIO=0.25 \
  bash GraphhopSimhash/scripts/run_arxiv_lower_t_policy_sweep.sh
```
