# Graph-Aware BFP Progressive Refinement Results

本文档记录当前 `W4BFPA*_B128` embedding pools 上的 progressive refinement 实验结果。核心问题是：

```text
是否可以默认使用低成本 BFPA4，
只把少量高风险节点提升到 BFPA6 / BFPA8，
在较低额外 cost 下恢复下游 GNN 精度。
```

## 1. 实验设置

使用已经生成好的 LLaMA-7B rowwise BFP embedding pools：

```text
P8 = W4BFPA8_B128
P6 = W4BFPA6_B128
P4 = W4BFPA4_B128
```

这里的 `B128` 表示每 128 个 activation value 共享一个 block exponent。当前结果是 embedding-pool 级 accuracy validation：

```text
先离线生成不同 BFP precision 的 embedding pools，
再按节点级 routing 策略从 P8/P6/P4 pool 中选择 embedding，
最后真实运行下游 GNN 分类并统计 Acc / Drop。
```

它验证的是节点级 BFP routing 对下游精度是否有效；不是在线 kernel latency，也不是真实 mixed-BFP NPU 实测时间。

## 2. Cost Proxy

当前 cost proxy 沿用 precision-depth 评估中的设置：

| Format | Cost | Extra Cost vs BFPA4 |
|---|---:|---:|
| BFPA4 | 0.287 | - |
| BFPA6 | 0.394 | +0.107 |
| BFPA8 | 0.500 | +0.213 |

因此：

```text
BFPA4 -> BFPA6: 约 +37.3% cost
BFPA4 -> BFPA8: 约 +74.2% cost
```

BFPA6 与 BFPA8 的下游精度非常接近，说明 BFP 的 shared exponent 已经保住了主要动态范围；BFPA8 多出的低位 mantissa 对当前 Cora/PubMed 分类边界帮助有限。

## 3. Cora / LLaMA-7B

日志目录：

```text
output/graphbfp_progressive_refinement/cora/
```

5-run 主表如下：

| Setting | Cost | Rand Drop | Degree Drop | TSER Drop | Best |
|---|---:|---:|---:|---:|---|
| All BFPA4 | 0.287 | - | - | - | 0.99% baseline |
| 10% BFPA6 + 90% BFPA4 | 0.298 | 0.89% | 0.82% | 0.76% | TSER |
| 20% BFPA6 + 80% BFPA4 | 0.309 | 0.74% | 0.68% | 0.54% | TSER |
| 30% BFPA6 + 70% BFPA4 | 0.319 | 0.64% | 0.55% | 0.49% | TSER |
| 10% BFPA8 + 90% BFPA4 | 0.309 | 0.91% | 0.77% | 0.76% | TSER |
| 20% BFPA8 + 80% BFPA4 | 0.330 | 0.69% | 0.53% | 0.47% | TSER |
| 30% BFPA8 + 70% BFPA4 | 0.351 | 0.72% | 0.58% | 0.44% | TSER |
| 10% BFPA8 + 20% BFPA6 + 70% BFPA4 | 0.330 | 0.72% | 0.63% | 0.46% | TSER |
| 20% BFPA8 + 20% BFPA6 + 60% BFPA4 | 0.351 | 0.68% | 0.74% | 0.36% | TSER |

Cora 结论：

```text
All BFPA4 已经只有约 1% drop。
只提升 20%-30% 高风险节点到 BFPA6/BFPA8，可以把 drop 拉回到 0.36%-0.55%。
TSER 在 Cora 上比 Degree 更强，说明拓扑感知语义风险对 BFP refinement 有额外价值。
```

推荐 Cora 配置：

| Goal | Config | Selector | Cost | Drop |
|---|---|---|---:|---:|
| 低成本 | 20% BFPA6 + 80% BFPA4 | TSER | 0.309 | 0.54% |
| 平衡 | 30% BFPA6 + 70% BFPA4 | TSER | 0.319 | 0.49% |
| 更低掉点 | 20% BFPA8 + 20% BFPA6 + 60% BFPA4 | TSER | 0.351 | 0.36% |

## 4. PubMed / LLaMA-7B

日志目录：

```text
output/graphbfp_progressive_refinement/pubmed/
output/graphbfp_progressive_refinement/pubmed_sweep_extended/
```

PubMed 单档 BFPA6 refinement：

| Setting | Cost | Rand Drop | Degree Drop | TSER Drop | Best |
|---|---:|---:|---:|---:|---|
| All BFPA4 | 0.287 | - | - | - | 1.19% baseline |
| 5% BFPA6 + 95% BFPA4 | 0.293 | 1.12% | 0.93% | 0.96% | Degree |
| 10% BFPA6 + 90% BFPA4 | 0.298 | 1.08% | 0.79% | 0.83% | Degree |
| 15% BFPA6 + 85% BFPA4 | 0.303 | 1.01% | 0.70% | 0.75% | Degree |
| 20% BFPA6 + 80% BFPA4 | 0.309 | 0.95% | 0.61% | 0.68% | Degree |
| 25% BFPA6 + 75% BFPA4 | 0.314 | 0.89% | 0.58% | 0.62% | Degree |
| 30% BFPA6 + 70% BFPA4 | 0.319 | 0.81% | 0.54% | 0.58% | Degree |
| 40% BFPA6 + 60% BFPA4 | 0.330 | 0.69% | 0.44% | 0.46% | Degree |

PubMed 单档 BFPA8 refinement：

| Setting | Cost | Rand Drop | Degree Drop | TSER Drop | Best |
|---|---:|---:|---:|---:|---|
| 5% BFPA8 + 95% BFPA4 | 0.298 | 1.11% | 0.92% | 0.96% | Degree |
| 10% BFPA8 + 90% BFPA4 | 0.309 | 1.06% | 0.77% | 0.81% | Degree |
| 15% BFPA8 + 85% BFPA4 | 0.319 | 1.00% | 0.69% | 0.74% | Degree |
| 20% BFPA8 + 80% BFPA4 | 0.330 | 0.95% | 0.62% | 0.67% | Degree |
| 25% BFPA8 + 75% BFPA4 | 0.341 | 0.89% | 0.58% | 0.59% | Degree |
| 30% BFPA8 + 70% BFPA4 | 0.351 | 0.81% | 0.53% | 0.56% | Degree |
| 40% BFPA8 + 60% BFPA4 | 0.373 | 0.68% | 0.42% | 0.44% | Degree |

PubMed hybrid refinement：

| Setting | Cost | Rand Drop | Degree Drop | TSER Drop | Best |
|---|---:|---:|---:|---:|---|
| 5% BFPA8 + 10% BFPA6 + 85% BFPA4 | 0.309 | 1.01% | 0.70% | 0.75% | Degree |
| 10% BFPA8 + 10% BFPA6 + 80% BFPA4 | 0.319 | 0.95% | 0.60% | 0.67% | Degree |
| 15% BFPA8 + 15% BFPA6 + 70% BFPA4 | 0.335 | 0.81% | 0.53% | 0.56% | Degree |
| 10% BFPA8 + 30% BFPA6 + 60% BFPA4 | 0.341 | 0.69% | 0.42% | 0.44% | Degree |
| 20% BFPA8 + 20% BFPA6 + 60% BFPA4 | 0.351 | 0.70% | 0.43% | 0.43% | Degree / TSER |

PubMed 结论：

```text
Degree 是 PubMed 上最稳定的 selector。
BFPA8 相比 BFPA6 只有很小的额外精度收益，但 cost 明显更高。
PubMed 的性价比前沿主要在 BFPA6 refinement，而不是 BFPA8 refinement。
```

推荐 PubMed 配置：

| Goal | Config | Selector | Cost | Drop |
|---|---|---|---:|---:|
| 极低成本 | 10% BFPA6 + 90% BFPA4 | Degree | 0.298 | 0.79% |
| 平衡 | 20% BFPA6 + 80% BFPA4 | Degree | 0.309 | 0.61% |
| 稳健 | 30% BFPA6 + 70% BFPA4 | Degree | 0.319 | 0.54% |
| 更低掉点 | 40% BFPA6 + 60% BFPA4 | Degree | 0.330 | 0.44% |

## 5. Cross-Dataset Takeaway

当前 Cora/PubMed 的共同规律：

```text
1. BFPA4 是强低成本底座。
2. BFPA6 是最有性价比的 refinement 档。
3. BFPA8 的额外收益很小，除非追求极低 drop，否则不应作为默认 refinement 档。
4. Cora 更适合 TSER，因为拓扑感知语义风险更明显。
5. PubMed 更适合 Degree，因为图更同质，传播风险与 degree 更强相关。
```

统一主线可以写成：

```text
Default: BFPA4
Refinement: graph-risk top 20%-30% -> BFPA6
Selector:
    Cora: TSER 更强
    PubMed: Degree 更强
    跨数据集叙事可使用 TSER，或使用 Degree 作为 hardware-friendly baseline
```

建议主表配置：

| Dataset | Config | Selector | Cost | Drop |
|---|---|---|---:|---:|
| Cora | 30% BFPA6 + 70% BFPA4 | TSER | 0.319 | 0.49% |
| PubMed | 30% BFPA6 + 70% BFPA4 | Degree | 0.319 | 0.54% |

## 6. SimHash / Residual-Gate + Progressive BFP Full Stack

前面的表只评估 BFP routing 本身；这一节接入当前前端：

```text
SimHash:
    8 heads x 16 bits, radius = 2

Score gate:
    T = 31, TSER weights = 3 / 1 / 1

Support split:
    support >= 5  -> direct reuse
    support = 3..4 -> residual-gate candidate
    support < 3   -> encoder

Encoder:
    miss / reject nodes 默认 BFPA4
    top 30% miss nodes 提升到 BFPA6
```

Cora 10-run 日志：

```text
output/progressive_bfp_fullstack/cora_h8_53_T31_bfpa6_r0.30/logs/cora_runs10.log
```

主表如下：

| Config | Reuse | Direct | Residual | P6 | P4 | Cost | Acc | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 | 39.5% | 18.6% | 20.9% | 0.0% | 0.0% | 0.304 | 0.6964 | 1.64% |
| AllP6 | 39.5% | 18.6% | 20.9% | 60.5% | 0.0% | 0.239 | 0.6953 | 1.76% |
| AllP4 | 39.5% | 18.6% | 20.9% | 0.0% | 60.5% | 0.175 | 0.6883 | 2.46% |
| Rand | 39.5% | 18.6% | 20.9% | 18.1% | 42.3% | 0.194 | 0.6917 | 2.12% |
| Deg | 39.5% | 18.6% | 20.9% | 18.1% | 42.3% | 0.194 | 0.6906 | 2.22% |
| TSER | 39.5% | 18.6% | 20.9% | 18.1% | 42.3% | 0.194 | 0.6905 | 2.24% |
| Uniq | 39.5% | 18.6% | 20.9% | 18.1% | 42.3% | 0.194 | 0.6908 | 2.21% |

这里 `Deg / TSER / Rand` 的 P6/P4 比例相同，区别只是 miss nodes 内部谁被提升到 BFPA6：

```text
Deg:
    按 propagation / degree risk 排序。

TSER:
    按 TSER sensitivity score 排序。

Rand:
    在同一 miss-node 集合中随机选择同样数量的节点。
```

这组结果说明：

```text
1. 前端 residual reuse 已经约简约 39.5% encoder calls。
2. 后端 Progressive BFP 可以把 cost 从 FullP8 的 0.304 降到 0.194。
3. 在当前 Cora full-stack 设置下，Deg / TSER / Rand 差距很小。
4. 纯 degree 或 TSER 对 BFPA4/BFPA6 refinement 的指导性不足。
```

原因是后端 BFP refinement 的误差来源不只来自图传播风险，还来自 BFP 数值格式本身：

```text
activation dynamic range
shared exponent stress
outlier block
P4/P6 embedding damage
```

因此当前已经实现的 full-stack 路径可以作为 baseline。后续更合理的后端 selector 应该从：

```text
graph risk only
```

推进到：

```text
graph risk + BFP numerical stress
```

也就是在 miss nodes 中联合考虑节点传播风险和 BFP block / embedding damage proxy，再决定哪些节点从 BFPA4 refine 到 BFPA6。

## 7. Reproduction Command

SimHash / Residual-Gate + Progressive BFP full-stack:

```bash
DATASET=cora RUNS=10 REFINE_BIT=6 REFINE_RATIO=0.30 FORCE=1 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

其中：

```text
REFINE_BIT=6:
    miss nodes 默认 BFPA4，top-risk 30% 提升到 BFPA6。

REFINE_BIT=5:
    miss nodes 默认 BFPA4，top-risk 30% 提升到 BFPA5。
```

下面命令只评估 BFP routing 本身，不接 SimHash / residual 前端。

P6 refinement example:

```bash
python -m GraphhopSimhash \
  --datasets pubmed \
  --runs 5 \
  --experiment_suite precision_depth_ablation \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4BFPA8_B128 \
  --precision_depth_reference_bits 8 \
  --precision_depth_tags W4BFPA6_B128 W4BFPA4_B128 \
  --precision_depth_bits 6 4 \
  --precision_depth_high_ratio 0.0 \
  --precision_depth_mid_ratio 0.30 \
  --precision_depth_low_ratio 0.0
```

P8 refinement example:

```bash
python -m GraphhopSimhash \
  --datasets pubmed \
  --runs 5 \
  --experiment_suite precision_depth_ablation \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4BFPA8_B128 \
  --precision_depth_reference_bits 8 \
  --precision_depth_tags W4BFPA6_B128 W4BFPA4_B128 \
  --precision_depth_bits 6 4 \
  --precision_depth_high_ratio 0.30 \
  --precision_depth_mid_ratio 0.0 \
  --precision_depth_low_ratio 0.0
```

Hybrid refinement example:

```bash
python -m GraphhopSimhash \
  --datasets pubmed \
  --runs 5 \
  --experiment_suite precision_depth_ablation \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4BFPA8_B128 \
  --precision_depth_reference_bits 8 \
  --precision_depth_tags W4BFPA6_B128 W4BFPA4_B128 \
  --precision_depth_bits 6 4 \
  --precision_depth_high_ratio 0.10 \
  --precision_depth_mid_ratio 0.20 \
  --precision_depth_low_ratio 0.0
```
