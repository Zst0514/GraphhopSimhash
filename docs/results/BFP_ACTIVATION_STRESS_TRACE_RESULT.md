# BFP Activation Block Stress Trace

本文记录把 BFP stress 从 final embedding proxy 下沉到真实 LLaMA activation block trace 后的验证结果。

## 1. 目的

前一版 BFP stress 使用 final embedding 的 block 分布近似 BFP 风险。这个 proxy 能解释一部分数值误差，但不一定对应 LLaMA encoder 内部 activation 的真实 shared-exponent 压力。

本实验直接 hook LLaMA-7B 的 Linear 输入 activation，统计每个节点在真实 encoder GEMM 前的 BFP block stress，再检查它是否能解释：

- BFPA4 相对 BFPA8 的数值误差；
- BFPA4 到 BFPA6 的可挽救空间；
- 下游 GNN hidden / logit / damage 的变化。

## 2. 实现

新增脚本：

```bash
GraphhopSimhash/scripts/diagnose_graphbfp_activation_stress.py
```

流程：

```text
sample graph nodes
    -> run LLaMA forward on raw node text
    -> hook selected Linear input activations
    -> compute per-node BFP block stress
    -> align with BFPA8 / BFPA4 / BFPA6 embedding pools
    -> train/evaluate GNN
    -> correlate activation stress with downstream damage
```

当前 hook 的模块：

```text
layers = 0 / 15 / 31
modules = q_proj / o_proj / up_proj / down_proj
```

每个 sampled node 的 activation trace 会统计：

```text
act_stress_mean / p90
act_outlier_mean / p90
act_zero_pressure
act_bfpa4_err
act_bfpa6_err
act_rescue = act_bfpa4_err - act_bfpa6_err
```

这里的 BFP block 是 rowwise hidden-dim B128，与当前 W4BFPAx_B128 pool 的 activation format 对齐。

## 3. 复现命令

Cora:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/diagnose_graphbfp_activation_stress.py \
  --dataset cora \
  --runs 3 \
  --sample_nodes 128 \
  --batch_size 2 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj o_proj up_proj down_proj \
  --output_dir output/graphbfp_activation_stress/cora_s128_runs3
```

PubMed:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/diagnose_graphbfp_activation_stress.py \
  --dataset pubmed \
  --runs 3 \
  --sample_nodes 128 \
  --batch_size 2 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj o_proj up_proj down_proj \
  --output_dir output/graphbfp_activation_stress/pubmed_s128_runs3
```

## 4. Cora 结果

输出目录：

```text
output/graphbfp_activation_stress/cora_s128_runs3
```

整体精度：

| Target | Acc | Drop |
| --- | ---: | ---: |
| BFPA8 reference | 0.7007 | 0.00% |
| All BFPA4 | 0.6958 | 0.48% |
| All BFPA6 | 0.7018 | -0.11% |

activation trace 信号质量：

| Signal | act_err | hidden_err | hidden_gain | margin_drop | damage AUC | top25 gain overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ActStress | -0.140 | 0.220 | 0.181 | -0.105 | 0.602 | 0.552 |
| ActRescue | 0.966 | -0.176 | -0.156 | -0.065 | 0.300 | 0.146 |
| Degree | -0.160 | 0.091 | 0.084 | 0.027 | 0.480 | 0.302 |
| TSER | -0.190 | 0.138 | 0.138 | 0.011 | 0.573 | 0.312 |
| TSER+ActStress | -0.195 | 0.163 | 0.161 | -0.002 | 0.605 | 0.312 |

观察：

- `ActStress` 对 hidden error / hidden gain / damage 的解释明显强于 embedding-level stress proxy。
- `ActStress` 的 top25 gain overlap 达到 0.552，说明真实 activation block stress 能更好找出 BFPA4 下可能需要 BFPA6 refinement 的节点。
- `ActRescue` 几乎完美解释 activation-level quantization rescue，但它不等价于下游任务 damage；这说明仅看局部 activation 数值误差还不够，仍需要图任务风险配合。

## 5. PubMed 结果

输出目录：

```text
output/graphbfp_activation_stress/pubmed_s128_runs3
```

整体精度：

| Target | Acc | Drop |
| --- | ---: | ---: |
| BFPA8 reference | 0.7522 | 0.00% |
| All BFPA4 | 0.7406 | 1.17% |
| All BFPA6 | 0.7518 | 0.05% |

activation trace 信号质量：

| Signal | act_err | hidden_err | hidden_gain | margin_drop | damage AUC | top25 gain overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ActStress | -0.103 | 0.140 | 0.142 | 0.082 | 0.611 | 0.135 |
| ActRescue | -0.103 | 0.140 | 0.142 | 0.082 | 0.611 | 0.135 |
| Degree | -0.265 | 0.112 | 0.114 | 0.072 | 0.684 | 0.271 |
| TSER | -0.332 | 0.139 | 0.139 | 0.042 | 0.704 | 0.260 |
| Degree+ActStress | -0.271 | 0.098 | 0.099 | 0.033 | 0.751 | 0.156 |
| TSER+ActStress | -0.322 | 0.130 | 0.128 | -0.005 | 0.761 | 0.219 |

观察：

- PubMed 上 graph risk 对 downstream damage 的解释更强，TSER / Degree 的 damage AUC 高于纯 activation stress。
- `TSER+ActStress` 和 `Degree+ActStress` 的 damage AUC 分别达到 0.761 / 0.751，说明 activation stress 和图风险结合后能更稳定定位脆弱节点。
- top-k rescue overlap 不高，说明 PubMed 的“数值可挽救节点”和“任务真正脆弱节点”并不完全一致。

## 6. 结论

这一步验证了两件事：

1. embedding-level BFP stress proxy 不够精确，真实 LLaMA activation block trace 更接近 BFP shared-exponent 的风险来源。
2. 单独 activation stress 仍然不足以决定 refinement，必须和图风险结合：

```text
activation stress:
    当前 block 是否容易被 shared exponent 污染。

graph risk:
    该节点的 embedding 误差是否会放大到 GNN 下游。
```

因此更合理的后端策略不是只按 Degree / TSER，也不是只按 BFP stress，而是：

```text
Graph-risk × Activation-stress refinement
```

当前结果支持继续把 BFP controller 从 embedding proxy 改成 activation-trace / activation-stress guided 的版本。

## 7. Full-Node / Large-Sample Policy Test

上面的 signal validation 说明 activation stress 有信息量，但还不能说明它直接就是最好的 refinement selector。因此继续做了 policy-level 测试。

新增脚本：

```bash
GraphhopSimhash/scripts/evaluate_graphbfp_activation_policy_from_trace.py
```

它不重新跑 LLaMA，而是读取已经保存的：

```text
activation_node_trace.tsv
```

然后重新训练/评估 GNN，测试哪些 traced nodes 应该从 BFPA4 lift 到 BFPA6。

### 7.1 Cora Full-Node Policy

输入 trace：

```text
output/graphbfp_activation_stress/cora_fullnodes_runs3/activation_node_trace.tsv
```

这里 `trace_nodes=2708`，也就是 Cora 全图节点。

复现命令：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/evaluate_graphbfp_activation_policy_from_trace.py \
  --dataset cora \
  --runs 3 \
  --trace_path output/graphbfp_activation_stress/cora_fullnodes_runs3/activation_node_trace.tsv \
  --ratios 0.10 0.20 0.25 0.30 0.40 \
  --output_dir output/graphbfp_activation_policy/cora_fullnodes_runs3
```

结果：

| Refine Ratio | Best Deployable Policy | Best Drop | Random Drop | Degree Drop | TSER Drop | ActStress Drop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10% | Degree | 0.42% | 0.45% | 0.42% | 0.42% | 0.45% |
| 20% | Random | 0.34% | 0.34% | 0.40% | 0.53% | 0.35% |
| 25% | Random | 0.29% | 0.29% | 0.32% | 0.53% | 0.35% |
| 30% | TSERXAct | 0.31% | 0.35% | 0.40% | 0.58% | 0.32% |
| 40% | TSERXAct | 0.24% | 0.37% | 0.53% | 0.52% | 0.29% |

观察：

- Cora 的 all-BFPA4 drop 只有 0.48%，本身已经很低，因此 selector 差异容易被随机波动稀释。
- 当 refinement ratio 较高时，`TSERXAct` 优于 Degree / TSER / Random，说明图风险和 activation stress 的乘积能更好定位“既任务敏感、又 BFP block 压力大”的节点。
- 但在 20%-25% ratio，Random 仍然很强，说明 Cora 的 BFPA4 误差已经不够尖锐，单靠 selector 很难稳定拉开差距。

### 7.2 PubMed Large-Sample Policy

输入 trace：

```text
output/graphbfp_activation_stress/pubmed_s4096_runs3/activation_node_trace.tsv
```

这里 `trace_nodes=4096`，不是 PubMed 全图，但比 128-node signal validation 更稳定。

复现命令：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/evaluate_graphbfp_activation_policy_from_trace.py \
  --dataset pubmed \
  --runs 3 \
  --trace_path output/graphbfp_activation_stress/pubmed_s4096_runs3/activation_node_trace.tsv \
  --ratios 0.10 0.20 0.25 0.30 0.40 \
  --output_dir output/graphbfp_activation_policy/pubmed_s4096_runs3
```

结果：

| Refine Ratio | Best Deployable Policy | Best Drop | Random Drop | Degree Drop | TSER Drop | ActStress Drop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10% | Degree | 1.00% | 1.11% | 1.00% | 1.01% | 1.13% |
| 20% | Degree | 0.90% | 1.07% | 0.90% | 0.93% | 1.10% |
| 25% | DegreeXAct | 0.89% | 1.03% | 0.89% | 0.89% | 1.09% |
| 30% | TSER | 0.87% | 1.02% | 0.88% | 0.87% | 1.09% |
| 40% | TSER | 0.84% | 1.01% | 0.85% | 0.84% | 1.07% |

观察：

- PubMed 上 graph risk 是主导信号，Degree / TSER 明显优于 Random 和 ActStress。
- activation stress 单独作为 selector 不可靠；它能描述 BFP block 数值压力，但不能直接描述 GNN 任务脆弱性。
- `DegreeXAct` 在 25% ratio 与最优点持平，说明 activation stress 可以作为 graph-risk selector 的辅助项，但不应取代 graph risk。

## 8. 当前判断

真实 activation trace 解决了一个关键问题：embedding proxy 确实太粗，activation block stress 才是 BFP shared-exponent 风险的正确观测对象。

但 policy-level 结果也说明：

```text
activation stress alone is not enough.
```

更稳的设计应写成：

```text
graph risk:
    判断节点误差是否会影响下游 GNN。

activation stress:
    判断该节点在 BFP activation block 中是否容易受到 shared exponent 污染。

refinement policy:
    graph risk primary,
    activation stress auxiliary.
```

因此当前最合理的后端方向不是“只用 BFP stress 做 refinement”，而是：

```text
Graph-risk-guided BFPA4/BFPA6 refinement
with activation-stress assisted tie-breaking / gating.
```

## 9. 当前边界

当前已经完成：

- Cora full-node activation trace；
- PubMed 4096-node large-sample activation trace；
- graph risk / activation stress / 二者组合的 policy-level refinement 测试。

仍需补齐：

- PubMed full-node trace；
- Arxiv trace；
- 接入前端 SimHash / residual-gate 后，只对 miss / rejected nodes 做 BFPA4/BFPA6 refinement；
- 把 activation-stress-assisted policy 写入最终 progressive BFP full-stack 表。
