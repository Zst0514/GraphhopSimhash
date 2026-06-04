# Graph-Aware BFP Refinement Policy

本文档说明当前后端 BFP encoder 的核心策略：如何把图任务风险和真实 LLaMA activation block stress 结合起来，决定 miss / rejected nodes 走 BFPA4 还是 BFPA6。

## 1. 背景

前端 SimHash / LRU-CAM / residual-gate 会把节点分成三类：

```text
direct reuse:
    直接读 embedding cache。

residual reuse:
    读 anchor embedding，并用轻量 residual adapter 修正。

miss / rejected:
    仍然需要运行 LLaMA encoder。
```

BFP encoder 只服务第三类节点。当前后端不把所有 miss nodes 都用 W4A8 / BFPA8 计算，而是采用：

```text
default path:
    W4BFPA4

refinement path:
    W4BFPA6
```

也就是大多数 miss nodes 走 BFPA4 低成本路径，只把少量更脆弱的节点提升到 BFPA6。

## 2. 为什么需要 Activation Stress

BFP 的核心特点是：

```text
一个 activation block 共享一个 exponent。
```

如果同一个 block 里存在 outlier，大值会拉高 shared exponent，小值的 mantissa 有效位会被压缩。因此，BFPA4 是否可靠，不只取决于节点本身，还取决于 LLaMA 内部 activation block 的数值分布。

对一个 activation block，可以定义轻量 stress：

```text
stress = log2(max_abs / median_abs)

outlier = log2(max_abs / mean_abs)

zero_pressure =
    block 中明显小于 max_abs 的值所占比例
```

这些量只依赖当前 activation tile 的数值分布，不依赖最终 embedding，也不需要 oracle label。

当前实验中通过 hook LLaMA activation 收集 trace，是为了验证这个信号是否有用。部署时，activation stress 可以由 BFP exponent selector / activation loader 顺带统计：

```text
activation tile arrives
    -> compute block max for BFP exponent
    -> compute mean/median approximation or range bucket
    -> generate stress metadata
```

因此它不是“额外跑一次 encoder 得到的 oracle”，而是 BFP 执行路径中的 tile-level runtime statistic。

## 3. 为什么还需要 Graph Risk

Activation stress 只能回答：

```text
这个 activation block 在 BFPA4 下是否容易量化受损？
```

它不能回答：

```text
这个节点的 embedding 误差是否会影响 GNN 下游？
```

图风险负责这个问题。当前可用的图风险包括：

```text
Degree / propagation risk:
    节点错误是否容易传播给邻居。

TSER:
    propagation risk + graph context + low-unique risk。
```

因此两个信号的职责不同：

```text
activation stress:
    数值侧风险，描述 BFP shared-exponent 是否危险。

graph risk:
    任务侧风险，描述该节点误差是否会被 GNN 放大。
```

最终 refinement 策略应该选择：

```text
既有 BFP 数值风险，
又有图任务风险
```

的节点。

## 4. Policy 设计

### 4.1 单信号策略

最简单的策略是单独按某个分数排序：

```text
Random:
    随机 lift 一部分 miss nodes 到 BFPA6。

Degree:
    高 degree / 高传播风险节点 lift 到 BFPA6。

TSER:
    高 TSER 节点 lift 到 BFPA6。

ActStress:
    高 activation stress 节点 lift 到 BFPA6。
```

单信号策略用于消融。它们分别回答：

```text
Random:
    没有图信息时的下限。

Degree / TSER:
    只看图风险是否足够。

ActStress:
    只看 BFP 数值风险是否足够。
```

### 4.2 联合策略

联合策略把 graph risk 和 activation stress 组合：

```text
TSERXAct = normalize(TSER) * normalize(ActStress)

DegreeXAct = normalize(Degree) * normalize(ActStress)
```

含义：

```text
高 graph risk 但 activation stress 低:
    节点重要，但 BFPA4 数值上可能已经足够。

高 activation stress 但 graph risk 低:
    数值压力大，但下游影响小。

高 graph risk 且 high activation stress:
    优先 BFPA6。
```

也可以采用两阶段策略：

```text
1. 用 graph risk 选出高任务风险候选池。
2. 在候选池内部按 activation stress 排序。
3. 选 top-ratio 进入 BFPA6 refinement。
```

这种设计更贴近硬件实现：

```text
graph risk:
    来自前端 metadata / node scheduler。

activation stress:
    来自 BFP loader / exponent selector。

refinement tag:
    写入 row tag，控制当前 row 是否执行 BFPA6 extra mantissa planes。
```

## 5. NPU 执行接口

### 5.1 输入流

前端输出给 BFP encoder 的不是完整图节点，而是 compacted miss-node stream：

```text
node_id
token rows
graph risk metadata
route tag = miss / rejected
```

reuse / residual 节点不进入 BFP encoder array。

### 5.2 Activation Loader

对每个 miss-node token row：

```text
load activation tile
compute BFP exponent
collect stress metadata
emit BFPA4 mantissa
```

stress metadata 可以是近似量，不要求完整浮点统计。硬件上可以用：

```text
max bucket
mean bucket
range bucket
zero-pressure counter
```

### 5.3 Refinement Controller

controller 接收：

```text
graph risk score
activation stress score
refinement budget / ratio
```

输出：

```text
row_refine_tag:
    0 -> BFPA4 only
    1 -> BFPA4 base + BFPA6 extra mantissa refinement
```

### 5.4 BFPA4 / BFPA6 Array

阵列默认执行 BFPA4 base：

```text
BFPA4:
    high 4 mantissa bits x W4
```

对被标记的 rows 追加 BFPA6 refinement：

```text
BFPA6:
    high 4 mantissa bits x W4
  + extra 2 mantissa bits x W4
```

BFPA4 和 BFPA6 共享同一份 W4 tile，不需要加载另一套权重。

## 6. 当前验证结果

结果文档：

```text
docs/results/BFP_ACTIVATION_STRESS_TRACE_RESULT.md
```

### 6.1 Signal Validation

Cora 128-node activation trace：

```text
ActStress:
    damage AUC = 0.602
    top25 BFPA4->BFPA6 gain overlap = 0.552
```

PubMed 128-node activation trace：

```text
TSER+ActStress:
    damage AUC = 0.761

Degree+ActStress:
    damage AUC = 0.751
```

结论：

```text
真实 LLaMA activation stress 比 final embedding proxy 更接近 BFP 风险来源。
但 activation stress 单独不等于任务风险。
```

### 6.2 Policy Validation

Cora full-node trace：

```text
trace_nodes = 2708
All BFPA4 drop = 0.48%
All BFPA6 drop = -0.11%
```

在 30%-40% refinement ratio 下：

```text
TSERXAct 最好或接近最好。

30%: drop = 0.31%
40%: drop = 0.24%
```

PubMed 4096-node trace：

```text
All BFPA4 drop = 1.16%
All BFPA6 drop = 0.02%
```

在 PubMed 上：

```text
Degree / TSER 是主导信号。
DegreeXAct 在 25% ratio 与最优点持平。
```

解释：

```text
Cora:
    BFPA4 已经很稳，selector 差异容易被随机波动稀释。
    较高 refinement ratio 下，TSERXAct 开始体现优势。

PubMed:
    图传播风险更主导。
    activation stress 更适合作为辅助 tie-breaking / gating。
```

## 7. 实验流程

### Step 1: 生成 activation trace

Cora full-node：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/diagnose_graphbfp_activation_stress.py \
  --dataset cora \
  --runs 3 \
  --sample_nodes 2708 \
  --batch_size 8 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj o_proj up_proj down_proj \
  --output_dir output/graphbfp_activation_stress/cora_fullnodes_runs3
```

PubMed large-sample：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/diagnose_graphbfp_activation_stress.py \
  --dataset pubmed \
  --runs 3 \
  --sample_nodes 4096 \
  --batch_size 8 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj o_proj up_proj down_proj \
  --output_dir output/graphbfp_activation_stress/pubmed_s4096_runs3
```

输出：

```text
activation_node_trace.tsv
activation_signal_summary.tsv
activation_sample_routing.tsv
```

### Step 2: 从 trace 评估 policy

Cora：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/evaluate_graphbfp_activation_policy_from_trace.py \
  --dataset cora \
  --runs 3 \
  --trace_path output/graphbfp_activation_stress/cora_fullnodes_runs3/activation_node_trace.tsv \
  --ratios 0.10 0.20 0.25 0.30 0.40 \
  --output_dir output/graphbfp_activation_policy/cora_fullnodes_runs3
```

PubMed：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/evaluate_graphbfp_activation_policy_from_trace.py \
  --dataset pubmed \
  --runs 3 \
  --trace_path output/graphbfp_activation_stress/pubmed_s4096_runs3/activation_node_trace.tsv \
  --ratios 0.10 0.20 0.25 0.30 0.40 \
  --output_dir output/graphbfp_activation_policy/pubmed_s4096_runs3
```

输出：

```text
policy_summary.tsv
summary.txt
```

### Step 3: 接入 full-stack miss-node path

下一步需要把 policy 从全图节点切换到前端之后的 miss / rejected nodes：

```text
SimHash / CAM / residual-gate
    -> direct reuse nodes
    -> residual reuse nodes
    -> miss / rejected nodes
            -> BFPA4/BFPA6 policy
```

对应实验应输出：

```text
Reuse %
Direct %
Residual %
Miss %
BFPA4 miss %
BFPA6 miss %
Cost
Acc / Drop
```

这一步会把当前 activation-stress-assisted policy 接入 `residual_precision_depth` / progressive BFP full-stack。

## 8. 后续实验计划

### 8.1 PubMed full-node trace

当前 PubMed 使用 4096-node trace。需要补全：

```text
sample_nodes = 19717
```

用于验证 4096-node policy 是否稳定。

### 8.2 Arxiv trace

Arxiv 节点多，优先做分层采样：

```text
random nodes
high TSER nodes
low TSER nodes
high degree nodes
```

先验证 signal，再决定是否跑 full-node trace。

### 8.3 Miss-only policy

当前 policy 测的是 traced nodes。最终论文主表要测：

```text
only miss / rejected nodes are eligible for BFPA6 refinement
```

这更符合真实硬件路径，因为 direct / residual nodes 不进入 encoder array。

### 8.4 Hardware Cost Model

每个 BFPA6 refined row 的额外开销：

```text
BFPA4:
    4 mantissa planes

BFPA6:
    4 base mantissa planes + 2 refinement mantissa planes
```

若 refinement ratio 为 `r`：

```text
avg_mantissa_bits = 4 * (1 - r) + 6 * r
                  = 4 + 2r
```

例如：

```text
r = 0.25:
    avg_mantissa_bits = 4.5
    相对 BFPA4 片上 mantissa MAC 增加 12.5%
    相对 BFPA6 减少 25%
```

后续 cost 表需要同时报告：

```text
encoder nodes saved by reuse
BFPA4/BFPA6 refined-node ratio
average mantissa bits
array activity proxy
end-to-end cost
```

## 9. 当前推荐表述

当前最稳的技术表述是：

```text
Graph-aware BFP refinement uses graph risk as the primary task-risk signal
and activation stress as a runtime numeric-risk signal.
```

中文表述：

```text
图风险判断节点误差是否会影响下游 GNN；
activation stress 判断该节点在 BFP block 中是否容易受到 shared exponent 污染；
二者共同决定 miss node 是否从 BFPA4 提升到 BFPA6。
```

这条线的关键不是直接迁移 BFP，而是把 GFM 前端产生的图任务风险接入 BFP refinement controller，使 BFPA6 预算优先服务于“任务敏感且数值脆弱”的节点。
