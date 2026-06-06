# Graph-Aware BFP Validation Plan

本文档定义一组实验，用来验证下面这个核心判断：

```text
activation stress 是 BFP / Transformer accelerator 中本来就存在的数值风险；
graph risk 是 GFM 场景新增的任务传播风险；
二者结合，才能决定哪些 miss nodes 值得从 BFPA4 refine 到 BFPA6。
```

因此验证目标不是证明 BFP 格式本身新，而是证明：

```text
普通 Transformer accelerator 只能看到 activation numeric stress；
GFM accelerator 还能看到节点传播风险、TSER 风险和前端 reuse/miss 路由；
这些图任务信号可以改变 BFP refinement 的节点选择。
```

## 1. 需要验证的问题

### Q1. Activation stress 是否真实存在？

BFP activation 的核心问题是 block 内共享 exponent。如果 block 内存在 outlier，小值 mantissa 会被压缩，BFPA4 更容易产生误差。

要验证：

```text
高 activation stress 的节点，
是否更容易出现 BFPA4 数值误差，
是否更容易被 BFPA6 挽救。
```

### Q2. Graph risk 是否提供了 activation stress 没有的信息？

Activation stress 只描述数值风险，不知道这个节点在图任务里是否重要。

要验证：

```text
只看 activation stress 是否不够；
Degree / TSER 是否能解释下游 GNN damage；
Graph risk × activation stress 是否比单独任一信号更稳。
```

### Q3. 这个机制接入前端后是否仍然有效？

前端 SimHash / residual-gate 已经把一部分节点绕过 encoder：

```text
direct reuse:
    cache read

residual reuse:
    anchor + residual correction

miss / rejected:
    BFPA4/BFPA6 encoder
```

最终后端 BFP refinement 只应该作用于第三类 miss / rejected nodes。

## 2. 实验输入

### 2.1 Embedding pools

需要已有 LLaMA-7B BFP embedding pools：

```text
W4BFPA8_B128:
    reference / high precision target

W4BFPA6_B128:
    refinement path

W4BFPA4_B128:
    default low-cost path
```

### 2.2 Activation trace

activation stress 由真实 LLaMA forward hook 得到。当前脚本：

```bash
GraphhopSimhash/scripts/diagnose_graphbfp_activation_stress.py
```

它 hook LLaMA selected Linear inputs，并对每个 node 统计：

```text
act_stress_mean / p90
act_outlier_mean / p90
act_zero_pressure
act_bfpa4_err
act_bfpa6_err
act_rescue
```

这些是 activation block 的数值统计，不来自 final embedding oracle。

### 2.3 Graph risk

当前主要比较：

```text
Degree:
    传播风险，degree 越高，节点误差越可能影响更多邻居。

TSER:
    propagation risk + graph context + low-unique risk。
```

Context 单独策略不作为主线。

## 3. 实验一：Activation Stress 数值有效性

### 3.1 目的

证明 activation stress 是 BFP shared-exponent 的真实数值风险，而不是 final embedding proxy。

### 3.2 方法

对 Cora / PubMed / Arxiv 采样节点，运行 LLaMA forward，hook activation block：

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

PubMed 先用 sampled trace：

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

Arxiv 先用 sampled trace，避免一次跑全图：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/diagnose_graphbfp_activation_stress.py \
  --dataset arxiv \
  --runs 1 \
  --sample_nodes 8192 \
  --batch_size 8 \
  --max_length 128 \
  --layers 0 15 31 \
  --module_suffixes q_proj o_proj up_proj down_proj \
  --output_dir output/graphbfp_activation_stress/arxiv_s8192_run1
```

### 3.3 指标

```text
act_err:
    activation stress 与 activation-level BFPA4 error 的相关性。

hidden_err:
    activation stress 与 GNN hidden error 的相关性。

hidden_gain:
    activation stress 与 BFPA4 -> BFPA6 rescue 的相关性。

damage AUC:
    activation stress 是否能区分下游受损节点。

top-k gain overlap:
    按 activation stress 选出的 top-k 节点，
    与真实 BFPA6 可挽救节点的重合度。
```

### 3.4 判据

如果 activation stress 对 `act_err / hidden_gain` 有明显相关性，说明：

```text
BFP shared-exponent 风险可以被 activation trace 捕捉。
```

如果 activation stress 对 `damage AUC` 不稳定，说明：

```text
纯数值风险不足以决定下游任务风险，
需要 graph risk。
```

## 4. 实验二：Graph Risk 是否补充 Activation Stress

### 4.1 目的

固定同一个 BFPA4/BFPA6 refine budget，比较不同 selector：

```text
Random
ActStress only
Degree only
TSER only
Degree × ActStress
TSER × ActStress
```

这个实验直接回答：

```text
图风险是否比普通 Transformer activation stress 多提供了下游信息？
```

### 4.2 方法

读取实验一保存的 `activation_node_trace.tsv`，不重新跑 LLaMA：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/evaluate_graphbfp_activation_policy_from_trace.py \
  --dataset cora \
  --runs 3 \
  --trace_path output/graphbfp_activation_stress/cora_fullnodes_runs3/activation_node_trace.tsv \
  --ratios 0.05 0.10 0.15 0.20 0.25 0.30 0.40 \
  --output_dir output/graphbfp_activation_policy/cora_fullnodes_runs3
```

PubMed：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/evaluate_graphbfp_activation_policy_from_trace.py \
  --dataset pubmed \
  --runs 3 \
  --trace_path output/graphbfp_activation_stress/pubmed_s4096_runs3/activation_node_trace.tsv \
  --ratios 0.05 0.10 0.15 0.20 0.25 0.30 0.40 \
  --output_dir output/graphbfp_activation_policy/pubmed_s4096_runs3
```

### 4.3 输出主表

每个 refine ratio 输出：

```text
Ratio | Random Drop | ActStress Drop | Degree Drop | TSER Drop | DegreeXAct Drop | TSERXAct Drop
```

同时输出：

```text
BFPA6 ratio
estimated cost
selected node overlap
damage AUC
```

### 4.4 判据

如果：

```text
Degree / TSER 优于 ActStress:
    图传播风险比数值 stress 更能解释下游 damage。

ActStress 优于 Random:
    BFP 数值风险本身有效。

DegreeXAct / TSERXAct 在部分 ratio 上优于单信号:
    图风险和数值风险存在互补。
```

那么可以得到结论：

```text
Graph-aware BFP refinement 不是简单搬 BFP，
而是在 BFP 数值风险上叠加图任务风险。
```

## 5. 实验三：2×2 Interaction Analysis

### 5.1 目的

验证 activation stress 和 graph risk 的职责确实不同。

把节点按两个维度划分：

```text
graph risk:
    high / low

activation stress:
    high / low
```

得到四类节点：

```text
high graph, high stress
high graph, low stress
low graph, high stress
low graph, low stress
```

### 5.2 要统计什么

每一类统计：

```text
node count
BFPA4 drop contribution
BFPA4 -> BFPA6 gain
misclassification flip rate
average GNN margin drop
```

### 5.3 预期解释

如果结果呈现：

```text
high graph + high stress:
    damage / gain 最大。

low graph + high stress:
    有数值误差，但下游 damage 不一定大。

high graph + low stress:
    节点重要，但 BFPA4 数值上可能已经安全。
```

就能说明：

```text
activation stress 是数值风险；
graph risk 是任务传播风险；
二者结合才是合理 refinement 条件。
```

## 6. 实验四：接入 SimHash / Residual-Gate 前端

### 6.1 目的

前面实验是全图节点 selector。论文主线需要验证 full-stack：

```text
direct reuse:
    不进 encoder。

residual reuse:
    轻量 correction，不进 BFPA4/BFPA6 encoder。

miss / rejected:
    进入 Graph-aware BFP encoder。
```

因此 BFPA4/BFPA6 selector 必须只作用于 miss / rejected nodes。

### 6.2 方法

先固定前端策略，导出 per-node route：

```text
node_id
route = direct / residual / miss
support
accept_score
graph risk
```

然后只在：

```text
route == miss
```

的节点上运行 BFPA4/BFPA6 selector。

### 6.3 需要比较的 full-stack baseline

```text
FullP8-miss:
    direct/residual nodes bypass encoder；
    all miss nodes use BFPA8。

AllBFPA4-miss:
    direct/residual nodes bypass encoder；
    all miss nodes use BFPA4。

Random-refine:
    miss nodes default BFPA4；
    random top ratio lift to BFPA6。

Degree-refine:
    miss nodes default BFPA4；
    high degree miss nodes lift to BFPA6。

TSER-refine:
    miss nodes default BFPA4；
    high TSER miss nodes lift to BFPA6。

GraphXAct-refine:
    miss nodes default BFPA4；
    high graph-risk × activation-stress miss nodes lift to BFPA6。
```

### 6.4 输出主表

```text
Dataset
Reuse %
Direct %
Residual %
Miss %
BFPA4 %
BFPA6 %
Cost
Acc
Drop
```

关键不是只看 BFPA6 selector 自己，而是看：

```text
前端 reuse 减少 encoder 节点数；
后端 Graph-aware BFP 降低 miss-node encoder cost；
二者叠加后的 end-to-end cost/drop。
```

## 7. 实验五：硬件开销与 NPU 接口验证

### 7.1 Activation stress 采集开销

需要说明 stress 不是额外 oracle。硬件路径为：

```text
activation tile enters BFP loader
    -> max_abs already needed for shared exponent
    -> add low-cost range / zero-pressure counters
    -> emit stress bucket
```

需要估算：

```text
extra comparator / counter
metadata bits per row
controller lookup cost
```

### 7.2 BFPA4/BFPA6 阵列开销

当前设计：

```text
BFPA4 base:
    default execution。

BFPA6 refinement:
    extra 2 mantissa-bit planes for selected rows。
```

需要统计：

```text
BFPA6 selected ratio
extra activation bits
extra MAC activity
additional cycles if refinement lane serialized
additional area if refinement lane parallelized
```

### 7.3 和普通 Transformer BFP accelerator 的区别

普通 Transformer BFP accelerator：

```text
activation stress -> precision decision
```

Graph-aware BFP accelerator：

```text
front-end route -> only miss nodes enter encoder
graph risk -> task importance
activation stress -> numeric fragility
graph risk × activation stress -> BFPA6 refinement
```

也就是说：

```text
图信息不是替代 BFP stress，
而是决定 BFP stress 造成的误差是否值得被保护。
```

## 8. 推荐实验顺序

第一阶段先不接前端，验证 signal：

```text
1. Cora full-node activation trace
2. PubMed sampled activation trace
3. Random / Degree / TSER / ActStress / XAct policy sweep
4. 2×2 interaction table
```

第二阶段接入前端：

```text
5. 固定统一 SimHash / residual-gate 前端
6. 导出 direct / residual / miss route
7. 只对 miss nodes 做 BFPA4/BFPA6 selector
8. 输出 full-stack cost/drop table
```

第三阶段做硬件解释：

```text
9. stress metadata 采集开销
10. BFPA4/BFPA6 refinement activity model
11. W tile service-window / miss-node compaction model
```

## 9. 最终要证明的结论

实验最终需要支撑三句话：

```text
1. Activation stress 是 BFP shared-exponent 的真实数值风险，
   但它只描述 numeric fragility。

2. Graph risk 描述节点误差在 GNN 中的传播后果，
   能补充 activation stress 无法看到的任务风险。

3. 在 SimHash / residual-gate 前端之后，
   Graph-aware BFP refinement 只服务 miss nodes，
   用少量 BFPA6 refinement 换取更稳定的 BFPA4 encoder path。
```

