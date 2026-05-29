# Graph-Bit Proxy Experiments

本文档记录 Graph-Bit 的 embedding-pool proxy 实验。它不是最终硬件仿真，而是用不同 activation precision-depth 的 embedding pool 近似 bit-serial early termination，先验证 graph risk 是否能指导 NPU 算术努力分配。

核心问题：

```text
高风险节点使用完整 P8 / W4A8 计算；
中风险节点使用较高 bit-depth；
低风险节点提前终止到更低 bit-depth。
```

这里的 P8/P6/P5/P4 是对 bit-serial / bit-grained early termination 的离线近似：先生成不同 activation bit-depth 的 embedding pool，再在同一个下游 GNN 上比较不同路由策略的精度和成本。

真正的 NPU 内部 bit-serial 设计见：

```text
docs/npu/GRAPH_CONDITIONED_BIT_SERIAL_EXECUTION.md
```

## 1. 实现位置

主要代码改动：

```text
generate_real_quant_pools.py
    新增 W4A6 / W4A5，以及 W4A6_FAKE / W4A5_FAKE 配置。

cli.py
    新增 --experiment_suite precision_depth_ablation 及相关参数。

runner.py
    新增 run_precision_depth_ablation。

routing.py
    precision-depth 实验日志名加入 bit-depth / budget / predictor target。
```

## 2. 实验含义

参考 bit-serial / bit-grained accelerator 的思路，硬件不一定一次性跑满所有 bit-plane，而是可以按风险决定算到几位：

```text
P8: safest path, equivalent to full W4A8
P6: medium-high precision path
P5: medium precision path
P4: aggressive early termination path
```

当前 cost model 是：

```text
cost(bit) = cost_scale * (fixed_cost + (1 - fixed_cost) * bit / reference_bits)
```

本轮设置：

```text
reference_bits = 8
cost_scale     = 0.50
fixed_cost     = 0.15

P8 cost = 0.500
P6 cost = 0.394
P5 cost = 0.341
P4 cost = 0.287
```

预算路由固定为：

```text
top 20% risk nodes     -> P8
next 30% risk nodes    -> P6
remaining 50% nodes    -> P4
```

P5 当前作为 uniform ablation 保留，用来检查 bit-depth profile 是否单调。

## 3. 路由策略

当前比较的策略：

```text
FullP8
AllP6 / AllP5 / AllP4
RandomDepthBudget
DegreeDepthBudget
TSERDepthBudget
ContextDepthBudget
LowUniqueDepthBudget
```

主线只使用不需要全图 reference embedding、不需要额外训练校准器的策略：

```text
Random / Degree / TSER / Context / LowUnique
```

`PredictorDepthBudget` 和 `OracleDamageBudget` 只保留为 debug / oracle 行，不作为系统主策略：

```text
PredictorDepthBudget:
    需要额外 calibration nodes 拟合 damage predictor。
    它可以帮助判断手写 proxy 是否还有空间，但不作为主线 deployable policy。

OracleDamageBudget:
    使用真实 embedding damage 排序。
    它要求已经有 reference embedding 和低精度 embedding 的差值，因此不能作为实际系统策略。
```

`PredictorDepthBudget` 用 512 个 calibration nodes 拟合一个轻量 ridge predictor。当前支持两个目标：

```text
embedding: 预测 embedding L2 damage
margin:    预测下游分类 margin damage
```

## 4. 复现实验命令

生成 Cora/ST precision-depth pools：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs W4A8 W4A6 W4A5 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

生成 PubMed/ST precision-depth pools：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets pubmed \
  --llm_name ST \
  --configs W4A8 W4A6 W4A5 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

运行 embedding-target predictor：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite precision_depth_ablation \
  --real_quant_model_name ST \
  --precision_depth_reference_tag W4A8 \
  --precision_depth_tags W4A6 W4A5 W4A4 \
  --precision_depth_bits 6 5 4 \
  --precision_depth_reference_bits 8 \
  --precision_depth_high_ratio 0.20 \
  --precision_depth_mid_ratio 0.30 \
  --precision_depth_cost_scale 0.50 \
  --precision_depth_fixed_cost 0.15 \
  --precision_depth_predictor_calib_samples 512 \
  --precision_depth_predictor_target embedding
```

把 `--datasets cora` 改成 `--datasets pubmed` 即可跑 PubMed。把最后一行改成：

```bash
  --precision_depth_predictor_target margin
```

即可跑 margin-target predictor。

## 5. Cora/ST 结果

10 runs，baseline 是 FullP8。

注意：表中的 `PredictorDepthBudget` / `OracleDamageBudget` 是 debug/oracle 行，不进入主策略比较。

| Config | Cost | Drop, embedding predictor | Drop, margin predictor | AvgErr, embedding predictor |
| --- | ---: | ---: | ---: | ---: |
| FullP8 | 0.500 | 0.00% | 0.00% | 0.00000 |
| AllP6 | 0.394 | 0.49% | 0.49% | 0.00837 |
| AllP5 | 0.341 | 0.13% | 0.13% | 0.02067 |
| AllP4 | 0.287 | 3.45% | 3.45% | 0.09387 |
| RandomDepthBudget | 0.362 | 1.57% | 1.57% | 0.04923 |
| DegreeDepthBudget | 0.362 | 1.37% | 1.37% | 0.05099 |
| TSERDepthBudget | 0.362 | 1.32% | 1.32% | 0.05004 |
| ContextDepthBudget | 0.362 | 1.26% | 1.26% | 0.04632 |
| LowUniqueDepthBudget | 0.362 | 1.81% | 1.81% | 0.05021 |
| PredictorDepthBudget | 0.362 | 1.16% | 1.01% | 0.03965 |
| OracleDamageBudget | 0.362 | 0.89% | 0.89% | 0.02464 |

观察：

```text
1. P6/P5 对 Cora/ST 基本安全，P4 开始明显掉点。
2. 相同 cost=0.362 下，Context / TSER / Degree 都优于 Random。
3. Predictor 只能作为 debug 行，用来估计校准式 predictor 的上限收益。
4. Oracle 只能作为不可部署上界，用来判断真实 damage 信息还有多少空间。
```

## 6. PubMed/ST 结果

10 runs，baseline 是 FullP8。

注意：表中的 `PredictorDepthBudget` / `OracleDamageBudget` 是 debug/oracle 行，不进入主策略比较。

| Config | Cost | Drop, embedding predictor | Drop, margin predictor | AvgErr, embedding predictor |
| --- | ---: | ---: | ---: | ---: |
| FullP8 | 0.500 | 0.00% | 0.00% | 0.00000 |
| AllP6 | 0.394 | 1.08% | 1.08% | 0.00455 |
| AllP5 | 0.341 | 3.49% | 3.49% | 0.01564 |
| AllP4 | 0.287 | 1.34% | 1.34% | 0.01322 |
| RandomDepthBudget | 0.362 | 0.90% | 0.90% | 0.00797 |
| DegreeDepthBudget | 0.362 | 0.86% | 0.86% | 0.00805 |
| TSERDepthBudget | 0.362 | 0.85% | 0.85% | 0.00789 |
| ContextDepthBudget | 0.362 | 0.85% | 0.85% | 0.00794 |
| LowUniqueDepthBudget | 0.362 | 0.89% | 0.89% | 0.00783 |
| PredictorDepthBudget | 0.362 | 0.78% | 0.86% | 0.00777 |
| OracleDamageBudget | 0.362 | 0.39% | 0.39% | 0.00403 |

观察：

```text
1. PubMed/ST 的 P4 比 P5 更稳，bit-depth damage 不完全单调。
2. Degree / TSER / Context 在 PubMed 上差距很小，但都略优于 Random。
3. Predictor/Oracle 不作为主策略：前者需要额外校准，后者需要真实 reference damage。
4. Oracle 明显更好，说明真实 damage 信息有诊断价值，但不能用于实际部署。
```

## 7. 当前结论

这组实验支持以下判断：

```text
Precision-depth execution 是值得保留的 NPU 路径。
它比单纯 FFN gating 更深入 datapath，因为它直接模拟 bit-plane / datatype effort。
```

但它也暴露了三个问题：

```text
1. 不同 backend 的 bit-depth profile 不一定单调，必须实际测量。
2. 图风险分数能带来小幅收益，尤其用于证明 graph risk 可以控制 arithmetic effort。
3. Predictor/Oracle 只作为 debug/oracle 上界，不进入主线结论。
```

因此更稳的论文表述是：

```text
Graph risk can condition arithmetic precision depth.
The deployable mainline should use Degree / TSER / Context style graph scores,
while Predictor / Oracle rows are only diagnostic baselines.
```

## 8. 下一步

建议继续补三类实验：

```text
1. LLaMA-7B / Cora, PubMed:
   检查大模型 embedding 下 P6/P5/P4 是否更有区分度。

2. Budget sweep:
   扫 high_ratio / mid_ratio，画 cost-drop curve。

3. True bit-serial simulator:
   从 embedding-pool 近似推进到 per-layer / per-GEMM bit-plane early termination 模型。
```
