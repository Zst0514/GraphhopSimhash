# Final BFP Validation Result

本文档记录当前后端 BFP 主线的收束实验。目标是把三个问题拆开验证：

1. `BFPA4` 是否是安全低精度 base path，`BFPA3` 是否越过安全边界。
2. `BFPA4 -> BFPA6` dynamic refinement 是否有必要，以及 graph-risk / activation-stress selector 是否优于随机选择。
3. 固定 SimHash / TSER / residual-gate 前端后，miss nodes 接入 dynamic BFP encoder 的全栈精度和成本。

## 1. 实验脚本

统一入口：

```bash
cd /home/zhangshangtong/Transformer/OFA/GraphhopSimhash
bash scripts/run_final_bfp_validation_suite.sh
```

默认配置：

```text
datasets = cora pubmed arxiv
Cora runs = 5
PubMed runs = 3
Arxiv runs = 1

Reference path = W4BFPA8_B128
Base path      = W4BFPA4_B128
Refine path    = W4BFPA6_B128
Boundary tags  = W4BFPA6/5/4/3_B128

Frontend:
    8 x 16-bit SimHash heads
    radius = 2
    score weights = 3/1/1
    Cora/PubMed T = 31
    Arxiv T = 22
```

输出根目录：

```text
output/final_bfp_validation/
```

## 2. BFPA Safety Boundary

命令由总控脚本自动调用：

```bash
python -m GraphhopSimhash \
  --experiment_suite precision_depth_ablation \
  --precision_depth_reference_tag W4BFPA8_B128 \
  --precision_depth_tags W4BFPA6_B128 W4BFPA5_B128 W4BFPA4_B128 W4BFPA3_B128 \
  --precision_depth_bits 6 5 4 3
```

结果日志：

```text
output/final_bfp_validation/boundary/cora/bfpa8_vs_6543_runs5.log
output/final_bfp_validation/boundary/pubmed/bfpa8_vs_6543_runs3.log
output/final_bfp_validation/boundary/arxiv/bfpa8_vs_6543_runs1.log
```

实际执行时每个 bit-depth 单独输出，避免自动生成的内部日志名过长：

```text
output/final_bfp_validation/boundary/{dataset}/bfpa8_vs_p6_runs*.log
output/final_bfp_validation/boundary/{dataset}/bfpa8_vs_p5_runs*.log
output/final_bfp_validation/boundary/{dataset}/bfpa8_vs_p4_runs*.log
output/final_bfp_validation/boundary/{dataset}/bfpa8_vs_p3_runs*.log
```

该实验回答：

```text
BFPA6 是否接近 BFPA8；
BFPA4 是否仍在安全区间；
BFPA3 是否明显崩塌。
```

## 3. Dynamic Refinement Necessity

该实验固定：

```text
reference = W4BFPA8_B128
base      = W4BFPA4_B128
refine    = W4BFPA6_B128
```

比较 selector：

```text
Random
Stress
Degree
TSER
DegreeXStress
TSERXStress
DegreePlusStress
TSERPlusStress
```

refinement ratio：

```text
5%, 10%, 15%, 20%, 25%, 30%, 40%
```

结果目录：

```text
output/final_bfp_validation/refinement/cora_runs5/
output/final_bfp_validation/refinement/pubmed_runs3/
output/final_bfp_validation/refinement/arxiv_runs1/
```

该实验回答：

```text
同样 BFPA6 refinement budget 下，
graph-risk / activation-stress 是否比 random 更能恢复 BFPA4 的精度损失。
```

## 4. Full-Stack Dynamic BFP

全栈路径：

```text
Graph text node
  -> SimHash + LRU/CAM
  -> TSER score gate
  -> direct reuse / residual-gate reuse / miss
  -> miss nodes run dynamic BFPA4->BFPA6 encoder
  -> final embedding
  -> GNN classifier
```

结果日志：

```text
output/final_bfp_validation/fullstack/cora_T31_runs5.log
output/final_bfp_validation/fullstack/pubmed_T31_runs3.log
output/final_bfp_validation/fullstack/arxiv_T22_runs1.log
```

对应 dynamic pool 和 array trace 会复用或生成到：

```text
cache_data/{dataset}_llama2_7b_oracle_W4GraphBFPA4to6_B128_deg_t0.20.pt
output/dynamic_bfp_fullstack/{dataset}_T*_W4GraphBFPA4to6_B128_deg_t0.20_runs*/
```

## 5. Current Running Status

本轮统一脚本执行后在本节补充最终摘要。
