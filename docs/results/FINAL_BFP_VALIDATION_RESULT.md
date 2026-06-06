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

## 5. BFPA Boundary Summary

该表只看 encoder target pool 本身，不叠加 SimHash / residual 前端。`FullP8` 是 `W4BFPA8_B128` reference。

| Dataset | Runs | FullP8 Acc | BFPA6 Drop | BFPA5 Drop | BFPA4 Drop | BFPA3 Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 5 | 0.7106 | 0.09% | 0.35% | 0.99% | 23.13% |
| PubMed | 3 | 0.7522 | 0.02% | 0.25% | 1.16% | 27.43% |
| Arxiv | 1 | 0.6896 | 0.03% | 0.13% | 0.04% | 35.31% |

结论：

```text
BFPA6 基本贴近 BFPA8；
BFPA5 仍很稳；
BFPA4 在 Cora/PubMed 上有可见但可控掉点，在 Arxiv 上几乎无损；
BFPA3 在三个数据集上都明显崩塌，不适合作为默认路径。
```

对应日志：

```text
output/final_bfp_validation/boundary/{dataset}/bfpa8_vs_p{6,5,4,3}_runs*.log
```

## 6. Refinement Necessity Summary

该表固定 `BFPA4` 为 base、`BFPA6` 为 refinement path，比较不同 selector 在相同 lift ratio 下能挽回多少 BFPA4 掉点。

| Dataset | Runs | BFPA4 Drop | BFPA6 Drop | 5% Best | 10% Best | 15% Best | 20% Best | 25% Best | 30% Best | 40% Best |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| Cora | 5 | 0.99% | 0.09% | Degree+Stress 0.81% | Degree×Stress 0.77% | Degree+Stress 0.70% | Degree 0.68% | Degree 0.59% | Degree 0.66% | Random 0.62% |
| PubMed | 3 | 1.16% | 0.02% | TSER 0.91% | TSER 0.79% | TSER 0.69% | TSER 0.62% | TSER 0.55% | TSER 0.51% | TSER 0.44% |
| Arxiv | 1 | 0.06% | 0.03% | Degree 0.00% | Degree 0.00% | TSER 0.00% | Stress 0.02% | Stress 0.00% | TSER+Stress 0.00% | TSER+Stress -0.04% |

结论：

```text
Cora/PubMed 上 BFPA4 有约 1% 级别掉点，BFPA6 refinement 可以稳定挽回一部分。
PubMed 上 TSER selector 最稳定，说明图语义风险对 refinement 选择有价值。
Cora 上 Degree/Stress 组合在低 lift ratio 下更好，较高 ratio 后 Degree 足够。
Arxiv 上 BFPA4 本身已经安全，refinement 空间很小；它更适合作为 BFPA4 安全性的第三数据点。
```

对应目录：

```text
output/final_bfp_validation/refinement/cora_runs5/
output/final_bfp_validation/refinement/pubmed_runs3/
output/final_bfp_validation/refinement/arxiv_runs1/
```

## 7. Full-Stack Dynamic BFP Summary

该表固定当前前端参数，比较：

```text
FullP8:
    miss nodes 全部走 BFPA8。

AllP4:
    miss nodes 全部走 BFPA4。

Dynamic:
    miss nodes 中一部分 BFPA4 block 追加到 BFPA6。
    当前 full-stack 脚本中对应 Deg / TSER / Rand 三行。
```

| Dataset | Runs | Reuse | Direct | Residual | Miss | FullP8 Cost | FullP8 Drop | AllP4 Cost | AllP4 Drop | Best Dynamic | Dynamic Cost | Dynamic Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Cora | 5 | 51.6% | 18.1% | 33.5% | 48.4% | 0.244 | 2.22% | 0.141 | 2.72% | Deg | 0.154 | 2.57% |
| PubMed | 3 | 41.9% | 41.9% | 0.0% | 58.1% | 0.290 | 1.86% | 0.167 | 2.42% | Rand | 0.182 | 2.31% |
| Arxiv | 1 | 44.9% | 19.8% | 25.1% | 55.1% | 0.277 | 2.29% | 0.160 | 2.36% | TSER | 0.174 | 2.34% |

Array trace:

| Dataset | Refined Blocks | Effective Bits | Dynamic/BFPA4 | Dynamic/BFPA6 | Dynamic/BFPA8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 20.79% | 4.416 | 1.102x | 0.735x | 0.551x |
| PubMed | 13.88% | 4.278 | 1.070x | 0.713x | 0.535x |
| Arxiv | 21.46% | 4.429 | 1.105x | 0.737x | 0.553x |

解释：

```text
FullP8 drop 不是 BFPA8 encoder 本身的掉点，而是前端 reuse/residual 后的 full-stack drop。
Dynamic path 的成本接近 BFPA4，精度介于 FullP8 和 AllP4 之间。
Cora/PubMed 上 dynamic refinement 可以在 AllP4 的低成本基础上回收部分精度。
Arxiv 上 AllP4 已经接近 FullP8，因此 dynamic refinement 的精度收益很小。
```

对应日志：

```text
output/final_bfp_validation/fullstack/cora_T31_runs5.log
output/final_bfp_validation/fullstack/pubmed_T31_runs3.log
output/final_bfp_validation/fullstack/arxiv_T22_runs1.log

output/dynamic_bfp_fullstack/*/array_trace/summary.md
```

## 8. Takeaways

当前实验支持如下论文主线：

```text
1. BFPA4 是合理的低成本 miss-node encoder base path，但不是所有数据集都完全无损。
2. BFPA3 明显越过安全边界，因此 refinement 主要应在 BFPA4/BFPA6 之间设计。
3. Dynamic BFPA4->BFPA6 refinement 的必要性主要来自 Cora/PubMed；Arxiv 作为 BFPA4 足够安全的反例说明机制可以退化为低成本路径。
4. 图风险 selector 在 PubMed 上最明显，TSER 在所有 refinement ratio 下优于其它 selector。
5. full-stack drop 由前端 reuse/residual 和后端 BFP 共同决定；后续若要继续压低 full-stack drop，应优先同时调前端 T/gate 与后端 refinement ratio。
```
