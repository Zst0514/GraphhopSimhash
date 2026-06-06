# Dynamic Graph-Aware BFP Full-Stack Result

本文档记录当前最终 full-stack 版本的实现方式和复现命令。该版本把前端 SimHash / CAM / residual-gate 与后端 graph-aware dynamic BFP encoder pool 接起来：

```text
Graph text node
  -> SimHash + LRU/CAM
  -> direct reuse / residual-gate / miss
  -> miss nodes use dynamic BFPA4/BFPA6 encoder pool
  -> GNN classifier
```

## 1. Dynamic BFP Backend

后端不再只用静态的 `W4BFPA4_B128` 或 `W4BFPA6_B128` pool，而是在 LLaMA forward 的每个 Linear wrapper 内部做 block-level 选择：

```text
base path:
    BFPA4 activation block

refine path:
    BFPA6 activation block

selection:
    degree_risk(node) * activation_stress(block) >= threshold
```

当前默认配置：

```text
base mantissa    = 4
refine mantissa  = 6
block size       = 128
stress scale     = 8.0
threshold        = 0.20
```

Cora 当前已生成并注册：

```text
cache_data/cora_llama2_7b_oracle_W4GraphBFPA4to6_B128_deg_t0.20.pt
cache_data/cora_llama2_7b_oracle_W4GraphBFPA4to6_B128_deg_t0.20.json
```

metadata:

```text
refined blocks = 1469101489 / 7067017984
refined ratio  = 20.79%
effective bits = 4.416
```

解释：

```text
effective bits = 4 + 2 * refined_ratio
```

也就是说，该 pool 在数值上是动态 BFPA4/BFPA6，在硬件成本估算中对应平均 activation mantissa depth 约 4.416 bit。

## 2. Generate Dynamic Pool

单独生成 / 注册 Cora dynamic pool：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py \
  --dataset cora \
  --llm_name llama2_7b \
  --threshold 0.20 \
  --stress_scale 8.0 \
  --block_size 128 \
  --base_mantissa 4 \
  --refine_mantissa 6 \
  --cache_tag W4GraphBFPA4to6_B128_deg_t0.20 \
  --save_to_cache \
  --runs 1 \
  --seed 42
```

Cora single-run pool-level check:

```text
Baseline Acc: 0.6987
Dynamic Acc:  0.6973
Dynamic Drop: 0.15%
```

该结果只验证 dynamic BFP encoder pool 本身相对 `W4BFPA8_B128` 的下游影响。

## 3. Full-Stack Command

总控脚本：

```bash
GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh
```

Cora smoke test：

```bash
DATASETS=cora RUNS=1 FORCE_FULLSTACK=1 FORCE_DYNAMIC=0 \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh
```

三数据集版本：

```bash
DATASETS="cora pubmed arxiv" \
FORCE_DYNAMIC=0 \
FORCE_FULLSTACK=1 \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh
```

默认 dataset-level front-end threshold：

```text
Cora   T = 31
PubMed T = 31
Arxiv  T = 22
```

这沿用 `UNIFIED_FRONTEND_POLICY_RESULT.md` 中当前主线的 dataset-level policy register 设置。

## 4. Cora Full-Stack Smoke Test

日志：

```text
output/dynamic_bfp_fullstack/cora_T31_W4GraphBFPA4to6_B128_deg_t0.20_runs1/logs/cora_runs1.log
```

结果：

| Config | Reuse | Direct | Residual | P8 | Dynamic | Cost | Acc | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 | 50.0% | 17.9% | 32.1% | 50.0% | 0.0% | 0.252 | 0.6668 | 1.98% |
| Dynamic | 50.0% | 17.9% | 32.1% | 0.0% | 50.0% | 0.145 | 0.6668 | 1.98% |

说明：

- `FullP8`：reuse/residual 前端固定，所有 miss nodes 使用 `W4BFPA8_B128`。
- `Dynamic`：reuse/residual 前端固定，所有 miss nodes 使用动态 BFPA4/BFPA6 pool。
- 这一轮 `FullP8` 与 `Dynamic` drop 相同，说明动态 BFP 后端没有额外恶化，主要 drop 来自前端 reuse/residual。
- runner 表中原始行名为 `AllP4`，但这里加载的是动态 pool `W4GraphBFPA4to6_B128_deg_t0.20`，不是纯 `W4BFPA4_B128`。

## 5. Cost Accounting

现有 runner 为了兼容旧 `precision_depth` 接口，会把 dynamic pool 当作 `P4` 行加载。因此表中 `Cost=0.145` 是纯 P4 成本口径。

硬件解释时应使用 metadata 中的 effective bits：

```text
effective bits = 4.416
```

若沿用当前 cost model：

```text
precision_cost(bits) = cost_scale * (fixed_cost + (1 - fixed_cost) * bits / 8)
cost_scale = 0.50
fixed_cost = 0.15
```

则 dynamic backend 每个 miss node 的 encoder cost 为：

```text
0.50 * (0.15 + 0.85 * 4.416 / 8) = 0.2346
```

相比：

```text
BFPA4: 0.2875
BFPA6: 0.3938
BFPA8: 0.5000
```

注意这里的 full-stack cost 还需要乘以 miss-node 比例；本次 Cora smoke test 中 miss-node 比例约 50%。

## 6. Next Runs

需要补齐：

```text
Cora   10 runs
PubMed 3 runs
Arxiv  1 run
```

推荐命令：

```bash
DATASETS=cora RUNS=10 FORCE_DYNAMIC=0 FORCE_FULLSTACK=1 \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh

DATASETS=pubmed RUNS=3 FORCE_DYNAMIC=1 FORCE_FULLSTACK=1 \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh

DATASETS=arxiv RUNS=1 FORCE_DYNAMIC=1 FORCE_FULLSTACK=1 \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh
```

PubMed / Arxiv 如果没有 dynamic pool，会先生成 LLaMA embedding，耗时明显更长。
