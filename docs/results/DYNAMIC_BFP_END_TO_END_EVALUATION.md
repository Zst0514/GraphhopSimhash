# Dynamic BFP End-to-End Evaluation

本文档记录 SimHash / residual-gate 前端接入 graph-aware dynamic BFP miss-node encoder 后，如何做端到端时间估计。重点是把三类数据连接起来：

```text
1. frontend reuse trace:
   direct / residual / miss 比例

2. accuracy trace:
   FullP8-miss drop 与 dynamic BFP drop

3. array trace:
   BFPA4 base blocks、BFPA6 refined blocks、dynamic/BFPA8 cycles ratio
```

---

## 1. 端到端时间模型

以“全图所有节点都运行 BFPA8/W4BFPA8 LLaMA encoder”为归一化 baseline：

```text
T_encoder_baseline = 1.0
```

前端把节点分成三条路径：

```text
direct reuse:
    cache read，近似 cost 很小

residual reuse:
    anchor embedding + MLP delta，远小于 LLaMA encoder

miss / reject:
    dynamic BFPA4/BFPA6 encoder
```

因此 encoder 端归一化时间可以写成：

```text
T_encoder =
    direct_ratio   * C_cache
  + residual_ratio * C_residual
  + miss_ratio     * C_dynamic_bfp
```

当前快速估计中：

```text
C_cache    ~= 0
C_residual ~= 0
C_dynamic_bfp = dynamic_cycles / BFPA8_cycles
```

如果考虑 GNN 后端占原始端到端时间约 1%，LLM encoder 占约 99%：

```text
T_e2e =
    0.99 * T_encoder
  + 0.01
```

端到端加速比：

```text
Speedup = 1 / T_e2e
```

如果要加入 residual unit / scheduler 的实际开销，可以使用：

```text
T_encoder =
    direct_ratio   * C_cache
  + residual_ratio * C_residual
  + miss_ratio     * C_dynamic_bfp
  + C_scheduler
```

其中 `C_residual` 和 `C_scheduler` 后续由硬件面积/频率估计补充。

---

## 2. Cora Current Result

### 2.1 Full-stack accuracy trace

日志：

```text
output/dynamic_bfp_fullstack/cora_T31_W4GraphBFPA4to6_B128_deg_t0.20_runs3/logs/cora_runs3.log
```

结果：

| Config | Reuse | Direct | Residual | Miss | Cost(row) | Acc | Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullP8-miss | 51.2% | 17.8% | 33.4% | 48.8% | 0.246 | 0.6786 | 1.81% |
| Dynamic BFP | 51.2% | 17.8% | 33.4% | 48.8% | 0.142* | 0.6757 | 2.10% |

`Dynamic BFP` 在 runner 表里显示为 `AllP4`，但实际加载的是：

```text
cache_data/cora_llama2_7b_oracle_W4GraphBFPA4to6_B128_deg_t0.20.pt
```

因此 `Cost(row)` 仍是旧 runner 的 P4 口径。硬件时间估计应使用 array trace 的 `dynamic/BFPA8 cycles`。

相对 FullP8-miss：

```text
extra drop = 2.10% - 1.81% = 0.29%
```

### 2.2 Dynamic BFP array trace

命令：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/simulate_dynamic_bfp_array_trace.py \
  --metadata cache_data/cora_llama2_7b_oracle_W4GraphBFPA4to6_B128_deg_t0.20.json \
  --output_dir output/dynamic_bfp_fullstack/cora_array_trace_t020
```

输出：

```text
output/dynamic_bfp_fullstack/cora_array_trace_t020/summary.md
output/dynamic_bfp_fullstack/cora_array_trace_t020/summary.json
output/dynamic_bfp_fullstack/cora_array_trace_t020/module_array_trace.tsv
output/dynamic_bfp_fullstack/cora_array_trace_t020/kind_array_trace.tsv
```

整体统计：

| Metric | Value |
| --- | ---: |
| refined blocks | 1469101489 / 7067017984 |
| refined ratio | 20.79% |
| effective mantissa bits | 4.416 |
| dynamic / BFPA4 cycles | 1.102x |
| dynamic / BFPA6 cycles | 0.735x |
| dynamic / BFPA8 cycles | 0.551x |

按模块：

| Module kind | Refined | Dynamic/BFPA4 | Dynamic/BFPA6 | Dynamic/BFPA8 |
| --- | ---: | ---: | ---: | ---: |
| down_proj | 35.20% | 1.198x | 0.799x | 0.599x |
| gate_proj | 11.32% | 1.064x | 0.709x | 0.532x |
| up_proj | 11.32% | 1.064x | 0.709x | 0.532x |
| q_proj | 17.29% | 1.097x | 0.731x | 0.549x |
| k_proj | 17.29% | 1.097x | 0.731x | 0.549x |
| v_proj | 17.29% | 1.097x | 0.731x | 0.549x |
| o_proj | 11.49% | 1.065x | 0.710x | 0.532x |

### 2.3 Cora end-to-end estimate

使用：

```text
miss_ratio = 48.8%
C_dynamic_bfp = 0.551
```

忽略 cache / residual / scheduler 开销：

```text
T_encoder = 0.488 * 0.551 = 0.269
```

相对全图 BFPA8 encoder：

```text
encoder speedup = 1 / 0.269 = 3.72x
```

如果 LLM encoder 占原始端到端时间 99%，GNN 占 1%：

```text
T_e2e = 0.99 * 0.269 + 0.01 = 0.276
e2e speedup = 1 / 0.276 = 3.62x
```

如果对比“前端 reuse 后，miss nodes 全部走 BFPA8”：

```text
T_fullp8_miss = 0.488
T_dynamic_miss = 0.488 * 0.551 = 0.269

speedup over FullP8-miss = 0.488 / 0.269 = 1.81x
```

---

## 3. PubMed Command and Fill-in Template

PubMed 当前命令：

```bash
FORCE_DYNAMIC=1 FORCE_FULLSTACK=1 DATASETS=pubmed RUNS=3 \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh
```

如果只想生成 dynamic pool 和 array trace：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py \
  --dataset pubmed \
  --llm_name llama2_7b \
  --threshold 0.20 \
  --stress_scale 8.0 \
  --block_size 128 \
  --base_mantissa 4 \
  --refine_mantissa 6 \
  --cache_tag W4GraphBFPA4to6_B128_deg_t0.20 \
  --save_to_cache \
  --runs 3 \
  --seed 42 \
  --overwrite
```

然后：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/simulate_dynamic_bfp_array_trace.py \
  --metadata cache_data/pubmed_llama2_7b_oracle_W4GraphBFPA4to6_B128_deg_t0.20.json \
  --output_dir output/dynamic_bfp_fullstack/pubmed_array_trace_t020
```

结果填表：

| Config | Reuse | Direct | Residual | Miss | Acc | Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FullP8-miss | TBD | TBD | TBD | TBD | TBD | TBD |
| Dynamic BFP | TBD | TBD | TBD | TBD | TBD | TBD |

Array trace：

| Metric | Value |
| --- | ---: |
| refined ratio | TBD |
| effective mantissa bits | TBD |
| dynamic / BFPA4 cycles | TBD |
| dynamic / BFPA6 cycles | TBD |
| dynamic / BFPA8 cycles | TBD |

端到端估计：

```text
T_encoder = miss_ratio * (dynamic / BFPA8 cycles)
T_e2e = 0.99 * T_encoder + 0.01
Speedup = 1 / T_e2e
```

---

## 4. Arxiv Command and Fill-in Template

Arxiv 默认前端使用更保守的 `T=22`：

```bash
FORCE_DYNAMIC=1 FORCE_FULLSTACK=1 DATASETS=arxiv RUNS=1 \
  bash GraphhopSimhash/scripts/run_dynamic_bfp_fullstack.sh
```

只生成 dynamic pool：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py \
  --dataset arxiv \
  --llm_name llama2_7b \
  --threshold 0.20 \
  --stress_scale 8.0 \
  --block_size 128 \
  --base_mantissa 4 \
  --refine_mantissa 6 \
  --cache_tag W4GraphBFPA4to6_B128_deg_t0.20 \
  --save_to_cache \
  --runs 1 \
  --seed 42 \
  --overwrite
```

Array trace：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/simulate_dynamic_bfp_array_trace.py \
  --metadata cache_data/arxiv_llama2_7b_oracle_W4GraphBFPA4to6_B128_deg_t0.20.json \
  --output_dir output/dynamic_bfp_fullstack/arxiv_array_trace_t020
```

结果填表：

| Config | Reuse | Direct | Residual | Miss | Acc | Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FullP8-miss | TBD | TBD | TBD | TBD | TBD | TBD |
| Dynamic BFP | TBD | TBD | TBD | TBD | TBD | TBD |

Array trace：

| Metric | Value |
| --- | ---: |
| refined ratio | TBD |
| effective mantissa bits | TBD |
| dynamic / BFPA4 cycles | TBD |
| dynamic / BFPA6 cycles | TBD |
| dynamic / BFPA8 cycles | TBD |

---

## 5. Reading Checklist

### 5.1 From full-stack log

Use:

```bash
rg -n "FINAL RESIDUAL|Baseline Acc|FullP8|AllP4|Rand|ReuseTrace" \
  output/dynamic_bfp_fullstack/<dataset>*/logs/*.log
```

Read:

```text
Reuse
Direct
Residual
Miss = 1 - Reuse
FullP8 Drop
Dynamic BFP Drop
```

### 5.2 From array trace

Use:

```bash
cat output/dynamic_bfp_fullstack/<dataset>_array_trace_t020/summary.md
```

Read:

```text
refined ratio
effective mantissa bits
dynamic / BFPA8 cycles
dynamic / BFPA6 cycles
dynamic / BFPA4 cycles
```

### 5.3 Final end-to-end number

Use:

```text
T_encoder = miss_ratio * dynamic_vs_BFPA8
T_e2e = 0.99 * T_encoder + 0.01
Speedup = 1 / T_e2e
```

The `0.99 / 0.01` split should be replaced by measured workload profiling if a more precise platform profile is available.

