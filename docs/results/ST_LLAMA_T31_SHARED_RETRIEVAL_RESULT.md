# ST full 768d 与 Llama 共享检索骨架结果

日期：2026-06-01

本文档记录当前 `GraphhopSimhash-main` 主线代码下，`ST full embedding` 与 `Llama2-7B W4A16` 共享同一套在线检索骨架时的有效结果。

## 重要说明

这里的 `ST` 不再是 `data.x` 这条 384 维缓存特征线，而是显式读取的 768 维目标 embedding：

```text
cache_data/cora_ST_oracle_W4A16.pt
cache_data/pubmed_ST_oracle_W4A16.pt
```

也就是说，本文里的 `ST` 含义是：

```text
完整 ST 前向应得到的目标 embedding 真值
维度 = 768
当前使用的池 = W4A16 版本
```

当前脚本也已经改成显式读取这两份文件：

- `st_cora -> cache_data/cora_ST_oracle_W4A16.pt`
- `st_pubmed -> cache_data/pubmed_ST_oracle_W4A16.pt`

## 共享在线检索骨架

四条结果都基于同一套在线检索骨架：

```text
8 heads x 16 bits
radius = 2
关闭结构检查
开启 score gate
score reuse threshold T = 31

support >= 5   -> hard direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute
```

对应关键参数：

```bash
--hash_heads_per_route 8
--main_hash_head_bits 16 16 16 16 16 16 16 16
--radius 2
--disable_structure_check
--enable_score_gate
--allow_rare_fuzzy
--score_reuse_threshold 31
--score_propagation_weight 3
--score_graph_context_weight 1
--score_low_unique_weight 1
--score_pair_confidence_discount 1
--residual_hard_min_support_hits 5
--residual_soft_min_support_hits 3
```

离线采样逻辑也统一为当前版本：

```text
先取同 bucket 样本
不够再放宽到邻近 bucket
保留 support floor
不允许 <=2 head 的样本混入 residual 离线训练
```

## 当前有效结果

以下均为 `3-run` 均值。

| Embedding 源 | 数据集 | Baseline Acc | ResidualReuse | Acc | Drop | TrainPairs | Alpha | gate 设置 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `ST:W4A16 768d` | Cora | 0.6789 | 39.4% | 0.6555 | 2.34% | 217.0 | 0.390 | `separate`, tau=0.575 | 未达标 |
| `ST:W4A16 768d` | PubMed | 0.7710 | 40.3% | 0.7491 | 2.18% | 62.3 | 0.042 | `shared`, tau=0.65 | 复用达标，掉点未达标 |
| `Llama2-7B:W4A16` | Cora | 0.7310 | 41.1% | 0.7136 | 1.74% | 330.7 | 0.089 | `separate`, tau=0.40 | 达标 |
| `Llama2-7B:W4A16` | PubMed | 0.7000 | 40.7% | 0.6816 | 1.84% | 317.3 | 0.037 | `shared`, tau=0.91 | 达标 |

## 补充对比

### ST:W4A16 768d / Cora

| 配置 | Reuse | Acc | Drop |
|---|---:|---:|---:|
| DirectReuse | 12.2% | 0.6688 | 1.02% |
| SoftDirectReuse | 39.4% | 0.6491 | 2.98% |
| ResidualReuse | 39.4% | 0.6555 | 2.34% |

### ST:W4A16 768d / PubMed

| 配置 | Reuse | Acc | Drop |
|---|---:|---:|---:|
| DirectReuse | 26.3% | 0.7587 | 1.22% |
| SoftDirectReuse | 71.6% | 0.7181 | 5.29% |
| ResidualReuse | 40.3% | 0.7491 | 2.18% |

### Llama2-7B:W4A16 / Cora

| 配置 | Reuse | Acc | Drop |
|---|---:|---:|---:|
| DirectReuse | 16.2% | 0.7263 | 0.47% |
| SoftDirectReuse | 52.5% | 0.6999 | 3.11% |
| ResidualReuse | 41.1% | 0.7136 | 1.74% |

### Llama2-7B:W4A16 / PubMed

| 配置 | Reuse | Acc | Drop |
|---|---:|---:|---:|
| DirectReuse | 36.2% | 0.6829 | 1.71% |
| SoftDirectReuse | 79.9% | 0.6442 | 5.58% |
| ResidualReuse | 40.7% | 0.6816 | 1.84% |

## 当前结论

基于真正的 `ST full 768d` 目标 embedding，可以直接得到下面的结论：

1. `Llama2-7B W4A16` 这条线在 `Cora / PubMed` 上都达标：
   - 复用率 `40%+`
   - 掉点 `< 2%`
2. `ST full 768d` 这条线在同一套在线骨架下没有同时达标：
   - `Cora`：复用率没到 `40%`，掉点也超过 `2%`
   - `PubMed`：复用率刚过 `40%`，但掉点还是高于 `2%`
3. 也就是说，**“同一套 T31 在线配置同时适配 ST full 768d 与 Llama W4A16，并在 Cora/PubMed 全部达到 40%+ 复用、2% 内掉点” 这个结论当前不成立。**

一句话总结：

```text
一旦 ST 真正切回 full 768d embedding，
共享 T31 骨架对 Llama 仍成立，
但对 ST 不成立。
```

## 为什么会这样

现在的现象很清楚：

- `data_x 384d` 那条 ST 线更容易复用，因为目标空间更平、更粗糙
- `ST full 768d` 目标更尖锐，中间态候选更容易越过分类边界
- 所以同样的 `3..4 head -> residual` 策略，在 `ST full 768d` 上更难压住掉点

从结果上看：

- `Cora/ST full` 更像是召回不够，复用率上不去
- `PubMed/ST full` 更像是 accept gate 不够准，复用率勉强够时掉点就超线

## 当前复现实验入口

当前推荐入口：

```bash
cd /home/qiumingzhi/Simhash-S/OneForAll
python GraphhopSimhash-main/scripts/run_graphhopsimhash_main.py ...
```

批量脚本：

```bash
bash GraphhopSimhash-main/scripts/run_t31_shared_frontend_reuse.sh
```

如果要改 ST oracle 路径，可以覆盖：

```bash
ST_CORA_PATH=cache_data/cora_ST_oracle_W4A16.pt
ST_PUBMED_PATH=cache_data/pubmed_ST_oracle_W4A16.pt
```

## 当前日志

```text
/tmp/st_oracle_cora_t31_current_3run.log
/tmp/st_oracle_pubmed_t31_current_3run.log
/tmp/llama_cora_t31_current_3run.log
/tmp/llama_pubmed_t31_current_3run.log
```

对应 trace cache：

```text
/home/qiumingzhi/Simhash-S/OneForAll/cache_data/reuse_traces/st_oracle_cora_t31_current_cora_seed{42,43,44}.pt
/home/qiumingzhi/Simhash-S/OneForAll/cache_data/reuse_traces/st_oracle_pubmed_t31_current_pubmed_seed{42,43,44}.pt
/home/qiumingzhi/Simhash-S/OneForAll/cache_data/reuse_traces/llama_cora_t31_current_cora_seed{42,43,44}.pt
/home/qiumingzhi/Simhash-S/OneForAll/cache_data/reuse_traces/llama_pubmed_t31_current_pubmed_seed{42,43,44}.pt
```
