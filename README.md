# GraphHop SimHash

`GraphhopSimhash` 当前主线做三件事：

1. **GraphHop SimHash reuse**：用图上下文 hash 找可复用节点，减少 embedding 计算。
2. **TSER score gate**：用图风险分数过滤危险复用，降低复用带来的精度掉点。
3. **AWQ W4A8/W4A4 embedding pool**：生成真实低精度 embedding，并评估固定预算下的 W4A8/W4A4 路由。

## 文档结构

当前 root 目录只保留主线文档：

```text
README.md
    项目入口、常用命令、文档索引。

SCORE_DEFINITIONS.md
    TSER reuse gate 与 TSER quant routing 的分数定义。

AWQ_W4A8_W4A4_GENERATION.md
    当前 AWQ-based W4A16/W4A8/W4A4 embedding pool 生成方式。

量化+哈希命令.md
    reuse_real_quant 联合实验命令与结果解释。
```

旧的 `REAL_QUANT.md`、`PTQ_EMBEDDING_QUANT.md`、`量化配置.md` 已经合并到上面几个文档中，避免重复和过时说明。

## 代码结构

```text
cli.py
    命令行参数。

runner.py
    实验流程、baseline 训练、reuse / quant 评估。

controller.py
    SimHash cache、候选检索、结构检查、score gate。

scoring.py
    propagation / graph context / low-degree uniqueness 分数。

generate_real_quant_pools.py
    FP16 / W4A16 / W4A8 / W4A4 embedding pool 生成。

real_quant.py
    真实 embedding pool 的固定预算评估和联合实验装配。

activation_outlier_calibration.py
    activation outlier 统计，可选用于 W4A4 outlier channel 保护。

features.py / projections.py
    cheap feature、hash feature、多头 learned hash projection。
```

## 1. 生成 AWQ Embedding Pool

ST / Cora + PubMed：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora pubmed \
  --llm_name ST \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

LLaMA-7B / Cora：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --overwrite
```

输出路径格式：

```text
cache_data/{dataset}_{model}_oracle_{tag}.pt
```

例如：

```text
cache_data/cora_ST_oracle_W4A16.pt
cache_data/cora_ST_oracle_W4A8.pt
cache_data/cora_ST_oracle_W4A4.pt
```

## 2. 真实量化固定预算评估

这组实验不做 hash reuse，只看 W4A8/W4A4 路由本身：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag W4A16 \
  --real_quant_int8_tag W4A8 \
  --real_quant_int4_tag W4A4 \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0
```

主表只保留可部署策略：

```text
AllFP
UniformW4A8
UniformW4A4
RandomTopK_W4A8
DegreeTopK_W4A8
TSERTopK_W4A8
```

`DegreeErrorTopK` / `TSERErrorTopK` 这类真实误差 oracle 行不作为主线，因为它们需要提前知道每个节点的 FP-vs-W4A4 误差。

## 3. Hash Reuse + TSER Gate

只评估 reuse，不叠加真实量化：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite score_ablation \
  --radius 2 \
  --hash_heads_per_route 4 \
  --main_hash_head_bits 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor
```

输出会比较：

```text
R2_NoScore
R2_DegreeOnly
R2_TSER
```

含义：

```text
NoScore:
    只看 hash/hamming，复用率高但掉点大。

DegreeOnly:
    只保护高传播节点。

TSER:
    degree + graph context + low-degree uniqueness。
```

## 4. Reuse + Real Quant 联合实验

联合实验中：

```text
reuse hit:
    直接读 cache，cost = 0。

reuse miss:
    再进入 W4A8 / W4A4 / FP 路径。
```

命令：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite reuse_real_quant \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag W4A16 \
  --real_quant_int8_tag W4A8 \
  --real_quant_int4_tag W4A4 \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --route_min_support_hits 3
```

## 5. TSER 参数探索

Cora：

```bash
RUNS=5 bash GraphhopSimhash/run_cora_tser_reuse_sweep.sh
```

PubMed：

```bash
RUNS=5 bash GraphhopSimhash/run_pubmed_tser_reuse_sweep.sh
```

结果目录：

```text
output/tser_reuse_sweep/cora/
output/tser_reuse_sweep/pubmed/
```

重点看：

```text
Reuse %
Acc
Drop %
Reuse n/d
```

## 6. 当前实验口径

论文叙事建议分清边界：

```text
SimHash:
    负责找可复用节点，贡献是减少计算。

TSER score gate:
    负责过滤危险复用，贡献是在复用率和精度之间做可调折中。

AWQ W4A8/W4A4:
    负责让低精度 embedding pool 本身可用。

Degree / TSER quant routing:
    负责在固定 W4A8 预算下选择哪些节点走安全路径。
```

不要把 oracle error-aware 策略写成主系统贡献；它们只能作为上界或 debug 参考。
