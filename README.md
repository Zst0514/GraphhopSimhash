# GraphHop SimHash

`GraphhopSimhash` 当前主线做三件事：

1. **GraphHop SimHash reuse**：用图上下文 hash 找可复用节点，减少 embedding 计算。
2. **TSER score gate**：用图风险分数过滤危险复用，降低复用带来的精度掉点。
3. **Hierarchical encoder execution**：把节点路由到 exact reuse、residual reuse、FFN-gated W4A8 或 full W4A8 encoder。

## 文档结构

当前 root 目录只保留 `README.md`，其他项目文档统一放在 `docs/`：

```text
README.md
    项目入口、常用命令、文档索引。

docs/SCORE_DEFINITIONS.md
    TSER reuse gate 的分数定义，以及量化路由中 Degree/TSER 的边界说明。

docs/AWQ_W4A8_W4A4_GENERATION.md
    当前 AWQ-based W4A16/W4A8/W4A4 embedding pool 生成方式。

docs/RESIDUAL_CORRECTED_REUSE.md
    fuzzy hash hit 上的 low-rank residual correction 机制与实验结果。

docs/FFN_CHANNEL_GATING.md
    面向 W4A8 encoder NPU 的 FFN channel gating 原型。

docs/HIERARCHICAL_ENCODER_NPU_DESIGN.md
    当前完整系统思路：P0/P1/P2/P3 分层 encoder 执行路径、硬件落点和端到端结果。

docs/GRAPH_AWARE_ENCODER_NPU_PROPOSAL.md
    基于当前实验和加速器综述提炼的 Graph-aware encoder NPU 方案建议。

docs/NPU_ADAPTIVE_ENCODER_EXPERIMENTS.md
    面向 graph-aware encoder NPU 的实验设计与验证路线。

docs/LLM_ACCELERATOR_SURVEY.md
    Encoder / 通用 Transformer / NPU 加速器综述。

docs/LLM_DECODER_ACCELERATOR_SURVEY.md
    Decoder / serving / KV-cache 相关加速器综述。

docs/量化+哈希命令.md
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

generate_real_quant_pools_ptq_legacy.py
    保留旧 PTQ_TEST 生成路径，用于复现已验证过的低误差 W4A8 pool。

real_quant.py
    真实 embedding pool 的固定预算评估和联合实验装配。

activation_outlier_calibration.py
    activation outlier 统计，可选用于 W4A4 outlier channel 保护。

features.py / projections.py
    cheap feature、hash feature、多头 learned hash projection。
```

## 0. 当前系统总览

当前更推荐把系统讲成 graph-aware hierarchical encoder execution：

```text
P0: exact hash reuse
    cost ~= 0

P1: fuzzy hash reuse + residual correction
    cost ~= tiny adapter

P2: W4A8 encoder + FFN channel gating
    cost < full W4A8

P3: full W4A8 encoder
    精度兜底路径
```

对应完整说明见：

```text
docs/HIERARCHICAL_ENCODER_NPU_DESIGN.md
docs/RESIDUAL_CORRECTED_REUSE.md
docs/FFN_CHANNEL_GATING.md
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

当前量化路由结论要和 reuse gate 分开：

```text
量化掉点主要由两件事决定：
    1. 节点自身量化误差
    2. 量化误差沿图传播的范围

第 1 项在线不可得；第 2 项最直接的可部署代理是 degree / propagation risk。
```

因此固定预算量化路由中，实验结果显示 `DegreeTopK_W4A8` 优于 `TSERTopK_W4A8`。
`TSERTopK_W4A8` 作为图语义修正消融保留。

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
    当前实验中 DegreeTopK_W4A8 优于 TSERTopK_W4A8；
    TSER quant routing 主要作为图语义修正消融。
```

不能把 oracle error-aware 策略当成可部署系统策略；它们只能作为离线上界或 debug 参考。
