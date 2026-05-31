# Graph-Bit Full-Stack Reproduction Guide

本文档记录 Graph-Bit full-stack 实验的复现流程、参数入口和输出位置，覆盖以下实验产物：

```text
1. residual reuse + Graph-Bit accuracy / drop
2. per-node stop-depth trace
3. ONNXim component lookup
4. trace-driven scheduler replay
5. FullP8-miss / GraphBit-now / FullP8-bucket / RiskBucket cycles 表
```

相关机制说明：

```text
docs/npu/GRAPH_BIT_NPU_DESIGN.md
docs/npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
```

本文档重点是运行流程、参数修改方式和结果读取路径。

## 1. 当前主线实验是什么

默认主线是：

```text
Dataset:
    cora

Backend:
    LLaMA-7B embedding pools

Front-end reuse:
    8 heads x 16 bits
    radius = 2
    score gate on
    score weights = 3 / 1 / 1

Residual split:
    hard direct reuse: support >= 5
    residual reuse:    support == 4
    compute/miss:      support < 4

Graph-Bit:
    miss nodes only
    priority = Degree / propagation_q
    predictor-free bound enabled
    high/mid/low min depths = 8 / 6 / 4
    high/mid/low tolerances = 0.00 / 0.02 / 0.04

Trace replay:
    baseline tile batch = 16
    risk bucket candidate batch = 32 / 64
```

这套配置的短名：

```text
cora_h8_54_T40
```

含义：

```text
h8:
    8 hash heads

54:
    hard>=5, soft=4

T40:
    score_reuse_threshold = 40
```

## 2. 输入文件要求

### 2.1 LLaMA precision-depth embedding pools

Graph-Bit accuracy validation 需要以下 cache：

```text
cache_data/cora_llama2_7b_oracle_W4A8.pt
cache_data/cora_llama2_7b_oracle_W4A6.pt
cache_data/cora_llama2_7b_oracle_W4A5.pt
cache_data/cora_llama2_7b_oracle_W4A4.pt
```

如果跑 PubMed，需要对应：

```text
cache_data/pubmed_llama2_7b_oracle_W4A8.pt
cache_data/pubmed_llama2_7b_oracle_W4A6.pt
cache_data/pubmed_llama2_7b_oracle_W4A5.pt
cache_data/pubmed_llama2_7b_oracle_W4A4.pt
```

如果缺少这些 pool，先生成 embedding pool。生成命令见：

```text
docs/core/AWQ_W4A8_W4A4_GENERATION.md
docs/tools/量化+哈希命令.md
```

### 2.2 ONNXim component results

trace replay 需要 ONNXim component lookup。默认路径是：

```text
output/onnxim_graphbit/risk_bucket_components_s8
```

如果这个目录不存在，先跑：

```bash
cd /home/zhangshangtong/Transformer/OFA

SEQ_LEN=8 \
STATIONARY_TILE_BATCHES="32 64" \
OUT_ROOT=output/onnxim_graphbit/risk_bucket_components_s8 \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_risk_bucket_components.sh
```

它会生成：

```text
full_p8/
p8_now/ p6_now/ p5_now/ p4_now/
p8_ws_b32/ p6_ws_b32/ p5_ws_b32/ p4_ws_b32/
p8_ws_b64/ p6_ws_b64/ p5_ws_b64/ p4_ws_b64/
```

每个目录里有 ONNXim 的 `aggregate.json`。

## 3. 一键跑当前 Cora 主线

最简单命令：

```bash
cd /home/zhangshangtong/Transformer/OFA

bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

默认会做两件事：

```text
1. 跑 residual_precision_depth，导出 per-node trace。
2. 调用 replay_graphbit_trace_scheduler.py，生成 trace replay cycles 表。
```

默认输出目录：

```text
output/graphbit_trace_replay/cora_h8_54_T40_quick
```

如果使用当前文档里的 boundclean quick 目录，可以显式指定：

```bash
cd /home/zhangshangtong/Transformer/OFA

RUNS=1 \
DATASET=cora \
THRESHOLD=40 \
HARD_SUPPORT=5 \
SOFT_SUPPORT=4 \
FRONTEND_ID=h8_54_T40 \
OUT_DIR=output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick \
bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

主要输出：

```text
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/node_traces/cora_seed42_DegBound.jsonl
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/predictor_free_main.txt
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_trace_replay.txt
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_component_lookup.tsv
```

如果要查看 bit-depth-sensitive 的片上活动分解，可继续运行：

```bash
cd /home/zhangshangtong/Transformer/OFA

/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/model_graphbit_activity_breakdown.py \
  --replay-json output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_trace_replay.json \
  --output-dir output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/activity_breakdown
```

输出：

```text
output/.../activity_breakdown/graphbit_activity_breakdown.txt
output/.../activity_breakdown/graphbit_activity_breakdown.tsv
output/.../activity_breakdown/graphbit_activity_breakdown.json
```

它会把每个方法拆成：

```text
W_HBM / A_HBM / A_RF / PE / W_RF / Psum / Out / Scheduler
```

用于判断 mixed-depth 相比 FullP8-bucket 是否真的减少片上活动。

如果要排查 P8/P6/P5 的 bit-depth 为什么没有明显转成 ONNXim wall cycles，可运行：

```bash
cd /home/zhangshangtong/Transformer/OFA

/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/diagnose_graphbit_cycle_sensitivity.py \
  --component-root output/onnxim_graphbit/risk_bucket_components_s8 \
  --output-dir output/onnxim_graphbit/cycle_sensitivity
```

输出：

```text
output/onnxim_graphbit/cycle_sensitivity/cycle_sensitivity.txt
output/onnxim_graphbit/cycle_sensitivity/measured_components.tsv
output/onnxim_graphbit/cycle_sensitivity/roofline_sensitivity.tsv
```

这个诊断会同时报告：

```text
ONNXim wall cycles:
    当前 simulator 看到的组件总 cycles。

Effective compute cycles:
    Graph-Bit 按 depth 缩小后的 bit-plane compute 总量。

PE critical proxy:
    如果 bit-serial compute 成为瓶颈，理论上应暴露出来的 PE issue path。

Roofline sensitivity:
    memory path 压低到什么程度后，A8 -> A6/A5 才会转成 latency 收益。
```

## 4. 分步跑法

如果要调参数，建议分步跑，方便检查每一步。

### 4.1 Step A: 跑算法并导出 node trace

```bash
cd /home/zhangshangtong/Transformer/OFA

RUNS=1 \
RUN_ALGO=1 \
RUN_ONNXIM=0 \
DATASET=cora \
THRESHOLD=40 \
HARD_SUPPORT=5 \
SOFT_SUPPORT=4 \
FRONTEND_ID=h8_54_T40 \
BUDGET=boundclean \
HIGH_RATIO=0.20 \
MID_RATIO=0.50 \
LOW_RATIO=0.0 \
OUT_DIR=output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick \
TRACE_EXPORT=1 \
TRACE_EXPORT_DIR=output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/node_traces \
TRACE_EXPORT_CONFIGS='DegBound' \
BOUND_ENABLE=1 \
BOUND_PRIORITIES='degree' \
BOUND_HIGH_MIN=8 \
BOUND_MID_MIN=6 \
BOUND_LOW_MIN=4 \
BOUND_HIGH_TOL=0.0 \
BOUND_MID_TOL=0.02 \
BOUND_LOW_TOL=0.04 \
BOUND_TILE_K=128 \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```

检查输出：

```text
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/predictor_free_main.txt
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/node_traces/cora_seed42_DegBound.jsonl
```

`predictor_free_main.txt` 主要看：

```text
FullP8 / DegBound / TSERBound 等 accuracy/drop
Reuse %
Direct %
Residual %
P8/P6/P5/P4 %
Cost
FinalErr
```

`node_traces/*.jsonl` 供 replay 使用。

### 4.2 Step B: 跑 ONNXim component lookup

通常只需要跑一次。当前默认 `SEQ_LEN=8` 是为了快速生成 component cost。

```bash
cd /home/zhangshangtong/Transformer/OFA

SEQ_LEN=8 \
STATIONARY_TILE_BATCHES="32 64" \
OUT_ROOT=output/onnxim_graphbit/risk_bucket_components_s8 \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_risk_bucket_components.sh
```

输出：

```text
output/onnxim_graphbit/risk_bucket_components_s8/full_p8/aggregate.json
output/onnxim_graphbit/risk_bucket_components_s8/p6_ws_b32/aggregate.json
output/onnxim_graphbit/risk_bucket_components_s8/p6_ws_b64/aggregate.json
...
```

### 4.3 Step C: replay trace 得到 cycles 表

```bash
cd /home/zhangshangtong/Transformer/OFA

/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/replay_graphbit_trace_scheduler.py \
  --trace output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/node_traces/cora_seed42_DegBound.jsonl \
  --components-root output/onnxim_graphbit/risk_bucket_components_s8 \
  --output-dir output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay \
  --fullp8-drop-percent 0.77 \
  --drop-percent 2.13 \
  --baseline-tile-batch 16 \
  --candidate-batches 32 64
```

输出：

```text
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_trace_replay.txt
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_trace_replay.tsv
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_trace_replay.json
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_component_lookup.tsv
```

注意：

```text
--fullp8-drop-percent:
    填 Step A 里 FullP8-miss 对应 drop。

--drop-percent:
    填 Step A 里 Graph-Bit policy 对应 drop，比如 DegBound drop。
```

这两个参数只用于 replay 表展示 accuracy/drop，不参与 cycles 计算。

## 5. 如何调参数

### 5.1 调 score threshold T

参数：

```text
THRESHOLD
```

示例：

```bash
THRESHOLD=35 FRONTEND_ID=h8_54_T35 ...
THRESHOLD=40 FRONTEND_ID=h8_54_T40 ...
THRESHOLD=45 FRONTEND_ID=h8_54_T45 ...
```

影响：

```text
T 越小:
    score gate 越严格
    reuse 更少
    drop 更低
    miss 更多
    encoder/NPU workload 更大

T 越大:
    score gate 越宽松
    reuse 更多
    drop 更高
    miss 更少
    encoder/NPU workload 更小
```

### 5.2 调 hard / soft support

参数：

```text
HARD_SUPPORT
SOFT_SUPPORT
```

当前主线：

```text
HARD_SUPPORT=5
SOFT_SUPPORT=4
```

含义：

```text
support >= 5:
    hard direct reuse

support == 4:
    residual correction

support < 4:
    compute / Graph-Bit miss
```

示例：

```bash
HARD_SUPPORT=5 SOFT_SUPPORT=4 FRONTEND_ID=h8_54_T40
HARD_SUPPORT=6 SOFT_SUPPORT=4 FRONTEND_ID=h8_64_T40
HARD_SUPPORT=4 SOFT_SUPPORT=3 FRONTEND_ID=h8_43_T40
```

影响：

```text
hard 更低:
    direct reuse 更多，但可能引入更多错误。

soft 更低:
    residual candidates 更多，复用率提高，但 residual/gate 压力变大。

hard 更高:
    更保守，drop 低，但 reuse 少。
```

### 5.3 调 Graph-Bit high/mid/low 比例

参数：

```text
HIGH_RATIO
MID_RATIO
LOW_RATIO
```

当前 quick flow：

```text
HIGH_RATIO=0.20
MID_RATIO=0.50
LOW_RATIO=0.0
```

代码中 `bound_budget` 会先把所有 miss nodes 初始化为 low bucket，再覆盖 high 和 mid。因此：

```text
low share = 1 - HIGH_RATIO - MID_RATIO
```

即使 `LOW_RATIO=0.0`，剩余未覆盖节点仍然进入 low bucket。

当前：

```text
high = 20%
mid  = 50%
low  = 30%
```

常用配置：

```bash
# conservative
HIGH_RATIO=0.60 MID_RATIO=0.30 LOW_RATIO=0.10

# balanced
HIGH_RATIO=0.20 MID_RATIO=0.50 LOW_RATIO=0.0

# aggressive
HIGH_RATIO=0.10 MID_RATIO=0.50 LOW_RATIO=0.0
```

### 5.4 调 predictor-free bound 阈值

参数：

```text
BOUND_HIGH_MIN
BOUND_MID_MIN
BOUND_LOW_MIN
BOUND_HIGH_TOL
BOUND_MID_TOL
BOUND_LOW_TOL
BOUND_SCALE
BOUND_TILE_K
```

默认：

```text
BOUND_HIGH_MIN=8
BOUND_MID_MIN=6
BOUND_LOW_MIN=4
BOUND_HIGH_TOL=0.0
BOUND_MID_TOL=0.02
BOUND_LOW_TOL=0.04
BOUND_SCALE=1.0
BOUND_TILE_K=128
```

影响：

```text
tolerance 越小:
    early stop 越保守
    AvgD 更高
    drop 更低
    cycles 更高

tolerance 越大:
    early stop 越激进
    AvgD 更低
    drop 更高
    cycles 更低
```

### 5.5 调 scheduler batch size

参数：

```text
CANDIDATE_BATCHES
```

示例：

```bash
CANDIDATE_BATCHES="16 32 64 128"
```

影响：

```text
batch 越大:
    Wloads 越少
    Wscale 越低
    cycles / traffic 可能更低
    但 SRAM 压力和 tail padding 可能更大

batch 越小:
    更容易硬件落地
    W tile reuse 收益较小
```

replay 表里会显示：

```text
Wloads
Wscale
Tail
SRAM
```

如果 `SRAM=no`，说明该 batch size 在当前 buffer model 下不可行。

## 6. 输出文件怎么读

### 6.1 predictor_free_main.txt

路径：

```text
output/.../predictor_free_main.txt
```

用途：

```text
查看算法层面的 accuracy/drop。
```

重点字段：

```text
Reuse:
    direct + residual 总复用率。

Direct:
    hard direct reuse 比例。

Residual:
    residual correction 比例。

P8/P6/P5/P4:
    miss nodes 里不同 Graph-Bit depth 的比例，按全图节点归一。

Cost:
    embedding-pool proxy cost，不是 ONNXim cycles。

Drop:
    accuracy drop。
```

### 6.2 node trace JSONL

路径：

```text
output/.../node_traces/cora_seed42_DegBound.jsonl
```

用途：

```text
真实 per-node workload trace。
```

每个节点包含：

```text
role:
    direct / residual / miss

is_miss:
    是否进入 Graph-Bit NPU

action_bit / stop_depth:
    miss node 的实际 depth

depth_bucket:
    p8 / p6 / p5 / p4

support_hits:
    多 head 支持数

degree_q / tser_q / context_q / low_unique_q:
    risk score
```

### 6.3 trace_replay.txt

路径：

```text
output/.../replay/*_trace_replay.txt
```

用途：

```text
硬件主表。
```

重点行：

```text
FullP8-miss:
    固定 reuse/residual 前端，所有 miss nodes 都完整 P8。

GraphBit-now:
    使用真实 stop_depth，但不扩大 W tile batch。

FullP8-bucket-b32/b64:
    所有 miss nodes 仍完整 P8，但使用更大的 W-stationary service window。
    这行用于隔离 W tile batching 的收益。

OriginalOrder-b32/b64:
    不分桶，按原始节点顺序组 batch。
    mixed batch 按最高 depth 执行。

RiskBucket-b32/b64:
    按真实 stop_depth 分桶，再组 batch。
```

重点字段：

```text
Cycles:
    归一化到 all-node FullP8/W4A8 encoder。

Traffic:
    归一化 DRAM read + write request。

Energy:
    当前 proxy = 0.5 * Cycles + 0.5 * Traffic。

AvgD:
    miss nodes 平均 stop depth。

Hist(miss):
    miss nodes 的 D5/D6/D8 分布。

Wloads:
    trace replay 中 W tile load 次数。

Wscale:
    Wloads / FullP8-miss baseline Wloads。

Tail:
    bucket batch 的有效填充率。

SRAM:
    当前 buffer model 下 batch 是否可容纳。
```

### 6.4 component_lookup.tsv

路径：

```text
output/.../replay/*_component_lookup.tsv
```

用途：

```text
查看 ONNXim component cost。
```

示例：

```text
full_p8
p5_now / p6_now / p8_now
p5_ws_b32 / p6_ws_b32 / p8_ws_b32
p5_ws_b64 / p6_ws_b64 / p8_ws_b64
```

`trace_replay.txt` 中的 `Cycles` 就是用这些 component rows 按真实 node trace 加权组合出来的。

## 7. 当前 Cora 主线结果

当前 quick trace：

```text
nodes = 2708
reuse = 27.8%
miss = 72.2%
```

主表：

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgD | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 27.8% | 72.2% | 0.722 | 0.722 | 0.722 | 0.77% | 8.00 | 123 | 1.000 |
| GraphBit-now | 27.8% | 72.2% | 0.716 | 0.719 | 0.717 | 2.13% | 6.10 | 123 | 1.000 |
| FullP8-bucket-b32 | 27.8% | 72.2% | 0.385 | 0.368 | 0.377 | 0.77% | 8.00 | 62 | 0.504 |
| RiskBucket-b32 | 27.8% | 72.2% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | 63 | 0.512 |
| FullP8-bucket-b64 | 27.8% | 72.2% | 0.290 | 0.191 | 0.241 | 0.77% | 8.00 | 31 | 0.252 |
| RiskBucket-b64 | 27.8% | 72.2% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | 33 | 0.268 |

加速计算：

```text
相对 FullP8-miss:
    GraphBit-now:    0.722 / 0.716 = 1.01x
    FullP8-bucket-b32: 0.722 / 0.385 = 1.88x
    RiskBucket-b32:  0.722 / 0.384 = 1.88x
    FullP8-bucket-b64: 0.722 / 0.290 = 2.49x
    RiskBucket-b64:  0.722 / 0.289 = 2.50x

相对 all-node FullP8/W4A8:
    FullP8-miss:     1 / 0.722 = 1.38x
    GraphBit-now:    1 / 0.716 = 1.40x
    FullP8-bucket-b32: 1 / 0.385 = 2.60x
    RiskBucket-b32:  1 / 0.384 = 2.60x
    FullP8-bucket-b64: 1 / 0.290 = 3.45x
    RiskBucket-b64:  1 / 0.289 = 3.46x
```

这张消融的关键结论是：

```text
FullP8-bucket 和 RiskBucket 的 cycles 几乎相同；
RiskBucket 的 AvgD 更低，但 accuracy drop 更高。
```

因此当前 trace 下，主要硬件收益来自 W-stationary bucket service window，而不是 mixed-depth early stop。mixed-depth / predictor-free bound 目前更适合定位为片上算术/能耗优化，需要继续用更细粒度 RF、psum、PE 活动模型验证其额外收益。

### 7.1 Activity Breakdown

用 `model_graphbit_activity_breakdown.py` 对同一 replay JSON 做片上活动拆分：

| Compare | ONNX-C Save | Activity-C Save | Activity-E Save | PE/W_RF/Psum Save | Extra Drop |
|---|---:|---:|---:|---:|---:|
| RiskBucket-b32 vs FullP8-bucket-b32 | 0.1% | 12.1% | 15.6% | 23.7% | +1.36% |
| RiskBucket-b64 vs FullP8-bucket-b64 | 0.3% | 13.9% | 16.8% | 23.7% | +1.36% |

这说明：

```text
ONNX cycles:
    目前主要反映 W tile batching，对 P8/P6/P5 depth 不敏感。

activity model:
    mixed-depth 能明确减少 PE issue、W RF/broadcast 和 psum update。
```

所以当前结论是：

```text
W-stationary bucket scheduler 负责 latency / traffic 主收益；
mixed-depth predictor-free early stop 负责片上活动和能耗收益。
```

## 8. 常见复现实验模板

### 8.1 扫 T

```bash
for T in 35 40 45; do
  RUNS=1 \
  DATASET=cora \
  THRESHOLD=${T} \
  HARD_SUPPORT=5 \
  SOFT_SUPPORT=4 \
  FRONTEND_ID=h8_54_T${T} \
  OUT_DIR=output/graphbit_trace_replay/cora_h8_54_T${T}_quick \
  bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
done
```

### 8.2 扫 hard / soft

```bash
for cfg in "5 4" "6 4" "4 3"; do
  set -- ${cfg}
  H=$1
  S=$2
  RUNS=1 \
  DATASET=cora \
  THRESHOLD=40 \
  HARD_SUPPORT=${H} \
  SOFT_SUPPORT=${S} \
  FRONTEND_ID=h8_${H}${S}_T40 \
  OUT_DIR=output/graphbit_trace_replay/cora_h8_${H}${S}_T40_quick \
  bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
done
```

### 8.3 只 replay，不重跑算法

如果已经有 trace：

```bash
SKIP_EXPORT=1 \
TRACE_PATH=output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/node_traces/cora_seed42_DegBound.jsonl \
REPLAY_DIR=output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay_b128 \
CANDIDATE_BATCHES="32 64 128" \
bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

### 8.4 PubMed

PubMed 更慢，建议先 `RUNS=1`：

```bash
RUNS=1 \
DATASET=pubmed \
THRESHOLD=40 \
HARD_SUPPORT=5 \
SOFT_SUPPORT=4 \
FRONTEND_ID=h8_54_T40 \
OUT_DIR=output/graphbit_trace_replay/pubmed_h8_54_T40_quick \
bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

如果要多 seed，再改：

```bash
RUNS=3
```

## 9. 调参时最容易犯的错

### 9.1 忘记改 FRONTEND_ID

如果你改了：

```text
THRESHOLD
HARD_SUPPORT
SOFT_SUPPORT
```

建议同步改：

```text
FRONTEND_ID
```

否则 output 目录名字会误导。

### 9.2 replay 的 drop 参数没有更新

`replay_graphbit_trace_scheduler.py` 的：

```text
--fullp8-drop-percent
--drop-percent
```

只是展示字段，不影响 cycles。调参后应该从 `predictor_free_main.txt` 里读取对应 drop，再填入 replay。

如果只是看 cycles / traffic / Wloads，可以先用默认值；如果要出论文表，需要填准确 drop。

### 9.3 component lookup 和 candidate batch 不匹配

如果 replay 里写：

```text
CANDIDATE_BATCHES="32 64 128"
```

但 component root 里只有：

```text
p*_ws_b32
p*_ws_b64
```

那么 b128 会缺 component。需要先跑 ONNXim component：

```bash
STATIONARY_TILE_BATCHES="32 64 128" \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_risk_bucket_components.sh
```

### 9.4 把 Wscale 当成 speedup

不要这样写：

```text
speedup = 1 / Wscale
```

正确是：

```text
speedup = baseline_cycles / method_cycles
```

`Wscale` 只表示 W tile load 次数缩放。

### 9.5 把 Cost 当成 Cycles

`predictor_free_main.txt` 里的 `Cost` 是 embedding-pool proxy cost。  
`trace_replay.txt` 里的 `Cycles` 才是 ONNXim component + trace replay 得到的硬件 cycles proxy。

## 10. 推荐调参顺序

建议按这个顺序调，不要一口气全扫：

```text
1. 固定 Graph-Bit，先调 reuse 前端：
   T, HARD_SUPPORT, SOFT_SUPPORT

2. 固定 reuse 前端，调 Graph-Bit risk budget：
   HIGH_RATIO, MID_RATIO

3. 固定 budget，调 bound：
   BOUND_MID_TOL, BOUND_LOW_TOL

4. 固定算法输出，调硬件 scheduler：
   CANDIDATE_BATCHES, BASELINE_TILE_BATCH, SRAM/buffer 参数
```

推荐先用 Cora `RUNS=1` 快速扫，再对候选点跑：

```text
Cora RUNS=3/10
PubMed RUNS=1/3
Arxiv 最后再跑
```

## 11. 最小复现 checklist

跑完后至少检查以下文件：

```text
1. predictor_free_main.txt
   是否有 FullP8 / DegBound 结果。

2. node_traces/*.jsonl
   是否存在 per-node trace。

3. *_component_lookup.tsv
   是否包含 full_p8, p5/p6/p8 now/ws_b32/ws_b64。

4. *_trace_replay.txt
   是否包含 FullP8-miss, GraphBit-now, RiskBucket-b32, RiskBucket-b64。
```

最小成功输出应该类似：

```text
FullP8-miss
GraphBit-now
RiskBucket-b32
RiskBucket-b64
```

如果这四行都有，说明 full-stack reproduction flow 已经跑通。
