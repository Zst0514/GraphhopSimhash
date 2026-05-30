# Graph-Bit Full-Stack Hardware Table

本文档记录当前 Graph-Bit NPU 的最终硬件合成表。它把三类信息合在一起：

```text
1. residual / reuse workload profile
   真实前端比例：direct reuse、residual reuse、miss nodes。

2. ONNXim stop-depth trace
   每个 P8 / P6 / P5 / P4 bucket 在 bit-serial early-stop datapath 下的 cycles、traffic、AvgDepth、DepthHist。

3. bucket scheduler feasibility
   同风险 miss nodes 是否足够形成 32 / 64 级别 micro-batch，以及 SRAM 是否能容纳 W tile + activation plane buffer + psum。
```

归一化口径：

```text
Cycles / Traffic / Energy 都相对于“全图每个节点都执行 FullP8 encoder”的成本。
Reuse / residual 节点被视为不进入 encoder；只有 miss nodes 进入 Graph-Bit NPU。
Drop 来自对应 workload 的 embedding-pool accuracy profile；bucket scheduler 不改变数值结果，只改变硬件执行成本。
```

## 1. 生成命令

先生成 ONNXim risk-bucket component trace：

```bash
SEQ_LEN=8 \
STATIONARY_TILE_BATCHES="32 64" \
OUT_ROOT=output/onnxim_graphbit/risk_bucket_components_s8 \
LOG_LEVEL=info \
bash GraphhopSimhash/scripts/run_onnxim_graphbit_risk_bucket_components.sh
```

生成 bucket feasibility：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/model_graphbit_bucket_scheduler.py \
  --workload-json output/graphbit_bound_runtime/cora_h8_54_T40_boundclean_quick/predictor_free_workload.json \
  --workload-json output/graphbit_bound_runtime/pubmed_h8_54_T40_boundclean_runs10/predictor_free_workload.json \
  --profile-match degree_runtime-bound \
  --manual-profile arxiv_h20_m30_l30_Deg:arxiv:0.0:0.20:0.30:0.30:0.20 \
  --tile-batches 16 32 64 \
  --baseline-tile-batch 16 \
  --sram-kb 512 \
  --tile-k 128 \
  --tile-n 128 \
  --weight-bits 4 \
  --fetch-depth 6 \
  --psum-bits 32 \
  --output-bits 16 \
  --buffer-factor 2 \
  --output-dir output/onnxim_graphbit/bucket_feasibility
```

合成 Cora 主表：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/summarize_graphbit_fullstack_hardware.py \
  --workload-json output/graphbit_bound_runtime/cora_h8_54_T40_boundclean_quick/predictor_free_workload.json \
  --dataset cora \
  --components-root output/onnxim_graphbit/risk_bucket_components_s8 \
  --feasibility-tsv output/onnxim_graphbit/bucket_feasibility/bucket_scheduler_feasibility.tsv \
  --output-dir output/onnxim_graphbit/fullstack_hardware
```

PubMed 有两个版本：

```text
h8_54_T40:
    reuse 高，但 PubMed/LLaMA 的 FullP8-miss 本身已经超过 3% drop。
    适合作为 datapath stress point，不建议作为主线精度点。

h8_76_T40:
    reuse 低，但 FullP8-miss drop 很小。
    适合作为 PubMed/LLaMA 的 accuracy-safe hardware point。
```

PubMed h8_76 的 feasibility 使用手动 profile：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/model_graphbit_bucket_scheduler.py \
  --manual-profile pubmed_h8_76_degree:pubmed:0.082:0.184:0.459:0.276:0.0 \
  --tile-batches 16 32 64 \
  --baseline-tile-batch 16 \
  --sram-kb 512 \
  --tile-k 128 \
  --tile-n 128 \
  --weight-bits 4 \
  --fetch-depth 6 \
  --psum-bits 32 \
  --output-bits 16 \
  --buffer-factor 2 \
  --output-dir output/onnxim_graphbit/bucket_feasibility_pubmed_h8_76
```

合成 PubMed h8_76：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/summarize_graphbit_fullstack_hardware.py \
  --workload-json output/graphbit_bound_runtime/pubmed_h8_76_T40_boundclean_runs10/predictor_free_workload.json \
  --dataset pubmed \
  --components-root output/onnxim_graphbit/risk_bucket_components_s8 \
  --feasibility-tsv output/onnxim_graphbit/bucket_feasibility_pubmed_h8_76/bucket_scheduler_feasibility.tsv \
  --feasibility-profile-match pubmed_h8_76_degree \
  --output-dir output/onnxim_graphbit/fullstack_hardware
```

## 2. Cora / LLaMA h8_54_T40

输出：

```text
output/onnxim_graphbit/fullstack_hardware/cora_h8_54_T40_fullstack_hardware.tsv
output/onnxim_graphbit/fullstack_hardware/cora_h8_54_T40_fullstack_hardware.txt
```

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgDepth | DepthHist | Wscale | SRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| FullP8-miss | 27.8% | 72.2% | 0.721 | 0.721 | 0.721 | 0.77% | 8.00 | D8:100.0% | 1.000 | yes |
| GraphBit-now | 27.8% | 72.1% | 0.715 | 0.718 | 0.716 | 2.13% | 6.10 | D5:30.0%, D6:50.1%, D8:20.0% | 1.000 | yes |
| GraphBit-bucket32 | 27.8% | 72.1% | 0.384 | 0.365 | 0.374 | 2.13% | 6.10 | D5:30.0%, D6:50.1%, D8:20.0% | 0.508 | yes |
| GraphBit-bucket64 | 27.8% | 72.1% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | D5:30.0%, D6:50.1%, D8:20.0% | 0.266 | yes |

解读：

```text
GraphBit-now:
    只启用 bit-plane early stop + demand fetch。
    AvgDepth 从 8.0 降到 6.1，但 Wscale=1.0，weight-side traffic 没摊薄，
    所以总体 cycles 只从 0.721 降到 0.715。

GraphBit-bucket32:
    进一步启用 same-risk bucket scheduler。
    Wscale=0.508，说明同风险 bucket 足够让 W tile 近似服务 2x node blocks。
    cycles 从 0.721 降到 0.384，相对 FullP8-miss 降低约 46.7%。

GraphBit-bucket64:
    Wscale=0.266，cycles 降到 0.289。
    这是更激进的 scheduler capacity point。
```

## 3. PubMed / LLaMA h8_76_T40

输出：

```text
output/onnxim_graphbit/fullstack_hardware/pubmed_h8_76_T40_fullstack_hardware.tsv
output/onnxim_graphbit/fullstack_hardware/pubmed_h8_76_T40_fullstack_hardware.txt
```

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgDepth | DepthHist | Wscale | SRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| FullP8-miss | 8.2% | 91.8% | 0.919 | 0.919 | 0.919 | 0.26% | 8.00 | D8:100.0% | 1.000 | yes |
| GraphBit-now | 8.2% | 91.9% | 0.911 | 0.916 | 0.913 | 1.24% | 6.10 | D5:30.0%, D6:49.9%, D8:20.0% | 1.000 | yes |
| GraphBit-bucket32 | 8.2% | 91.9% | 0.489 | 0.465 | 0.477 | 1.24% | 6.10 | D5:30.0%, D6:49.9%, D8:20.0% | 0.501 | yes |
| GraphBit-bucket64 | 8.2% | 91.9% | 0.368 | 0.240 | 0.304 | 1.24% | 6.10 | D5:30.0%, D6:49.9%, D8:20.0% | 0.251 | yes |

解读：

```text
h8_76_T40 的 reuse 不高，但它是 PubMed/LLaMA 上更干净的 accuracy point。
GraphBit-bucket32 相对 FullP8-miss:
    cycles 0.919 -> 0.489，降低约 46.8%
    traffic 0.919 -> 0.465，降低约 49.4%
    drop 0.26% -> 1.24%
```

## 4. PubMed / LLaMA h8_54_T40 Stress Point

输出：

```text
output/onnxim_graphbit/fullstack_hardware/pubmed_h8_54_T40_fullstack_hardware.tsv
```

这组 reuse 高：

```text
Reuse = 54.1%
Miss  = 45.9%
```

但 FullP8-miss 已经有 `3.01%` drop，说明前端复用过宽。它可以用于说明 hardware datapath 的收益，但不应该作为 PubMed/LLaMA 的主精度点。

## 5. 关键结论

1. `GraphBit-now` 证明 predictor-free bit-plane early stop 真的改变了硬件执行深度：

```text
AvgDepth: 8.0 -> 6.1
DepthHist: D5/D6/D8 = 30% / 50% / 20%
```

2. 仅靠 early stop 不够，因为 weight-side traffic 仍然主导：

```text
Wscale = 1.0
cycles 只小幅下降
```

3. Graph-Bit 真正完整的硬件闭环是：

```text
graph risk -> same-risk buckets
same-risk buckets -> larger weight-stationary tile service window
runtime bound -> activation bit-plane early stop
bucket scheduler + early stop -> cycles / traffic / energy 同时下降
```

4. 当前主线可以这样表述：

```text
Reuse/residual reduces how many nodes enter the LLM encoder.
Graph-Bit reduces arithmetic and memory effort for the remaining miss nodes.
The key hardware enabler is graph-risk bucket scheduling, which makes predictor-free bit-plane early stop visible at full-stack cycles and traffic.
```
