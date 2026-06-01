# Project Roadmap

当前主线是：

```text
Graph-conditioned hierarchical LLM encoder execution for graph-text workloads.
```

不是所有节点都完整运行 LLM encoder。系统先用 SimHash/LRU-CAM 判断能否复用，再用 Residual-Gate 修复中等置信 fuzzy hit，最后只让 miss / reject 节点进入 Graph-Bit NPU。

## 1. System Path

```text
P0 Direct reuse:
    high-confidence CAM hit
    read cached embedding

P1 Residual-Gate reuse:
    fuzzy CAM hit
    MLP predicts delta and accept/reject

P2 Graph-Bit NPU:
    miss/reject nodes
    graph risk controls scheduling and activation-depth execution

P3 Full W4A8:
    conservative reference / fallback
```

## 2. Current Stable Pieces

### Residual-Gate Front-End

Shared online configuration:

```text
8 heads x 16 bits
radius = 2
score gate = on
score weights = 3 / 1 / 1
score threshold T = 30
support >= 5   -> direct reuse
support = 3..4 -> residual candidate
support < 3    -> encoder / Graph-Bit
gate_accept_threshold = 0.575
```

Current ST result:

| Dataset | Reuse | Drop |
|---|---:|---:|
| Cora/ST | 46.5% | 0.93% |
| PubMed/ST | 42.3% | 1.96% |

### Graph-Bit NPU

Current first-line policy:

```text
node tolerance:
    degree / propagation risk

runtime bound:
    A_low_bound(depth) * W_tile_abs_bound

op sensitivity:
    1
```

Graph risk sets tolerance. The numerical bound decides actual stop depth. Operator-specific sensitivity is kept as a later ablation, not part of the current mainline.

## 3. Documentation Layout

```text
docs/core/
    CAM, score definitions, residual reuse, AWQ pool generation

docs/npu/
    Graph-Bit NPU design, dataflow, early-stop implementation, reproduction guide

docs/results/
    current main results and residual/Graph-Bit progress

docs/survey/
    encoder/general accelerator survey and decoder/serving survey

docs/archive/
    historical sweeps, old proxy experiments, and superseded design notes
```

Root `README.md` stays short. Detailed commands and long tables belong in `docs/`.

## 4. Current Experiment Priorities

### A. Front-End Alignment

Use the shared online residual-gate configuration as the default ST front-end. LLaMA-7B must use LLaMA target embeddings for residual/gate training and must pass the `FullP8-miss` sanity check before adding Graph-Bit.

### B. Graph-Bit Full-Stack Replay

For each selected front-end:

```text
1. export route profile:
       direct / residual / miss

2. export stop-depth trace:
       D8 / D7 / D6 / D5 / D4 for miss nodes

3. replay risk-bucket scheduler:
       baseline order
       b32 / b64 W tile service windows

4. report:
       cycles / traffic / energy / drop / avg depth / Wloads
```

### C. PubMed And Arxiv

```text
PubMed:
    run the same full-stack trace replay as Cora.

Arxiv:
    first run feasibility-only:
        reuse/miss profile
        risk bucket size
        stop-depth histogram
        Wloads/Wscale
        SRAM feasibility
```

## 5. Output Policy

Main results use stable directories:

```text
output/graphbit_trace_replay/
output/residual_reuse/
output/onnxim_graphbit/
```

Temporary sweeps can stay under `output/`; summarized results enter `docs/results/`.

## 6. Main Reporting Table

The final hardware-facing table compares:

```text
FullP8-miss
GraphBit-now
FullP8-bucket-b32 / b64
RiskBucket-b32 / b64
```

Required columns:

```text
Reuse
Miss
Cycles
Traffic
Energy
Drop
AvgDepth
Wloads
Wscale
```
