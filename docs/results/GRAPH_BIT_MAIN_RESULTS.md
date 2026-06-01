# Graph-Bit Main Results

本文档只保留当前主线结果。历史 support-split sweep、旧 proxy、旧 h8_54/T40 表格保留在 `docs/archive/` 或对应专项文档中。

## 1. Shared Online Residual-Gate Front-End

当前共享在线控制流：

```text
8 heads x 16 bits
radius = 2
score gate = on
score weights = 3 / 1 / 1
score threshold T = 30

support >= 5   -> direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute / Graph-Bit
gate_accept_threshold = 0.575
```

ST 3-run result:

| Dataset | Baseline | Reuse | Acc | Drop | TrainPairs | Alpha |
|---|---:|---:|---:|---:|---:|---:|
| Cora/ST | 0.7200 | 46.5% | 0.7107 | 0.93% | 464.7 | 0.263 |
| PubMed/ST | 0.7587 | 42.3% | 0.7392 | 1.96% | 151.3 | 0.309 |

Detailed document:

```text
docs/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md
```

## 2. Cora/LLaMA T31 Full-Stack Trace Replay

Current Cora/LLaMA trace replay uses the T31 shared retrieval front-end:

```text
front-end:
    h8_53_T31
    hard direct: support >= 5
    residual:    support = 3..4
    compute:     support < 3

Graph-Bit:
    degree / propagation risk tolerance
    predictor-free runtime bound
    high/mid/low min depth = 8 / 6 / 4
    high/mid/low tolerance = 0.00 / 0.02 / 0.04
```

Accuracy profile:

| Config | Reuse | Direct | Residual | P8 | P6 | P5 | P4 | Cost | Acc | Drop | FinalErr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 | 52.4% | 17.8% | 34.6% | 47.6% | 0.0% | 0.0% | 0.0% | 0.240 | 0.7012 | 2.77% | 0.10602 |
| DegBound | 52.4% | 17.8% | 34.6% | 9.5% | 23.8% | 14.3% | 0.0% | 0.192 | 0.6908 | 3.80% | 0.10950 |

Seed-42 trace-driven hardware replay, normalized to all graph nodes running FullP8:

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgDepth | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 52.4% | 47.6% | 0.476 | 0.476 | 0.476 | 2.77% | 8.00 | 81 | 1.000 |
| GraphBit-now | 52.4% | 47.6% | 0.471 | 0.474 | 0.473 | 3.80% | 6.10 | 81 | 1.000 |
| FullP8-bucket-b32 | 52.4% | 47.6% | 0.254 | 0.243 | 0.248 | 2.77% | 8.00 | 41 | 0.506 |
| RiskBucket-b32 | 52.4% | 47.6% | 0.253 | 0.241 | 0.247 | 3.80% | 6.10 | 43 | 0.531 |
| FullP8-bucket-b64 | 52.4% | 47.6% | 0.191 | 0.126 | 0.159 | 2.77% | 8.00 | 21 | 0.259 |
| RiskBucket-b64 | 52.4% | 47.6% | 0.191 | 0.124 | 0.158 | 3.80% | 6.10 | 23 | 0.284 |

Activity-level breakdown:

| Compare | ONNX Cycles Save | Activity Cycles Save | Activity Energy Save | PE/Psum Save | Extra Drop |
|---|---:|---:|---:|---:|---:|
| RiskBucket-b32 vs FullP8-bucket-b32 | 0.1% | 11.0% | 15.0% | 23.7% | +1.03% |
| RiskBucket-b64 vs FullP8-bucket-b64 | 0.3% | 13.1% | 16.4% | 23.7% | +1.03% |

Interpretation:

```text
FullP8-miss:
    front-end accepted hits are fixed; all miss nodes run P8.

GraphBit-now:
    adds predictor-free stop depth, without extra W tile service-window reuse.

FullP8-bucket:
    isolates W-stationary bucket scheduling while keeping all miss nodes at P8.

RiskBucket:
    combines runtime stop depth and risk-bucket scheduling.
```

The current table shows two different effects:

```text
W-stationary risk-bucket scheduling:
    dominates normalized cycles / traffic reduction.

Variable activation depth:
    mainly reduces PE / psum / activation-side activity.
```

## 3. Nodewise Bound Policy Validation

The current nodewise bound sweep fixes the T31 reuse/residual front-end and only changes the miss-node Graph-Bit bound policy.

Across Cora and PubMed, the useful policy family is `module_p90_*`:

| Policy | Cora Drop | PubMed Drop | Max Drop | Mean Cost Save vs FullP8-miss | Mean AvgDepth |
|---|---:|---:|---:|---:|---:|
| no_w_node | 2.55% | 3.27% | 3.27% | 28.00% | 5.35 |
| module_p75_w20 | 2.30% | 2.94% | 2.94% | 21.24% | 6.01 |
| module_p90_node | 2.27% | 2.85% | 2.85% | 21.05% | 6.02 |
| module_p90_w50 | 2.40% | 2.85% | 2.85% | 21.08% | 6.02 |

Interpretation:

```text
no_w_node:
    lower AvgDepth and higher cost saving,
    but PubMed receives many P5 stops and exceeds 3% drop.

module_p90_*:
    uses W tile strength in the runtime bound,
    keeps most miss nodes around P6/P7,
    and keeps both Cora and PubMed below 3% drop.
```

Output:

```text
output/graphbit_weighted_bound_validation_cora_runs3/summary.tsv
output/graphbit_weighted_bound_validation_pubmed_runs3/summary.tsv
```

Output:

```text
output/graphbit_trace_replay/cora_h8_53_T31_t31/
```

Reproduction:

```bash
cd /home/zhangshangtong/Transformer/OFA
RUNS=3 DATASET=cora bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

## 4. PubMed And Arxiv Scope

PubMed uses the same full-stack trace replay table as Cora:

```text
FullP8-miss
GraphBit-now
FullP8-bucket-b32 / b64
RiskBucket-b32 / b64
```

Arxiv first uses feasibility-only reporting:

```text
reuse / miss profile
risk bucket size
stop-depth histogram
Wloads / Wscale
SRAM feasibility
```

## 4. Current Boundary

The current main result is not:

```text
degree directly selects low precision
```

The current main result is:

```text
Graph-aware front-end decides whether the encoder runs.
Graph risk sets node-level tolerance for miss nodes.
Predictor-free bound decides stop depth.
Risk-bucket scheduler improves W tile service window.
```
