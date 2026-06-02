# Project Roadmap

当前论文目标：

```text
减少 text-attributed graph inference 中 LLM encoder 的执行次数和剩余执行成本。
```

主线不是单独的 degree-guided quantization，而是一个图风险控制的 encoder execution hierarchy。

---

## 1. Paper Logic

### 1.1 SimHash/LRU-CAM

```text
input:
    graph text node

output:
    anchor candidate + support count

role:
    快速判断节点是否可以复用已有 embedding。
```

### 1.2 TSER-Guided Residual Reuse

```text
input:
    fuzzy CAM hit
    TSER / support / distance / cheap feature signals

output:
    corrected embedding or reject

role:
    在 fuzzy hit 上做 lightweight correction 和 accept/reject。
```

TSER 在这里负责判断 fuzzy reuse 的风险：

```text
TSER = 3 * propagation
     + 1 * graph_context
     + 1 * low_unique
```

### 1.3 TSER / Graph-Risk-Guided BFP Encoder

```text
input:
    residual gate reject / CAM miss nodes

output:
    encoder embedding

role:
    默认 BFPA4，只把高风险 miss nodes 提升到 BFPA6/BFPA8。
```

这部分把 TSER 从前端 reuse 决策扩展到后端 encoder precision 决策：

```text
front-end:
    TSER controls reuse safety.

backend:
    TSER / Degree controls BFP refinement.
```

---

## 2. Current Stable Results

### 2.1 Cora/LLaMA BFPA Target Residual Reuse

当前 T31 前端：

```text
8 heads x 16 bits
radius = 2
T = 31
score = 3 / 1 / 1
support >= 5  -> direct reuse
support = 3..4 -> residual candidate
support < 3   -> compute
```

| Target | Residual Reuse | Residual Drop |
|---|---:|---:|
| BFPA6 | 39.1% | 1.24% |
| BFPA4 | 38.5% | 1.74% |

### 2.2 BFP Progressive Refinement

| Dataset | Policy | Selector | Cost | Drop |
|---|---|---|---:|---:|
| Cora | 30% BFPA6 + 70% BFPA4 | TSER | 0.319 | 0.49% |
| PubMed | 30% BFPA6 + 70% BFPA4 | Degree | 0.319 | 0.54% |

Takeaway:

```text
BFPA4 is the low-cost baseline.
BFPA6 is the most cost-effective refinement.
BFPA8 adds little extra accuracy for current Cora/PubMed.
```

---

## 3. Current Experiment Priorities

### A. Complete Missing BFP Pools

Current missing pools:

```text
arxiv:
    W4BFPA7_B128
    W4BFPA6_B128
    W4BFPA5_B128

pubmed:
    W4BFPA7_B128
```

Script:

```bash
bash GraphhopSimhash/scripts/generate_missing_llama_bfp_pools.sh
```

### B. Arxiv BFP Accuracy

After missing pools are generated:

```text
1. All BFPA4 / BFPA6 / BFPA8 accuracy.
2. 20%-30% BFPA6 refinement from BFPA4 baseline.
3. Selector comparison:
       Random
       Degree
       TSER
```

### C. Full-Stack Table

For Cora/PubMed/Arxiv:

```text
Full LLaMA/BFPA8 baseline
SimHash direct reuse
TSER-guided residual reuse
TSER-guided residual reuse + BFP refinement for miss nodes
```

Required columns:

```text
Reuse
Direct / residual / compute ratio
BFPA4 / BFPA6 / BFPA8 ratio
Cost
Acc
Drop
AvgErr
```

---

## 4. Documentation Policy

Primary docs:

```text
docs/core/SCORE_DEFINITIONS.md
docs/core/CAM设计.md
docs/core/RESIDUAL_CORRECTED_REUSE.md
docs/npu/GRAPH_BIT_NPU_DESIGN.md
docs/npu/BFP_ACTIVATION_FORMAT.md
docs/results/GRAPH_BFP_PROGRESSIVE_REFINEMENT_RESULT.md
```

Archived docs:

```text
partial-depth encoder
token compaction
FFN gating
prediction-free early stop
cross-row BFP packing
old trace replay
```

Historical docs remain useful for tracing decisions, but they should not appear as the paper's mainline.
