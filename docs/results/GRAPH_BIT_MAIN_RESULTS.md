# Main Results

本文档汇总当前论文主线结果。旧 predictor-free early-stop / trace replay 结果已移到 `docs/archive/`。

---

## 1. Front-End: T31 SimHash + TSER + Residual-Gate

当前共享检索前端：

```text
SimHash:
    8 heads x 16 bits
    radius = 2

TSER score gate:
    T = 31
    weights = 3 / 1 / 1

Support split:
    support >= 5  -> direct reuse
    support = 3..4 -> residual candidate
    support < 3   -> compute / BFP encoder
```

Residual candidate 会经过 MLP adapter 和 accept gate：

```text
delta, accept_score = MLP(pair_feature)

if accept:
    E_hat = E_anchor + alpha * delta
else:
    compute / BFP encoder
```

---

## 2. Cora/LLaMA BFPA Target Residual Reuse

这组实验把 residual target 显式换成 BFPA embedding pool，而不是旧 W4A8/W4A16 pool。

| Target | Config | Reuse | Acc | Drop | AvgErr | HitErr |
|---|---|---:|---:|---:|---:|---:|
| BFPA6 | DirectReuse | 15.7% | 0.7086 | 0.71% | 0.03029 | 0.19270 |
| BFPA6 | SoftDirectReuse | 51.6% | 0.6900 | 2.56% | 0.12040 | 0.23313 |
| BFPA6 | ResidualReuse | 39.1% | 0.7033 | 1.24% | 0.07214 | 0.18433 |
| BFPA4 | DirectReuse | 15.7% | 0.7018 | 0.42% | 0.03237 | 0.20641 |
| BFPA4 | SoftDirectReuse | 50.3% | 0.6779 | 2.80% | 0.12077 | 0.23997 |
| BFPA4 | ResidualReuse | 38.5% | 0.6886 | 1.74% | 0.07025 | 0.18253 |

Logs:

```text
output/residual_reuse/bfpa_target_reuse/cora_bfpa6_runs3.log
output/residual_reuse/bfpa_target_reuse/cora_bfpa4_runs3.log
```

Takeaway:

```text
SoftDirectReuse receives too many fuzzy hits.
ResidualReuse rejects dirty fuzzy hits and keeps Cora drop below 2%.
The residual adapter/gate is backend-specific and is retrained for BFPA4/BFPA6 target embeddings.
```

---

## 3. BFP Progressive Refinement

BFP path targets nodes that are not safely handled by direct/residual reuse.

```text
Default:
    BFPA4

Refinement:
    top-risk nodes -> BFPA6 / BFPA8
```

Current cost proxy:

| Format | Cost |
|---|---:|
| BFPA4 | 0.287 |
| BFPA6 | 0.394 |
| BFPA8 | 0.500 |

Recommended result:

| Dataset | Policy | Selector | Cost | Drop |
|---|---|---|---:|---:|
| Cora | 30% BFPA6 + 70% BFPA4 | TSER | 0.319 | 0.49% |
| PubMed | 30% BFPA6 + 70% BFPA4 | Degree | 0.319 | 0.54% |

Historical detailed sweep:

```text
docs/archive/results/GRAPH_BFP_PROGRESSIVE_REFINEMENT_RESULT.md
```

---

## 4. Cross-Module Interpretation

The current full-stack story is:

```text
1. SimHash/LRU-CAM removes high-confidence repeated nodes.
2. TSER prevents high-risk fuzzy reuse.
3. Residual gate repairs accepted fuzzy hits and rejects dirty fuzzy hits.
4. BFPA4/BFPA6 path lowers the cost of remaining miss-node encoder execution.
```

TSER is used twice:

```text
front-end:
    reuse safety

backend:
    BFP refinement priority
```

This is the key difference from treating the graph-text encoder as a plain Transformer batch.
