# Project TODO

本文只保留当前仍然需要补齐的实验。已经验证效果不佳的路径放在 `docs/archive/`。

## 1. Immediate

### 1.1 Finish Missing LLaMA BFP Pools

缺失：

```text
arxiv:
    W4BFPA7_B128
    W4BFPA6_B128
    W4BFPA5_B128

pubmed:
    W4BFPA7_B128
```

运行：

```bash
bash GraphhopSimhash/scripts/generate_missing_llama_bfp_pools.sh
```

### 1.2 Arxiv BFP Progressive Refinement

在 Arxiv/LLaMA 上补：

```text
All BFPA4
All BFPA6
All BFPA8
20% BFPA6 + 80% BFPA4
30% BFPA6 + 70% BFPA4
```

Selector：

```text
Random
Degree
TSER
```

### 1.3 Full-Stack Cora/PubMed Table

固定 T31 front-end：

```text
8 heads x 16 bits
radius = 2
T = 31
score = 3 / 1 / 1
support >= 5  -> direct
support = 3..4 -> residual
support < 3   -> BFP encoder
```

比较：

```text
BFPA8 full encoder
direct reuse only
TSER-guided residual reuse
TSER-guided residual reuse + BFPA4/BFPA6 miss-node refinement
```

## 2. Medium Priority

### 2.1 Residual-Gate BFPA Target For PubMed

Cora/BFPA4/BFPA6 已跑完。PubMed 需要完整 3-run：

```text
pubmed BFPA6 target
pubmed BFPA4 target
```

输出同 Cora：

```text
DirectReuse
SoftDirectReuse
ResidualReuse
```

### 2.2 BFP Cost Model Refinement

当前 BFP cost proxy：

```text
BFPA4 = 0.287
BFPA6 = 0.394
BFPA8 = 0.500
```

下一步可以用 ONNXim / roofline 进一步估计：

```text
mantissa datapath cost
block exponent / shift overhead
BFPA4 vs BFPA6 PE utilization
```

## 3. Historical / Not Mainline

以下方向不再作为主线推进：

```text
partial-depth encoder L4/L8/L16
token compaction / prefix truncation
FFN channel gating as main contribution
oracle error routing
calibration-node learned damage predictor
cross-row BFPA4 packing as full-layer default
prediction-free early stop as primary NPU contribution
```

它们可以在论文中简短说明为设计空间探索，不进入主贡献和主表。
