# Progressive BFP Encoder Interface

本文档定义当前后端 NPU 如何接入 SimHash / LRU-CAM / Residual-Gate 前端。核心目标是：

```text
reuse 命中节点不进入 encoder；
fuzzy hit 先尝试 residual-gate；
剩余 miss / reject 节点进入 progressive BFP encoder。
```

---

## 1. Overall Interface

前端输出的是节点级 route decision：

```text
node_id
anchor_id
support
score_risk
route:
    direct_reuse
    residual_candidate
    encoder_miss
```

后端 encoder 只接收 `encoder_miss` 节点。完整路径如下：

```text
Graph text node
    |
    v
SimHash + LRU/HD-CAM
    |
    +-- support >= 5
    |       -> direct cache reuse
    |
    +-- support = 3..4
    |       -> residual-gate
    |           accept -> anchor embedding + delta
    |           reject -> encoder request
    |
    +-- support < 3
            -> encoder request

encoder request
    |
    v
Progressive BFP NPU
    |
    v
final embedding -> GNN classifier
```

当前主线前端参数：

```text
SimHash:
    8 heads x 16 bits
    radius = 2

Score gate:
    T = 31
    TSER weights = 3 / 1 / 1

Support split:
    support >= 5  -> direct reuse
    support = 3..4 -> residual candidate
    support < 3   -> encoder
```

---

## 2. Encoder Request Format

进入后端的请求可以抽象为：

```text
EncoderRequest {
    node_ids
    token_ids / text payload
    graph_risk_score
    selector_score
    base_format
    refine_format
    refine_mask
}
```

其中：

```text
base_format:
    W4BFPA4_B128

refine_format:
    W4BFPA6_B128

selector_score:
    Degree / TSER / LowUnique / Random
```

后端默认规则：

```text
all encoder nodes:
    execute BFPA4 base

top-risk encoder nodes:
    execute additional mantissa refinement to BFPA6
```

这里的 `top-risk` 只在 encoder miss nodes 内排序，不会影响已经 direct / residual reuse 的节点。

---

## 3. Progressive BFP Execution

传统固定精度路径是：

```text
all encoder nodes -> BFPA6
```

Progressive BFP 路径是：

```text
low-risk miss node:
    BFPA4 base

high-risk miss node:
    BFPA4 base + extra mantissa planes -> BFPA6
```

数值上可以理解为：

```text
Y4 = A_high4 @ W4
Y6 = A_high4 @ W4 + A_extra2 @ W4
```

其中：

```text
W4:
    固定 AWQ W4 weight path。

A_high4:
    BFP activation 的高 4-bit mantissa。

A_extra2:
    BFPA6 相比 BFPA4 多出来的 2 个 mantissa planes。
```

这不是学习一个新的 correction matrix。它是同一组 activation mantissa 的数值精化。

---

## 4. Hardware Interpretation

HBM 侧保持硬件友好的 byte-aligned container：

```text
activation container:
    8-bit aligned storage / transfer

on-chip execution:
    BFPA4 base cycles
    optional extra mantissa-plane cycles
```

因此 BFPA6 是 execution depth，不要求 HBM 以 6-bit 非对齐格式存储。这样可以避免非整字节压缩带来的地址生成、burst 传输和 packing/unpacking 复杂度。

阵列层面采用 W4-capable bit-sliced / bit-serial data path：

```text
base:
    4 mantissa planes x W4

refine:
    extra 1 or 2 mantissa planes x W4
```

当前硬件设计重点不是任意 W/A 混合精度，而是：

```text
W fixed W4
A progressive BFP mantissa refinement
graph risk controls which miss nodes need refinement
```

---

## 5. Software Validation Path

当前代码用 embedding pools 做 accuracy validation：

```text
W4BFPA8_B128:
    reference P8 pool

W4BFPA6_B128:
    refined pool

W4BFPA4_B128:
    base pool
```

`residual_precision_depth` suite 组合三部分：

```text
1. SimHash / CAM / TSER front-end
2. residual-gate reuse
3. progressive BFP encoder routing for miss nodes
```

组合逻辑：

```text
direct reuse:
    final_embedding = E(anchor)

residual accepted:
    final_embedding = E(anchor) + alpha * delta

encoder miss:
    final_embedding = selected BFP pool embedding
```

因此软件评估不是只测 BFP 本身，而是测完整 full-stack：

```text
reuse/residual saves encoder calls
progressive BFP reduces remaining encoder cost
GNN classifier measures final drop
```

当前 Cora 10-run full-stack 结果：

| Config | Reuse | P6 | P4 | Cost | Drop |
|---|---:|---:|---:|---:|---:|
| RefP8 | 39.5% | 0.0% | 0.0% | 0.304 | 1.64% |
| AllP6 | 39.5% | 60.5% | 0.0% | 0.239 | 1.76% |
| AllP4 | 39.5% | 0.0% | 60.5% | 0.175 | 2.46% |
| Rand | 39.5% | 18.1% | 42.3% | 0.194 | 2.12% |
| Deg | 39.5% | 18.1% | 42.3% | 0.194 | 2.22% |
| TSER | 39.5% | 18.1% | 42.3% | 0.194 | 2.24% |

这说明当前接口已经能把前端 reuse / residual gate 和后端 BFPA4/BFPA6 encoder path 接起来。`Deg / TSER / Rand` 在这组 Cora full-stack 结果中差距很小，后端 refinement selector 后续应加入 BFP 数值压力项，而不是只依赖图传播风险。

---

## 6. Reproduction

主线 BFPA6 refinement：

```bash
DATASET=cora RUNS=10 REFINE_BIT=6 REFINE_RATIO=0.30 FORCE=1 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

PubMed：

```bash
DATASET=pubmed RUNS=3 REFINE_BIT=6 REFINE_RATIO=0.30 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

输出目录：

```text
output/progressive_bfp_fullstack/
```

脚本使用的默认前端就是当前 T31 shared retrieval skeleton：

```text
8 heads x 16 bits
T = 31
hard support >= 5
soft support = 3..4
```

---

## 7. Code Entry Points

```text
GraphhopSimhash/progressive_bfp.py
    Progressive BFP routing/cost interface utilities.

GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
    End-to-end SimHash/residual + progressive BFP experiment script.

GraphhopSimhash/runner.py
    residual_precision_depth suite composes front-end reuse and BFP pool selection.
```

The interface utility is intentionally small. The runner remains the source of truth for actual model/GNN evaluation, while `progressive_bfp.py` provides reusable definitions for cost accounting and future NPU scheduler code.
