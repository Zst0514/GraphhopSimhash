# Residual-Gate Reuse and Graph-Bit NPU Progress

本文记录当前两条主线的技术机制和实验进展：

```text
1. SimHash residual-gate:
   用 residual adapter 和 learned accept gate 拯救 fuzzy match。

2. Graph-Bit NPU:
   对必须进入 encoder 的 miss nodes，使用图风险控制可变 activation-depth 执行。
```

---

## 1. SimHash Residual-Gate

### 1.1 问题

SimHash/CAM 前端会产生三类结果：

```text
exact / high-confidence hit:
    锚点和目标节点高度相似，可以直接复用 embedding。

fuzzy / medium-confidence hit:
    CAM 找到了相近锚点，但直接复用有误差风险。

miss / rejected hit:
    复用不可靠，需要重新运行 encoder。
```

仅使用 direct reuse 时，fuzzy hit 是主要矛盾：

```text
放开 fuzzy hit:
    reuse 提高，但 drop 增大。

只保留 exact hit:
    drop 很低，但 reuse 太低。
```

Residual-gate 的目标是在 fuzzy hit 上加入中间路径：

```text
anchor embedding + residual correction + accept/reject gate
```

---

### 1.2 在线执行路径

对目标节点 `v`，CAM/SimHash 找到锚点节点 `u` 后：

```text
support high:
    E_hat(v) = E(u)

support medium:
    residual path

support low:
    compute / Graph-Bit encoder
```

Residual path 的在线计算为：

```text
z_vu = pair_feature(v, u)

delta_vu, accept_score = Adapter(z_vu)

if accept_score >= tau:
    E_hat(v) = E(u) + alpha * delta_vu
else:
    reject -> compute / Graph-Bit encoder
```

其中：

```text
E(u):
    已缓存的 anchor embedding。

delta_vu:
    residual adapter 预测的 embedding 修正量。

alpha:
    residual 强度，可按 support bucket 调整。

tau:
    learned accept gate 的在线阈值。
```

代码实现中，验证阶段的 reject 会用 reference embedding 替代；部署语义上对应重新运行 encoder。

---

### 1.3 Pair Feature

Residual adapter 不从 hash bit 直接还原 embedding。Hash/CAM 只负责找锚点，adapter 使用目标节点和锚点之间的差异特征：

```text
cheap_delta:
    cheap_feature(v) - cheap_feature(u)

context_delta:
    context_signature(v) - context_signature(u)

scalar_stats:
    hamming distance
    support count
    base support count
    table hit count
    cheap/context cosine
    degree ratio
    risk / sensitivity score
```

这里的 cheap feature 通常来自轻量文本模型或浅层编码结果；context signature 来自节点自身特征和邻居均值。

---

### 1.4 Adapter 和 Gate

当前主要使用 MLP residual adapter：

```text
delta_vu = MLP(z_vu)
accept_score = sigmoid(gate_head(z_vu))
```

训练目标包括：

```text
1. correction loss:
   让 E(u) + delta_vu 接近目标 E(v)。

2. gate / accept loss:
   学习哪些 fuzzy candidate 适合继续复用，
   哪些 candidate 应该打回 compute。

3. residual regularization:
   避免修正量过大导致 embedding 空间整体扰动。
```

当前支持两类 accept mode：

```text
separate:
    correction gate 和 accept gate 分开学习。
    适合 soft hit 相对干净、希望尽量保留复用的场景。

shared:
    correction strength 和 accept/reject 共用 learned signal。
    更保守，适合 fuzzy hit 污染更重的场景。
```

---

### 1.5 技术作用

Residual-gate 提供两个能力：

```text
1. 修正 fuzzy hit:
   不改变复用率的前提下降低 embedding error / classification drop。

2. 拒绝不可靠 fuzzy hit:
   在高污染数据集上，把低质量 fuzzy candidate 打回 compute。
```

因此它不是单纯的 residual correction，而是：

```text
CAM provides anchor.
Residual adapter corrects accepted fuzzy hits.
Accept gate filters unsafe fuzzy hits.
```

---

### 1.6 当前实验进展

共享在线 residual-gate 配置：

```text
8 heads x 16 bit
radius = 2
score gate on
score weights = 3 / 1 / 1
support >= 5   -> direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute
```

在当前 residual-gate 实验记录中，Cora / PubMed 3-run 结果如下：

| Dataset | Baseline Acc | ResidualReuse | Acc | Drop | TrainPairs | Alpha |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.7200 | 46.5% | 0.7107 | 0.93% | 464.7 | 0.263 |
| PubMed | 0.7587 | 42.3% | 0.7392 | 1.96% | 151.3 | 0.309 |

对应消融：

| Dataset | Config | Reuse | Drop |
|---|---|---:|---:|
| Cora | DirectReuse | 16.0% | 0.40% |
| Cora | SoftDirectReuse | 46.5% | 1.77% |
| Cora | ResidualReuse | 46.5% | 0.93% |
| PubMed | DirectReuse | 25.7% | 1.04% |
| PubMed | SoftDirectReuse | 69.5% | 4.48% |
| PubMed | ResidualReuse | 42.3% | 1.96% |

解释：

```text
Cora:
    soft hit 相对干净。
    residual 主要负责在相同 reuse 下拉回精度。

PubMed:
    soft hit 更脏。
    accept gate 拒绝一部分 fuzzy hit，
    用更低 reuse 换回 2% 内 drop。
```

需要注意的 target 对齐问题：

```text
ST oracle 纠错后，严格 T31 共享配置尚未完全恢复到
“Cora/PubMed 同时 40%+ reuse 且 2% 内 drop”。

Llama2-7B W4A16 旧日志显示该方向可行，
但需要在当前代码和当前 target pool 下重新复核。
```

因此 residual-gate 的机制已经跑通；最终主结果仍需要按目标 encoder backend 分别复核。

---

## 2. Graph-Bit Variable-Depth NPU Unit

### 2.1 问题

SimHash residual-gate 只能处理可复用节点。对于 reject / miss nodes，仍然需要运行 LLM encoder。

Graph-Bit NPU 处理的是这部分 miss nodes：

```text
reuse / residual 前端:
    减少进入 encoder 的节点数量。

Graph-Bit NPU:
    降低剩余 miss nodes 的 encoder 执行成本。
```

---

### 2.2 基本思想

Graph-Bit 不把节点静态分配成固定比例的 P8/P6/P4。当前主线改为 nodewise predictor-free bound：

```text
graph risk -> min_depth + tolerance
runtime bound -> actual stop depth
```

对每个 miss node：

```text
risk_norm(v) = clamp(risk(v) / risk_max, 0, 1)

tolerance(v) =
    min_tol + (max_tol - min_tol) * (1 - risk_norm(v))^gamma

for depth in min_depth..8:
    if remaining_low_bit_bound(depth) <= tolerance(v):
        stop at depth
        break
```

这意味着：

```text
高风险节点:
    tolerance 小，更接近 P8。

低风险节点:
    tolerance 大，更容易提前停止。

实际 P8/P7/P6/P5/P4 分布:
    由 runtime bound 逐节点产生，
    不是预设比例。
```

---

### 2.3 Predictor-Free Bound

activation 逻辑上是 A8：

```text
A8 = b7 b6 b5 b4 b3 b2 b1 b0
```

bit-serial 执行从高位到低位：

```text
P8: b7..b0
P7: b7..b1
P6: b7..b2
P5: b7..b3
P4: b7..b4
```

执行到某个 depth 后，剩余低位 bit-plane 的最大可能贡献有一个上界：

```text
remaining_low_bit_bound(depth)
```

当前 validation 侧使用归一化低位范围作为近似：

```text
omitted = 2^(8 - depth) - 1
bound   = scale * omitted / (2^8 - 1) * sqrt(tile_k / 128)
```

例子：

```text
depth = 6:
    omitted = 3
    bound = 3 / 255 = 0.01176

depth = 5:
    omitted = 7
    bound = 7 / 255 = 0.02745
```

如果当前节点的 tolerance 大于该 bound，就停止低位执行。

这个过程不使用 learned predictor，也不使用 oracle error。

---

### 2.4 NPU 数据流

Graph-Bit NPU 使用 weight-stationary systolic dataflow。

LLM encoder 中 projection / FFN 的核心计算是：

```text
Y = X @ W
```

其中：

```text
W:
    模型权重，对所有节点和 token rows 共享。

X:
    miss nodes 的 token rows。
```

数据流：

```text
for each layer:
    for each GEMM:
        for each W tile:
            load W tile into SRAM / RF
            stream token rows from risk bucket
            execute bit-serial GEMM
            apply predictor-free early stop
            evict W tile
```

核心收益来自两部分：

```text
1. variable activation depth:
   减少 PE MAC、W RF broadcast、psum update 等片上活动。

2. risk-bucket W-stationary scheduling:
   让同风险 miss nodes 连续消费同一个 W tile，
   减少 W tile reload。
```

---

### 2.5 与 FlashAttention 思想的关系

Graph-Bit 借鉴的是 FlashAttention 的 IO-aware 原则：

```text
keep reusable tile on chip
stream consumers through it
avoid repeated HBM traffic
```

区别在于作用对象不同：

```text
FlashAttention:
    面向 attention tile 和 online softmax。

Graph-Bit:
    面向 encoder Linear / FFN 的 W tile。
    graph risk 决定哪些 miss-node token rows 应该一起消费该 W tile。
```

因此 Graph-Bit 不是直接复用 FlashAttention 算法，而是把它的 tile-stationary 思想迁移到 graph-text encoder workload。

---

### 2.6 当前实现进展

当前代码路径已经支持：

```text
1. W4A8 / W4A7 / W4A6 / W4A5 / W4A4 embedding pool 作为 accuracy validation target。
2. nodewise predictor-free bound assignment。
3. per-node stop-depth trace 导出。
4. residual/reuse 前端与 Graph-Bit miss-node 后端连接。
5. ONNXim component simulation 和 trace-driven scheduler replay。
```

关键入口：

```text
GraphhopSimhash/scripts/run_t31_graphbit_nodewise_bound_sweep.sh
GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```

---

### 2.7 当前快速结果

固定 T31 前端，在 Cora/Llama2-7B W4A8 family 上做 nodewise bound quick sweep：

| Policy | P8 | P7 | P6 | P5 | P4 | AvgDepth | Drop | Cost Save vs FullP8-miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mild | 0.1% | 4.8% | 55.5% | 0.0% | 0.0% | 6.08 | 2.85% | 20.13% |
| normal | 0.0% | 0.3% | 22.3% | 37.7% | 0.0% | 5.38 | 3.53% | 27.72% |
| steep | 0.3% | 4.6% | 42.7% | 12.7% | 0.0% | 5.88 | 3.05% | 22.44% |
| strong | 0.0% | 0.1% | 0.5% | 33.5% | 26.0% | 4.58 | 5.27% | 36.21% |

解释：

```text
mild:
    当前 Cora quick 下最稳，
    AvgDepth 从 FullP8 的 8.00 降到 6.08，
    总 drop 控制在 3% 内。

normal / strong:
    bit-depth 更低，cost saving 更高，
    但 drop 超过当前稳健目标。
```

这组比例是逐节点 bound 运行后得到的分布，不是预设 P8/P6/P5/P4 比例。

---

## 3. 当前进展状态

### 3.1 Residual-Gate

已经完成：

```text
1. CAM fuzzy hit 的 residual correction 路径。
2. learned accept gate。
3. support-aware alpha / adapter。
4. Cora/PubMed shared online residual-gate 实验。
```

仍需继续：

```text
1. 在当前代码下重跑 Llama2-7B W4A16/W4A8 family 的 residual-gate 结果。
2. 明确 ST oracle / Llama oracle 的 target embedding 差异。
3. 将最终前端配置固定为 Graph-Bit full-stack 的默认入口。
```

### 3.2 Graph-Bit NPU

已经完成：

```text
1. nodewise predictor-free bound 逻辑。
2. W4A7 支持，补齐 P8/P7/P6/P5/P4 validation。
3. Cora quick sweep。
4. ONNXim component + trace-driven replay 的初版链路。
```

仍需继续：

```text
1. PubMed / Arxiv 上复核 nodewise bound policy。
2. 用更真实的 M = batch_nodes * seq_len 重跑 NPU component profile。
3. 进一步区分 compute-bound 与 memory-bound GEMM 下 variable depth 的收益。
4. 将 risk-bucket scheduler 的 W tile service window 和 SRAM 约束写成稳定主表。
```

---

## 4. 当前推荐叙述

两条线可以合成一个完整 encoder execution hierarchy：

```text
SimHash/CAM:
    找可复用 anchor。

Residual-gate:
    把 fuzzy match 从高风险 direct reuse 变成可控中间路径。

Graph-Bit NPU:
    对剩余 miss nodes，利用 graph risk 控制 bit-serial arithmetic effort 和 W-stationary scheduling。
```

这形成了从算法到硬件的闭环：

```text
reuse fewer encoder calls
repair fuzzy reuse
schedule remaining encoder work by graph risk
reduce arithmetic effort and W tile reload inside NPU
```
