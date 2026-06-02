# Graph-Conditioned Prediction-Free NPU Design Space

本文档梳理 GraphhopSimhash 场景下可以进入 NPU datapath 的 prediction-free 机制。这里的 prediction-free 指：

```text
不训练额外 learned predictor
不使用 oracle embedding error
只使用在线可得的图信号、CAM/reuse 元数据、静态权重统计和运行时数值上界
```

目标不是把 PADE 的 sparse-attention threshold 直接替换成 degree，而是从图文本 encoder workload 出发，回答：

```text
SimHash / LRU-CAM / residual-gate 之后，
剩余 miss nodes 必须运行 LLM encoder 时，
图任务信息如何改变 NPU 内部的数据流、调度和算术 effort？
```

---

## 1. 可用信号

Graph-Bit 可以使用的信号分成三类。

### 1.1 图任务信号

这些信号来自图结构和当前节点在 GNN 后端中的传播风险，不需要目标 encoder 的 full embedding。

```text
degree / propagation risk:
    节点错误 embedding 会传播到多少邻居。

graph context / boundary risk:
    节点是否处在语义或结构边界。

low-degree unique:
    低度但语义稀有节点是否容易被错误复用破坏自身分类。
```

当前实验更支持 degree / propagation risk 作为主线，因为它稳定、可部署、硬件友好。

### 1.2 前端复用信号

这些信号来自 SimHash / HD-CAM 前端。

```text
route:
    direct reuse / residual reuse / miss

support:
    命中的 hash head 数。

Hamming distance:
    fuzzy match 的距离。

residual accept/reject:
    fuzzy hit 是否被 residual gate 接收。
```

这些信号决定节点是否进入 encoder。Graph-Bit NPU 只处理 miss / reject 节点。

### 1.3 NPU 运行时数值信号

这些信号来自当前 GEMM tile，不是 learned predictor。

```text
remaining low-bit bound:
    停在某个 activation depth 后，剩余低位 bit-plane 的最大可能贡献。

W tile strength:
    当前 W tile 对低位 activation 误差的放大能力。

partial-sum magnitude:
    当前已累加结果的数值尺度。

tile shape / service window:
    当前 W tile 能连续服务多少 token rows。
```

---

## 2. 机制 A：Graph-Guarded Activation Depth

这是目前最直接的 PADE-style 迁移。

PADE 的核心是：

```text
不额外运行 predictor。
在 bit-serial QK 执行过程中，用已算 partial result 和剩余 bit uncertainty 判断是否可以 prune。
```

Graph-Bit 的对应版本是：

```text
不额外训练 predictor。
在 bit-serial GEMM 执行过程中，用图风险 tolerance 和低位剩余贡献上界判断是否可以停止低位 activation effort。
```

### 2.1 基本规则

对 miss node `v`：

```text
risk(v) -> tolerance(v)
```

风险越高，tolerance 越小；风险越低，tolerance 越大。

对当前 GEMM tile：

```text
bound(depth, tile) =
    A_low_bound(depth) * W_tile_strength(tile)
```

停止规则：

```text
for depth in 8, 7, 6, 5, 4:
    if bound(depth, tile) <= tolerance(v):
        stop at depth
        skip lower activation bit effort
        break
```

这个机制不是“全图 P7/P6”，而是 node / tile 级 guarded stop。

### 2.2 为什么必须加入 W tile strength

只看 activation 低位幅度是不够的。真正影响输出的是：

```text
A_low @ W_tile
```

同样的 activation low bits：

```text
弱 W tile:
    低位贡献小，可以更早停止。

强 W tile / outlier tile:
    低位可能被放大，应更保守。
```

因此第一版主线应使用：

```text
node risk + W tile strength + remaining low-bit budget
```

而不是只用：

```text
omitted_low_bits / 255
```

### 2.3 可实现版本

推荐先实现三档：

```text
G0 FullP8:
    所有 miss nodes 完整 P8。

G1 NodeRiskBound:
    tolerance 只由 degree / propagation risk 决定。

G2 NodeRisk+WTileBound:
    tolerance 由 node risk 决定，
    bound 由 A_low_bound * W_tile_strength 决定。
```

验证顺序：

```text
1. truncation embedding pools 验证 stop depth 的下游精度影响。
2. tile-level numeric validation 验证 bound 是否覆盖真实 A_low @ W_tile。
3. ONNXim / trace replay 验证 bit effort 和 W tile service-window 收益。
```

---

## 3. 机制 B：Risk-Bucket W-Stationary Scheduling

这个机制不依赖 early stop，本身也是 prediction-free。

核心思想：

```text
图前端知道哪些节点已经被复用过滤掉，
也知道剩余 miss nodes 的图风险。
因此可以把同风险 miss nodes 聚成 bucket，
让它们连续消费同一个 W tile。
```

### 3.1 为什么这不是普通 Transformer batch 自然具备的能力

如果把 GFM 前端当作普通 LLM encoder 来跑，batch 只是一组文本 sequence / token rows，系统不会利用：

```text
哪些节点已经通过 SimHash/CAM 被 bypass
哪些 miss nodes 对下游 GNN 更敏感
哪些节点应该保守执行
哪些节点可以更激进执行
```

Graph-Bit 把这些图任务信息变成 NPU scheduler 的输入。

### 3.2 数据流

```text
for each layer:
    for each GEMM:
        for each W tile:
            load W tile into on-chip buffer
            stream token rows from a risk bucket
            execute GEMM with bucket-specific tolerance
            keep W tile until service window is reached
            evict W tile
```

收益来源：

```text
1. W tile reload 减少。
2. 同一 bucket 内 stop-depth / tolerance 更一致，减少 batch 内高风险节点拖慢低风险节点。
3. miss-node token rows 更集中，W-stationary dataflow 更容易摊薄 HBM weight traffic。
```

### 3.3 b16 / b32 / b64 的含义

这里的 `b16 / b32 / b64` 不是 bit-width，而是 W tile 的 service window：

```text
b16:
    一个 W tile 服务 16 个 token-row blocks 后换出。

b32:
    一个 W tile 服务 32 个 token-row blocks 后换出。

b64:
    一个 W tile 服务 64 个 token-row blocks 后换出。
```

service window 越大，W tile HBM load 越能被更多 token rows 摊薄；代价是更高的 buffer / scheduler 压力，以及 risk bucket 需要足够大。

---

## 4. 机制 C：Partial-Sum Guard

只用固定 tolerance 仍然偏静态。更动态的 prediction-free 信号是当前 partial sum。

直觉：

```text
如果当前 partial sum 已经很大，
剩余低位对相对输出的影响很小，
可以更早停止。

如果当前 partial sum 很小，
低位可能改变方向或比例，
需要继续执行。
```

可选规则：

```text
relative_bound(depth, tile) =
    remaining_bound(depth, tile) / (abs(partial_sum) + eps)

stop if:
    relative_bound <= node_tolerance(v)
```

这个机制仍然 prediction-free，因为它只使用当前已经计算出的 accumulator，不额外训练 predictor。

实现优先级：

```text
第二阶段。
第一阶段先用 node risk + W tile strength，
等数值验证稳定后再加入 partial-sum guard。
```

---

## 5. 机制 D：Graph-Aware Conservative Tile Routing

不是所有 tile 都适合 early stop。对 outlier W tile，可以直接进入保守路径。

```text
if W_tile_strength > outlier_threshold:
    force P8 or force stricter tolerance
else:
    use normal guarded stop
```

这个机制的作用是保护极端权重 tile，避免少数 outlier tile 破坏整体精度。

它不需要 learned predictor，只需要模型部署前统计 W tile metadata：

```text
mean_abs
row_l1_p90 / p95
max_abs
```

---

## 6. 机制 E：Reuse-Aware Encoder Bypass

这不是 NPU 内部机制，但它决定 NPU workload 的形状。

```text
direct reuse:
    不进入 encoder。

residual reuse:
    不进入 full encoder，只运行 tiny adapter。

miss / reject:
    进入 Graph-Bit NPU。
```

因此 Graph-Bit NPU 的评估必须固定前端 reuse/residual 参数，否则会混淆：

```text
前端少算节点带来的收益
NPU 内部变 bit / W tile reuse 带来的收益
```

---

## 7. 哪些方向不作为第一版主线

### 7.1 Learned Damage Predictor

可以作为 oracle / debug baseline，但不适合作为主线：

```text
需要 calibration nodes
增加部署复杂度
和 prediction-free 目标冲突
```

### 7.2 复杂 Operator Sensitivity 表

例如：

```text
Q/K high sensitivity
FFN-up mid sensitivity
late layer higher sensitivity
```

这类规则可能有效，但第一版不引入。原因是：

```text
1. 参数空间会变大。
2. 容易被质疑是经验调参。
3. 当前 graph risk + W tile bound 已经能形成清晰主线。
```

第一版统一：

```text
op_sensitivity = 1
```

### 7.3 全局统一 P7/P6

全图统一 truncation 只是 stress test，不是 early stop 机制。它不能代表 PADE-style guarded execution。

---

## 8. 推荐主线

当前最合理的实现顺序：

```text
Step 1. 固定前端
    使用当前共享 residual-gate / SimHash / CAM 参数。

Step 2. 实现 G1 / G2
    G1: node risk tolerance only
    G2: node risk tolerance + W tile strength bound

Step 3. 使用 truncation pools 做 accuracy validation
    P8  = W4A8
    P7  = W4A8_TRUNC7
    P6  = W4A8_TRUNC6
    P5  = W4A8_TRUNC5
    P4  = W4A8_TRUNC4

Step 4. 做 tile-level numeric validation
    采样 A8 activation / W tile，
    统计 actual_delta = A_low @ W_tile，
    检查 bound coverage 和 tightness。

Step 5. 做 trace-driven scheduler replay
    比较 FullP8-miss、Graph-Bit no bucket、Graph-Bit risk bucket b32/b64。
```

主表应拆成两部分：

```text
Accuracy table:
    Reuse, P8/P7/P6/P5/P4, AvgDepth, Acc, Drop

Hardware table:
    Wloads, Wscale, cycles, traffic, activity energy
```

---

## 9. 设计结论

Graph-conditioned prediction-free NPU 不应该只等价于 early stop。更完整的设计是：

```text
1. Graph risk controls numerical tolerance.
2. W tile strength guards low-bit stopping.
3. Risk bucket scheduling controls W tile locality.
4. Residual/CAM front-end controls whether encoder is invoked.
```

其中第一版最值得实现和验证的是：

```text
NodeRisk + WTileBound guarded activation depth
Risk-bucket W-stationary scheduling
```

这两项分别对应：

```text
算术 effort 减少
W tile memory traffic 摊薄
```

二者合在一起，才是 Graph-Bit 相对普通 Transformer accelerator 的核心机会。
