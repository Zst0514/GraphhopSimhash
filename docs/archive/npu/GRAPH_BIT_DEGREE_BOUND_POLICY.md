# Graph-Bit Degree-Bound Policy

本文档专门说明 Graph-Bit 中 Degree 如何控制 predictor-free runtime bound。核心点是：

```text
Degree 不直接决定最终 P8/P6/P5/P4。
Degree 只决定节点进入 high / mid / low 哪个风险 bucket。
风险 bucket 再决定 min_depth / tolerance。
最终 stop depth 由 runtime remaining-bit bound 决定。
```

相关机制背景见：

```text
docs/archive/npu/GRAPH_BIT_PREDICTOR_FREE_WTILE.md
docs/npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
```

## 1. Degree 在代码里具体是什么

Graph-Bit 的 Degree priority 在代码里对应：

```text
scores["propagation_q"]
```

位置：

```text
GraphhopSimhash/runner.py
    select_precision_depth_actions(...)
```

当前映射：

```python
if priority_name == "degree":
    priority = scores["propagation_q"].to(dtype=torch.float32)
```

因此这里的 Degree 不是直接使用原始 `degree(v)`，而是使用已经归一化/量化后的 propagation risk score。直觉上：

```text
degree / propagation risk 越高：
    节点 embedding 错误越可能通过 GNN message passing 影响更多节点
    -> 应该更保守

degree / propagation risk 越低：
    传播影响更小
    -> 可以允许更激进 early stop
```

## 2. Eligible Nodes

Degree-bound policy 只作用在当前需要分配 bit-depth 的 eligible nodes 上。

不同实验里 eligible nodes 不同：

```text
pure precision-depth ablation:
    eligible nodes = all nodes

residual + Graph-Bit full-stack:
    eligible nodes = miss / compute nodes
    direct reuse 和 residual reuse 命中的节点不再进入 encoder bit-depth 分配
```

这点很重要。Graph-Bit 的目标不是让所有节点都跑低 bit，而是：

```text
reuse/residual:
    先减少进入 encoder 的节点数

Graph-Bit:
    只对剩余 miss nodes 控制 bit-plane arithmetic effort
```

## 3. Degree 如何分 high / mid / low bucket

### 3.1 排序

代码先取 eligible nodes，然后按 priority 从高到低排序：

```python
eligible_idx = torch.nonzero(eligible).flatten()
order = eligible_idx[torch.argsort(priority[eligible_idx], descending=True)]
```

如果 priority 是 degree，则：

```text
高 degree / 高 propagation_q 节点排在前面
低 degree / 低 propagation_q 节点排在后面
```

### 3.2 分桶比例

核心参数：

```text
--precision_depth_high_ratio
--precision_depth_mid_ratio
--precision_depth_low_ratio
```

代码：

```python
high_count = round(high_ratio * eligible_count)
mid_count  = round(mid_ratio  * eligible_count)
low_count  = round(low_ratio  * eligible_count)
```

然后：

```text
order[:high_count]
    -> high bucket

order[high_count : high_count + mid_count]
    -> mid bucket

remaining eligible nodes
    -> low bucket
```

注意：对 `bound_budget` 来说，代码会先把所有 eligible nodes 初始化成 low bucket：

```python
actions[eligible_idx] = bucket_bits["low"]["pool_bit"]
```

然后再覆盖 high 和 mid：

```python
actions[order[:high_count]] = bucket_bits["high"]["pool_bit"]
actions[order[high_count : high_count + mid_count]] = bucket_bits["mid"]["pool_bit"]
```

所以实际 low bucket 通常是：

```text
effective_low_ratio = 1 - high_ratio - mid_ratio
```

`low_ratio` 在 bound policy 下更多是和旧 budget policy 保持参数接口一致，不应把它理解成唯一的 low 节点比例。

## 4. 每个 bucket 如何映射到 min_depth / tolerance

默认配置：

```text
high-risk:
    min_depth = 8
    tolerance = 0.00

mid-risk:
    min_depth = 6
    tolerance = 0.02

low-risk:
    min_depth = 4
    tolerance = 0.04
```

对应参数：

```bash
--precision_depth_bound_high_min_depth 8
--precision_depth_bound_mid_min_depth 6
--precision_depth_bound_low_min_depth 4

--precision_depth_bound_high_tolerance 0.0
--precision_depth_bound_mid_tolerance 0.02
--precision_depth_bound_low_tolerance 0.04
```

含义：

```text
high-risk:
    至少执行到 P8，且 tolerance=0
    -> 基本不允许 early stop

mid-risk:
    至少执行到 P6
    只要剩余低位 bound <= 0.02 就停止

low-risk:
    至少执行到 P4
    bound <= 0.04 即可停止
```

## 5. Runtime Bound 如何决定实际 stop depth

实际函数：

```text
select_runtime_bound_depth(min_depth, tolerance, ref_bit, args)
```

逻辑：

```python
for depth in range(min_depth, ref_bit + 1):
    if remaining_low_bit_bound(depth, ref_bit, args) <= tolerance:
        return depth
return ref_bit
```

其中：

```text
ref_bit = 8
```

bound 函数：

```text
remaining_low_bit_bound(depth)
    = bound_scale
      * omitted_low_bit_value(depth) / full_8bit_value
      * sqrt(tile_k / 128)
```

代码：

```python
omitted = (2 ** (ref_bit - depth)) - 1
denom = (2 ** ref_bit) - 1
tile_scale = sqrt(tile_k / 128)
bound = bound_scale * omitted / denom * tile_scale
```

默认：

```text
bound_scale = 1.0
tile_k = 128
tile_scale = 1.0
```

因此默认 bound 近似为：

| depth | omitted low-bit value | bound |
|---:|---:|---:|
| 4 | 15 | 15/255 = 0.0588 |
| 5 | 7 | 7/255 = 0.0275 |
| 6 | 3 | 3/255 = 0.0118 |
| 7 | 1 | 1/255 = 0.0039 |
| 8 | 0 | 0 |

所以默认 bucket 的实际 stop depth 通常是：

```text
high:
    min_depth = 8, tolerance = 0.00
    depth 8 bound = 0
    -> stop_depth = 8

mid:
    min_depth = 6, tolerance = 0.02
    depth 6 bound = 0.0118 <= 0.02
    -> stop_depth = 6

low:
    min_depth = 4, tolerance = 0.04
    depth 4 bound = 0.0588 > 0.04
    depth 5 bound = 0.0275 <= 0.04
    -> stop_depth = 5
```

这就是为什么当前常见结果里：

```text
low-risk 经常不是 P4，而是 P5。
```

## 6. stop depth 如何映射到 embedding pools

accuracy validation 不能真的逐 bit-plane 重跑 LLaMA，所以使用已生成 embedding pools 近似：

```text
available pools = {P8, P6, P5, P4}
P8 = W4A8
P6 = W4A6
P5 = W4A5
P4 = W4A4
```

代码会把 runtime depth 映射到不低于该 depth 的最近可用 pool：

```text
nearest_available_precision_depth(requested_depth, bits, ref_bit)
```

例子：

```text
runtime_depth = 5
available = {4, 5, 6, 8}
-> pool_bit = 5

runtime_depth = 7
available = {4, 5, 6, 8}
-> pool_bit = 8
```

因此：

```text
runtime bound 决定 stop_depth
embedding pool 决定 accuracy proxy 使用哪个 cached embedding
```

两者语义接近，但不是严格等价。硬件里真实执行的是 A8 activation 的高位 bit-plane early stop；accuracy validation 里用 W4A{8,6,5,4} pools 近似不同 stop depth。

## 7. 默认 Degree-Bound 示例

假设 eligible miss nodes 有 1000 个，默认采用：

```text
high_ratio = 0.20
mid_ratio = 0.50
```

则：

```text
top 200 degree nodes:
    high bucket
    min_depth = 8
    tolerance = 0.00
    stop_depth = 8

next 500 degree nodes:
    mid bucket
    min_depth = 6
    tolerance = 0.02
    stop_depth = 6

remaining 300 degree nodes:
    low bucket
    min_depth = 4
    tolerance = 0.04
    stop_depth = 5 under default bound
```

最终 stop-depth histogram 可能类似：

```text
D8: 20%
D6: 50%
D5: 30%
```

这正是当前 Cora trace 里常见的 `AvgDepth ~= 6.10` 的来源。

## 8. 和静态 Degree Precision 的区别

### 8.1 静态 degree-guided precision

静态策略是：

```text
degree high -> P8
degree mid  -> P6
degree low  -> P4
```

最终 bit-depth 由 degree 直接决定。

### 8.2 Graph-Bit Degree-Bound

Graph-Bit 是：

```text
degree high -> min_depth=8, tolerance=0.00
degree mid  -> min_depth=6, tolerance=0.02
degree low  -> min_depth=4, tolerance=0.04

runtime bound -> actual stop depth
```

最终 bit-depth 不由 degree 直接指定，而由 runtime bound 决定。

一句话：

```text
Degree 控制 early-stop 激进程度；
Bound 控制实际停止位置。
```

这也是和 HEAT-like static degree precision 拉开边界的关键。

## 9. 可调参数

### 9.1 调风险比例

更保守：

```bash
--precision_depth_high_ratio 0.60
--precision_depth_mid_ratio 0.30
```

更激进：

```bash
--precision_depth_high_ratio 0.10
--precision_depth_mid_ratio 0.50
```

解释：

```text
high_ratio 越大:
    更多节点至少 P8
    精度更稳
    cycles/traffic 节省更小

high_ratio 越小:
    更多节点进入 mid/low
    AvgDepth 更低
    精度风险更高
```

### 9.2 调 min_depth

更保守：

```bash
--precision_depth_bound_mid_min_depth 7
--precision_depth_bound_low_min_depth 5
```

更激进：

```bash
--precision_depth_bound_mid_min_depth 5
--precision_depth_bound_low_min_depth 4
```

### 9.3 调 tolerance

更保守：

```bash
--precision_depth_bound_mid_tolerance 0.01
--precision_depth_bound_low_tolerance 0.02
```

更激进：

```bash
--precision_depth_bound_mid_tolerance 0.03
--precision_depth_bound_low_tolerance 0.06
```

解释：

```text
tolerance 越小:
    bound 更难满足
    stop depth 更深

tolerance 越大:
    更容易 early stop
    stop depth 更浅
```

### 9.4 调 tile_k / scale

```bash
--precision_depth_bound_tile_k 128
--precision_depth_bound_scale 1.0
```

如果 `tile_k` 更大：

```text
tile_scale = sqrt(tile_k / 128)
bound 变大
early stop 更保守
```

如果 `bound_scale` 更大：

```text
bound 整体变大
early stop 更保守
```

## 10. 推荐 sweep

建议优先扫三组：

```text
conservative:
    high_ratio = 0.60
    mid_ratio  = 0.30
    high/mid/low = (8,0.00), (6,0.015), (4,0.03)

balanced:
    high_ratio = 0.20
    mid_ratio  = 0.50
    high/mid/low = (8,0.00), (6,0.02), (4,0.04)

aggressive:
    high_ratio = 0.10
    mid_ratio  = 0.50
    high/mid/low = (8,0.00), (5,0.03), (4,0.06)
```

每组需要同时报告：

```text
Drop
AvgDepth
DepthHist
Cycles
Traffic
Energy
Wloads / Wscale
```

只看 Drop 或只看 AvgDepth 都不够。

## 11. 论文中建议表述

推荐写法：

```text
Graph-Bit does not use degree to statically assign precision.
Instead, degree-derived propagation risk configures a per-node minimum bit depth
and a remaining-bit tolerance. A predictor-free runtime bound then determines
the actual stop depth.
```

中文解释：

```text
Degree 不是最终精度选择器，而是 early-stop 安全策略控制器。
最终实际算到几 bit，由 bit-plane runtime bound 决定。
```

不建议写成：

```text
degree high -> P8
degree mid -> P6
degree low -> P4
```

这种写法会退化成静态 degree-guided precision，和 Graph-Bit 的 runtime-bound 机制不一致。
