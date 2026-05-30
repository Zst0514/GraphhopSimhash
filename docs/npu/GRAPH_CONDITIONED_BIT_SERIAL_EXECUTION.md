# Graph-Conditioned Predictor-Free Bit-Serial Execution

本文档定义 Graph-Bit 的下一步硬件化版本：不是只把节点路由到 `W4A8/W4A6/W4A4` embedding pool，而是在 NPU 内部实现类似 PADE / BETA 的 bit-serial early termination，并让 graph risk 控制每个 node batch 的 arithmetic effort。

核心目标：

```text
当节点必须执行 LLM encoder 时，
NPU 不对所有节点统一跑满 A8 activation bit-plane；
而是根据图任务风险决定 bit-serial GEMM 算到多深。
```

一句话版本：

```text
PADE-style predictor-free bound controls whether low-bit computation is still needed;
graph risk controls how strict this bound must be.
```

## 0. Current Runtime-Bound Implementation

当前代码已经把 Graph-Bit 从“静态 P8/P6/P4 路由”推进到 runtime-bound 路由：

```text
Degree / TSER 不直接决定最终执行 P8/P6/P4。

它们只决定节点属于 high / mid / low 哪个风险桶；
每个风险桶只给出：
    1. min_depth: 最低安全执行深度
    2. tolerance: 剩余低位 bit-plane 的可容忍上界

最终执行到 P8/P6/P5/P4 中哪一个，
由 predictor-free remaining-bit bound 在运行时决定。
```

当前 runner 入口是：

```text
--precision_depth_bound_enable
--precision_depth_bound_priorities degree tser
--precision_depth_bound_high_min_depth 8
--precision_depth_bound_mid_min_depth 6
--precision_depth_bound_low_min_depth 4
--precision_depth_bound_high_tolerance 0.0
--precision_depth_bound_mid_tolerance 0.02
--precision_depth_bound_low_tolerance 0.04
--precision_depth_bound_tile_k 128
--precision_depth_bound_scale 1.0
```

示例 bound 映射：

```text
high: min=8, tau=0.00 -> runtime P8
mid:  min=6, tau=0.02 -> runtime P6
low:  min=4, tau=0.04 -> runtime P5
```

注意 low bucket 不是被手工指定为 P5。它先给 `min_depth=4`，然后 bound 判断 P4 的剩余误差界仍偏大，于是继续执行到 P5。

### Cora/LLaMA Quick Validation

固定前端：

```text
h8_54_T40
8 heads x 16-bit
R = 2
hard direct: support >= 5
residual: support == 4
miss nodes -> Graph-Bit
```

命令：

```bash
RUNS=1 RUN_ALGO=1 RUN_ONNXIM=0 \
DATASET=cora THRESHOLD=40 HARD_SUPPORT=5 SOFT_SUPPORT=4 \
FRONTEND_ID=h8_54_T40 BUDGET=boundclean \
HIGH_RATIO=0.20 MID_RATIO=0.50 LOW_RATIO=0.0 \
OUT_DIR=output/graphbit_bound_runtime/cora_h8_54_T40_boundclean_quick \
BOUND_ENABLE=1 BOUND_PRIORITIES='degree tser' \
BOUND_MID_TOL=0.02 BOUND_LOW_TOL=0.04 \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```

关键结果：

| Method | Reuse | P8 | P6 | P5 | P4 | Cost | Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullP8-miss | 27.8% | 72.2% | 0.0% | 0.0% | 0.0% | 0.361 | 0.77% |
| Degree static | 27.8% | 14.4% | 36.1% | 0.0% | 21.6% | 0.277 | 2.71% |
| Degree runtime-bound | 27.8% | 14.4% | 36.1% | 21.6% | 0.0% | 0.288 | 2.13% |
| TSER runtime-bound | 27.8% | 14.4% | 36.1% | 21.6% | 0.0% | 0.288 | 2.85% |

这个表的意义：

```text
Static Degree:
    低风险 miss 节点被硬压到 P4，drop=2.71%。

Runtime-bound Degree:
    Degree 只给 low bucket 的 min_depth=4/tolerance=0.04；
    bound 发现 P4 不够安全，最终执行到 P5；
    drop 降到 2.13%。
```

因此当前实现已经体现了核心机制：graph risk 配置安全下限和容忍度，实际 bit-depth 由 runtime bound 决定。

## 1. Motivation

当前系统已经有三层：

```text
P0: exact hash reuse
P1: fuzzy hash reuse + residual correction
P2/P3: miss nodes 执行 LLM encoder
```

`P0/P1` 解决的是“能不能不跑 encoder”。Graph-Bit 解决的是：

```text
如果节点必须跑 encoder，NPU 内部是否还需要完整执行 A8？
```

这比普通 degree-guided quantization 更深入。Degree / propagation risk 不只是决定选哪个 embedding pool，而是直接进入 NPU datapath：

```text
low-risk node:
    允许更早停止低位 bit-plane

high-risk node:
    保留更多 bit-plane，甚至完整 P8
```

## 2. Relation to PADE and FACT

### 2.1 PADE 的启发

PADE 的核心是 predictor-free sparse attention：

```text
不额外训练/运行 predictor；
在 bit-serial 执行过程中维护 partial sum；
用 bit-level upper/lower bound 判断后续低位是否还能改变决策；
如果不能改变，就提前停止。
```

Graph-Bit 借鉴这个思想，但迁移目标不同：

```text
PADE:
    bound 用来判断 attention pair 是否需要继续计算。

Graph-Bit:
    bound 用来判断 node-batch 的 GEMM 低位 bit-plane 是否值得继续算。
```

### 2.2 FACT 的启发

FACT 的核心是 eager prediction：

```text
在 Transformer layer 早期预测 correlation，
提前指导 QKV / attention / FFN 计算省略或降精度。
```

Graph-Bit 不直接预测 attention top-k，而是把图后端信息作为 eager control signal：

```text
Graph risk is available before encoder execution.
It configures precision-depth before GEMM starts.
```

因此它比纯模型内部 predictor 更适合 graph-text workload：图结构天然给出节点误差传播风险。

## 3. Core Design

### 3.1 Bit-serial W4A8 GEMM

以 LLaMA encoder 的线性层为例：

```text
Y = A8 x W4
```

其中 weight 固定为 W4，activation 逻辑上是 A8。NPU 将 activation 拆成 bit-plane：

```text
A8 = b7 * 2^7 + b6 * 2^6 + ... + b0 * 2^0
```

bit-serial PE 从高位到低位执行：

```text
partial_sum = 0

for bit in [7, 6, 5, 4, 3, 2, 1, 0]:
    partial_sum += GEMM(A_bit, W4) * 2^bit
    remaining_bound = estimate_remaining_bound(bit - 1)
    if can_stop(remaining_bound, graph_tolerance):
        break
```

这里的 `can_stop` 是 predictor-free 的：它不需要额外预测器，只比较当前剩余 bit-plane 的理论误差上界和 graph tolerance。

### 3.2 Precision-depth levels

当前离线实验里的 `P8/P6/P4` 对应硬件里的执行深度：

```text
P8:
    execute b7..b0

P6:
    execute b7..b2

P4:
    execute b7..b4
```

注意：`P6/P4` 不一定是独立硬件 datatype。它可以只是 bit-serial PE 少执行若干低位 plane。

推荐论文表述：

```text
P8/P6/P4 are precision-depth levels, not necessarily fixed ISA datatypes.
```

在 predictor-free early-stop 版本里，`P6/P4` 进一步退化为安全下限，而不是固定终点：

```text
high-risk:
    start from max_depth=8, min_depth=8

mid-risk:
    start from max_depth=8, min_depth=6
    if bound is still large, continue below/above the nominal boundary as needed

low-risk:
    start from max_depth=8, min_depth=4
    stop once the bit-level bound is small enough
```

因此真正的硬件机制是动态执行深度：

```text
always start from high bit-plane;
execute at least min_depth;
then stop by predictor-free bound.
```

`P6/P4` 在论文中应表述为 validation anchors / minimum-depth floors，而不是必须永久固定成 6-bit 或 4-bit datatype。

## 4. Risk Definition

第一版主线建议从 Degree / propagation risk 开始，因为它最稳、最 deployable、无需 calibration。

### 4.1 Degree risk

原始 degree 不直接用，先做 log 压缩，再做分位桶：

```text
degree_raw(v) = in_degree(v) + out_degree(v)
degree_score(v) = log(1 + degree_raw(v))
deg_q(v) = quantile_bucket(degree_score(v), bins=16)  # 0..15
```

直觉：

```text
high degree:
    embedding 误差通过 GNN message passing 影响更多邻居；
    需要更高 precision-depth。

low degree:
    误差主要局部影响；
    可以更早停止。
```

### 4.2 Propagation risk

如果已有 normalized propagation score，可以直接替代 degree：

```text
propagation_q(v) in [0, 15]
```

这比 raw degree 更贴近“误差传播范围”，建议作为主线 deployable risk。

### 4.3 TSER risk

TSER 是复合风险：

```text
TSER(v) =
    w_p * propagation_q(v)
  + w_c * graph_context_q(v)
  + w_u * low_degree_unique_q(v)
```

它可以用于消融，但目前实验更支持：

```text
Degree / propagation risk 是更稳定的 deployable Graph-Bit 路由依据；
TSER 更适合作为图语义修正对照。
```

### 4.4 Context and low-unique risk

这两项可以继续作为 ablation：

```text
graph_context_q:
    边界/上下文异质性风险

low_degree_unique_q:
    低度但语义稀有节点风险
```

但硬件主线不建议一开始就依赖太复杂的组合项。先证明 degree/propagation 能稳定工作。

## 5. Threshold Management

### 5.1 Absolute threshold mode

最直观的配置是固定阈值：

```text
if deg_q >= 12:
    min_depth = 8
    tolerance = small
elif deg_q >= 6:
    min_depth = 6
    tolerance = medium
else:
    min_depth = 4
    tolerance = large
```

参数可以写成：

```text
risk_bins = 16
high_threshold = 12
mid_threshold = 6
depths = [8, 6, 4]
tolerances = [tau_high, tau_mid, tau_low]
```

含义：

```text
deg_q >= high_threshold:
    高风险，P8 或接近 P8

mid_threshold <= deg_q < high_threshold:
    中风险，P6

deg_q < mid_threshold:
    低风险，P4
```

### 5.2 Budget mode

为了公平比较不同策略，也可以用 budget mode：

```text
high_ratio = 0.20
mid_ratio = 0.50
low_ratio = 0.30
```

即在 miss nodes 里：

```text
top 20% risk -> P8
next 50% risk -> P6
last 30% risk -> P4
```

当前 Cora full-stack 表就是这个模式：

```text
P8: 20% miss nodes
P6: 50% miss nodes
P4: 30% miss nodes
```

优点：

```text
不同 risk proxy 在同样 cost 下比较；
更适合画 cost-drop curve。
```

缺点：

```text
实际硬件部署时需要维护排序/分位统计。
```

因此建议：

```text
实验主表:
    budget mode，保证公平。

硬件落地:
    absolute threshold mode，规则更简单。
```

### 5.3 Tolerance table

`min_depth` 决定至少算到几位；`tolerance` 决定是否还要继续算低位。

示例：

| Risk bucket | Condition | min_depth | tolerance |
|---|---|---:|---|
| high | `deg_q >= 12` | 8 | `tau_high = 0` or very small |
| mid | `6 <= deg_q < 12` | 6 | `tau_mid` |
| low | `deg_q < 6` | 4 | `tau_low` |

解释：

```text
min_depth:
    硬下限，保证低风险节点也至少算若干高位。

tolerance:
    软条件，如果剩余 bound 已经足够小，可以在 min_depth 后继续早停。
```

第一版可以先不启用连续 bound，只用固定 `min_depth`，也就是当前 `P8/P6/P4` proxy。

第二版再启用 predictor-free bound：

```text
P6 node:
    通常算到 6 bit；
    如果 bound 仍大于 tau_mid，可以继续算到 P7/P8。

P4 node:
    通常算到 4 bit；
    如果 bound 仍大于 tau_low，可以继续算到 P5/P6。
```

这会比固定 P4/P6 更安全。

## 6. Predictor-Free Bound

### 6.1 Bound estimator

对于某个 output tile：

```text
partial_sum_t = 已处理 bit-plane 的累加结果
remaining_bits = 未处理低位 bit-plane
```

剩余误差上界可以估计为：

```text
remaining_bound <= scale_a * scale_w * abs_weight_sum * remaining_activation_max
```

其中：

```text
abs_weight_sum:
    当前 W4 tile 的 |W| 行/列和，可离线预计算或 tile load 时顺带计算。

remaining_activation_max:
    未处理低位 bit-plane 的最大可能值。
```

对 A8，从 bit `k` 停止时，剩余低位最大值为：

```text
remaining_activation_max(k) = 2^k - 1
```

例如已经算完 `b7..b4`，剩下 `b3..b0`：

```text
remaining_activation_max = 2^4 - 1 = 15
```

### 6.2 Normalized bound

为了让不同层/不同 tile 可比，建议使用归一化 bound：

```text
norm_bound = remaining_bound / (abs(partial_sum) + eps)
```

或者：

```text
norm_bound = remaining_bound / output_scale
```

第一版硬件 proxy 可以用更简单的 bit-depth cost model；后续 ONNXim 内部实现再细化。

### 6.3 Stop rule

```text
if executed_depth >= min_depth(v) and norm_bound <= tolerance(v):
    stop lower bit-planes for this node/tile
```

高风险节点：

```text
min_depth = 8
tolerance = 0
```

低风险节点：

```text
min_depth = 4
tolerance = larger
```

## 7. NPU Microarchitecture

### 7.1 New hardware state

```text
Graph Risk Buffer:
    node_id -> risk_q / min_depth / tolerance

Risk-aware Batch Scheduler:
    groups similar-risk nodes into the same micro-batch

Bit-Plane Sequencer:
    controls activation bit-plane order

Partial-Sum Scoreboard:
    stores tile partial sums across bit rounds

Bound Estimator:
    computes remaining error upper bound

Early-Stop Mask:
    disables lanes / tiles whose bound satisfies tolerance

Compaction / OOO Scheduler:
    repacks active lanes to avoid PE under-utilization
```

### 7.2 Dataflow

```text
1. Scheduler receives miss nodes after CAM/residual gate.
2. Read graph risk for each node.
3. Assign min_depth and tolerance.
4. Group nodes by risk bucket.
5. Execute W4A8 GEMM in bit-serial order.
6. After each bit-plane:
       update partial sum
       estimate remaining bound
       update early-stop mask
7. Finished nodes emit final encoder embedding.
```

### 7.3 Batch divergence

Per-node stopping can hurt utilization if lanes stop at different times. Therefore:

```text
First implementation:
    per-micro-batch depth mode
    batch nodes with similar risk

Second implementation:
    per-lane early-stop mask
    compact active lanes after each bit round
```

This matches PADE's motivation: bit-level execution creates irregular workloads, so hardware needs OOO / compaction to preserve utilization.

## 8. Validation Plan

### 8.1 Stage A: offline embedding-pool proxy

这是当前已经在做的路径：

```text
P8 -> W4A8 embedding pool
P6 -> W4A6 embedding pool
P4 -> W4A4 embedding pool
```

固定 residual front-end：

```text
R = 2
heads = 8 x 16-bit
T = 40
hard >= 5
soft = 4
```

然后只调 Graph-Bit 的 miss-node 策略：

```text
Risk source:
    Random / Degree / TSER / Context / LowUnique

Budget:
    P8/P6/P4 = 20/50/30
    P8/P6/P4 = 30/50/20
    P8/P6/P4 = 20/60/20
```

这一步回答：

```text
graph risk 是否比 random 更适合决定 precision depth？
```

### 8.2 Stage B: threshold sweep

基于 degree / propagation 做 absolute threshold：

```text
deg_q bins = 16

Sweep:
    high_threshold in [10, 11, 12, 13]
    mid_threshold  in [4, 5, 6, 7, 8]
```

输出：

```text
P8/P6/P4 ratio
normalized cost
accuracy drop
FinalErr
```

目标：

```text
找到不依赖 dataset-specific tuning 的固定 threshold。
```

### 8.3 Stage C: ONNXim internal bit-serial proxy

在 ONNXim 的 `GemmWS` / systolic datapath 内部加入：

```text
graphbit_enable
graphbit_precision_depth
graphbit_bound_enable
graphbit_bound_tolerance
graphbit_min_depth
```

先做固定 depth：

```text
P8: execute 8 bit-planes
P6: execute 6 bit-planes
P4: execute 4 bit-planes
```

再做 bounded early termination：

```text
execute until min_depth;
continue only if remaining_bound > tolerance.
```

输出硬件指标：

```text
cycles
read requests
write requests
average executed bit-depth
saved bit-planes
PE utilization proxy
energy proxy
```

### 8.3.1 Current Cora early-stop validation

固定 full-stack 前端：

```text
Dataset: Cora
Reuse front-end: h8_54_T40
    8 heads x 16-bit
    R = 2
    T = 40
    support >= 5 -> direct reuse
    support == 4 -> residual correction
    support < 4  -> Graph-Bit / full encoder

Graph-Bit risk: Degree
Miss-node budget proxy: P8/P6/P4 = 20/50/30
ONNXim sequence length: 64
```

命令：

```bash
FORCE_ONNXIM=1 bash GraphhopSimhash/scripts/run_cora_graphbit_earlystop_sweep.sh
```

结果文件：

```text
output/graphbit_predictor_free/cora_h8_54_T40/earlystop_sweep/earlystop_sweep.txt
```

当前结果：

| Method | Reuse | AvgD | Saved | Stop | Cycles | Traffic | Energy | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 40.0% | 8.00 | 0.00 | 0.0% | 0.601 | 0.602 | 0.602 | 1.53% |
| Static Degree P8/P6/P4 | 40.0% | 5.80 | 2.20 | 0.0% | 0.575 | 0.581 | 0.578 | 2.39% |
| EarlyStop conservative | 40.0% | 6.90 | 1.10 | 100.0% | 0.588 | 0.591 | 0.589 | 2.39% |
| EarlyStop balanced | 40.0% | 6.10 | 1.90 | 100.0% | 0.576 | 0.583 | 0.580 | 2.39% |
| EarlyStop aggressive | 40.0% | 5.80 | 2.20 | 100.0% | 0.575 | 0.581 | 0.578 | 2.39% |

解释：

```text
FullP8-miss:
    miss nodes 全部完整执行 A8 bit-plane。

Static Degree P8/P6/P4:
    旧的固定深度 proxy，用来估计精度 drop。

EarlyStop:
    所有 miss nodes 都从 max_depth=8 开始；
    degree risk 只控制 min_depth 和 tolerance；
    ONNXim 在 GEMM 内部按 bit-bound 早停。
```

这个结果说明：

```text
1. P6/P4 不再必须作为固定最终位宽。
2. Balanced early-stop 的 AvgD=6.10，已经接近静态 Degree proxy 的 AvgD=5.80。
3. Cycles/Traffic/Energy 也接近静态 proxy，说明 predictor-free early-stop 在 NPU datapath 中有真实节省潜力。
4. Drop 当前仍沿用静态 Degree proxy 的 embedding 结果；它用于精度保守估计。若要精确评估动态 AvgD=6.10，需要后续生成动态深度 embedding 或用 nearest-depth conservative mapping。
```

### 8.3.1 Miss-only breakdown

上面的 full-stack 表会被 reuse/residual 比例、weight read、output write 等因素稀释。为了确认 Graph-Bit 是否真的节省了 NPU 内部 bit-plane 计算，ONNXim 现在额外输出 miss-only 分解统计：

```text
output/graphbit_predictor_free/cora_h8_54_T40/earlystop_sweep/miss_only_breakdown.txt
```

该表只看必须执行 encoder 的 miss nodes，不再把 direct reuse / residual reuse 混进分母。

| Method | AvgD | Saved | Cycles | MatMul | BitComp | ActRd | ActSave | WeightRd | OutWr | Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 8.00 | 0.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0% | 1.000 | 1.000 | 1.000 |
| Static Degree P8/P6/P4 | 5.80 | 2.20 | 0.957 | 1.000 | 0.726 | 0.725 | 27.5% | 1.000 | 1.000 | 0.964 |
| EarlyStop conservative | 6.90 | 1.10 | 0.978 | 1.000 | 0.864 | 0.862 | 13.8% | 1.000 | 1.000 | 0.982 |
| EarlyStop balanced | 6.10 | 1.90 | 0.959 | 1.000 | 0.764 | 0.763 | 23.7% | 1.000 | 1.000 | 0.969 |
| EarlyStop aggressive | 5.80 | 2.20 | 0.957 | 1.000 | 0.726 | 0.725 | 27.5% | 1.000 | 1.000 | 0.964 |

字段解释：

```text
AvgD:
    miss-node GEMM 平均执行的 activation bit-depth。

BitComp:
    Graph-Bit effective bit-serial compute cycles / raw full-depth compute cycles。
    这是判断 bit-plane 算术是否真的下降的主指标。

ActRd:
    activation/input read requests，相对 FullP8-miss 归一化。

ActSave:
    相比 full-depth activation input requests 的节省比例。

WeightRd / OutWr:
    weight read 和 output write。当前 Graph-Bit 不压缩这两项，所以保持 1.000。

Traffic:
    总 DRAM read+write requests，相对 FullP8-miss 归一化。
```

关键结论：

```text
1. Balanced early-stop 把 AvgD 从 8.00 降到 6.10。
2. BitComp 从 1.000 降到 0.764，说明 miss-only bit-serial 算术量约降 23.6%。
3. ActRd 从 1.000 降到 0.763，activation bit-plane 读取约降 23.7%。
4. Traffic 只降到 0.969，是因为 weight read 和 output write 不随 activation bit-depth 下降。
5. 因此机制本身是有效的；full-stack 总收益较小主要来自 weight/output/static scheduling 的稀释。
```

### 8.3.2 Risk-bucket batching

Bit-serial early-stop 还需要 scheduler 配合。若 high/mid/low risk miss nodes 随机混在一个 micro-batch，硬件通常必须跑到该 batch 的最大 bit-depth。低风险节点的省算会被同 batch 的高风险节点抵消。

新增脚本：

```bash
bash GraphhopSimhash/scripts/run_cora_graphbit_risk_bucket_batching.sh
```

输出：

```text
output/graphbit_predictor_free/cora_h8_54_T40/risk_bucket_batching/risk_bucket_batching.txt
```

在 Cora `h8_54_T40` 下，miss nodes 内部风险比例为：

```text
high-risk P8 floor: 20%
mid-risk  P6 floor: 50%
low-risk  P4 floor: 30%
```

micro-batch size 64 的核心结果：

| Method | Schedule | UsefulD | ExecD | Util | Cycles | BitComp | ActRd | Traffic | Drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomOrder Static | random mixed | 5.80 | 8.00 | 72.5% | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| DegreeBucket Static | risk bucket | 5.80 | 5.80 | 100.0% | 0.957 | 0.726 | 0.725 | 0.964 | 2.39% |
| RandomRisk Bucket | risk bucket | 5.80 | 5.80 | 100.0% | 0.957 | 0.726 | 0.725 | 0.964 | 2.79% |
| RandomOrder EarlyStop | random mixed | 6.10 | 8.00 | 76.3% | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| DegreeBucket EarlyStop | risk bucket | 6.10 | 6.10 | 100.0% | 0.959 | 0.764 | 0.763 | 0.969 | 2.39% |

结论：

```text
1. Graph-Bit 不能只做 per-node depth decision，还必须做 risk-bucket batching。
2. Random mixed batch 在 batch=64 时几乎退化回 P8，bit-plane saving 消失。
3. DegreeBucket 让 executed depth 等于 useful depth，才真正把 bit-plane saving 映射到 NPU datapath。
4. RandomRisk Bucket 硬件成本一样，但 drop 更高，说明 degree proxy 仍然有必要。
```

三组 early-stop 参数：

| Name | Mid-risk | Low-risk |
|---|---|---|
| conservative | `min_depth=6, tau=0.006` | `min_depth=4, tau=0.02` |
| balanced | `min_depth=6, tau=0.02` | `min_depth=4, tau=0.04` |
| aggressive | `min_depth=6, tau=0.06` | `min_depth=4, tau=0.06` |

### 8.4 Stage D: full-stack cost integration

把软件路径比例和 ONNXim microbenchmark 结合：

```text
total_cost =
    direct_reuse_ratio   * cache_read_cost
  + residual_ratio       * residual_engine_cost
  + p8_ratio             * ONNXim_cost(P8)
  + p6_ratio             * ONNXim_cost(P6)
  + p4_ratio             * ONNXim_cost(P4)
```

最终主图：

```text
x-axis: normalized cycles / traffic / energy
y-axis: accuracy drop
curves: Random / Degree / TSER / Context
```

## 9. Code Interface Proposal

为了后续灵活调参，建议把 Graph-Bit 参数分成三类。

### 9.1 Risk source

```bash
--graphbit_risk_source degree
--graphbit_risk_source propagation
--graphbit_risk_source tser
--graphbit_risk_source context
--graphbit_risk_source low_unique
--graphbit_risk_source random
```

### 9.2 Budget mode

```bash
--graphbit_mode budget
--graphbit_depths 8 6 4
--graphbit_depth_ratios 0.20 0.50 0.30
```

对应当前 `precision_depth_high_ratio/mid_ratio/low_ratio`。

### 9.3 Threshold mode

```bash
--graphbit_mode threshold
--graphbit_risk_bins 16
--graphbit_thresholds 12 6
--graphbit_depths 8 6 4
```

含义：

```text
risk_q >= 12 -> P8
6 <= risk_q < 12 -> P6
risk_q < 6 -> P4
```

### 9.4 Bounded early-stop mode

```bash
--graphbit_bound_enable
--graphbit_min_depths 8 6 4
--graphbit_tolerances 0.0 0.02 0.05
--graphbit_bound_norm output
```

含义：

```text
high-risk:
    min_depth=8, tolerance=0.0

mid-risk:
    min_depth=6, tolerance=0.02

low-risk:
    min_depth=4, tolerance=0.05
```

## 10. Recommended Experiment Matrix

第一组，验证固定 budget：

| Dataset | Frontend | Risk | Budget | Runs |
|---|---|---|---|---:|
| Cora/LLaMA | `h8_54_T40` | Random/Degree/TSER/Context | P8/P6/P4=20/50/30 | 10 |
| PubMed/LLaMA | `h8_76_T40` | Random/Degree/TSER/Context | P8/P6/P4=20/50/30 | 3 then 10 |

这里特意把 Cora 和 PubMed 的 front-end 分开。`h8_54_T40` 是 ST/data.x residual-reuse 的共同参数，但 PubMed/LLaMA full-stack 下过宽，`FullP8-miss` 会先掉到 5% 以上。Graph-Bit 是 miss-node NPU 优化，不应该替过宽的 fuzzy reuse 背锅。因此 full-stack 表必须先保证 `FullP8-miss` 在精度预算内，再比较 Degree/Random/TSER 的 bit-plane routing。

第二组，验证 degree threshold：

| high | mid | Expected |
|---:|---:|---|
| 12 | 6 | default |
| 11 | 6 | more P8 |
| 13 | 6 | less P8 |
| 12 | 5 | more P6 |
| 12 | 7 | more P4 |

第三组，验证 bounded early stop：

| min depths | tolerance | Meaning |
|---|---|---|
| 8/6/4 | 0/0/0 | fixed P8/P6/P4 |
| 8/6/4 | 0/0.01/0.03 | conservative bound |
| 8/6/4 | 0/0.02/0.05 | balanced bound |
| 8/6/4 | 0/0.04/0.08 | aggressive bound |

当前优先执行的 Cora 快速闭环已经脚本化：

```bash
RUN_ALGO=0 RUN_ONNXIM=0 bash GraphhopSimhash/scripts/run_cora_graphbit_predictor_free_flow.sh
```

它完成三件事：

```text
1. 固定 h8_54_T40 前端，读取或重跑 Cora residual + Graph-Bit 软件结果。
2. 读取或重跑 LLaMA-7B QKV/O/FFN GEMM ONNXim microbenchmark。
3. 生成 FullP8-miss / Random static / Degree static / Degree predictor-free EarlyStop 主表。
```

这个脚本是后续调 `min_depth/tolerance` 的默认入口。调参时只需要改：

```bash
--bounded-save-p6
--bounded-save-p4
```

或者进一步把 ONNXim config 中的：

```text
graphbit_min_depth
graphbit_bound_tolerance
graphbit_bound_scale
```

暴露成环境变量即可。

## 11. Paper Claim Boundary

能安全声称：

```text
Graph-Bit maps graph propagation risk to NPU bit-serial precision depth.
It reduces normalized encoder cost under controlled accuracy drop.
```

暂时不能声称：

```text
真实运行时间下降 23%
真实能耗下降 23%
```

这些必须由 ONNXim / cycle-level simulator 支撑。

建议论文措辞：

```text
Our software full-stack experiment shows that graph-risk-guided precision-depth routing reduces normalized encoder cost.
Our ONNXim-backed microbenchmark further estimates the corresponding cycle and memory-traffic reduction of the bit-serial datapath.
```

## 12. Current Default

当前默认主线建议：

```text
Reuse front-end:
    h8_54_T40

Graph-Bit risk:
    Degree / propagation

Graph-Bit mode:
    budget mode first

Budget:
    P8/P6/P4 = 20/50/30

Hardware next step:
    implement bounded early termination in ONNXim GemmWS
```
