# Graph-Bit NPU Design

本文档把 Graph-Bit 从实验现象整理成正式的 NPU 设计。主线目标是：

```text
当 graph-text node 必须执行 LLM encoder 时，
图任务风险控制 NPU 内部 activation bit-plane 的执行深度。
```

Graph-Bit 不是普通的 W4A8/W4A4 路由，也不是 FFN channel gating。它深入到 GEMM datapath：同一个 W4 weight encoder path 下，activation 逻辑上仍是 A8，但硬件可以减少低位 activation bit-plane 的执行。

当前有两种验证层次：

```text
Static precision-depth proxy:
    P8/P6/P5/P4 作为固定执行深度，用离线 embedding pool 评估精度。

Predictor-free early-stop:
    所有节点从 P8 高位开始执行；
    P6/P4 只是 min_depth 安全下限；
    是否继续执行低位由 bit-level bound 和 graph tolerance 决定。
```

因此最终硬件主线不是“必须设计 6-bit/4-bit datatype”，而是“bit-serial datapath 支持 graph-conditioned early termination”。

当前实现已经支持 runtime-bound 策略：

```text
Degree / TSER risk -> high/mid/low bucket
bucket -> min_depth + tolerance
runtime bound -> actual_depth
actual_depth -> nearest generated validation pool P8/P6/P5/P4
```

例如：

```text
high: min_depth=8, tolerance=0.00 -> P8
mid:  min_depth=6, tolerance=0.02 -> P6
low:  min_depth=4, tolerance=0.04 -> P5
```

这里 low bucket 并不是静态指定 P5。硬件逻辑是从高位 bit-plane 开始执行，至少执行到 P4，然后由 remaining-bit bound 判断是否可以停。如果 P4 的剩余误差界仍超过 tolerance，就继续执行到 P5。

更细的 predictor-free bit-serial 实现、degree/propagation risk 阈值管理和 ONNXim 验证接口见：

```text
docs/npu/GRAPH_CONDITIONED_BIT_SERIAL_EXECUTION.md
```

更具体的 NPU 数据流组件模型见：

```text
docs/npu/GRAPH_BIT_NPU_DATAFLOW_MODEL.md
```

它把 early stop 拆成：

```text
byte-major compute mask
bit-plane / plane-group-major activation demand fetch
risk-bucket batching
weight-stationary tile reuse
psum update gating
```

用于解释“跳过 bit-plane”如何真正减少 A/W/psum 访问和 PE issue cycles。

## 1. 系统位置

完整系统可以分成四级：

```text
P0: exact hash reuse
    命中 exact anchor，直接读 embedding cache，cost ~= 0

P1: fuzzy hash reuse + residual correction
    读 anchor embedding，再用 low-rank residual adapter 修正，cost 很小

P2: Graph-Bit precision-depth encoder
    必须跑 encoder，但根据 graph risk 执行 P8/P6/P5/P4 bit-depth

P3: full W4A8 encoder
    高风险兜底路径，完整执行 P8
```

本文档只定义 P2/P3 的 NPU 内部设计。P0/P1 由 SimHash/CAM 和 residual reuse engine 负责。

## 2. 主线策略边界

主线 deployable policy 只能使用在线可得信息：

```text
Degree / propagation risk
TSER = propagation + graph context + low-degree uniqueness
Context-only
LowUnique-only
Random baseline
```

以下策略降级为 debug/oracle，不作为主策略：

```text
PredictorDepthBudget:
    需要 calibration nodes 拟合 damage predictor。
    可用于诊断 hand-crafted proxy 还有多少空间，但增加部署前校准成本。

OracleDamageBudget:
    需要全图 reference embedding 和 low-depth embedding 的真实误差。
    只能作为不可部署上界。
```

论文主表应主要报告 Random / Degree / TSER / Context / LowUnique；Predictor/Oracle 放在 debug 或 upper-bound 小节。

## 3. Datapath 定义

### 3.1 Precision Depth

Graph-Bit 使用 W4 weight path，activation 以 A8 逻辑格式进入。bit-serial / bit-grained datapath 从高位到低位执行：

```text
A8 activation bit planes:
    b7 b6 b5 b4 b3 b2 b1 b0

P8:
    execute b7..b0

P6:
    execute b7..b2

P5:
    execute b7..b3

P4:
    execute b7..b4
```

P6/P5/P4 可以理解成提前终止低位 activation bit-plane。当前精度实验用离线 embedding pools 近似这个过程：

```text
P8 = W4A8
P6 = W4A6
P5 = W4A5
P4 = W4A4
```

在 ONNXim predictor-free early-stop 实现里，`P6/P4` 不再是固定终点，而是：

```text
min_depth:
    至少执行到这个深度，保证风险桶的安全下限。

tolerance:
    如果剩余低位 bit-plane 的 bound 已经足够小，就停止。

actual_depth:
    由运行时 bound 决定，可落在 P8 和 min_depth 之间。
```

### 3.2 PE Array

每个 PE 支持 W4 x A-bit 的 bit-plane accumulation：

```text
for bit_plane in MSB_to_LSB:
    activation_slice = A[bit_plane]
    partial_sum += W4 * activation_slice * bit_weight
    if precision_depth reached:
        stop remaining lower bit planes
```

硬件需要的新增状态很少：

```text
mode register:
    2-bit precision mode: P8/P6/P5/P4

bit-plane sequencer:
    控制当前 batch 执行到哪一位

partial-sum buffer:
    保存每个 output tile 的累加值

early-stop mask:
    当前 batch 达到指定 precision depth 后停止低位计算
```

第一版不要求 per-element dynamic stopping。更稳的实现是 per-node-batch mode，即一个 micro-batch 内节点共享 P8/P6/P5/P4。这样调度简单，array utilization 更高。

## 4. Scheduler

### 4.1 输入

每个节点进入 encoder 前，scheduler 已经有：

```text
node id
reuse status: exact / fuzzy / miss
graph risk score: Degree / TSER / Context / LowUnique
target path: P0/P1/P2/P3
```

Graph-Bit 只处理 miss nodes，也就是仍然需要跑 encoder 的节点。

当前 Cora/LLaMA 快速验证的 reuse 前端固定为：

```text
R = 2
8 heads x 16-bit
score threshold T = 40
support >= 5 -> direct reuse
support == 4 -> residual correction
support < 4  -> Graph-Bit / full encoder
```

因此 Graph-Bit 的输入集合不是全图节点，而是 support 小于 4 或被 score gate 拒绝、仍需执行 encoder 的节点。这个固定前端避免把 Graph-Bit 结果和 reuse 参数调优混在一起。

但这个前端不是所有 backend/dataset 的无条件默认。LLaMA-7B full-stack 的第一原则是：

```text
先检查 FullP8-miss：
    accepted hit -> direct/residual reuse
    miss         -> P8 encoder

只有 FullP8-miss drop 已经安全，
才继续评估 Graph-Bit 对 miss nodes 的 bit-plane 省算。
```

当前结果中，Cora/LLaMA 可以用 `h8_54_T40`；PubMed/LLaMA 需要更严格的：

```text
h8_76_T40:
    8 heads x 16-bit
    R = 2
    T = 40
    support >= 7 -> direct reuse
    support == 6 -> residual correction
    support < 6  -> Graph-Bit / full encoder
```

这不是改变 NPU datapath，而是 scheduler 的安全阈值配置；Graph-Bit datapath 仍然相同。

### 4.2 Risk-to-depth Mapping

固定 budget 映射：

```text
sort nodes by risk descending

top high_ratio:
    P8

next mid_ratio:
    P6

next low_ratio:
    P5

remaining:
    P4
```

典型 budget：

```text
10/20/30/40  -> aggressive
20/30/30/20  -> balanced-low-cost
30/40/20/10  -> balanced-safe
50/30/20/0   -> near-lossless
```

更硬件友好的执行顺序：

```text
1. 收集一批 miss nodes
2. 根据 risk 排序或桶化
3. 分成 P8/P6/P5/P4 mode queues
4. 同 mode nodes 组成 micro-batch
5. mode register 写入 NPU
6. 执行对应 bit-plane depth
```

### 4.3 为什么不主打 learned predictor

Predictor routing 会引入 calibration nodes 和训练成本，影响架构通用性。Graph-Bit 主线使用 Degree/TSER 这类无需训练的 graph proxy；Predictor 只作为 debug/profiling baseline。

## 5. Buffer 设计

Graph-Bit 需要四类 buffer：

```text
1. Node mode queue
   保存 P8/P6/P5/P4 四个队列的 node ids。

2. Activation bit-plane buffer
   将 A8 activation 按 bit-plane 或 packed group 读入。
   P4/P5/P6 batch 不读取或不送入低位 bit-plane。

3. Partial-sum buffer
   保存当前 GEMM tile 的累加结果。
   P4/P5/P6 仍输出同样 shape，只是低位贡献被省略。

4. Embedding output buffer
   保存 encoder output embedding，供后续 GNN 或 embedding cache 使用。
```

关键点：Graph-Bit 不改变 embedding shape，不改变 GNN 后端接口。它只改变 encoder 内部算术努力。

## 6. Cost Model

当前实验使用一个简单但可解释的 cost model：

```text
cost(bit) = cost_scale * (fixed_cost + (1 - fixed_cost) * bit / reference_bits)
```

本轮 LLaMA-7B 设置：

```text
reference_bits = 8
cost_scale     = 0.50
fixed_cost     = 0.15

P8 cost = 0.500
P6 cost = 0.394
P5 cost = 0.341
P4 cost = 0.287
```

含义：

```text
cost_scale:
    W4A8 encoder 相对 FP/full baseline 的归一化成本。

fixed_cost:
    不随 activation bit-plane 减少而消失的固定开销，
    包括 weight fetch、control、LayerNorm、softmax、pooling、scale/repack 等。

bit / reference_bits:
    可被 bit-plane early termination 缩减的 activation-side GEMM effort。
```

对于一个 batch：

```text
BatchCost =
    ratio_P8 * cost(P8)
  + ratio_P6 * cost(P6)
  + ratio_P5 * cost(P5)
  + ratio_P4 * cost(P4)
```

与 reuse 组合时：

```text
TotalCost =
    reuse_exact_ratio * 0
  + reuse_fuzzy_ratio * residual_adapter_cost
  + miss_ratio * GraphBitCost(miss nodes)
```

## 7. Step 3: Bit-Plane Early-Termination Simulation

Step 3 的目标是从 embedding-pool 近似推进到真正的 bit-plane 模型。

第一阶段已经完成：

```text
W4A8/W4A6/W4A5/W4A4 embedding pools
    ~= P8/P6/P5/P4 fixed precision depth
```

下一阶段软件仿真：

```text
for each GEMM tile:
    compute high bit-plane partial sums
    estimate remaining low-bit contribution bound
    compare with graph-conditioned tolerance
    stop low bit-planes when safe
```

Graph-conditioned tolerance：

```text
high-risk node:
    strict tolerance -> more bit-planes

low-risk node:
    loose tolerance -> early stop
```

这一步的验证指标：

```text
executed_bit_planes / full_bit_planes
partial-sum error
embedding error
downstream GNN accuracy drop
array utilization
mode-switch overhead
```

## 8. Step 4: Mode-Adaptive PE Array

Step 4 把已有 FFN gating 降级为一个 mode-adaptive PE array 的实例，而不是主贡献本身。

Graph-Bit PE array 应支持：

```text
P8 mode:
    full A8 bit-plane execution

P6/P5/P4 mode:
    early-stop lower bit-plane

optional FFN-gated mode:
    在低风险 batch 上减少部分 FFN channel
```

FFN gating 的定位：

```text
不是主线机制，
而是证明 mode-adaptive PE array 可以支持多种 low-effort execution mode。
```

最终硬件故事：

```text
Graph risk -> mode scheduler -> mode-adaptive PE array

mode can control:
    activation precision depth
    optional FFN channel budget
    optional outlier protection budget
```

其中最核心、最应该主打的是 activation precision depth，因为它直接作用于 GEMM bit-plane datapath，适用 QKV / attention projection / FFN 等主要线性层。

## 9. ONNXim Hardware Flow

当前已经接入两层硬件仿真闭环：

```text
Layer A: workload-level hardware proxy
    从算法结果导出 direct / residual / P8 / P6 / P4 比例，
    再叠加到 ONNXim Full-P8 GEMM baseline 上。

Layer B: ONNXim internal Graph-Bit execution
    直接修改 ONNXim GemmWS / SystolicWS，
    在 GEMM instruction 内部模拟 activation bit-plane early termination。
```

### 9.1 Workload-Level Proxy

```text
1. scripts/export_graphbit_workload.py
   从 residual / Graph-Bit 结果中导出 direct、residual、P8/P6/P4 比例。

2. scripts/onnxim_graphbit_microbench.py
   为 LLaMA-7B 主要 GEMM 生成 ONNX microbenchmark，并调用 ONNXim 得到 Full-P8 baseline cycles/traffic。

3. scripts/summarize_onnxim_graphbit.py
   将 workload profile 叠加到 ONNXim baseline，输出 Graph-Bit normalized cycles、traffic 和 energy proxy。
```

一键运行：

```bash
bash scripts/run_onnxim_graphbit_sim.sh
```

默认使用 `SEQ_LEN=64` 做快速 microbenchmark；需要更接近长文本 encoder 时可以：

```bash
SEQ_LEN=128 bash scripts/run_onnxim_graphbit_sim.sh
```

输出路径：

```text
output/onnxim_graphbit/microbench_s64/summary.tsv
output/onnxim_graphbit/microbench_s64/aggregate.json
output/onnxim_graphbit/workloads/three_depth_deg_profiles.json
output/onnxim_graphbit/summary/three_depth_deg_profiles_hardware.tsv
output/onnxim_graphbit/summary/three_depth_deg_profiles_compact.txt
```

### 9.1.1 Fixed Cora h8_54_T40 predictor-free flow

为了把当前主线参数固定下来，新增了 Cora 快速硬件验证脚本：

```bash
RUN_ALGO=0 RUN_ONNXIM=0 bash GraphhopSimhash/scripts/run_cora_graphbit_predictor_free_flow.sh
```

默认设置：

```text
frontend:
    h8_54_T40
    R = 2
    hard direct: support >= 5
    residual: support == 4

Graph-Bit:
    risk = Degree
    budget = P8/P6/P4 = 20/50/30 on miss nodes
```

常用开关：

```bash
# 重新跑 Cora residual + Graph-Bit 软件实验
RUN_ALGO=1 bash GraphhopSimhash/scripts/run_cora_graphbit_predictor_free_flow.sh

# 重新跑 ONNXim GEMM microbenchmark
RUN_ONNXIM=1 bash GraphhopSimhash/scripts/run_cora_graphbit_predictor_free_flow.sh

# 如果 ONNXim 尚未构建，先尝试构建再跑
BUILD_ONNXIM=1 RUN_ONNXIM=1 bash GraphhopSimhash/scripts/run_cora_graphbit_predictor_free_flow.sh
```

输出：

```text
output/graphbit_predictor_free/cora_h8_54_T40/summary.tsv
output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_main.tsv
output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_main.txt
output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_workload.json
```

当前 Cora 主表：

```text
Method                         Reuse   P8     P6     P4     AvgBit Saved  Cycles Traffic Energy Drop
FullP8-miss                     40.0%  60.0%   0.0%   0.0%   8.00  0.00   0.601   0.602  0.602  1.53%
Random static P8/P6/P4          40.0%  12.0%  30.0%  18.0%   5.80  2.20   0.436   0.544  0.485  2.79%
Degree static P8/P6/P4          40.0%  12.0%  30.0%  18.0%   5.80  2.20   0.436   0.544  0.485  2.39%
Degree predictor-free EarlyStop 40.0%  12.0%  30.0%  18.0%   5.47  2.53   0.412   0.536  0.468  2.39%
```

解释：

```text
FullP8-miss:
    accepted reuse/residual hits 走 P0/P1；
    miss nodes 全部完整执行 P8。

Degree static P8/P6/P4:
    same reuse set；
    miss nodes 按 degree risk 静态分配到 P8/P6/P4。

Degree predictor-free EarlyStop:
    same software assignment；
    NPU 内部用 bounded bit-plane early stop 估计额外低位省算。
```

这里的 `Cycles/Traffic/Energy` 是相对全图 Full-P8 encoder 的归一化硬件 proxy。`Drop` 仍来自静态 embedding proxy；bounded early-stop 的真实数值精度需要后续 bit-serial numerical kernel 或更细的 ONNXim numerical model 支撑。

### 9.1.2 PubMed/LLaMA robust front-end

PubMed/LLaMA 上，`h8_54_T40` 会让 reuse 命中过多，`FullP8-miss` 自身已经掉到 `5.34%`，因此不能作为 Graph-Bit 主线结果。按照同一套流程扫 support split 后，当前稳健点是：

```text
h8_76_T40:
    hard direct: support >= 7
    residual:    support == 6
```

3-run 结果：

| Front-end | Reuse | FullP8 Cost | FullP8 Drop | Degree Cost | Degree Drop | PF AvgBit | PF Cycles | PF Traffic | PF Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `h8_54_T40` | 74.9% | 0.127 | 5.34% | 0.098 | 5.76% | 5.47 | 0.173 | 0.226 | 0.197 |
| `h8_65_T40` | 50.6% | 0.249 | 3.62% | 0.191 | 4.56% | 5.48 | 0.340 | 0.443 | 0.386 |
| `h8_76_T40` | 22.3% | 0.389 | 1.26% | 0.298 | 2.54% | 5.47 | 0.532 | 0.692 | 0.604 |

结论：

```text
1. PubMed/LLaMA 的 residual hits 更脏，必须提高 support 门槛。
2. h8_76_T40 下 FullP8-miss 已安全，Degree Graph-Bit 仍低于 3% drop。
3. predictor-free early-stop 在同一精度 drop 下，把 static Degree 的
   cycles/traffic/energy 从 0.563/0.703/0.626 降到 0.532/0.692/0.604。
```

Miss-only ONNXim 分解进一步确认了这个收益不是表面 cost model：

```text
output/graphbit_predictor_free/cora_h8_54_T40/earlystop_sweep/miss_only_breakdown.txt
```

核心结果：

```text
FullP8-miss:
    AvgD=8.00, BitComp=1.000, ActRd=1.000, Traffic=1.000

EarlyStop balanced:
    AvgD=6.10, BitComp=0.764, ActRd=0.763, Traffic=0.969

EarlyStop aggressive:
    AvgD=5.80, BitComp=0.726, ActRd=0.725, Traffic=0.964
```

这里 `BitComp` 是 bit-serial effective compute cycles / raw full-depth compute cycles。它说明在只看 miss nodes 时，balanced early-stop 已经减少约 `23.6%` 的 bit-plane 算术量和约 `23.7%` 的 activation input reads。总 traffic 下降较小，是因为 weight read 和 output write 没有随 activation bit-depth 下降。

### 9.1.2 Risk-bucket batching

Miss-only 分解还暴露了另一个硬件问题：如果 high/mid/low risk 节点随机混在同一个 micro-batch 里，bit-serial 控制器通常必须跑到该 batch 的最大 bit-depth。这样低风险节点虽然被判为 P6/P4 或 early-stop，但实际会被同 batch 的 high-risk 节点拖回 P8。

因此 Graph-Bit 需要一个简单但关键的 scheduler：

```text
1. reuse/residual 前端先筛掉 cache hit 节点；
2. 对剩余 miss nodes 计算 degree risk；
3. 按 high / mid / low risk 分桶；
4. 每个桶单独形成 micro-batch；
5. NPU 对每个 bucket 使用对应 min_depth / tolerance。
```

命令：

```bash
bash GraphhopSimhash/scripts/run_cora_graphbit_risk_bucket_batching.sh
```

结果文件：

```text
output/graphbit_predictor_free/cora_h8_54_T40/risk_bucket_batching/risk_bucket_batching.txt
```

Cora `h8_54_T40`，miss-node mix 为 `high=20% / mid=50% / low=30%`，micro-batch size 64：

| Method | Assign | Schedule | Mode | UsefulD | ExecD | Util | Waste | Cycles | BitComp | ActRd | Traffic | Drop |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomOrder Static | degree | random mixed | static | 5.80 | 8.00 | 72.5% | 27.5% | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| DegreeBucket Static | degree | risk bucket | static | 5.80 | 5.80 | 100.0% | 0.0% | 0.957 | 0.726 | 0.725 | 0.964 | 2.39% |
| RandomRisk Bucket | random | risk bucket | static | 5.80 | 5.80 | 100.0% | 0.0% | 0.957 | 0.726 | 0.725 | 0.964 | 2.79% |
| RandomOrder EarlyStop | degree | random mixed | early-stop | 6.10 | 8.00 | 76.3% | 23.7% | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| DegreeBucket EarlyStop | degree | risk bucket | early-stop | 6.10 | 6.10 | 100.0% | 0.0% | 0.959 | 0.764 | 0.763 | 0.969 | 2.39% |

解释：

```text
RandomOrder:
    仍然使用 Degree assignment，但 high/mid/low 混在一个 batch。
    batch=64 时几乎每个 batch 都包含 high-risk 节点，因此整批跑到 P8。

DegreeBucket:
    同样的 Degree assignment，但 high/mid/low 分桶执行。
    useful depth 和 executed depth 对齐，bit-plane waste 变成 0。

RandomRisk Bucket:
    硬件 cost 和 DegreeBucket 一样，但风险分配随机，drop 更高。
    这说明 graph proxy 不只是为了凑比例，而是真的保护了精度。
```

这个实验把 Graph-Bit 从“precision-depth policy”推进到更像硬件论文的 “graph-aware NPU scheduler/dataflow”：

```text
graph risk decides not only how many bit-planes to execute,
but also how miss nodes are batched so the bit-serial array can realize that saving.
```

### 9.1.3 FFN block-gating hardware probe

前面的结果说明 activation bit-plane early-stop 能省 `BitComp` 和 `ActRd`，但总 traffic 被 weight read / output write 稀释。要继续降低 weight traffic，需要让部分 FFN weight block 不被读取。

这里先做硬件探针，不声明精度结论：

```bash
bash GraphhopSimhash/scripts/run_onnxim_ffn_block_gating_microbench.sh
```

输出：

```text
output/onnxim_graphbit/ffn_block_gating/ffn_block_gating.txt
```

ONNXim LLaMA-7B FFN intermediate block-gating 结果：

| Keep | Intermediate | Cycles | MatMul | Traffic | InRead | WeightRd | OutWr | GFLOPs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 74% | 8192 | 0.826 | 0.829 | 0.830 | 0.834 | 0.829 | 0.867 | 0.829 |
| 50% | 5504 | 0.629 | 0.654 | 0.669 | 0.687 | 0.666 | 0.741 | 0.666 |

解释：

```text
1. FFN block gating 改变 intermediate dimension，因此 weight read 会真实下降。
2. keep≈74% 时，weight read 降到 82.9%，cycles 降到 82.6%。
3. keep=50% 时，weight read 降到 66.6%，cycles 降到 62.9%。
4. 这条线能解决 Graph-Bit early-stop 不降低 weight traffic 的问题。
5. 但它会改变模型函数，必须单独做 embedding/GNN 精度验证后才能进入主线。
```

因此当前建议：

```text
主线:
    Graph-Bit predictor-free bit-plane early-stop
    + degree-risk bucket batching

下一阶段 Graph-Bit+:
    对低风险 miss nodes 进一步启用 FFN block gating，
    用 accuracy proxy 验证 keep≈74% 是否能在 <3% drop 内换取 weight traffic 下降。
```

当前论文主线不依赖 FFN block gating。它只是一个硬件探针，用来说明如果未来要继续压 `WeightRd`，必须触及 FFN/block 级数据流；主线仍然是 predictor-free bit-serial early-stop 和 risk-bucket scheduler。

### 9.1.4 Memory dataflow breakdown

为了区分“算术省算”和“访存省流量”，新增 memory-dataflow 汇总：

```bash
bash GraphhopSimhash/scripts/run_cora_graphbit_memory_dataflow.sh
```

输出：

```text
output/graphbit_predictor_free/cora_h8_54_T40/memory_dataflow/memory_dataflow.txt
```

结果：

| Method | AvgD | BitComp | Cycles | ActRd | WeightRd | OutWr | Traffic | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 miss | 8.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.53% |
| EarlyStop compute-only | 6.10 | 0.764 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| EarlyStop + ActPack | 6.10 | 0.764 | 0.959 | 0.763 | 1.000 | 1.000 | 0.969 | 2.39% |
| EarlyStop + ActPack + FFNBypass | 6.10 | 0.764 | 0.959 | 0.763 | 1.000 | 1.000 | 0.938 | 2.39% |

解释：

```text
compute-only:
    只体现 bit-serial 算术省算；如果 activation 仍按 A8 存取，traffic 不变。

ActPack:
    当前 Graph-Bit 路径。activation bit-plane packed read 让 ActRd 降到 0.763。

FFNBypass:
    exact dataflow upper bound，不改变模型函数。
    它假设 FFN intermediate 留在片上，避免 ffn_up output write 和 ffn_down input read。
```

结论：

```text
Graph-Bit 的主收益来自 BitComp / ActRd；
若片上 SRAM 允许 FFN intermediate bypass，traffic 可从 0.969 进一步到 0.938。
这不是 FFN channel gating，不改变权重或输出维度，只是减少中间结果往返 HBM。
```

### 9.1.5 Batch-size amortization

risk-bucket scheduler 还有一个访存收益：同一个 risk bucket 连续处理时，weight tile 可以被更多 node tokens 复用。用 ONNXim 对 FullP8 GEMM 做 batch-size sweep：

```bash
bash GraphhopSimhash/scripts/run_onnxim_batch_amortization.sh
```

输出：

```text
output/onnxim_graphbit/batch_amortization/batch_amortization.txt
```

结果：

| Seq / micro-batch | Cyc/Node norm | Traffic/Node norm | Weight/Node norm | Input/Node norm |
|---:|---:|---:|---:|---:|
| 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16 | 0.507 | 0.510 | 0.500 | 1.000 |
| 32 | 0.266 | 0.265 | 0.250 | 1.000 |
| 64 | 0.143 | 0.143 | 0.125 | 1.000 |
| 128 | 0.080 | 0.082 | 0.062 | 1.000 |

解释：

```text
Input/Node 基本不变：
    每个 node token 都有自己的 activation。

Weight/Node 随 micro-batch 近似按 1/B 摊薄：
    权重 tile 被更多节点复用。

这说明 risk-bucket scheduler 不只是为了避免 bit-depth divergence；
它还把相同执行模式的 miss nodes 聚成大 batch，从而提高 weight-stationary 数据复用。
```

### 9.2 Internal GemmWS Bit-Plane Execution

ONNXim 内部已经加入 Graph-Bit knobs：

```json
{
  "graphbit_enable": true,
  "graphbit_precision_depth": 6,
  "graphbit_full_depth": 8,
  "graphbit_min_depth": 4,
  "graphbit_bound_enable": false,
  "graphbit_bound_tolerance": 0.0,
  "graphbit_bound_scale": 1.0,
  "graphbit_memory_scale": 1.0
}
```

代码路径：

```text
ONNXim/src/SimulationConfig.h
ONNXim/src/Common.h
ONNXim/src/Common.cc
ONNXim/src/operations/GemmWS.cc
ONNXim/src/SystolicWS.cc
ONNXim/src/SystolicWS.h
```

实现方式：

```text
GemmWS:
    生成 MOVIN activation 和 GEMM_PRELOAD instruction 时，
    标注 graphbit_full_depth / config_depth / effective_depth / remaining_bound。

    activation MOVIN 的 src_addrs 根据 effective_depth/full_depth 截短，
    用来模拟低位 activation bit-plane 不再进入 SRAM/array。

SystolicWS:
    get_inst_compute_cycles() 按 effective_depth/full_depth 缩放 GEMM cycle，
    代表 bit-serial PE array 少执行低位 bit-plane。

    print_stats() 输出：
        GraphBit Inst
        BoundStops
        AvgDepth
        AvgSavedBitplanes
```

固定 depth 运行：

```bash
python scripts/onnxim_graphbit_microbench.py \
  --seq-len 64 \
  --workspace output/onnxim_graphbit/microbench_s64_internal_p6 \
  --graphbit-depth 6 \
  --action all
```

predictor-free bound estimator 运行：

```bash
python scripts/onnxim_graphbit_microbench.py \
  --seq-len 64 \
  --workspace output/onnxim_graphbit/microbench_s64_internal_bound_t006 \
  --graphbit-depth 8 \
  --graphbit-bound-enable \
  --graphbit-bound-tolerance 0.06 \
  --action all
```

这个 bound mode 的含义是：虽然外部给的是 P8 上限，但 GemmWS 会逐 depth 检查 remaining low-bit bound，只要 bound 小于 tolerance，就提前停止低位 bit-plane。

当前 smoke result，LLaMA-7B GEMM microbench，seq_len=64，32 layers：

| Mode | Cycles | Cycle Ratio | DRAM Read Req | Read Ratio | AvgDepth |
|---|---:|---:|---:|---:|---:|
| P8 baseline | 43,159,520 | 1.000 | 466,386,944 | 1.000 | - |
| Internal P6 | 41,112,672 | 0.953 | 450,977,792 | 0.967 | 6.0 |
| Internal P4 | 40,355,424 | 0.935 | 435,568,640 | 0.934 | 4.0 |
| Bound P8, tol=0.06 | 42,442,784 | 0.983 | 458,682,368 | 0.983 | 4.0 |

P6/P4 的 read request 下降来自 activation MOVIN bit-plane 截短；cycle 下降来自 GEMM_PRELOAD 的 effective bit-plane execution。Bound mode 的日志中 `BoundStops` 等于 GraphBit GEMM instruction 数，说明 early termination 是在 ONNXim 内部触发的，而不是外部后处理。

当前 smoke result 使用 LLaMA-7B GEMM microbench：

```text
seq_len = 64
layers  = 32
Full-P8 encoder baseline cycles = 43,159,520
Full-P8 DRAM requests ~= 471,826,432
```

Degree Graph-Bit 的硬件 proxy 结果：

| Dataset | Setting | Reuse | Drop | Norm cycles | Norm traffic | Energy proxy | Bounded cycles |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | h4 T20 balanced | 4.5% | 2.45% | 0.692 | 0.863 | 0.769 | 0.653 |
| Cora | h4 T22 conservative | 12.1% | 2.95% | 0.683 | 0.812 | 0.741 | 0.650 |
| PubMed | h8 T16 conservative | 14.4% | 2.13% | 0.664 | 0.789 | 0.720 | 0.632 |
| PubMed | h8 T20 conservative | 33.5% | 3.43% | 0.517 | 0.616 | 0.562 | 0.492 |

这里：

```text
Norm cycles = 相对所有节点都跑 Full-P8 encoder 的周期比例
Norm traffic = 相对 Full-P8 的访存请求比例估计
Energy proxy = 0.55 * cycles + 0.45 * traffic
Bounded cycles = predictor-free bounded early termination 进一步允许 P6/P4 少跑少量低位 bit-plane 的估计
```

注意：上面的 workload-level proxy 仍然有用，因为它负责组合 direct/residual/P8/P6/P4 的全栈比例；但 ONNXim internal path 已经把 P8/P6/P4 下沉到 GemmWS/SystolicWS 内部。后续要做的是把 workload profile 自动映射成多次 internal ONNXim run 并汇总，而不是只在 Python 后处理里缩放。

### 9.3 Demand-Fetch and Utilization Model

最新建模已经把 Graph-Bit 从“位宽 cost proxy”进一步拆成四个硬件层级：

```text
compute-mask only:
    PE 跳过低位 MAC，但 activation 仍按完整 A8 读取。

demand-fetch:
    activation 采用 bit-plane-major layout，被 early-stop 跳过的低位 bit-plane 不再发起读取。

random-mixed batching:
    不同风险节点混在一个 bit-serial micro-batch，batch 执行深度被最深节点拖到 P8。

risk-bucket batching:
    high/mid/low risk nodes 分桶执行，使 useful depth 真正变成 executed depth。
```

建模脚本：

```bash
bash scripts/run_graphbit_demand_fetch_model.sh
```

默认输出：

```text
output/graphbit_predictor_free/cora_h8_53_T30/demand_fetch_model/
```

历史 balanced 前端复现实验：

```bash
WORKLOAD=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_workload.json \
OUT_DIR=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/cora_h8_54_T40/demand_fetch_model \
bash scripts/run_graphbit_demand_fetch_model.sh
```

关键结果：

| Workload | Method | UsefulD | ExecD | BitComp | ActRd | Full cycles | Full traffic | Drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h8_53_T30 p8heavy | Degree compute-mask only | 7.60 | 7.60 | 0.950 | 1.000 | 0.707 | 0.707 | 2.18% |
| h8_53_T30 p8heavy | Degree demand-fetch | 7.60 | 7.60 | 0.950 | 0.950 | 0.700 | 0.703 | 2.18% |
| h8_54_T40 balanced | Degree compute-mask only | 5.80 | 5.80 | 0.726 | 1.000 | 0.601 | 0.602 | 2.39% |
| h8_54_T40 balanced | Degree random-mixed | 6.10 | 8.00 | 1.000 | 1.000 | 0.601 | 0.602 | 2.39% |
| h8_54_T40 balanced | Degree demand-fetch | 6.10 | 6.10 | 0.764 | 0.762 | 0.576 | 0.583 | 2.39% |

这张表给出当前最重要的硬件判断：

```text
1. 只做 compute mask 不够，cycles/traffic 不会自动下降。
2. 只做 demand-fetch 也不够，若 batch 随机混合风险等级，会被 P8 节点拖回 full-depth。
3. Graph-Bit 必须同时包含：
       bit-plane-major demand fetch
       predictor-free early-stop
       risk-bucket scheduler
   才能把 graph risk 变成 NPU 可见的 cycle/traffic 收益。
```

完整公式和可靠性边界见：

```text
docs/npu/GRAPH_BIT_DEMAND_FETCH_MODEL.md
```

## 10. 当前验证结果

LLaMA-7B / Arxiv，10 runs：

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 10/20/30/40 | 0.346 | 0.82 | 0.59 | 0.66 | 0.75 | 0.87 | Degree |
| 20/30/30/20 | 0.378 | 0.52 | 0.36 | 0.38 | 0.45 | 0.57 | Degree |
| 30/40/20/10 | 0.404 | 0.36 | 0.22 | 0.26 | 0.34 | 0.43 | Degree |
| 50/30/20/0  | 0.436 | 0.21 | 0.12 | 0.13 | 0.18 | 0.21 | Degree |

这个结果说明：

```text
同样 precision-depth budget 下，
graph risk 尤其 Degree/propagation risk 能比 Random 更稳地保护精度。
```

更多 Cora/PubMed/Arxiv 结果见：

```text
../results/GRAPH_BIT_VALIDATION_SUMMARY.md
```

## 11. 论文表述建议

推荐主贡献表述：

```text
Graph-Bit is a graph-conditioned precision-depth NPU for graph-text LLM encoders.
It maps graph propagation and semantic risk to activation bit-plane execution depth,
allowing low-risk nodes to terminate bit-serial GEMM earlier while preserving high-risk nodes with full W4A8 execution.
```

不要写成：

```text
Degree-guided W4A8/W4A4 quantization routing.
```

更准确的是：

```text
Graph risk controls arithmetic effort inside the NPU datapath.
```
