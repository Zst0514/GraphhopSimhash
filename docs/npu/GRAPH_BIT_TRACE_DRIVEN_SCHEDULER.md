# Graph-Bit Trace-Driven Scheduler Replay

本文档记录新的 Graph-Bit NPU trace-driven 仿真层。目标是把之前的 `bucket32 / bucket64 Wscale` 从公式估算，推进到真实 workload trace replay：

```text
real miss node trace
    -> risk / stop-depth bucket
    -> micro-batch scheduling
    -> W tile load replay
    -> ONNXim component cost lookup
    -> cycles / traffic / energy / utilization
```

这不是 full-system cycle-accurate 仿真；更准确的定位是：

```text
ONNXim component-cycle simulation
    + real Graph-Bit node trace
    + trace-driven bucket scheduler replay
```

该层输出 `Wloads`、`Wscale`、`tail utilization`、`cycles`、`traffic` 和 `energy`，用于描述 bucket scheduler 在真实节点 trace 上的执行结果。

## 1. Trace 内容

`residual_precision_depth` 现在可以导出 per-node JSONL trace。第一行是 metadata，后面每行对应一个 graph node：

```text
node_id
role: direct / residual / miss
source_id
hit_kind
best_dist
route_hits / support_hits
action_bit
depth_bucket: p8 / p6 / p5 / p4
stop_depth
degree_q / tser_q / context_q / low_unique_q
```

关键点：

```text
role != miss:
    由 cache reuse / residual reuse 处理，不进入 LLM encoder

role == miss:
    进入 Graph-Bit NPU
    action_bit / stop_depth 是 predictor-free bound 后的实际执行深度
```

## 2. Replay 策略

调度器重放四类路径：

```text
FullP8-miss:
    同一批 miss nodes 全部强制 D8。

GraphBit-now:
    使用真实 stop_depth，但不扩大 W tile 服务窗口。

OriginalOrder-bN:
    保持原始节点顺序形成 batch。
    一个 batch 内如果混有 D5/D6/D8，则按 max depth 执行。

RiskBucket-bN:
    先按真实 stop_depth 分桶，再形成 batch。
    W tile load 直接由 bucket size / batch size 数出来。
```

其中 `Wloads` 和 `Wscale` 是 replay 统计结果：

```text
baseline_wloads = ceil(num_miss_nodes / baseline_tile_batch)
Wloads          = replayed number of W tile loads
Wscale          = Wloads / baseline_wloads
```

## 3. Cora Quick Result

命令：

```bash
bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

等价展开：

```bash
RUNS=1 \
RUN_ALGO=1 \
RUN_ONNXIM=0 \
DATASET=cora \
THRESHOLD=40 \
HARD_SUPPORT=5 \
SOFT_SUPPORT=4 \
FRONTEND_ID=h8_54_T40 \
BUDGET=boundclean \
HIGH_RATIO=0.20 \
MID_RATIO=0.50 \
LOW_RATIO=0.0 \
OUT_DIR=output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick \
TRACE_EXPORT=1 \
TRACE_EXPORT_CONFIGS='DegBound' \
BOUND_ENABLE=1 \
BOUND_PRIORITIES='degree' \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh

/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/replay_graphbit_trace_scheduler.py \
  --trace output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/node_traces/cora_seed42_DegBound.jsonl \
  --components-root output/onnxim_graphbit/risk_bucket_components_s8 \
  --output-dir output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay \
  --fullp8-drop-percent 0.77 \
  --drop-percent 2.13 \
  --baseline-tile-batch 16 \
  --candidate-batches 32 64
```

输出：

```text
output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick/replay/cora_seed42_DegBound_trace_replay.txt
```

结果：

| Method | Reuse | Miss | Cycles | Traffic | Energy | Drop | AvgD | Hist(miss) | Wloads | Wscale |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| FullP8-miss | 27.8% | 72.2% | 0.722 | 0.722 | 0.722 | 0.77% | 8.00 | D8:100.0% | 123 | 1.000 |
| GraphBit-now | 27.8% | 72.2% | 0.716 | 0.719 | 0.717 | 2.13% | 6.10 | D5:30.0%, D6:50.0%, D8:20.0% | 123 | 1.000 |
| OriginalOrder-b32 | 27.8% | 72.2% | 0.385 | 0.368 | 0.376 | 2.13% | 7.77 | D6:11.5%, D8:88.5% | 62 | 0.504 |
| RiskBucket-b32 | 27.8% | 72.2% | 0.384 | 0.366 | 0.375 | 2.13% | 6.10 | D5:30.0%, D6:50.0%, D8:20.0% | 63 | 0.512 |
| OriginalOrder-b64 | 27.8% | 72.2% | 0.290 | 0.191 | 0.241 | 2.13% | 7.87 | D6:6.5%, D8:93.5% | 31 | 0.252 |
| RiskBucket-b64 | 27.8% | 72.2% | 0.289 | 0.189 | 0.239 | 2.13% | 6.10 | D5:30.0%, D6:50.0%, D8:20.0% | 33 | 0.268 |

## 4. 如何解读

这张表最重要的不是单个 cycles 数字，而是 `Wloads / Wscale / AvgD` 已经由真实 trace replay 得到：

```text
FullP8-miss:
    1954 个 miss nodes，baseline tile batch=16
    Wloads = ceil(1954 / 16) = 123

RiskBucket-b32:
    按 stop-depth 分桶后重放
    Wloads = 63
    Wscale = 63 / 123 = 0.512

RiskBucket-b64:
    Wloads = 33
    Wscale = 33 / 123 = 0.268
```

这里的 `Wscale` 由 miss-node trace replay 得到，计算方式为 `Wloads / baseline_wloads`。

`OriginalOrder` 的作用是说明为什么需要 risk-bucket：

```text
OriginalOrder:
    Wloads 也能下降，因为 batch 变大了；
    但混合风险节点会让 batch 按最高 depth 执行。
    AvgD 接近 8，D5/D6 的 early-stop 优势被高风险节点拖住。

RiskBucket:
    Wloads 由真实 bucket size 决定；
    同时保留 D5/D6/D8 的 stop-depth 分布。
```

## 5. 当前结论

当前实现把 Graph-Bit 硬件证据链推进到：

```text
1. 真实 residual / miss trace
2. 真实 predictor-free stop-depth action
3. 真实 bucket replay W tile load
4. ONNXim component cost lookup
```

当前实现将 workload-level accuracy profile、per-node stop-depth trace 和 ONNXim component cost lookup 连接起来。后续如果要继续增强，可以把 replay 层进一步下沉到更细粒度的 ONNXim per-tile event trace。

```text
Graph risk is not only an accuracy proxy;
it also shapes NPU scheduling and weight-tile reuse.
```
