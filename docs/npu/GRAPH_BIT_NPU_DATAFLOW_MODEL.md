# Graph-Bit NPU Dataflow Model

本文档定义当前 Graph-Bit NPU 的可执行数据流模型。它回答的问题是：

```text
Graph risk 让 bit-serial GEMM 提前停止之后，
硬件到底跳过了什么？
```

结论先说清楚：

```text
只减少 MAC 不够。
Graph-Bit 必须同时设计：

1. bit-plane / plane-group-major activation buffer
2. bit-plane issue scheduler
3. risk-bucket micro-batch scheduler
4. weight-stationary tile reuse
5. psum update gating
```

否则 low-bit early stop 很容易只停留在软件 proxy，无法变成端到端的 cycles / traffic / energy 收益。

## 1. 为什么普通 A8 Byte Layout 不够

普通 activation layout 是 byte-major：

```text
A_byte = [a7 a6 a5 a4 a3 a2 a1 a0]
```

这种格式下，SRAM/HBM/NoC 通常按 byte、word、cacheline 或 burst 读取。即使节点只需要高 5 bit，也已经把低 3 bit 一起读进来了。

所以 byte-major 下最多只能做到：

```text
读完整 A8；
低位 MAC 不算或不更新；
activation traffic 基本不变。
```

这就是 `compute-mask only` 的问题。

## 2. Plane-Group-Major Activation Buffer

Graph-Bit 需要片上 activation buffer 采用 plane-group-major layout。不是零散读单个 bit，而是对一个 tile 的 bit-plane group 做连续 burst：

```text
group 0: A[7:6] for tile
group 1: A[5:4] for tile
group 2: A[3:2] for tile
group 3: A[1:0] for tile
```

推荐第一版使用 2-bit group：

```text
P8 -> 4 groups
P6 -> 3 groups
P4 -> 2 groups
```

P5 可以作为动态 stop 的软件 proxy；硬件上可以：

```text
读取到 P6 group；
在最后一个 group 内部 gate 掉 1 个 bit 的 issue/psum 更新。
```

这样比 1-bit plane 更硬件友好，也比 4-bit group 更有 early-stop 粒度。

## 3. Bit-Plane Issue Scheduler

如果 bound 满足停止条件，NPU 不能继续发空 cycle。正确逻辑是：

```text
for group in high_to_low_groups:
    fetch activation plane group
    issue PE cycles
    update psum

    if depth >= min_depth and bound < tolerance:
        stop lower groups
```

被跳过的低位 group 不会：

```text
1. 发 activation read request
2. 发 PE issue
3. 读/广播 weight RF/SRAM
4. 更新 partial sum
```

这和 “mask MAC but still issue cycle” 不同。

## 4. Weight Tile Reuse

W4 权重有两级读取：

```text
HBM/DRAM -> on-chip SRAM/RF -> PE
```

early stop 直接减少的是：

```text
低位 cycle 对应的 SRAM/RF read
低位 cycle 对应的 weight broadcast
PE input toggle
```

它不一定直接减少 HBM weight read，因为当前 tile 的高位计算仍然需要同一个 W tile。

要让 HBM weight traffic 也下降，需要 weight-stationary batching：

```text
load W tile once;
serve many same-risk node batches;
amortize W HBM traffic over more nodes.
```

因此 Graph-Bit scheduler 需要尽量形成较大的同风险 micro-batch。

## 5. Risk-Bucket Scheduler

如果 high-risk 和 low-risk 节点混在一个 bit-serial micro-batch，整个 batch 往往要跑到最深节点的 bit-depth：

```text
1 high-risk P8 node + 63 low-risk P5 nodes
    -> whole batch executes P8
```

这会抹掉 low-risk 节点的 early-stop 收益。

Graph-Bit 使用 risk bucket：

```text
Q_high -> P8 / strict bound
Q_mid  -> P6/P5 / medium bound
Q_low  -> P5/P4 / loose bound
```

调度规则：

```text
1. 优先形成同风险 batch；
2. bucket 太小时允许等待或相邻 bucket merge；
3. 离线 embedding 任务可以重排节点，因此比在线 serving 更适合 bucket batching；
4. 如果需要低延迟，可以用 subarray partition 同时跑不同 bucket。
```

## 6. Load Imbalance 处理

风险分桶会带来队列不均衡。第一版模型采用三个可调参数：

```text
batch_size:
    NPU micro-batch 大小

baseline_weight_tile_batch:
    普通调度下每个 W tile 平均服务的节点数

weight_stationary_tile_batch:
    risk-bucket + weight-stationary 下每个 W tile 平均服务的节点数
```

当某个 bucket 节点少时，有三种策略：

```text
wait-to-fill:
    离线任务中最简单，等 bucket 填满再发。

adjacent-bucket merge:
    high+mid 或 mid+low 合并，牺牲一点 depth saving 换 PE 利用率。

subarray partition:
    不同子阵列同时跑不同 bucket，控制更复杂。
```

论文主线建议先采用：

```text
offline graph embedding -> wait-to-fill + adjacent merge
```

因为图文本 embedding 通常是批处理，不是严格 token-level online serving。

## 7. 可执行模型

脚本：

```bash
bash scripts/run_graphbit_npu_dataflow_model.sh
```

默认输入：

```text
output/graphbit_predictor_free/cora_h8_54_T40_dynp5/predictor_free_workload.json
```

输出：

```text
output/graphbit_predictor_free/cora_h8_54_T40_dynp5/npu_dataflow_model/npu_dataflow_model.txt
output/graphbit_predictor_free/cora_h8_54_T40_dynp5/npu_dataflow_model/npu_dataflow_model.tsv
output/graphbit_predictor_free/cora_h8_54_T40_dynp5/npu_dataflow_model/npu_dataflow_model.json
```

PubMed replay：

```bash
WORKLOAD=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/pubmed_h8_76_T40/predictor_free_workload.json \
NODE_COUNT=19717 \
OUT_DIR=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/pubmed_h8_76_T40/npu_dataflow_model \
bash scripts/run_graphbit_npu_dataflow_model.sh
```

## 8. Conservative Cora / PubMed Result

这部分是当前更严谨的主线：**不默认给 weight HBM 任何额外 4x 约简**。

默认参数：

```text
plane_group_bits = 2
batch_size = 64
baseline_weight_tile_batch = 16
weight_stationary_tile_batch = 16
WHBM scale = 1.0
```

也就是说，下面的 `PlaneGroup risk bucket` 只声称三件事：

```text
1. plane-group activation fetch 减少 Aread；
2. bit-plane issue gating 减少 PE issue / WRF / Psum depth；
3. risk-bucket batching 防止 low-risk 节点被 high-risk 节点拖到 P8。
```

它**不声称** weight HBM 自动下降。

### 8.1 Cora h8_54_T40

输入：

```text
output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_workload.json
```

核心结果：

| Method | Layout | Schedule | PE | Aread | WHBM | WRF | Psum | FullC | FullT | FullE | Drop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 byte baseline | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.601 | 0.602 | 0.601 | 1.53% |
| ByteMajor mask only | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.601 | 0.602 | 0.601 | 2.39% |
| ByteMajor issue gate | byte | bucket | 0.725 | 1.000 | 1.000 | 0.725 | 0.725 | 0.510 | 0.602 | 0.510 | 2.39% |
| PlaneGroup random mixed | plane | random | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.601 | 0.602 | 0.610 | 2.39% |
| PlaneGroup risk bucket | plane | bucket | 0.725 | 0.725 | 1.000 | 0.725 | 0.725 | 0.486 | 0.544 | 0.486 | 2.39% |
| Random risk full NPU | plane | bucket | 0.725 | 0.725 | 1.000 | 0.725 | 0.725 | 0.486 | 0.544 | 0.486 | 2.79% |

相对 FullP8 byte baseline：

```text
PlaneGroup risk bucket:
    FullC-save = 19.2%
    FullT-save = 9.6%
    FullE-save = 19.1%
    ExtraDrop  = 0.86%
```

### 8.2 PubMed h8_76_T40

输入：

```text
output/graphbit_predictor_free/pubmed_h8_76_T40/predictor_free_workload.json
```

核心结果：

| Method | Layout | Schedule | PE | Aread | WHBM | WRF | Psum | FullC | FullT | FullE | Drop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 byte baseline | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.778 | 0.779 | 0.778 | 1.26% |
| ByteMajor mask only | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.777 | 0.778 | 0.777 | 2.54% |
| ByteMajor issue gate | byte | bucket | 0.725 | 1.000 | 1.000 | 0.725 | 0.725 | 0.660 | 0.778 | 0.660 | 2.54% |
| PlaneGroup random mixed | plane | random | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.777 | 0.778 | 0.789 | 2.54% |
| PlaneGroup risk bucket | plane | bucket | 0.725 | 0.725 | 1.000 | 0.725 | 0.725 | 0.627 | 0.703 | 0.628 | 2.54% |
| Random risk full NPU | plane | bucket | 0.725 | 0.725 | 1.000 | 0.725 | 0.725 | 0.627 | 0.703 | 0.628 | 2.75% |

相对 FullP8 byte baseline：

```text
PlaneGroup risk bucket:
    FullC-save = 19.3%
    FullT-save = 9.7%
    FullE-save = 19.2%
    ExtraDrop  = 1.28%
```

### 8.3 Bucket Realism Check

这个检查只回答“risk bucket 是否足够大、会不会严重 tail padding”，不声称额外 W HBM 下降。

Cora：

```text
P8: 325 nodes,  6 batches, tail_util 84.6%
P6: 812 nodes, 13 batches, tail_util 97.6%
P4: 487 nodes,  8 batches, tail_util 95.1%
bucket padding overhead = 1.064x
assumed W HBM scale = 1.000
```

PubMed：

```text
P8: 3056 nodes,  48 batches, tail_util 99.5%
P6: 7650 nodes, 120 batches, tail_util 99.6%
P4: 4594 nodes,  72 batches, tail_util 99.7%
bucket padding overhead = 1.004x
assumed W HBM scale = 1.000
```

结论：

```text
1. risk buckets 足够大，batch tail overhead 很小；
2. 这支持 risk-bucket batching 的可行性；
3. 但它不证明 W HBM 能自动降到 0.25。
```

## 9. Optional Weight-Stationary Sensitivity

`RiskBucket + WS sensitivity` 只用于参数扫描：

```text
baseline_weight_tile_batch = 16
weight_stationary_tile_batch = 32 / 48 / 64 / ...
```

如果设置 64，就得到：

```text
W HBM scale = 16 / 64 = 0.25
```

这不是主线结论，而是一个上界/敏感性分析。只有当后续有更具体的 scheduler / SRAM capacity / batch formation 证明 `W tile` 真的能服务 4x 更多 row，才能把它写成正式收益。

当前主线不依赖这个假设。

## 10. 设计边界

这个模型不是 RTL，也不是最终硅片能耗。它用于：

```text
1. 验证各个硬件机制是否必要；
2. 估算哪个 traffic component 是瓶颈；
3. 指导下一步 ONNXim / RTL 细化。
```

当前最重要的建模假设：

```text
1. 2-bit plane group 是片上 activation buffer layout；
2. risk-bucket batching 可以重排离线 graph embedding workload；
3. 主线 WHBM scale 固定为 1.0，不默认声称 weight HBM 下降；
4. P5 是 dynamic-depth proxy，不代表必须有 5-bit ISA；
5. 任何 WHBM < 1.0 都必须作为 sensitivity / 上界单独标注。
```

## 11. ONNXim / GemmWS 下沉实现

当前已经把组件级模型下沉到 ONNXim 的 `GemmWS` / `SystolicWS`：

```text
ONNXim/src/operations/GemmWS.cc
    1. 为 activation MOVIN 显式生成 plane-group fetch depth；
    2. 为 weight MOVIN 保留 original/actual 计数；
    3. 可选建模 weight-stationary HBM sensitivity；
    4. 给 GEMM instruction 标注 effective / fetch / issue / WRF / psum depth。

ONNXim/src/SystolicWS.cc
    1. 用 issue depth 缩放 bit-plane issue cycles；
    2. 统计 fetch / issue / WRF / psum 的平均 bit-depth；
    3. 输出 GraphBitDataflow 日志。

ONNXim/src/Core.cc
    1. 保留原 MemoryBreakdown 格式；
    2. 新增 GraphBitMemory，记录 weight actual/original request。
```

也就是说，现在不只是外部 Python proxy，而是 ONNXim 内部已经能区分：

```text
1. byte-major 只 mask，不减少 activation fetch；
2. plane-group-major demand fetch，减少 activation request；
3. bit-plane issue gating，减少 PE issue depth；
4. W RF / broadcast gating，减少片上权重广播深度；
5. psum update gating，减少低位 partial-sum 更新；
6. risk-bucket disabled 时，低风险节点被 P8 batch 拖住；
7. 可选 weight-stationary sensitivity 会降低 weight HBM actual request，但主线不默认使用。
```

### 11.1 Microbench 运行方式

```bash
SEQ_LEN=8 bash GraphhopSimhash/scripts/run_onnxim_graphbit_datapath_suite.sh
```

输出：

```text
output/onnxim_graphbit/datapath_suite_s8/datapath_summary.txt
output/onnxim_graphbit/datapath_suite_s8/datapath_summary.tsv
```

### 11.2 快速验证结果

`seq_len=8` 的 sanity-check 结果如下。这里重点看机制是否按预期改变 counter，而不是把这个小规模数值当最终论文数字：

| Case | Cycles | Act/orig | W/orig | Fetch | Issue | WRF | Psum | Meaning |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full_p8 | 37.65M | 1.000 | 1.000 | - | - | - | - | 完整 P8 baseline |
| byte_major_mask_only_p6 | 37.65M | 1.000 | 1.000 | 8.0 | 8.0 | 8.0 | 8.0 | 只 mask，基本无收益 |
| byte_major_issue_rf_psum_p6 | 37.63M | 1.000 | 1.000 | 8.0 | 6.0 | 6.0 | 6.0 | 少发 PE/WRF/Psum，但 Aread 不变 |
| plane_group2_issue_rf_psum_p6 | 37.23M | 0.750 | 1.000 | 6.0 | 6.0 | 6.0 | 6.0 | activation demand fetch 生效 |
| plane_group2_bound_low | 37.22M | 0.750 | 1.000 | 6.0 | 5.0 | 5.0 | 5.0 | predictor-free bound early stop 生效 |
| no_risk_bucket_p6 | 37.65M | 1.000 | 1.000 | 8.0 | 8.0 | 8.0 | 8.0 | 无 risk bucket 时被 P8 batch 拖住 |
| ws_sensitivity_4x_p6 | 15.09M | 0.750 | 0.250 | 6.0 | 6.0 | 6.0 | 6.0 | 4x WS sensitivity，上界/参数扫描 |

这个结果说明：

```text
1. 单纯“少算低位”不够；
2. byte-major layout 无法减少 activation fetch；
3. plane-group layout 能把 Aread 降到 6/8；
4. risk-bucket batching 是让低风险节点真的停在低 depth 的必要条件；
5. W HBM 不自动下降；`ws_sensitivity_4x_p6` 只是显式 4x WS 假设下的上界。
```

### 11.3 下一步

下一步不是再证明组件存在，而是把这个 ONNXim datapath counter 接回 full-stack workload：

```text
reuse/residual 输出 node path ratio
Graph-Bit 输出 high/mid/low miss-node ratio
ONNXim datapath suite 输出 per-path cycles/traffic
最终合成端到端 cycles / traffic / energy proxy
```

如果要进一步走 RTL，不需要一开始写完整 LLaMA encoder，只需要实现一个 `A8 x W4 GEMM tile` 微内核：

```text
bit-plane activation buffer
bit-plane sequencer
bound/stop controller
weight RF gated broadcast
psum gated update
risk-bucket queue
```
