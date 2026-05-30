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

## 8. Cora Result

Default:

```text
frontend = h8_54_T40_dynp5
plane_group_bits = 2
batch_size = 64
baseline_weight_tile_batch = 16
weight_stationary_tile_batch = 64
```

核心结果：

| Method | Layout | Schedule | PE | Aread | WHBM | WRF | Psum | FullC | FullT | FullE | Drop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 byte baseline | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.712 | 0.712 | 0.712 | 1.08% |
| ByteMajor mask only | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.711 | 0.711 | 0.711 | 1.93% |
| ByteMajor issue gate | byte | bucket | 0.762 | 1.000 | 1.000 | 0.762 | 0.762 | 0.618 | 0.711 | 0.618 | 1.93% |
| PlaneGroup random mixed | plane | random | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.711 | 0.711 | 0.721 | 1.93% |
| PlaneGroup risk bucket | plane | bucket | 0.762 | 0.800 | 1.000 | 0.762 | 0.762 | 0.597 | 0.662 | 0.600 | 1.93% |
| RiskBucket + WS | plane | bucket | 0.762 | 0.800 | 0.250 | 0.762 | 0.762 | 0.490 | 0.395 | 0.494 | 1.93% |
| Random risk full NPU | plane | bucket | 0.762 | 0.800 | 0.250 | 0.762 | 0.762 | 0.490 | 0.395 | 0.494 | 2.30% |

解读：

```text
ByteMajor mask only:
    几乎没有系统收益，说明只 mask MAC 不够。

ByteMajor issue gate:
    cycles/energy 降了，但 traffic 几乎没降，因为 A8 仍完整读取。

PlaneGroup random mixed:
    有 plane layout 但 batch 被 P8 节点拖回 full depth，没有收益。

PlaneGroup risk bucket:
    最小可行 Graph-Bit NPU。它真正减少 Aread、PE issue、WRF、Psum。

RiskBucket + WS:
    加入 weight-stationary tile reuse 后，WHBM 从 1.0 降到 0.25，
    这是让 full-stack traffic 明显下降的关键。

Random risk full NPU:
    同样硬件下随机风险分配 drop 更高，说明 graph risk 不是摆设。
```

## 9. PubMed Result

输入：

```text
frontend = pubmed_h8_76_T40
```

核心结果：

| Method | Layout | Schedule | PE | Aread | WHBM | WRF | Psum | FullC | FullT | FullE | Drop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 byte baseline | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.778 | 0.779 | 0.778 | 1.26% |
| ByteMajor mask only | byte | bucket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.777 | 0.778 | 0.777 | 2.54% |
| ByteMajor issue gate | byte | bucket | 0.725 | 1.000 | 1.000 | 0.725 | 0.725 | 0.660 | 0.778 | 0.660 | 2.54% |
| PlaneGroup risk bucket | plane | bucket | 0.725 | 0.725 | 1.000 | 0.725 | 0.725 | 0.627 | 0.703 | 0.628 | 2.54% |
| RiskBucket + WS | plane | bucket | 0.725 | 0.725 | 0.250 | 0.725 | 0.725 | 0.511 | 0.412 | 0.512 | 2.54% |
| Random risk full NPU | plane | bucket | 0.725 | 0.725 | 0.250 | 0.725 | 0.725 | 0.511 | 0.412 | 0.512 | 2.75% |

PubMed 和 Cora 的趋势一致：

```text
1. byte-major 不能省 activation traffic；
2. plane-group risk bucket 是最低可行 datapath；
3. weight-stationary tile reuse 是降低 weight HBM traffic 的关键；
4. random risk 在同样硬件成本下精度更差。
```

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
2. WHBM 降低来自 weight-stationary tile reuse；
3. RiskBucket + WS 的收益依赖足够大的同风险 batch；
4. P5 是 dynamic-depth proxy，不代表必须有 5-bit ISA。
```

## 11. 下一步

下一步应该把该模型中的三个机制接入更底层仿真：

```text
1. ONNXim GemmWS 中显式模拟 plane-group activation fetch；
2. 在 GemmWS tile loop 中加入 bit-plane issue gating 和 psum gating；
3. 在 model-level scheduler 中模拟 risk-bucket batching 与 W tile reuse。
```

如果要进一步走 RTL，不需要一开始写完整 LLaMA encoder，只需要实现：

```text
bit-plane activation buffer
bit-plane sequencer
bound/stop controller
weight RF gated broadcast
psum gated update
risk-bucket queue
```

先做一个 `A8 x W4 GEMM tile` 微内核就够支撑硬件机制验证。
