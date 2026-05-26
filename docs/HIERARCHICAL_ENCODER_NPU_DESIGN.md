# Graph-Aware Hierarchical Encoder NPU Design

本文档整理当前 GraphhopSimhash 的完整系统思路：不是只做 hash reuse，也不是只做 W4A8/W4A4 量化，而是面向 graph-text workload 设计一套分层 encoder 执行机制。

核心目标：

```text
对每个图节点，不默认运行完整 LLM/ST encoder；
先用 graph/hash 风险信息判断它应该走哪条执行路径。
```

当前推荐的系统主线是：

```text
P0: exact hash reuse
P1: fuzzy hash reuse + residual correction
P2: W4A8 encoder + FFN channel gating
P3: full W4A8 encoder
```

其中 P0/P1 减少 encoder 调用次数，P2/P3 解决必须执行 encoder 的节点该如何在 NPU 上更便宜地运行。

## 1. Why Not Degree-Guided Quantization as Main Point

固定预算 W4A8/W4A4 量化路由中，实验显示 Degree/propagation risk 比 TSERTopK 更稳定。原因是量化掉点主要由两项决定：

```text
1. 节点自身量化误差
2. 量化误差沿图传播的范围
```

第 1 项在线不可得；第 2 项最直接的 deployable proxy 就是 degree / propagation risk。因此 degree-guided quant routing 更适合作为 baseline 或硬件友好的路由策略，而不是论文的主创新。

当前更有价值的主线是：

```text
Graph-aware hierarchical encoder execution
```

也就是用图结构、hash 命中质量和 TSER 风险来决定节点是否需要运行 encoder，以及运行多重的 encoder 路径。

## 2. Execution Paths

### P0: Exact Reuse

当 CAM/SimHash 找到 exact hit 时：

```text
E_hat(v) = E(u)
cost ~= 0
```

这条路径只读 embedding cache，不运行 encoder。它非常便宜，也最安全；当前实验显示 exact hit 不适合再强行 residual correction，因为会引入额外扰动。

### P1: Fuzzy Reuse + Residual Correction

当 CAM/SimHash 找到 fuzzy hit 且 TSER gate 认为风险可接受时：

```text
E_hat(v) = normalize(E(u) + alpha * R(z_vu))
```

其中：

```text
u      = CAM/SimHash 找到的锚点节点
E(u)   = 已缓存的高质量锚点 embedding
z_vu   = cheap feature delta + graph context delta + hash/support statistics
R(.)   = low-rank residual adapter
```

重点：residual adapter 不是从 SimHash bits 还原 embedding。SimHash 只负责定位相似锚点；adapter 在锚点 embedding 的基础上做轻量修正。

Cora/ST 纯 reuse sweep 中，TSER `3/1/1` 下的结果：

| T | Direct Reuse | Direct Drop | Residual Reuse | Residual Drop |
|---:|---:|---:|---:|---:|
| 20 | 4.5% | 0.24% | 4.5% | 0.21% |
| 30 | 44.1% | 2.76% | 44.1% | 2.43% |
| 45 | 48.3% | 3.84% | 48.3% | 3.16% |
| 60 | 49.2% | 3.92% | 49.2% | 3.27% |
| 90 | 59.8% | 5.82% | 59.8% | 4.16% |

结论：residual correction 的价值不是单点大幅救回精度，而是在相同 reuse 率下把 reuse-drop 曲线整体下移。

### P2: W4A8 + FFN Channel Gating

对没有被 reuse 接收、但风险较低的节点，仍然运行 encoder，但只在 FFN 中保留一部分 channel group：

```text
h = FFN_up(x)
h = activation(h)
h = h * channel_group_mask
out = FFN_down(h)
```

`FFN75` 表示每层 FFN 中间通道保留 75%，跳过 25%。硬件上可对应为：

```text
不读被 mask channel 的权重
不做对应 MAC
减少 FFN activation/weight traffic
```

第一版选择 FFN channel gating，而不是 attention tile skipping，原因是：

```text
1. FFN 占 Transformer encoder 计算和权重流量的大头；
2. channel group mask 对硬件更规则；
3. 不改变 attention softmax 的精确语义，精度风险更可控；
4. 适合和 graph-aware scheduler 结合。
```

Cora/ST 初步结果显示，全图统一 gating 不成立，但图感知路由有效：

| Config | Cost | Drop |
|---|---:|---:|
| FullW4A8 | 0.500 | 0.08% |
| Uniform FFN75 | 0.419 | 6.06% |
| Uniform FFN50 | 0.338 | 9.75% |
| TSER20_FFN75 | 0.484 | 0.44% |
| Degree20_FFN75 | 0.484 | 0.52% |
| TSER40_FFN75 | 0.468 | 1.08% |
| Degree60_FFN75 | 0.451 | 1.82% |
| Random60_FFN75 | 0.451 | 3.16% |

结论：FFN channel gating 不能全图静态开启，必须由 scheduler 选择低风险节点使用。

### P3: Full W4A8 Encoder

高风险节点、reuse miss/reject 节点、以及不适合 FFN gating 的节点走完整 W4A8 encoder：

```text
E_hat(v) = W4A8Encoder(text_v)
```

这是系统的精度兜底路径。当前更推荐使用已经验证过的 `W4A8_PTQ_TEST` 作为 ST 实验的 full W4A8 路径；它相对 FP16 的误差明显低于早期裸 `W4A8` 生成方式。

## 3. Scheduler Inputs

当前 scheduler 可使用以下在线可得信号：

```text
hash hit type:
    exact / fuzzy / miss

candidate confidence:
    Hamming distance
    route support
    base support
    cosine proxy

graph risk:
    propagation_q
    graph_context_q
    low_degree_unique_q
    sensitivity_q
```

建议的路径分配逻辑：

```text
exact hit:
    P0 direct reuse

accepted fuzzy hit:
    P1 residual-corrected reuse

hash reject/miss + low graph risk:
    P2 W4A8 + FFN channel gating

hash reject/miss + high graph risk:
    P3 full W4A8 encoder
```

这里的阈值不是论文核心。论文核心应该是：

```text
同一套 graph/hash metadata 被用于分层执行调度；
不同路径对应不同硬件成本和精度风险；
系统在精度约束下自动减少 full encoder 调用和 FFN 计算。
```

## 4. NPU Hardware View

### 4.1 Front-End Router

Router 维护每个节点的轻量 metadata：

```text
node id
hash code
bucket id
candidate source id
hit type
TSER risk
route support counters
```

输出为节点执行路径：

```text
P0 / P1 / P2 / P3
```

### 4.2 Embedding Cache + CAM

P0/P1 使用 embedding cache：

```text
P0:
    read E(u)

P1:
    read E(u)
    run residual adapter
```

CAM/SimHash 的作用不是生成 embedding，而是快速定位锚点并给出命中置信信息。

### 4.3 Residual Engine

Residual engine 是一个小型 low-rank GEMM/vector unit：

```text
R(z) = W_up * GELU(W_down * z)
```

它的开销远小于完整 encoder。以 Cora/ST 为例：

```text
input dim ~= 1545
rank      = 32
emb dim   = 768
MAC       ~= 74K / corrected node
```

LLaMA-7B embedding dim 为 4096，adapter MAC 会增加，但相对完整 LLaMA encoder 仍然很小。

### 4.4 FFN-Gated W4A8 Array

P2 使用同一个 W4A8 encoder array，但在 FFN 阶段加载 channel-group mask：

```text
Full path:
    read all FFN channel weights
    compute all FFN channels

Gated path:
    read selected channel-group weights
    compute selected channels
```

硬件需要支持：

```text
channel-group mask buffer
grouped FFN weight fetch
grouped activation writeback
batch-level path scheduling
```

这比 token truncation 更像 NPU/array 级别的设计点，因为它直接减少 FFN MAC 和权重流量。

## 5. Current End-to-End Prototype

当前 Cora/ST hierarchical encoder 脚本：

```bash
bash GraphhopSimhash/run_cora_st_hierarchical_encoder.sh
```

默认设置：

```text
runs                    = 10
reference               = FP16
full encoder path        = W4A8_PTQ_TEST
gated encoder path       = W4A8_FFN75
TSER reuse gate          = 3/1/1, T=30
gated route policy       = TSER
gated route ratio        = 20%
router supervision       = data_x
```

当前 10-run 结果：

| Config | Reuse | Direct | Residual | FFN | Full | Cost | Acc | Drop | AvgErr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullW4A8 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.500 | 0.6779 | 0.40% | 0.00676 |
| DirectReuse | 37.1% | 37.1% | 0.0% | 0.0% | 62.9% | 0.315 | 0.6422 | 3.97% | 0.15671 |
| ResidualReuse | 37.1% | 4.8% | 32.2% | 0.0% | 62.9% | 0.316 | 0.6450 | 3.69% | 0.13641 |
| TserFFNGatingOnly | 0.0% | 0.0% | 0.0% | 12.6% | 87.4% | 0.490 | 0.6730 | 0.89% | 0.01874 |
| FullHierarchy | 37.1% | 4.8% | 32.2% | 12.6% | 50.3% | 0.306 | 0.6402 | 4.17% | 0.14839 |

这个结果说明：

```text
1. W4A8_PTQ_TEST 作为 full W4A8 路径已经接近 FP16。
2. residual reuse 相比 direct reuse 降低了 embedding error 和 drop。
3. FFN gating 单独作为低风险路径可用，但需要和 full W4A8 同源 backend 后再做最终端到端比较。
4. 当前 FullHierarchy 仍是结构验证版本；gated pool 来自 FFN75 路径，和 W4A8_PTQ_TEST full pool 不是完全同一生成 backend。
```

因此当前更稳的论文表述是：

```text
P0/P1 reuse hierarchy 已验证；
P2 FFN gating 作为 NPU 执行路径已初步验证；
P0/P1/P2/P3 的全链路组合仍需要同源 W4A8 backend 下的最终确认。
```

## 6. Reproduce Commands

生成 Cora/ST FP16 reference：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs fp16 \
  --batch_size 64 \
  --overwrite
```

生成旧 PTQ 风格的 Cora/ST `W4A8_PTQ_TEST` full encoder pool：

```bash
python -m GraphhopSimhash.generate_real_quant_pools_ptq_legacy \
  --datasets cora \
  --llm_name ST \
  --configs W4A8 \
  --batch_size 64 \
  --w4a_backend ptq \
  --w4a_calib_samples 256 \
  --calibration_strategy random \
  --seed 42 \
  --ptq_group_size 64 \
  --ptq_sample_rows 128 \
  --ptq_smooth_grid 0.0 0.25 0.5 \
  --ptq_clip_grid 1.0 0.999 \
  --ptq_output_clip_percentile 0.999 \
  --ptq_output_clip_multiplier 4.0 \
  --ptq_align_output \
  --tag_suffix PTQ_TEST \
  --overwrite
```

生成 Cora/ST FFN75 gated pool：

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs W4A8 \
  --batch_size 128 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --ffn_channel_gating \
  --ffn_gate_keep_ratio 0.75 \
  --ffn_gate_group_size 64 \
  --ffn_gate_calib_samples 256 \
  --ffn_gate_calibration_strategy random \
  --tag_suffix FFN75 \
  --overwrite
```

运行 hierarchical encoder 验证：

```bash
bash GraphhopSimhash/run_cora_st_hierarchical_encoder.sh
```

只看 reuse/residual，不混入 FFN gating：

```bash
GATED_ROUTE_RATIO=0 bash GraphhopSimhash/run_cora_st_hierarchical_encoder.sh
```

## 7. Next Steps

最重要的后续验证：

```text
1. 生成同源 backend 的 W4A8_FFN75 pool，让 FullW4A8 和 FFN75 只差 channel gating。
2. 在 PubMed/ST 上复现实验，验证 FFN gating 是否仍然只适合低风险节点。
3. 在 LLaMA-7B 上评估 residual adapter rank / calibration size。
4. 用 profiling 替换当前 cost model 中的 attn_weight / ffn_weight。
5. 报告硬件指标：MAC reduction、weight traffic、activation traffic、SRAM mask storage、array utilization。
```

最终论文可以落成一句话：

```text
GraphHopSimhash turns graph/text metadata into an encoder execution hierarchy:
zero-cost exact reuse, low-cost corrected fuzzy reuse, graph-aware FFN-gated W4A8 execution, and full W4A8 fallback.
```
