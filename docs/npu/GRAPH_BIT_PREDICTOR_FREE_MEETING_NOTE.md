# Graph-Bit Predictor-Free 变 Bit 位宽计算说明

本文聚焦 Graph-Bit NPU 中的 predictor-free variable bit-depth execution：当节点没有被 SimHash / residual reuse 命中、必须进入 LLM encoder 计算时，NPU 不再对所有节点固定执行完整 A8 activation bit-plane，而是由图任务风险和运行时数值上界共同决定实际执行到几 bit。

核心结论：

```text
Graph risk 不直接指定最终 P8/P6/P5/P4。
Graph risk 只指定最低安全深度 min_depth 和容忍度 tolerance。
最终 stop depth 由 runtime predictor-free bound 判断。
```

这条线的意义是把图后端信息从“外部调度策略”推进到 NPU datapath 内部，让图风险控制 bit-serial GEMM 的算术努力。

## 1. 背景问题

在 LLM-as-GNN-Encoder 场景中，每个节点文本都要经过 Transformer encoder 得到 embedding。前端已经有三段式复用路径：

```text
P0: exact/direct reuse
P1: fuzzy reuse + residual correction
P2: miss node -> LLM encoder compute
```

Graph-Bit 关注的是 P2：仍然需要跑 encoder 的 miss nodes。

传统做法是：

```text
all miss nodes -> full W4A8 encoder
```

但图任务里不同节点的数值误差容忍度不同：

```text
high-degree / high-risk node:
    错误会传播到更多邻居，需要更完整的 activation precision

low-degree / low-risk node:
    传播范围小，可以更早停止低位 bit-plane
```

因此 Graph-Bit 的目标是：

```text
让 graph risk 控制 NPU 内部 activation bit-plane execution depth。
```

## 2. 为什么不是 learned predictor

这里使用 predictor-free，而不是训练一个 learned predictor，原因是：

```text
1. 不需要额外 calibration nodes
2. 不需要离线学习每层/每 token 的误差模型
3. 硬件逻辑更接近 PADE/BETA 这类 bound-based early termination
4. 更容易解释：停止依据来自剩余低位 bit-plane 的理论上界
```

区别在于：

```text
普通 predictor:
    预测哪些计算重要
    预测错了可能误剪枝

predictor-free bound:
    已经执行高位 bit-plane
    计算剩余低位 bit-plane 最大可能贡献
    如果剩余贡献不足以显著改变结果，则停止
```

## 3. 执行流程

整体流程如下：

```text
node text + graph
        |
        v
SimHash / CAM retrieval
        |
        +-- exact / high-confidence hit -> direct reuse
        |
        +-- fuzzy hit -> residual correction
        |
        +-- miss -> Graph-Bit NPU
                    |
                    v
             graph risk bucket
                    |
        --------------------------------
        | high-risk | mid-risk | low-risk |
        --------------------------------
                    |
                    v
          bit-serial W4 x A8 GEMM
                    |
                    v
     execute high bit-plane first, then check bound
                    |
        -----------------------------
        | bound <= tolerance | else |
        | stop              | continue lower bits
        -----------------------------
```

## 4. Risk 到 Min-Depth / Tolerance

当前主线优先使用 Degree，因为实验里 Degree 对量化误差传播更稳定。TSER / Context 可以作为对照。

概念上：

```text
deg_q(v) = quantile_bucket(log(1 + degree(v)))  # 0..15

if high-risk:
    min_depth = 8
    tolerance = 0.00

elif mid-risk:
    min_depth = 6
    tolerance = 0.02

else:
    min_depth = 4
    tolerance = 0.04
```

这里 `min_depth` 是安全下限，不是最终深度。真正最终深度由 runtime bound 决定。

例如：

```text
mid-risk:
    至少执行到 A6
    如果 A6 后剩余低位上界已经足够小，则停在 A6
    否则继续到 A7/A8

low-risk:
    至少执行到 A4
    如果 A4 后 bound 不满足，可以继续到 A5/A6/...
```

因此 Graph-Bit 不是静态地把节点分到 P8/P6/P4，而是：

```text
risk bucket -> min_depth + tolerance
runtime bound -> actual stop depth
```

## 5. Predictor-Free Bound

bit-serial GEMM 中，activation A8 可以看成 8 个 bit-plane：

```text
A8 = b7 b6 b5 b4 b3 b2 b1 b0
```

高位对数值影响大，低位对数值影响小。执行到某个 depth 后，未执行低位的最大贡献可以被上界约束：

```text
remaining_low_bit_bound(depth)
  ~= remaining_activation_low_bits(depth)
    * weight_abs_bound
    * tile_scale
    * quant_scale
```

其中：

```text
remaining_activation_low_bits(depth):
    未执行低位 bit-plane 的最大可能数值贡献

weight_abs_bound:
    当前 W tile 的权重绝对值上界或统计界

tile_scale:
    K tile 长度、累加范围和量化 scale 相关的归一化因子

quant_scale:
    activation / weight scale 对最终 partial sum 的影响
```

停止条件：

```text
if remaining_low_bit_bound(depth) <= tolerance:
    stop
else:
    continue lower bit-plane
```

这个判断不依赖 learned predictor，也不需要知道 FP reference embedding。

## 6. NPU Datapath 需要什么

Graph-Bit 对 NPU 内部提出以下模块：

```text
1. Risk-bucket scheduler
   将 miss nodes 按 high/mid/low risk 聚成 batch。

2. Weight-stationary tile buffer
   同一个 W tile 服务同 bucket 中尽量多的 token rows。

3. Bit-plane issue controller
   控制 activation bit-plane 从高位到低位逐步发射。

4. Predictor-free bound estimator
   计算剩余低位贡献上界，并与 tolerance 比较。

5. Psum / RF / broadcast gating
   停止后不再执行低位 PE issue、W_RF broadcast 和 psum update。
```

当前主线中，最重要的收益来源有两个：

```text
W-stationary risk-bucket batching:
    减少 W tile reload / 摊薄 W HBM cost

variable activation bit-depth:
    减少 PE bit-plane issue、W_RF broadcast、A_RF access、psum update
```

activation plane-group buffer 可以进一步减少 activation demand fetch，但它不是当前主线必须依赖的唯一收益点。

## 7. 为什么 M 很关键

Transformer Linear 的 GEMM row 数不是节点数，而是：

```text
M = batch_nodes * padded_sequence_length
```

例如：

```text
batch_nodes = 4, seq_len = 512
M = 2048

batch_nodes = 32, seq_len = 2048
M = 65536
```

TAPE / DyLGNN 这类真实 LLM+GNN 前端中，M 通常是几千到几万：

```text
TAPE-like:
    M ≈ 2k 到 18k

DyLGNN-like:
    M ≈ 65k
```

因此早期 `M=16/32/64` microbench 会低估 mixed-depth 的价值。真实 encoder batch 下，同一个 W tile 会服务大量 token rows，compute path 更容易暴露，A-depth 减少更容易转化为 cycles saving。

## 8. Bound 开销不能忽略

理想情况下：

```text
A8 -> A6: PE bit-plane compute 约减少 25.0%
A8 -> A5: PE bit-plane compute 约减少 37.5%
A8 -> A4: PE bit-plane compute 约减少 50.0%
```

但完整 predictor-free datapath 需要额外开销：

```text
T_compute =
    T_bitserial(depth)
  + T_bound_estimator_visible
  + T_bound_control

T_total = max(T_compute, T_memory)
```

当前模型显式计入：

```text
bound_ops:
    remaining-bit bound estimation and compare

bound_control_cycles:
    per-tile stop-check control and issue decision

bound_overlap:
    bound estimator 与 GEMM bit-plane execution 的可重叠比例
```

默认设置：

```text
bound_ops_per_output = 8
bound_tops = 16 TOPS
bound_overlap = 0.5
bound_control_cycles_per_check = 4
```

在 `M >= 2048` 下，净收益为：

| Stop Depth | Ideal Compute Save | Net Cycle Save With Bound |
|---|---:|---:|
| A6 | 25.0% | about 23.6% |
| A5 | 37.5% | about 34.8% |
| A4 | 50.0% | about 48.6% |

如果 bound datapath 很弱：

```text
bound_ops_per_output = 16
bound_tops = 4 TOPS
bound_overlap = 0
control_cycles_per_check = 8
```

则净收益会下降到：

| Stop Depth | Net Cycle Save, Pessimistic Bound |
|---|---:|
| A6 | about 12.6% |
| A5 | about 12.8% |
| A4 | about 37.6% |

这个结果说明：Graph-Bit 的收益不仅取决于少算多少 bit，还取决于 bound estimator 是否足够轻、是否能和 bit-plane GEMM 重叠。

## 9. 当前实验结果如何解读

当前 roofline/activity 模型输出位置：

```text
output/graphbit_internal_roofline/tape_dylgnn_bound_overhead/
output/graphbit_internal_roofline/tape_dylgnn_bound_pessimistic/
```

默认 bound overhead 下，`M=2048..65536` 的 layer-level 结果稳定为：

```text
P8 -> P6: 23.6% net cycle saving
P8 -> P5: 34.8% net cycle saving
P8 -> P4: 48.6% net cycle saving
```

Per-stage 看：

```text
Q/K/V/O projection:
    roughly 33% of layer GEMM cycles
    P6 net save about 23.4%

FFN gate/up:
    roughly 45% of layer GEMM cycles
    P6 net save about 23.4%

FFN down:
    roughly 22% of layer GEMM cycles
    P6 net save about 24.4%
```

这说明 Graph-Bit 不是只优化 FFN。QKV/O projection 也会受到 activation bit-depth 的影响。

## 10. 和 HEAT / 静态 Degree 精度路由的区别

HEAT 类路线可以概括为：

```text
degree -> precision routing
```

也就是用 degree 指导不同节点走不同计算精度。

Graph-Bit 的区别是：

```text
degree / risk 不直接决定最终 bit-depth
degree / risk 决定 min_depth + tolerance
runtime predictor-free bound 决定最终 stop depth
```

同时，Graph-Bit 和前端复用形成 full stack：

```text
reuse hit:
    不进入 encoder

fuzzy hit:
    residual correction

miss node:
    Graph-Bit NPU predictor-free variable bit-depth
```

因此它不是单独的 degree-guided quantization，而是：

```text
graph-aware reuse + residual correction + predictor-free bit-serial encoder execution
```

## 11. 复现实验命令

默认 bound overhead：

```bash
cd /home/zhangshangtong/Transformer/OFA

python GraphhopSimhash/scripts/model_graphbit_internal_roofline.py \
  --batch-nodes 4 8 16 32 \
  --seq-lens 512 2048 \
  --activation-hbm-mode byte_major \
  --output-dir output/graphbit_internal_roofline/tape_dylgnn_bound_overhead
```

悲观 bound overhead：

```bash
python GraphhopSimhash/scripts/model_graphbit_internal_roofline.py \
  --batch-nodes 4 8 16 32 \
  --seq-lens 512 2048 \
  --activation-hbm-mode byte_major \
  --bound-ops-per-output 16 \
  --bound-tops 4 \
  --bound-overlap 0 \
  --bound-control-cycles-per-check 8 \
  --output-dir output/graphbit_internal_roofline/tape_dylgnn_bound_pessimistic
```

输出文件：

```text
graphbit_internal_roofline.txt
graphbit_internal_roofline.tsv
graphbit_internal_roofline.json
```

## 12. 汇报时建议强调的主线

可以按下面逻辑讲：

```text
1. LLM+GNN encoder 的真实 M 是 batch_nodes * seq_len，通常是几千到几万。

2. 当 M 足够大，同一个 W tile 被大量 token rows 复用，compute path 暴露。

3. Graph-Bit 不训练 predictor，而是用 graph risk 设置 min_depth/tolerance。

4. 真正是否停，由 runtime remaining-bit bound 判断。

5. 变 bit 位宽不仅减少 PE MAC，也减少 RF/broadcast/psum activity。

6. bound estimator 有开销，必须显式建模。

7. 默认 bound datapath 下，P8->P6 仍有约 23.6% net cycle saving。

8. 如果 bound datapath 很弱，收益会明显下降，因此硬件设计重点是轻量 bound estimator + overlap。
```

## 13. 当前风险和下一步

当前仍需继续强化的部分：

```text
1. 将 roofline/activity model 与 trace-driven workload replay 更紧密结合。

2. 对真实 Cora/PubMed/Arxiv miss-node depth histogram 加权，而不是只看单一 P6/P5/P4。

3. 更细粒度建模 bound estimator 的 tile-level 实现：
       per-output bound
       per-vector bound
       per-tile bound

4. 对比 byte-major 和 plane-group activation buffer：
       byte-major: 主要省 PE/RF/psum
       plane-group: 进一步省 activation demand fetch，但实现复杂

5. 结合 W-stationary risk-bucket scheduler，报告 full-stack cycles / traffic / energy。
```

当前可汇报的稳健结论：

```text
Graph-Bit mixed-depth 不是简单静态 P8/P6/P4。
它是 graph-risk-conditioned predictor-free early stop。

在真实 LLM+GNN encoder 的大 M 条件下，
扣除 bound overhead 后，activation bit-depth reduction 仍能带来明显 compute-cycle saving。

bound estimator 设计是关键：
轻量且可重叠时收益接近理想 bit-plane saving；
过重时收益会被控制开销吃掉。
```
