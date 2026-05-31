# Graph-Bit NPU Internal Roofline

本文档专门回答一个 NPU 内部问题：

```text
当 miss node 必须进入 LLaMA encoder 时，
Graph-Bit 在 Q/K/V/O projection 和 FFN GEMM 里到底减少了哪些计算/访存？
```

它和 full-stack GNN accuracy 表不同。这里不讨论分类准确率，只拆 NPU 内部的 GEMM row count、W HBM、activation HBM、PE bit-plane compute、RF/broadcast、psum update。

## 1. 为什么要重新建模 M

Transformer linear GEMM 的真实 row count 是：

```text
M = batch_nodes * padded_sequence_length
```

例如：

```text
batch_nodes = 4
seq_len = 512
M = 2048
```

每个 token row 都是一个 `4096` 维 hidden vector，和同一个 Linear 权重矩阵相乘：

```text
X: [M, 4096]
W: [4096, N]
Y: [M, N]
```

因此不能只用 `M=8/16/64` 的小 microbench 推断真实 LLaMA encoder。小 M 会夸大 W load 的影响；真实 `batch_nodes * seq_len` 足够大时，W tile 被更多 token row 复用，compute path 会显现。

## 2. 覆盖的 LLaMA GEMM

当前模型覆盖每层主要 Linear GEMM：

```text
Q/K/V/O projection:
    shape = [M, 4096] x [4096, 4096]
    count = 4

FFN gate/up:
    shape = [M, 4096] x [4096, 11008]
    count = 2

FFN down:
    shape = [M, 11008] x [11008, 4096]
    count = 1
```

默认假设 Q/K/V 输入读取可以融合：

```text
Q/K/V 共享一次 input activation read
O projection 读取 attention output
=> QKVO activation input read count = 2
```

如果要测试不融合，可以使用 `--no-qkv-fused`。

## 3. Graph-Bit 在 NPU 内部省什么

权重固定为 W4，activation depth 可变：

```text
P8: W4A8
P6: W4A6
P5: W4A5
P4: W4A4
```

对 bit-serial datapath：

```text
bit-serial compute ∝ M * K * N * W_bits * A_depth
```

因此理论 PE bit-plane compute reduction 是：

```text
A8 -> A6: 25.0%
A8 -> A5: 37.5%
A8 -> A4: 50.0%
```

但 latency 是否同幅下降取决于 roofline：

```text
T = max(T_HBM, T_PE_bitserial)
```

如果 `T_HBM` 占主导，A-depth 减少主要体现为片上 activity / energy；如果 `T_PE_bitserial` 占主导，A-depth 才直接变成 latency reduction。

## 4. 两种 activation 读取模式

### 4.1 byte-major

外部 activation 仍以普通 A8 byte 读取：

```text
activation HBM read = A8
```

即使运行时只执行到 P6/P5，低位 bit 已经被读入。此时 Graph-Bit 主要减少：

```text
PE bit-plane issue
A_RF access
W_RF broadcast
psum update
```

不明显减少 activation HBM。

### 4.2 plane-group

activation 在 NPU 内部或 layer-fused buffer 中按 bit-plane group 组织：

```text
group0: b7 b6
group1: b5 b4
group2: b3 b2
group3: b1 b0
```

如果 runtime bound 在 P5/P6 停止，则后续低位 group 不再 demand fetch。此时 Graph-Bit 同时减少：

```text
activation HBM / SRAM fetch
PE issue
A_RF / W_RF
psum update
```

这一路径收益更强，但实现复杂度也更高。当前更保守的主线是先报告 `byte_major`，把 `plane_group` 作为增强设计。

## 5. 复现命令

默认 sweep：

```bash
cd /home/zhangshangtong/Transformer/OFA
bash GraphhopSimhash/scripts/run_graphbit_internal_roofline.sh
```

输出：

```text
output/graphbit_internal_roofline/byte_major/graphbit_internal_roofline.txt
output/graphbit_internal_roofline/byte_major/graphbit_internal_roofline.tsv
output/graphbit_internal_roofline/plane_group/graphbit_internal_roofline.txt
output/graphbit_internal_roofline/plane_group/graphbit_internal_roofline.tsv
```

关键参数：

```bash
BATCH_NODES="1 2 4 8 16 32"
SEQ_LENS="128 256 512"
bash GraphhopSimhash/scripts/run_graphbit_internal_roofline.sh
```

也可以直接跑脚本：

```bash
python GraphhopSimhash/scripts/model_graphbit_internal_roofline.py \
  --batch-nodes 1 2 4 8 16 32 \
  --seq-lens 128 256 512 \
  --activation-hbm-mode byte_major \
  --output-dir output/graphbit_internal_roofline/byte_major
```

## 6. 当前默认结果

默认参数：

```text
weight = W4
output = 16 bit
peak = 131.1 TOPS
memory bandwidth = 614.4 GB/s
QKV input read fused = true
```

### 6.1 Layer total

`byte_major` 下的 layer total：

| M | 对应 batch/seq 示例 | P8 Bound | P8 W% | P8 A% | P6 Cycle Save | P5 Cycle Save | P4 Cycle Save |
|---:|---|---|---:|---:|---:|---:|---:|
| 128 | B1xS128 | compute | 87.6% | 3.0% | 4.8% | 4.8% | 4.8% |
| 256 | B1xS256 / B2xS128 | compute | 77.9% | 5.4% | 25.0% | 37.5% | 46.5% |
| 512 | B1xS512 / B2xS256 / B4xS128 | compute | 63.7% | 8.8% | 25.0% | 37.5% | 50.0% |
| 1024 | B2xS512 / B4xS256 / B8xS128 | compute | 46.8% | 13.0% | 25.0% | 37.5% | 50.0% |
| 2048 | B4xS512 / B8xS256 / B16xS128 | compute | 30.5% | 16.9% | 25.0% | 37.5% | 50.0% |
| 4096 | B8xS512 / B16xS256 / B32xS128 | compute | 18.0% | 20.0% | 25.0% | 37.5% | 50.0% |

关键含义：

```text
在真实 W4 datapath 和 M=batch*seq_len 下，
QKV/O + FFN 并不是一直 W-memory-bound。

当 M >= 256 后，A-depth reduction 可以直接转成明显 cycle reduction。
```

### 6.2 Per-stage 结论

P8 下每层 cycle share 稳定约为：

| Stage | Cycle Share |
|---|---:|
| Q/K/V/O projection | 33% |
| FFN gate/up | 45% |
| FFN down | 22% |

也就是说 Q/K/V/O generation 不是小项，约占每层三分之一。Graph-Bit 对 QKV/O 的影响不能忽略。

在 `M=2048` 时：

| Stage | W% | A% | P6 Cycle Save | P6 Activity Save |
|---|---:|---:|---:|---:|
| Q/K/V/O projection | 28.6% | 14.3% | 25.0% | 25.0% |
| FFN gate/up | 29.7% | 11.0% | 25.0% | 25.0% |
| FFN down | 36.4% | 36.4% | 25.0% | 25.0% |

这说明当 bucket 让 `M` 到达几千 token rows，A-depth 对 QKV/O 和 FFN 都能显著影响 compute cycles。

## 7. 高峰值 NPU sensitivity

如果把 P8 等效峰值提高到 `524.4 TOPS`，小 M 会重新变成 memory-bound：

| M | Bound | P6 Cycle Save |
|---:|---|---:|
| 128 | memory | 0.0% |
| 256 | memory | 0.0% |
| 512 | memory | 0.0% |
| 1024 | compute-exposed | 10.9% |
| 2048 | compute | 25.0% |

这说明：

```text
Graph-Bit mixed-depth 的 latency 收益依赖两个条件：
1. W4 权重已经被压缩/驻留/复用；
2. risk bucket 形成足够大的 M，让 compute path 暴露出来。
```

如果 NPU 算力极强而 batch 很小，mixed-depth 主要仍是 energy / activity 优化。

## 8. 对 Graph-Bit 设计的影响

这份模型给出的更准确设计边界是：

```text
reuse/residual:
    先减少进入 encoder 的节点。

risk-bucket scheduler:
    把 miss nodes 聚成同风险 bucket，增大有效 M。

bit-serial early stop:
    在足够大的 M 下，A-depth reduction 直接减少 QKV/O 和 FFN latency；
    在小 M 下，主要减少 PE/RF/psum activity。
```

因此后续论文里应避免只说：

```text
低 degree 节点用低 bit，所以更快。
```

更准确的说法是：

```text
graph risk controls both batching locality and arithmetic depth.
The batching locality exposes compute by amortizing W4 tiles;
the arithmetic-depth control then reduces bit-serial PE/RF/psum work.
```
