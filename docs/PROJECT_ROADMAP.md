# GraphHopSimhash Project Roadmap

本文档用于把当前项目主线、文档组织、output 管理和下一步硬件仿真实现计划放到一个地方。

## 1. 当前主线

当前论文故事不要再散成“hash、量化、打分、各种小实验”。更稳的主线是：

```text
Graph-conditioned hierarchical LLM encoder execution for graph-text workloads.
```

也就是：不是所有节点都应该完整跑 LLM encoder。图后端风险决定节点走哪条路径：

```text
P0: exact hash reuse
    直接读 embedding cache，cost 约为 0

P1: fuzzy hash reuse + residual correction
    CAM 找 anchor，low-rank adapter 修正 fuzzy hit

P2: Graph-Bit precision-depth encoder
    必须跑 encoder，但 Degree / TSER 等图风险控制 P8/P6/P4 bit-plane 深度

P3: full W4A8 encoder
    高风险兜底路径
```

当前最适合作为固定前端参数的配置：

```text
R = 2
8 heads x 16 bits
score threshold T = 40
hard direct: support >= 5
residual soft: support == 4
compute: support <= 3
```

这个参数来自 Cora / PubMed 共同参数探索，目标是在两个数据集上都保持低于约 3% drop，同时保留尽量高的 reuse。

## 2. 文档组织

文档现在分为：

```text
docs/core/
    核心算法：score、CAM、residual、AWQ embedding

docs/npu/
    NPU 设计：Graph-Bit、hierarchical encoder、实验路线

docs/results/
    主线结果汇总

docs/survey/
    LLM / Transformer 加速器综述

docs/tools/
    ONNXim 和命令说明

docs/figures/
    图片
```

根目录只保留 `README.md`，详细入口见 `docs/README.md`。

## 3. Output 管理原则

`output/` 只保留当前主线会继续引用的实验：

```text
output/residual_reuse/common_param_sweep_20260528
output/residual_precision_depth
output/residual_graphbit_head_threshold_sweep
output/residual_graphbit_three_depth_probe
output/llama7b_precision_depth_budget_sweep
output/graph_bit_validation
```

旧实验归档到：

```text
output/_archive_misc_20260528
```

归档不是删除。保留的原因是：旧实验可能仍有追溯价值，但不应该继续污染主线阅读。

后续新实验建议使用清晰目录：

```text
output/onnxim_graphbit/
output/graphbit_fullstack/
output/residual_reuse/
```

不要再用纯时间戳目录作为主结果目录。

## 4. 下一步硬件仿真要写什么

目标不是简单跑 ONNXim，而是搭一个能回答 Graph-Bit 是否真正节省 NPU 内部计算和访存的仿真流程。

### Step 1: 导出 Graph-Bit workload profile

从现有 `residual_precision_depth` 结果中导出每个 dataset/config 的节点路由统计：

```json
{
  "dataset": "cora",
  "model": "llama2_7b",
  "reuse": {
    "direct_ratio": 0.21,
    "residual_ratio": 0.18
  },
  "encoder": {
    "p8_ratio": 0.12,
    "p6_ratio": 0.30,
    "p4_ratio": 0.18
  },
  "baseline": {
    "full_p8_ratio": 1.0
  }
}
```

这一步只做 workload 汇总，不碰 ONNXim 内部。

### Step 2: 生成 LLaMA encoder GEMM microbenchmarks

为 LLaMA-7B encoder path 建立代表性 GEMM：

```text
Q/K/V projection
O projection
FFN gate/up/down projection
```

第一版不需要完整 ONNX LLaMA 图，只需要覆盖主要 GEMM shape，方便用 ONNXim 估计计算周期和访存。

### Step 3: 跑 Full P8 baseline

用 ONNXim 跑完整 P8 GEMM，得到：

```text
cycles
DRAM read/write
SRAM traffic
PE utilization
```

这是所有 Graph-Bit 节省比例的分母。

### Step 4: Graph-Bit postprocess cost model

在 ONNXim baseline 上叠加 Graph-Bit bit-depth 模型：

```text
P8: execute 8 activation bit-planes
P6: execute 6 activation bit-planes
P4: execute 4 activation bit-planes
```

估算：

```text
compute_cycles ~= baseline_cycles * active_bitplanes / 8
activation_traffic ~= baseline_activation_traffic * active_bitplanes / 8
weight_traffic ~= unchanged W4 path
```

这一步是最快能落地的版本，能把当前算法结果转换成硬件收益。

### Step 5: Predictor-free bounded early termination

第二版再加入更像 PADE / BETA 的 predictor-free early termination：

```text
for each bit-plane:
    update partial sum
    estimate remaining low-bit bound
    if bound < graph_tolerance:
        stop remaining bit-planes
```

新增 NPU 组件：

```text
bit-plane sequencer
partial-sum scoreboard
remaining-bound estimator
graph tolerance register
early-stop mask generator
```

其中 `graph_tolerance` 来自 Degree / TSER 风险：

```text
high-risk node -> strict tolerance -> P8/P6
medium-risk    -> medium tolerance -> P6
low-risk       -> loose tolerance  -> P4
```

这一步比固定 P8/P6/P4 更像真实硬件论文，因为它把图风险接入到了 GEMM datapath 的 runtime early termination。

### Step 6: ONNXim 内部扩展

如果前面结果成立，再改 ONNXim 内部：

```text
GraphBitGemm
GraphBitBitSerialArray
GraphBitScheduler
GraphBitBufferModel
```

第一版不急着改 C++ 核心。先用 wrapper + postprocess 确认收益边界，再决定是否深改。

## 5. 主表应该报告什么

硬件仿真主表不要只报 accuracy。建议固定报告：

```text
Dataset
Policy
Reuse %
Residual %
P8/P6/P4 ratio
Accuracy drop
Normalized cycles
Normalized DRAM traffic
Normalized activation traffic
Estimated energy proxy
```

关键对比：

```text
Full P8
Reuse + Full P8
Reuse + Random Graph-Bit
Reuse + Degree Graph-Bit
Reuse + TSER Graph-Bit
Reuse + Context Graph-Bit
```

这样能回答两个问题：

1. reuse front-end 省了多少 encoder 调用？
2. 对剩下必须跑 encoder 的节点，Graph-Bit 是否比 random 更会分配 bit-depth？

## 6. 当前不作为主线的方向

下面这些实验可以保留为探索，但不建议继续当主贡献：

```text
partial-depth hidden state as final embedding
    已验证 naive early-depth 效果差。

token compaction / token budget routing
    容易变成输入预处理工程，且引入额外策略复杂度。

learned predictor / oracle damage routing
    需要 calibration 或 reference error，不适合作为 deployable 主策略。

FFN channel gating
    可作为辅助硬件消融，但不如 Graph-Bit 深入 datapath。
```

## 7. 最近要完成的代码任务

当前已经完成第一版 wrapper flow：

```text
scripts/export_graphbit_workload.py
   从 residual_precision_depth / precision_depth 输出中提取 workload profile JSON。

scripts/onnxim_graphbit_microbench.py
   生成或调用 ONNX GEMM microbenchmark，并整理 ONNXim baseline 指标。

scripts/summarize_onnxim_graphbit.py
   把 workload profile 和 ONNXim baseline 合成 Graph-Bit cycles/traffic 表。

scripts/run_onnxim_graphbit_sim.sh
   一键执行上面三步。

output/onnxim_graphbit/
   保存 profile、ONNXim log、summary table。
```

这套流程让项目从“算法模拟”推进到“Graph-conditioned bit-serial NPU proxy 仿真”。

下一步代码优先级：

```text
1. 在 ONNXim 内部新增 GraphBitGemm / GraphBitGemmWS。
2. 将 P8/P6/P4 从后处理 scaling 下沉到 GEMM tile execution。
3. 加入 bit-plane sequencer、partial-sum scoreboard、remaining-bound estimator。
4. 让每个 node-batch 的 graph_tolerance 控制 low-bit early termination。
```
