# ONNXim 项目说明与仿真流程

本文档记录当前仓库中 ONNXim 的核心功能、主要模块、输入输出格式，以及如何用它跑端到端 NPU 仿真和 Linear/GEMM microbenchmark。

## 1. ONNXim 是什么

ONNXim 是一个 cycle-level multi-core NPU simulator。它的目标不是训练模型，也不是执行真实推理结果，而是读取 ONNX 图或内置语言模型描述，估计 DNN/Transformer 在给定 NPU 架构、片上 SRAM、NoC 和 DRAM 配置下的执行周期、阵列利用率、访存请求和带宽利用率。

它适合回答的问题包括：

- 一个 GEMM/Conv/Attention 在某个 systolic array 上需要多少 cycle；
- 多核 NPU 下不同 core 的负载是否均衡；
- 给定 DDR/HBM 配置后，算子是 compute-bound 还是 memory-bound；
- 矩阵 shape 与 array shape 是否匹配，PE utilization 是否被浪费；
- 不同 precision 字节数、SRAM 容量、tile mapping 对性能和访存的影响。

## 2. 顶层输入

ONNXim 一次仿真通常需要两个输入：

1. Hardware config：

```bash
--config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json
```

2. Model list：

```bash
--model ./example/models_list.json
```

model list 指向一个或多个模型目录，每个模型目录中包含同名 ONNX 文件。例如：

```json
{
  "models": [
    {
      "name": "gemm_64_4096_4096",
      "batch_size": 1,
      "request_time": 0
    }
  ]
}
```

对应文件路径为：

```text
models/gemm_64_4096_4096/gemm_64_4096_4096.onnx
```

## 3. 硬件配置文件

`configs/*.json` 描述 NPU 架构和系统参数。关键字段如下。

### 3.1 Core 和阵列

```json
"num_cores": 4,
"core_config": {
  "core_0": {
    "core_type": "systolic_ws",
    "core_width": 128,
    "core_height": 128,
    "spad_size": 32768,
    "accum_spad_size": 4096
  }
}
```

含义：

- `num_cores`：NPU core 数量；
- `core_type`：核心数据流，目前常用 `systolic_ws`，即 weight-stationary systolic array；
- `core_width/core_height`：阵列形状；
- `spad_size`：输入/权重 scratchpad；
- `accum_spad_size`：partial sum / output accumulator SRAM。

### 3.2 DRAM

```json
"dram_type": "ramulator2",
"dram_channels": 32,
"dram_req_size": 32,
"dram_config_path": "../configs/ramulator2_configs/HBM2.yaml"
```

ONNXim 可以接：

- simple memory model；
- Ramulator；
- Ramulator2。

因此输出中会包含 row hit/miss/conflict、read/write request、memory cycles 等统计。

### 3.3 NoC

```json
"icnt_type": "simple"
```

也可以接 BookSim2。多核下，NoC 会影响 core 和 memory 间的数据搬运延迟。

### 3.4 Precision

```json
"precision": 2
```

这里表示每个 tensor element 的字节数，而不是 ONNX 文件里的真实 dtype。比如：

- `precision=1` 可近似 INT8 / A8；
- `precision=2` 可近似 FP16 / BF16 / 16-bit 数据；
- `precision=4` 可近似 FP32。

在我们的 microbenchmark 中，ONNX 文件只承担 shape carrier 的角色；实际访存字节数由 config 的 `precision` 决定。

## 4. 主要源码模块

### 4.1 Simulator 主流程

核心文件：

```text
src/main.cc
src/Simulator.cc
src/Model.cc
src/Core.cc
```

职责：

- 解析命令行；
- 读取 hardware config；
- 读取 ONNX model list；
- 构建模型图；
- 为每个 operation 创建对应 simulator；
- 调度 core 执行；
- 收集 cycle、utilization、DRAM/NoC 统计。

### 4.2 Operation 层

路径：

```text
src/operations/
```

典型算子：

- `GemmWS.cc` / `GemmOS.cc`：GEMM systolic simulation；
- `ConvWS.cc` / `ConvOS.cc`：Conv simulation；
- `Attention.cc`：Attention simulation；
- `Softmax.cc`、`BiasGelu.cc`、`SkipLayerNorm.cc`：Transformer 常见非 GEMM 算子；
- `OperationFactory.cc`：根据 ONNX op type 创建对应 operation。

其中 Linear 在 ONNX 中通常会变成 `MatMul` 或 `Gemm`，最终由 GEMM 类处理。

### 4.3 Systolic Array

核心文件：

```text
src/SystolicWS.cc
src/SystolicOS.cc
```

职责：

- 模拟 weight-stationary / output-stationary 阵列；
- 统计 systolic instruction issue count；
- 统计 preload 次数；
- 统计 array utilization 和 PE utilization；
- 根据 tile mapping 推进 cycle。

### 4.4 Mapping

核心文件：

```text
src/Mapping.cc
src/Mapping.h
```

ONNXim 支持层级 tiling。若模型目录中没有手写 `.mapping` 文件，则使用默认 mapping 策略。手写 mapping 可控制：

```text
[T] total loop
[O] outer loop
[I] inner tile loop
```

这决定每次装入 scratchpad 的 input/weight/output tile 大小。

### 4.5 Memory 和 Interconnect

核心文件：

```text
src/Dram.cc
src/Sram.cc
src/Interconnect.cc
```

外部依赖：

```text
extern/ramulator_custom/
extern/ramulator2/
extern/booksim/
```

职责：

- 将 tensor tile 访问转成 memory request；
- 模拟 SRAM/DRAM 访问延迟；
- 统计 DRAM bandwidth utilization；
- 多核场景下模拟 core-memory 网络传输。

### 4.6 Language Model 模式

路径：

```text
src/models/LanguageModel.cc
src/scheduler/
```

ONNXim 还支持非 ONNX 图的 LLM serving/custom format，用 `--mode language` 启动。这部分更偏向 decoder serving、iteration-level batching 和 request scheduling。当前我们做 Graph-Bit / encoder GEMM microbenchmark 时，主要使用 ONNX 图路径。

## 5. 端到端仿真流程

标准 ONNX 路径如下：

```text
PyTorch/TensorFlow model
        |
        v
Export ONNX graph
        |
        v
models/<name>/<name>.onnx
        |
        v
model_lists/*.json
        |
        v
Simulator loads ONNX graph
        |
        v
OperationFactory creates op simulators
        |
        v
Mapping decides tile loops
        |
        v
Core executes systolic/vector ops
        |
        v
SRAM / NoC / DRAM requests
        |
        v
cycle, utilization, bandwidth report
```

运行命令示例：

```bash
cd /home/zhangshangtong/Transformer/OFA/ONNXim

./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json \
  --model ./model_lists/gemm_64_4096_4096.json
```

## 6. 当前 Linear/GEMM Microbenchmark

当前已生成一个 Linear/GEMM benchmark：

```text
Input:  (64, 4096)
Weight: (4096, 4096)
Output: (64, 4096)
Ops:    MatMul
FLOPs:  2.147 GFLOPs
```

文件：

```text
models/gemm_64_4096_4096/gemm_64_4096_4096.onnx
model_lists/gemm_64_4096_4096.json
```

已测配置：

1. `systolic_ws_8x8_c1_simple_noc_transformer.json`

```text
1 core, 8x8 array, precision=1
MatMul finish: 16,988,384 cycles
Simulated time: 16988 us
Systolic array utilization: 98.84%
PE utilization: 98.76%
DDR4 BW utilization: 14%
```

2. `systolic_ws_128x128_c4_simple_noc_tpuv4.json`

```text
4 cores, 128x128 array, precision=2
MatMul finish: 110,581 cycles
Simulated time: 110 us
Core array utilization: 23.26% / 34.88% / 34.88% / 31.01%
PE utilization: 11.11% - 16.67%
HBM2 BW utilization: 58%
```

解释：

- 8x8 小阵列几乎被喂满，compute utilization 很高；
- 128x128 四核配置更快，但这个 GEMM 的 `N=64` 无法充分填满 128 维阵列，因此 PE utilization 低；
- 这说明 Graph-Bit 或 LLM encoder cost model 不能只看 FLOPs，还要看 batch/sequence/channel shape 与 array tile 的匹配程度。

## 7. 和 Graph-Bit 方案的关系

Graph-Bit 的核心是：当节点必须执行 LLM encoder 时，NPU 不必对所有节点使用同样的 activation bit-depth。图任务风险可以控制 bit-serial / bit-grained GEMM 的执行深度：

```text
high-risk nodes   -> P8 / P6
medium-risk nodes -> P6 / P5
low-risk nodes    -> P5 / P4
```

在 ONNXim 层面，可以用两步近似验证：

1. 用不同 `precision` 或不同 GEMM cost 参数估计 P8/P6/P5/P4 的 execution cost；
2. 在 GraphhopSimhash 中用真实生成的 P8/P6/P5/P4 embedding pool 评估 downstream accuracy/drop。

ONNXim 负责回答硬件侧问题：

```text
每个 bit-depth 对 GEMM cycle / bandwidth / array utilization 的影响是多少？
```

GraphhopSimhash 负责回答任务侧问题：

```text
哪些节点可以安全少算 bit-plane？
图风险路由是否比 random 更能保护精度？
```

二者合起来形成 Graph-Bit 的完整验证链路。

## 8. 当前本地构建状态

当前环境下 `Simulator` target 已成功构建：

```text
build/bin/Simulator
```

完整 `make -j` 会在测试依赖 gtest 上遇到 warning-as-error，但不影响 `Simulator` 使用。为兼容当前 gcc，已在：

```text
extern/ramulator_custom/src/StatType.h
```

补充了：

```cpp
#include <cstdint>
```

