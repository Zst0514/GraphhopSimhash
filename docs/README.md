# GraphHopSimhash 文档入口

本文档是 `docs/` 的入口。当前文档按用途分组，避免所有 `.md` 平铺在根目录。

## Core

- [SCORE_DEFINITIONS.md](core/SCORE_DEFINITIONS.md): TSER / Degree / graph context / low-unique 分数定义。
- [CAM设计.md](core/CAM设计.md): 8-head HD-CAM、support 聚合、direct / residual / compute 三段式复用。
- [RESIDUAL_CORRECTED_REUSE.md](core/RESIDUAL_CORRECTED_REUSE.md): fuzzy hit residual correction 思路、参数和结果。
- [AWQ_W4A8_W4A4_GENERATION.md](core/AWQ_W4A8_W4A4_GENERATION.md): AWQ-based embedding pool 生成方式。

## NPU

- [GRAPH_BIT_NPU_DESIGN.md](npu/GRAPH_BIT_NPU_DESIGN.md): Graph-Bit NPU 主线设计，包含 datapath、scheduler、buffer、cost model。
- [GRAPH_BIT_END_TO_END_THEORY.md](npu/GRAPH_BIT_END_TO_END_THEORY.md): Graph-Bit 端到端算法、prediction-free bound、变 bit-depth 计算、W tile reuse 和 speedup 数值推导。
- [GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md](npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md): bit-plane early stop 从 CLI、runner、ONNXim/GemmWS 到 trace replay 的代码级实现说明。
- [GRAPH_BIT_TRACE_DRIVEN_SCHEDULER.md](npu/GRAPH_BIT_TRACE_DRIVEN_SCHEDULER.md): per-node trace 导出、bucket scheduler replay、真实 W tile load / Wscale 统计。
- [GRAPH_BIT_PROXY_EXPERIMENTS.md](npu/GRAPH_BIT_PROXY_EXPERIMENTS.md): Graph-Bit 的 embedding-pool proxy 实验、precision-depth ablation 和命令。

已删除/合并早期探索文档：partial-depth / token budget / FFN channel gating，以及 Graph-Bit 的 demand-fetch、bound、bucket、dataflow 分散文档。主线内容已合并到 [GRAPH_BIT_NPU_DESIGN.md](npu/GRAPH_BIT_NPU_DESIGN.md)、[GRAPH_BIT_END_TO_END_THEORY.md](npu/GRAPH_BIT_END_TO_END_THEORY.md)、[GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md](npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md) 和 [GRAPH_BIT_TRACE_DRIVEN_SCHEDULER.md](npu/GRAPH_BIT_TRACE_DRIVEN_SCHEDULER.md)。

## Results

- [GRAPH_BIT_VALIDATION_SUMMARY.md](results/GRAPH_BIT_VALIDATION_SUMMARY.md): Graph-Bit 在 Cora / PubMed / Arxiv 上的主线结果。

## Survey

- [LLM_ACCELERATOR_SURVEY.md](survey/LLM_ACCELERATOR_SURVEY.md): Encoder / 通用 Transformer / NPU 加速器综述。
- [LLM_DECODER_ACCELERATOR_SURVEY.md](survey/LLM_DECODER_ACCELERATOR_SURVEY.md): Decoder / serving / KV-cache 加速器综述。

## Tools

- [ONNXIM_PROJECT_GUIDE.md](tools/ONNXIM_PROJECT_GUIDE.md): ONNXim 项目功能、模块和仿真流程说明。
- [量化+哈希命令.md](tools/量化+哈希命令.md): 常用量化与哈希实验命令。

## Roadmap

- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md): 当前项目主线、output 管理原则和下一步硬件仿真实现计划。
