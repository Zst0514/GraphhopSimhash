# GraphHopSimhash 文档入口

本文档是 `docs/` 的入口。当前文档按用途分组，避免所有 `.md` 平铺在根目录。

## Core

- [SCORE_DEFINITIONS.md](core/SCORE_DEFINITIONS.md): TSER / Degree / graph context / low-unique 分数定义。
- [CAM设计.md](core/CAM设计.md): 8-head HD-CAM、support 聚合、direct / residual / compute 三段式复用。
- [RESIDUAL_CORRECTED_REUSE.md](core/RESIDUAL_CORRECTED_REUSE.md): fuzzy hit residual correction 思路、参数和结果。
- [AWQ_W4A8_W4A4_GENERATION.md](core/AWQ_W4A8_W4A4_GENERATION.md): AWQ-based embedding pool 生成方式。

## NPU

- [GRAPH_BIT_NPU_DESIGN.md](npu/GRAPH_BIT_NPU_DESIGN.md): Graph-Bit NPU 主线设计，包含 datapath、scheduler、buffer、cost model。
- [GRAPH_CONDITIONED_BIT_SERIAL_EXECUTION.md](npu/GRAPH_CONDITIONED_BIT_SERIAL_EXECUTION.md): Graph-conditioned predictor-free bit-serial execution，包含风险定义、阈值管理和验证接口。
- [HIERARCHICAL_ENCODER_NPU_DESIGN.md](npu/HIERARCHICAL_ENCODER_NPU_DESIGN.md): P0/P1/P2/P3 分层 encoder 执行框架。
- [GRAPH_AWARE_ENCODER_NPU_PROPOSAL.md](npu/GRAPH_AWARE_ENCODER_NPU_PROPOSAL.md): Graph-aware encoder NPU 方案探索。
- [NPU_ADAPTIVE_ENCODER_EXPERIMENTS.md](npu/NPU_ADAPTIVE_ENCODER_EXPERIMENTS.md): NPU 实验路线。
- [PRECISION_DEPTH_ABLATION.md](npu/PRECISION_DEPTH_ABLATION.md): precision-depth ablation。
- [FFN_CHANNEL_GATING.md](npu/FFN_CHANNEL_GATING.md): FFN gating 原型，当前作为辅助消融，不是主线。

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
