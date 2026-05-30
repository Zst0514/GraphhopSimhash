# GraphHopSimhash 文档入口

本文档是 `docs/` 的入口。当前文档按用途分组，避免所有 `.md` 平铺在根目录。

## Core

- [SCORE_DEFINITIONS.md](core/SCORE_DEFINITIONS.md): TSER / Degree / graph context / low-unique 分数定义。
- [CAM设计.md](core/CAM设计.md): 8-head HD-CAM、support 聚合、direct / residual / compute 三段式复用。
- [RESIDUAL_CORRECTED_REUSE.md](core/RESIDUAL_CORRECTED_REUSE.md): fuzzy hit residual correction 思路、参数和结果。
- [AWQ_W4A8_W4A4_GENERATION.md](core/AWQ_W4A8_W4A4_GENERATION.md): AWQ-based embedding pool 生成方式。

## NPU

- [GRAPH_BIT_NPU_DESIGN.md](npu/GRAPH_BIT_NPU_DESIGN.md): Graph-Bit NPU 主线设计，包含 datapath、scheduler、buffer、cost model。
- [GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md](npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md): bit-plane early stop 从 CLI、runner、ONNXim/GemmWS 到 trace replay 的代码级实现说明。
- [GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md](npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md): Graph-Bit full-stack 复现实验流程、调参入口、输出文件和结果解读。

历史 proxy、早期理论拆分和旧 scheduler 说明已移到 `docs/archive/npu/`。主线只维护上面三份文档。

## Results

- [GRAPH_BIT_MAIN_RESULTS.md](results/GRAPH_BIT_MAIN_RESULTS.md): 当前 Graph-Bit / residual reuse 主线结果。
- [SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md](results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md): Cora/PubMed 共享在线 residual reuse 配置。
- [ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md](results/ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md): ST/LLaMA 共享检索骨架的历史对照。

## Survey

- [LLM_ACCELERATOR_SURVEY.md](survey/LLM_ACCELERATOR_SURVEY.md): Encoder / 通用 Transformer / NPU 加速器综述。
- [LLM_DECODER_ACCELERATOR_SURVEY.md](survey/LLM_DECODER_ACCELERATOR_SURVEY.md): Decoder / serving / KV-cache 加速器综述。

## Tools

- [ONNXIM_PROJECT_GUIDE.md](../ONNXim/ONNXIM_PROJECT_GUIDE.md): ONNXim 项目功能、模块和仿真流程说明。
- [量化+哈希命令.md](tools/量化+哈希命令.md): 常用量化与哈希实验命令。

## Archive

- `archive/npu/`: Graph-Bit 早期 proxy、旧理论拆分和旧 scheduler 说明。
- `archive/results/`: 过往大规模 sweep 汇总，保留用于追溯，不作为主入口。

## Roadmap

- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md): 当前项目主线、output 管理原则和下一步硬件仿真实现计划。
- [PROJECT_TODO.md](PROJECT_TODO.md): 当前缺口、下一步实验优先级、可选增强和不建议继续投入的方向。
