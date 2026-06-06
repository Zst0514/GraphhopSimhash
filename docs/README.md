# GraphHopSimhash Documentation

当前论文主线：

```text
Graph-aware LLM encoder execution for text-attributed graph inference.

1. SimHash/LRU-CAM:
   找可复用 anchor，减少 encoder 调用次数。

2. TSER-guided residual reuse:
   用图传播风险控制 fuzzy reuse，并用轻量 adapter 修正中风险 fuzzy hit。

3. TSER / graph-risk-guided BFP encoder:
   对 reject / miss nodes，用 BFPA4/BFPA6 refinement 降低剩余 encoder 成本。
```

## Core Algorithm

- [SCORE_DEFINITIONS.md](core/SCORE_DEFINITIONS.md)
  TSER / Degree / graph context / low-unique 分数定义。TSER 是前端 reuse 和后端 BFP refinement 的共享风险信号。

- [CAM设计.md](core/CAM设计.md)
  Multi-head SimHash、LRU/HD-CAM、support 聚合和 direct / residual / compute 三段式复用。

- [RESIDUAL_CORRECTED_REUSE.md](core/RESIDUAL_CORRECTED_REUSE.md)
  residual adapter / accept gate 的实现、训练目标、参数和结果。

- [AWQ_W4A8_W4A4_GENERATION.md](core/AWQ_W4A8_W4A4_GENERATION.md)
  AWQ / real quant embedding pool 生成方式。

## NPU

- [npu/GRAPH_BIT_NPU_DESIGN.md](npu/GRAPH_BIT_NPU_DESIGN.md)
  当前后端主设计：BFPA4 低成本底座，高风险节点提升到 BFPA6/BFPA8。

- [npu/GRAPH_AWARE_DYNAMIC_BFP_REFINEMENT_NPU.md](npu/GRAPH_AWARE_DYNAMIC_BFP_REFINEMENT_NPU.md)
  Graph risk × activation stress 的 dynamic BFPA4-to-BFPA6 refinement NPU 实现。

- [npu/PROGRESSIVE_BFP_ARRAY_DESIGN_AND_EXPERIMENTS.md](npu/PROGRESSIVE_BFP_ARRAY_DESIGN_AND_EXPERIMENTS.md)
  BFP 阵列、PE 数据通路、service-window 和 array-level 实验设计。

## Results

- [results/FINAL_BFP_VALIDATION_RESULT.md](results/FINAL_BFP_VALIDATION_RESULT.md)
  BFPA safety boundary、graph-aware refinement 和 full-stack dynamic BFP 收束结果。

- [results/UNIFIED_FRONTEND_POLICY_RESULT.md](results/UNIFIED_FRONTEND_POLICY_RESULT.md)
  Cora/PubMed/Arxiv 的统一前端 policy register。

- [results/GRAPH_BIT_MAIN_RESULTS.md](results/GRAPH_BIT_MAIN_RESULTS.md)
  当前端到端主结果入口。

## Survey

- [survey/LLM_ACCELERATOR_SURVEY.md](survey/LLM_ACCELERATOR_SURVEY.md)
  Encoder / 通用 Transformer / NPU 加速器综述。

- [survey/LLM_DECODER_ACCELERATOR_SURVEY.md](survey/LLM_DECODER_ACCELERATOR_SURVEY.md)
  Decoder / serving / KV-cache 加速器综述。

## Tools

- [ONNXIM_PROJECT_GUIDE.md](../ONNXim/ONNXIM_PROJECT_GUIDE.md)
  ONNXim 项目功能、模块和仿真流程说明。

- [量化+哈希命令.md](tools/量化+哈希命令.md)
  常用量化与哈希实验命令。

## Archive

`docs/archive/` 保留历史探索，不作为当前主线入口：

```text
partial-depth encoder
token compaction
FFN channel gating
prediction-free early stop
cross-row BFP block packing
old Graph-Bit trace replay
```

## Planning

- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)
  当前论文逻辑、实验路线和文档组织。

- [PROJECT_TODO.md](PROJECT_TODO.md)
  当前缺口和下一步实验。
