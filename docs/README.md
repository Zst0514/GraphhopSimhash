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

- [npu/BFP_ACTIVATION_FORMAT.md](npu/BFP_ACTIVATION_FORMAT.md)  
  BFP activation 格式、block size 和相对普通 A4/A8 的差异。

- [npu/GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md](npu/GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md)  
  W-stationary systolic array 和类 FlashAttention 的 IO-aware W tile 数据流。

- [npu/LLAMA_ROOFLINE_PROFILE.md](npu/LLAMA_ROOFLINE_PROFILE.md)  
  LLaMA projection / FFN GEMM 的 roofline profile。

## Results

- [results/GRAPH_BFP_PROGRESSIVE_REFINEMENT_RESULT.md](results/GRAPH_BFP_PROGRESSIVE_REFINEMENT_RESULT.md)  
  Cora/PubMed LLaMA-7B BFPA4/BFPA6/BFPA8 progressive refinement 主结果。

- [results/RESIDUAL_GATE_GRAPHBIT_NPU_PROGRESS.md](results/RESIDUAL_GATE_GRAPHBIT_NPU_PROGRESS.md)  
  Residual-Gate 和 BFP NPU 技术进展说明。

- [results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md](results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md)  
  Cora/PubMed 共享在线 residual reuse 配置。

- [results/ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md](results/ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md)  
  ST/LLaMA T31 shared retrieval 的历史对照和 target 对齐说明。

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
