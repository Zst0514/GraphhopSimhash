# NPU Documentation

本目录只保留当前后端主线文档。早期 prediction-free early stop、cross-row BFP、旧 full-stack replay 等探索已移到 `docs/archive/npu/`。

## Mainline

- [GRAPH_BIT_NPU_DESIGN.md](GRAPH_BIT_NPU_DESIGN.md)  
  当前 NPU 主设计入口：TSER/graph-risk-guided BFPA4/BFPA6 encoder path，以及它和 SimHash / residual reuse 的端到端关系。

- [PROGRESSIVE_BFP_ENCODER_INTERFACE.md](PROGRESSIVE_BFP_ENCODER_INTERFACE.md)  
  SimHash / Residual-Gate 前端接入 Progressive BFP encoder 的接口定义和复现实验入口。

- [PROGRESSIVE_BFP_ARRAY_DESIGN_AND_EXPERIMENTS.md](PROGRESSIVE_BFP_ARRAY_DESIGN_AND_EXPERIMENTS.md)  
  BFP 阵列本身的设计、PE/dataflow/service-window 机制，以及需要补齐的 array-level 实验路线。

- [GRAPH_AWARE_BFP_REFINEMENT_POLICY.md](GRAPH_AWARE_BFP_REFINEMENT_POLICY.md)  
  Graph risk 与真实 LLaMA activation block stress 联合指导 BFPA4/BFPA6 refinement 的策略、接口和实验流程。

- [GRAPH_AWARE_DYNAMIC_BFP_REFINEMENT_NPU.md](GRAPH_AWARE_DYNAMIC_BFP_REFINEMENT_NPU.md)  
  当前后端 NPU 主实现：BFPA4 base always compute，按 `graph risk × activation stress` 在 activation block 内动态追加 BFPA6 refinement。

- [GRAPH_AWARE_BFP_VALIDATION_PLAN.md](GRAPH_AWARE_BFP_VALIDATION_PLAN.md)  
  验证 activation stress、graph risk 以及二者联合 refinement 是否有效的分层实验设计。

- [BFP_ACTIVATION_FORMAT.md](BFP_ACTIVATION_FORMAT.md)  
  BFP activation 格式、B64/B128/B256 block size、相对普通 W4A8/W4A4 的差异。

- [GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md](GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md)  
  W-stationary systolic array 与类 FlashAttention 的 IO-aware W tile 数据流。

- [LLAMA_ROOFLINE_PROFILE.md](LLAMA_ROOFLINE_PROFILE.md)  
  LLaMA-7B projection / FFN GEMM 的 roofline 和 large-M 解释。

## Archived Explorations

这些内容不作为当前论文主贡献，只用于追溯探索过程：

```text
docs/archive/npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
docs/archive/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
docs/archive/npu/GRAPH_CONDITIONED_PREDICTION_FREE_DESIGN.md
docs/archive/npu/GRAPH_AWARE_BFP_BLOCK_PACKING.md
```
