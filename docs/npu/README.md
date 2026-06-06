# NPU Documentation

本目录只保留当前后端主线文档。早期 prediction-free early stop、cross-row BFP、roofline 拆分、接口草案和旧 full-stack replay 等探索已移到 `docs/archive/npu/`。

## Mainline

- [GRAPH_BIT_NPU_DESIGN.md](GRAPH_BIT_NPU_DESIGN.md)
  当前 NPU 主设计入口：TSER/graph-risk-guided BFPA4/BFPA6 encoder path，以及它和 SimHash / residual reuse 的端到端关系。

- [BFP_TECHNICAL_BRIEF.md](BFP_TECHNICAL_BRIEF.md)
  BFP 格式、Transformer activation scale 选择、graph-aware dynamic BFPA4-to-BFPA6 refinement 和 NPU 通路说明。

- [PROGRESSIVE_BFP_ARRAY_DESIGN_AND_EXPERIMENTS.md](PROGRESSIVE_BFP_ARRAY_DESIGN_AND_EXPERIMENTS.md)
  BFP 阵列本身的设计、PE/dataflow/service-window 机制，以及需要补齐的 array-level 实验路线。

- [GRAPH_AWARE_DYNAMIC_BFP_REFINEMENT_NPU.md](GRAPH_AWARE_DYNAMIC_BFP_REFINEMENT_NPU.md)
  当前后端 NPU 主实现：BFPA4 base always compute，按 `graph risk × activation stress` 在 activation block 内动态追加 BFPA6 refinement。

## Archived Explorations

这些内容不作为当前论文主贡献，只用于追溯探索过程：

```text
docs/archive/npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
docs/archive/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
docs/archive/npu/GRAPH_CONDITIONED_PREDICTION_FREE_DESIGN.md
docs/archive/npu/GRAPH_AWARE_BFP_BLOCK_PACKING.md
docs/archive/npu/BFP_ACTIVATION_FORMAT.md
docs/archive/npu/GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md
docs/archive/npu/LLAMA_ROOFLINE_PROFILE.md
```
