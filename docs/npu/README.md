# NPU Documentation

This folder keeps the current Graph-Bit NPU documents.

## Main Design

- [GRAPH_BIT_NPU_DESIGN.md](GRAPH_BIT_NPU_DESIGN.md)  
  Main design entry: SimHash/LRU-CAM, Residual-Gate, Graph-Bit NPU, W-stationary dataflow.

- [GRAPH_CONDITIONED_PREDICTION_FREE_DESIGN.md](GRAPH_CONDITIONED_PREDICTION_FREE_DESIGN.md)
  Graph-conditioned prediction-free design space: node risk, W-tile bound, partial-sum guard, risk-bucket scheduling, and implementation priority.

- [GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md](GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md)  
  Systolic-array and FlashAttention-style IO-aware W tile service-window design.

- [GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md](GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md)  
  Code-level path for predictor-free stop-depth, CLI options, runner flow, ONNXim/replay outputs.

- [BFP_ACTIVATION_FORMAT.md](BFP_ACTIVATION_FORMAT.md)
  BFP activation format, B64/B128/B256 block-size meaning, W4A8 comparison, and Cora/LLaMA-7B block sweep.

## Reproduction And Profiling

- [GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md](GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md)  
  End-to-end reproduction guide for front-end route profile, stop-depth trace, scheduler replay, and activity breakdown.

- [LLAMA_ROOFLINE_PROFILE.md](LLAMA_ROOFLINE_PROFILE.md)  
  LLaMA projection/FFN roofline profile and large-M GEMM interpretation.

Historical NPU notes are in:

```text
docs/archive/npu/
```
