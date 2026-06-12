# Results Documentation

当前目录只保留与论文主线直接相关的结果入口。阶段性 sweep、历史 target 对齐、旧 dynamic/full-stack 拆分结果已移到 `docs/archive/results/`。

## Main Results

- [FINAL_BFP_VALIDATION_RESULT.md](FINAL_BFP_VALIDATION_RESULT.md)
  当前后端 BFP 主线的收束结果：BFPA safety boundary、graph-aware refinement、full-stack dynamic BFP。

- [UNIFIED_FRONTEND_POLICY_RESULT.md](UNIFIED_FRONTEND_POLICY_RESULT.md)
  固定在线控制流下的 dataset-level `T` policy register 收敛结果，包含 Cora/PubMed/Arxiv 当前主线参数。

- [CORA_CAM32_NODE_ORDER_RESULT.md](CORA_CAM32_NODE_ORDER_RESULT.md)
  Cora 在 `CACHE_SIZE=32` 下的节点访问顺序对比结果，包含默认、哈希、请求 METIS 三组日志和复现命令。

- [GRAPH_BIT_MAIN_RESULTS.md](GRAPH_BIT_MAIN_RESULTS.md)
  当前端到端主结果入口：front-end reuse、residual-gate、BFP miss-node path。

## Archived Results

以下结果已归档，不作为当前主线入口：

```text
docs/archive/results/GRAPH_BIT_TILE_BOUND_NUMERIC_VALIDATION.md
docs/archive/results/GRAPH_BIT_TILE_SCORE_V2_VALIDATION.md
docs/archive/results/GRAPH_BIT_VALIDATION_SUMMARY.md
docs/archive/results/GRAPH_BFP_PROGRESSIVE_REFINEMENT_RESULT.md
docs/archive/results/DYNAMIC_BFP_FULLSTACK_RESULT.md
docs/archive/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md
docs/archive/results/ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md
```
