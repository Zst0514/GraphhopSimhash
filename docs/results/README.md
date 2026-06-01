# Results Documentation

This folder keeps current result summaries. Long historical sweeps are under `docs/archive/results/`.

## Current

- [GRAPH_BIT_MAIN_RESULTS.md](GRAPH_BIT_MAIN_RESULTS.md)  
  Main residual/Graph-Bit result table and trace-replay interpretation.

- [RESIDUAL_GATE_GRAPHBIT_NPU_PROGRESS.md](RESIDUAL_GATE_GRAPHBIT_NPU_PROGRESS.md)  
  Technical progress note covering Residual-Gate and Graph-Bit NPU.

- [SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md](SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md)  
  Shared online residual-gate configuration and Cora/PubMed ST result.

- [ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md](ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md)  
  ST/LLaMA T31 shared retrieval result and backend-specific target notes.

## Scope

Current result docs separate three questions:

```text
1. How much front-end reuse is safe?
2. How much miss-node activity does Graph-Bit reduce?
3. How much W tile service-window reuse comes from risk-bucket scheduling?
```
