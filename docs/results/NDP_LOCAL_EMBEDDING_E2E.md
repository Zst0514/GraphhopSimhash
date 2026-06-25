# NDP-Local Embedding Store End-to-End Timing

## Architecture Contract

- NPU memory is reserved for streaming LLM weights and activations.
- The NDP local DRAM stores graph CSR indices and generated/reused node embeddings.
- CAM/LRU entries store compact SimHash signatures and pointers into the NDP embedding store.
- CAM hits read full embeddings from NDP local DRAM and bypass the NPU.
- CAM misses invoke the NPU encoder, then write the new embedding one-way into NDP local DRAM.
- Backend GNN aggregation reads graph indices and neighbor embeddings in the NDP memory domain.

## Configuration

- NPU clock: `500.0 MHz`; NDP clock: `500.0 MHz`.
- NDP local DRAM bandwidth: `256.0 GB/s`.
- NPU-to-NDP embedding write bandwidth: `64.0 GB/s`.
- Embedding: `4096` x `16`b = `8.0 KiB/node`.
- Graph index: `32`b CSR indices.
- Neighbor embedding read factor: `1.0` per edge.
- CAM cycles: search `1.0` + select `1.0` + miss update `1.0`.
- GNN compute proxy: `0.01` x full W4BFPA4 encoder cycles.
- BFPLift source: Available BFPLift traces are used as-is; Arxiv currently uses graphstress10, while Cora/PubMed/WikiCS use graphstress20.

## Aggregate Timing

| Policy | Reuse | Drop | Lift | Eff Bits | NPU Encoder | BFPLift Enc Extra | NPU->NDP Write | CAM/LRU | Hit Emb Read | Graph Index | Neighbor Emb Read | GNN Proxy | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoReuse+W4BFPA4 | 0.00% | 0.00% | 0.00% | 4.000 | 298424.081s | 0.000ns | 28.914ms | 0.000ns | 0.000ns | 26.845us | 47.749ms | 2984.241s | 301408.399s |
| TSER40+W4BFPA4 | 40.03% | 1.39% | 0.00% | 4.000 | 179132.403s | 0.000ns | 17.378ms | 1.175ms | 2.884ms | 26.845us | 47.749ms | 2984.241s | 182116.713s |
| TSER40+BFPLift | 40.03% | 1.39% | 18.33% | 4.367 | 192408.557s | 13276.154s | 17.378ms | 1.175ms | 2.884ms | 26.845us | 47.749ms | 2984.241s | 195392.867s |

## Per-Task TSER40 Breakdown

| Task | Policy | Reuse | Drop | BFP Tag | Lift | Eff Bits | NPU Encoder | BFPLift Enc Extra | NPU->NDP Write | CAM/LRU | Hit Emb Read | Graph Index | Neighbor Emb Read | GNN Proxy | Total |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | TSER40+W4BFPA4 | 39.90% | 0.98% | `W4BFPA4` | 0.00% | 4.000 | 1677.454s | 0.000ns | 208.321us | 14.087us | 34.576us | 207.266ns | 337.792us | 27.911s | 1705.366s |
| CN | TSER40+BFPLift | 39.90% | 0.98% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 1866.169s | 188.715s | 208.321us | 14.087us | 34.576us | 207.266ns | 337.792us | 27.911s | 1894.081s |
| CL | TSER40+W4BFPA4 | 39.46% | 1.59% | `W4BFPA4` | 0.00% | 4.000 | 1689.735s | 0.000ns | 209.846us | 14.111us | 34.194us | 207.266ns | 337.792us | 27.911s | 1717.647s |
| CL | TSER40+BFPLift | 39.46% | 1.59% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 1879.831s | 190.096s | 209.846us | 14.111us | 34.194us | 207.266ns | 337.792us | 27.911s | 1907.743s |
| PN | TSER40+W4BFPA4 | 39.90% | 1.67% | `W4BFPA4` | 0.00% | 4.000 | 21019.192s | 0.000ns | 1.517ms | 102.568us | 251.747us | 1.001us | 1.419ms | 349.737s | 21368.932s |
| PN | TSER40+BFPLift | 39.90% | 1.67% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 23383.952s | 2364.761s | 1.517ms | 102.568us | 251.747us | 1.001us | 1.419ms | 349.737s | 23733.692s |
| PL | TSER40+W4BFPA4 | 42.36% | 1.51% | `W4BFPA4` | 0.00% | 4.000 | 20158.839s | 0.000ns | 1.455ms | 101.598us | 267.268us | 1.001us | 1.419ms | 349.737s | 20508.579s |
| PL | TSER40+BFPLift | 42.36% | 1.51% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 22426.805s | 2267.967s | 1.455ms | 101.598us | 267.268us | 1.001us | 1.419ms | 349.737s | 22776.546s |
| AR | TSER40+W4BFPA4 | 39.69% | 1.47% | `W4BFPA4` | 0.00% | 4.000 | 122249.305s | 0.000ns | 13.073ms | 881.634us | 2.151ms | 20.869us | 37.320ms | 2027.016s | 124276.374s |
| AR | TSER40+BFPLift | 39.69% | 1.47% | `W4GraphBFPA4to6_B256_tser_graphstress10` | 10.00% | 4.200 | 129125.848s | 6876.543s | 13.073ms | 881.634us | 2.151ms | 20.869us | 37.320ms | 2027.016s | 131152.917s |
| WK | TSER40+W4BFPA4 | 38.90% | 1.15% | `W4BFPA4` | 0.00% | 4.000 | 12337.879s | 0.000ns | 915.112us | 61.103us | 145.654us | 3.560us | 6.916ms | 201.929s | 12539.816s |
| WK | TSER40+BFPLift | 38.90% | 1.15% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 13725.951s | 1388.073s | 915.112us | 61.103us | 145.654us | 3.560us | 6.916ms | 201.929s | 13927.889s |

## Read

- The graph index load is tiny; neighbor embedding reads dominate the NDP-side graph memory traffic.
- CAM/LRU lookup is colocated with the NDP embedding store and remains a microsecond-scale component.
- BFPLift changes only the miss-side encoder cycles in this model; CAM, hit embedding reads, graph index loads, neighbor reads, and NPU-to-NDP miss writes are unchanged for the same reuse point.
- The NPU is shielded from irregular graph/embedding reads; it only streams encoder data and writes miss embeddings to the NDP store.
- This is a trace-composition model, not a bank-conflict or NoC arbitration simulator.
