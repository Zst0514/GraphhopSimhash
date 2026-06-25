# TSER40 Full-Stack Timing Simulation

## Scope

This report fixes the frontend operating point to TSER-selected reuse near 40%.
Timing is composed from CAM/LRU frontend cycles, embedding movement, progressive-BFP encoder cycles, and a configurable GNN/task-head term.

## Timing Model

- Clock: `500 MHz`.
- Reuse input: `/home/zhangshangtong/Transformer/OFA/output/tser_reuse_drop_tradeoff_40pt_alignment.tsv`.
- Target reuse: `40.0% +/- 3.0%`.
- Frontend/query cycles: search `1` + select `1` + TSER `0` + miss_update `1 * miss_rate`.
- Embedding movement: `4096` x `16`b per node at `25.6 GB/s`.
- Scheduler overhead: `0.005` x full BFPA8 encoder cycles.
- GNN/task-head overhead: `0.01` x full BFPA8 encoder cycles.

## Main TSER40 Result

| Task | Nodes | Reuse | Drop | Dyn Tag | Eff. Bits | Frontend | Embed | Encoder | GNN | Total Time | Norm. | Speedup |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 2,708 | 39.90% | 0.98% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 4.400 | 7044 | 433280 | 933.084B | 27.911B | 1949.903s | 0.346x | 2.89x |
| CL | 2,708 | 39.46% | 1.59% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 4.400 | 7055 | 433280 | 939.916B | 27.911B | 1963.565s | 0.348x | 2.87x |
| PN | 19,717 | 39.90% | 1.67% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 4.400 | 51284 | 3.155M | 11.692T | 349.737B | 24433.169s | 0.346x | 2.89x |
| PL | 19,717 | 42.36% | 1.51% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 4.400 | 50799 | 3.155M | 11.213T | 349.737B | 23476.023s | 0.332x | 3.01x |
| AR | 169,343 | 39.69% | 1.47% | `W4GraphBFPA4to6_B256_tser_graphstress10` | 4.200 | 440817 | 27.095M | 64.563T | 2.027T | 135206.949s | 0.330x | 3.03x |
| WK | 11,701 | 38.90% | 1.15% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 4.400 | 30551 | 1.872M | 6.863T | 201.929B | 14331.743s | 0.351x | 2.85x |

Average normalized time: `0.342x` of NoReuse+BFPA8.
Average speedup: `2.92x` over NoReuse+BFPA8.

## Policy Breakdown

| Task | Policy | Reuse | Miss | Frontend | Embed | Scheduler | Encoder | GNN | Total Cycles | Time | Norm. | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | NoReuse+BFPA8 | 0.00% | 100.00% | 0 | 433280 | 0 | 2.791T | 27.911B | 2.819T | 5638.033s | 1.000x | 1.00x |
| CN | TSER40+BFPA8 | 39.90% | 60.10% | 7044 | 433280 | 13.956B | 1.677T | 27.911B | 1.719T | 3438.642s | 0.610x | 1.64x |
| CN | TSER40+DynBFP | 39.90% | 60.10% | 7044 | 433280 | 13.956B | 933.084B | 27.911B | 974.951B | 1949.903s | 0.346x | 2.89x |
| CL | NoReuse+BFPA8 | 0.00% | 100.00% | 0 | 433280 | 0 | 2.791T | 27.911B | 2.819T | 5638.033s | 1.000x | 1.00x |
| CL | TSER40+BFPA8 | 39.46% | 60.54% | 7055 | 433280 | 13.956B | 1.690T | 27.911B | 1.732T | 3463.204s | 0.614x | 1.63x |
| CL | TSER40+DynBFP | 39.46% | 60.54% | 7055 | 433280 | 13.956B | 939.916B | 27.911B | 981.783B | 1963.565s | 0.348x | 2.87x |
| PN | NoReuse+BFPA8 | 0.00% | 100.00% | 0 | 3.155M | 0 | 34.974T | 349.737B | 35.323T | 70646.873s | 1.000x | 1.00x |
| PN | TSER40+BFPA8 | 39.90% | 60.10% | 51284 | 3.155M | 174.868B | 21.019T | 349.737B | 21.544T | 43087.601s | 0.610x | 1.64x |
| PN | TSER40+DynBFP | 39.90% | 60.10% | 51284 | 3.155M | 174.868B | 11.692T | 349.737B | 12.217T | 24433.169s | 0.346x | 2.89x |
| PL | NoReuse+BFPA8 | 0.00% | 100.00% | 0 | 3.155M | 0 | 34.974T | 349.737B | 35.323T | 70646.873s | 1.000x | 1.00x |
| PL | TSER40+BFPA8 | 42.36% | 57.64% | 50799 | 3.155M | 174.868B | 20.159T | 349.737B | 20.683T | 41366.895s | 0.586x | 1.71x |
| PL | TSER40+DynBFP | 42.36% | 57.64% | 50799 | 3.155M | 174.868B | 11.213T | 349.737B | 11.738T | 23476.023s | 0.332x | 3.01x |
| AR | NoReuse+BFPA8 | 0.00% | 100.00% | 0 | 27.095M | 0 | 202.702T | 2.027T | 204.729T | 409457.187s | 1.000x | 1.00x |
| AR | TSER40+BFPA8 | 39.69% | 60.31% | 440817 | 27.095M | 1.014T | 122.249T | 2.027T | 125.290T | 250579.712s | 0.612x | 1.63x |
| AR | TSER40+DynBFP | 39.69% | 60.31% | 440817 | 27.095M | 1.014T | 64.563T | 2.027T | 67.603T | 135206.949s | 0.330x | 3.03x |
| WK | NoReuse+BFPA8 | 0.00% | 100.00% | 0 | 1.872M | 0 | 20.193T | 201.929B | 20.395T | 40789.717s | 1.000x | 1.00x |
| WK | TSER40+BFPA8 | 38.90% | 61.10% | 30551 | 1.872M | 100.965B | 12.338T | 201.929B | 12.641T | 25281.549s | 0.620x | 1.61x |
| WK | TSER40+DynBFP | 38.90% | 61.10% | 30551 | 1.872M | 100.965B | 6.863T | 201.929B | 7.166T | 14331.743s | 0.351x | 2.85x |

## Interpretation

- The CAM/LRU frontend is modeled with actual 500MHz query cycles rather than a fixed normalized filter penalty.
- Embedding traffic includes one 4096-d FP16 cached-embedding read for reuse nodes and one write for miss nodes.
- Encoder time dominates under the configured BFP backend; frontend and embedding movement remain visible but small.
- The absolute seconds are hardware-model seconds at the configured clock, not RTX4090 wall-clock seconds.

Raw TSV: `/home/zhangshangtong/Transformer/OFA/output/tser40_fullstack_sim/tser40_fullstack_timing.tsv`
Raw JSON: `/home/zhangshangtong/Transformer/OFA/output/tser40_fullstack_sim/tser40_fullstack_timing.json`
