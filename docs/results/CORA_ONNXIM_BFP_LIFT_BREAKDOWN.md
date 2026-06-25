# Cora ONNXim BFP Lift Runtime Breakdown

## Inputs

- ONNXim config: `/home/zhangshangtong/Transformer/OFA/GraphhopSimhash/ONNXim/configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json`.
- Cora BFP array trace: `/home/zhangshangtong/Transformer/OFA/output/e2e_time_breakdown_40reuse/array_cora_graphstress20`.
- Trace tag: `W4GraphBFPA4to6_B256_tser_graphstress20`.
- Trace block size: `256` activation values/block.
- Full Cora trace blocks: `3533508992` total, `706706144` refined (20.00%).
- ONNXim config core frequency field: `1000 MHz`; report clock override: `500.0 MHz`.

## Runtime BFP Loader Model

- Exponent select per block: `ceil(log2(block_size)) * add_tree_latency + exp_latency = 8 * 1 + 1 = 9` cycles.
- Mantissa pack/slice per block: `scalar_mul_latency + scalar_add_latency = 2` cycles.
- Stress/priority/refine-flag per block: `scalar_mul_latency + scalar_add_latency = 2` cycles.
- RefineQueue push per selected block: `1` cycle.
- `dynamic_mac` is the existing ONNXim-style BFP array trace: BFPA4 base MAC plus selected BFPA6 low-2-bit correction MAC.
- `serial_total` is conservative no-overlap time. `overlap_exposed_total` assumes the BFP loader is double-buffered with the MAC array; only loader work exceeding MAC time is exposed.

## Scenario Summary

| Scenario | Reuse | Miss | Exp Select | Pack/Slice | Stress+Flag | Queue Push | Loader Raw | BFPA4 MAC | BFPLift Extra MAC | Dynamic MAC | Serial Total | Overlap Exposed | Loader Raw / Dynamic MAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CoraFull+BFPLift | 0.00% | 100.00% | 63.603s | 14.134s | 14.134s | 1.413s | 93.285s | 2791.105s | 314.001s | 3105.106s | 3198.391s | 3105.106s | 3.00% |
| CN_TSER40_Miss+BFPLift | 39.90% | 60.10% | 38.226s | 8.495s | 8.495s | 849.461ms | 56.064s | 1677.454s | 188.715s | 1866.169s | 1922.233s | 1866.169s | 3.00% |
| CL_TSER40_Miss+BFPLift | 39.46% | 60.54% | 38.505s | 8.557s | 8.557s | 855.680ms | 56.475s | 1689.735s | 190.096s | 1879.831s | 1936.306s | 1879.831s | 3.00% |

## Cycle Details

| Scenario | Blocks | Refined Blocks | Exp Select | Pack/Slice | Stress+Flag | Queue Push | Loader Raw | BFPA4 MAC | BFPLift Extra MAC | Dynamic MAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CoraFull+BFPLift | 3.534B | 706.706M | 31.802B | 7.067B | 7.067B | 706.706M | 46.642B | 1.396T | 157.001B | 1.553T |
| CN_TSER40_Miss+BFPLift | 2.124B | 424.730M | 19.113B | 4.247B | 4.247B | 424.730M | 28.032B | 838.727B | 94.357B | 933.084B |
| CL_TSER40_Miss+BFPLift | 2.139B | 427.840M | 19.253B | 4.278B | 4.278B | 427.840M | 28.237B | 844.867B | 95.048B | 939.916B |

## Read

- The exponent-selection and lift-selection work is online runtime work, not Table VI offline preprocessing.
- For Cora CN at the 39.90% reuse point, the raw loader/control work is tens of seconds if serialized, but it is only a few percent of the dynamic MAC time and is hidden under a double-buffered loader/MAC pipeline in this model.
- If a reviewer asks for the unhidden worst case, use `Serial Total`; if discussing the actual pipelined NPU critical path, use `Overlap Exposed` plus separately report `Loader Raw` as work performed.
