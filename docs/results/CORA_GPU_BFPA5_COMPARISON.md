# Cora GPU BFPA5 Baseline Comparison

## Baseline

- GPU baseline log: `/home/zhangshangtong/Transformer/OFA/output/bfpa635_b256_generation/cora_W4BFPA5_B256_20260610_134558.log`.
- Parsed final encoding line: `Encoding: 100%, 677/677, [01:48<00:00,  6.26it/s]`.
- Baseline seconds used: `108.000s` for full Cora W4BFPA5_B256 embedding generation.
- The baseline is the encoding phase only. Model checkpoint loading, AWQ search, and one-time pseudo weight quantization are not counted.

## Calibration

- Cora BFP trace: `/home/zhangshangtong/Transformer/OFA/output/e2e_time_breakdown_40reuse/array_cora_graphstress20/summary.json`.
- Full BFPA5 cycles are interpolated from BFPA4 and BFPA6: `1744440570311.112` cycles.
- Dynamic BFPLift/BFPA5 cycle ratio: `0.890001`.
- Online loader raw/BFPA5 cycle ratio: `0.026738`.
- The GPU seconds are scaled by these ratios, so this is a GPU-calibrated trace-composition result rather than a raw single-array ONNXim wall time.

## Result

| Task | Reuse | Drop | GPU BFPA5 Baseline | BFPA4 Base | BFPLift Extra | Dynamic MAC | Online Loader Raw | CAM/LRU | NPU->NDP Write | Hit Emb Read | Graph Index | Neighbor Emb Read | E2E Overlap | E2E Serial | Speedup Overlap | Speedup Serial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 39.90% | 0.98% | 108.000s | 51.926s | 5.842s | 57.768s | 1.735s | 14.087us | 208.321us | 34.576us | 207.266ns | 337.792us | 57.769s | 59.504s | 1.87x | 1.81x |
| CL | 39.46% | 1.59% | 108.000s | 52.307s | 5.885s | 58.191s | 1.748s | 14.111us | 209.846us | 34.194us | 207.266ns | 337.792us | 58.192s | 59.940s | 1.86x | 1.80x |

## Read

- CN beats the measured GPU BFPA5 baseline as `108s -> about 57.8s` with double-buffered loader overlap, or `about 59.5s` under conservative serial accounting.
- The key reduction is not a raw clock-frequency claim: `miss_stream * dynamic/BFPA5 = 0.601 * 0.890`, before tiny CAM and NDP-local embedding traffic.
- Online exponent selection is modeled as runtime loader/control work. It is shown separately and is only exposed on the critical path if the loader cannot be hidden behind the MAC array.
- NDP local graph index and neighbor embedding reads are included as memory traffic here. For Cora they are sub-millisecond and do not move the conclusion.
- Backend GNN arithmetic is not included because the measured GPU BFPA5 baseline log is an embedding-generation baseline, not a full GNN training/inference wall time.
