# Encoder Memory-Bound Roofline Check

## Workload

- Nodes: `2708`.
- Batch size: `4` nodes.
- Sequence length: `512` tokens.
- Token rows per encoder batch: `2048`.
- Encoder batches: `677`.
- Model shape: LLaMA-style `32` layers, hidden `4096`, intermediate `11008`.
- Measured GPU BFPA5 encoding time: `108.000s`.
- GPU HBM bandwidth used for the check: `1008.0 GB/s`.
- GPU FP16 Tensor throughput used for compute lower bound: `165.2 TFLOP/s`.

## Operation Count

- Linear parameters counted in transformer blocks: `6.476B`.
- Linear MACs per encoder batch: `13.263T`.
- Attention MACs per encoder batch: `0.275T`.
- Total MACs over Cora: `9.165P`.
- Measured effective throughput: `84.86 TMAC/s` (`169.72 TFLOP/s` if one MAC is two FLOPs).

## HBM Lower Bounds

| Traffic model | Total bytes | HBM-only lower bound | Fraction of measured 108s |
| --- | ---: | ---: | ---: |
| FP16 weights only | 7.97 TiB | 8.699s | 8.05% |
| W4-packed weights only | 1.99 TiB | 2.175s | 2.01% |
| FP16 weights + activation lower bound | 12.62 TiB | 13.770s | 12.75% |
| W4 weights + activation lower bound | 6.64 TiB | 7.245s | 6.71% |

## Roofline Read

- FP16-weight arithmetic intensity: `1320.6` FLOP/byte.
- W4-weight arithmetic intensity: `2509.8` FLOP/byte.
- GPU ridge point: `163.9` FLOP/byte.
- FP16 compute lower bound: `110.957s`; measured time is `0.97x` this lower bound.

## Conclusion

- Even the conservative FP16 weight-stream assumption gives an HBM-only lower bound far below the measured time.
- The workload has high arithmetic intensity because `batch_size * seq_len = 2048` token rows reuse the same layer weights.
- The current encoder run is therefore not plausibly explained by HBM bandwidth alone; it is compute/kernel-overhead dominated rather than decoder-style memory-bound.
- GPU and NPU comparisons should keep the same node batch size and sequence length, then align full BFPA5 throughput before enabling TSER.
