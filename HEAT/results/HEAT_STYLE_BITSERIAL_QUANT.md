# HEAT-Style Bit-Serial Quantization Evaluation

## Scope

This is a mechanism-level HEAT-style baseline, not a full HEAT simulator reproduction.
It evaluates HEAT's topology-aware high/low vertex precision routing and the Sec. 5.2.1 bit-serial cost model.

## Configuration

- Tasks: `CN, CL, PN, PL, AR, WK`.
- Runs: `3` with seed base `42`.
- Key vertex fraction alpha: `0.1`.
- HEAT-style key precision: `10`-bit activation/token x `8`-bit weight.
- HEAT-style non-key precision: `2`-bit activation/token x `4`-bit weight.
- Accuracy proxy high pool: `W4BFPA8_B128`.
- Accuracy proxy low pools: `W4BFPA4_B256, W4BFPA3_B256`.

The exact HEAT bit-serial reduction is reported from the published bit widths.
Task-level drop is a local proxy based on existing LLaMA2-7B BFPA embedding pools.

## Aggregate Result

| Scope | Low Proxy | Policy | Key Rate | Drop | HEAT Norm vs INT8xINT8 | HEAT Speedup vs INT8xINT8 | HEAT Norm vs W4A8 | Proxy Norm vs High Pool |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AVG_HEAT5 | `W4BFPA3_B256` | AllLow | 0.00% | 37.85% | 0.1250x | 8.00x | 0.2500x | 0.3750x |
| AVG6 | `W4BFPA3_B256` | AllLow | 0.00% | 41.68% | 0.1250x | 8.00x | 0.2500x | 0.3750x |
| AVG_HEAT5 | `W4BFPA3_B256` | RandomKey10 | 10.00% | 35.17% | 0.2375x | 4.21x | 0.4751x | 0.4375x |
| AVG6 | `W4BFPA3_B256` | RandomKey10 | 10.00% | 38.19% | 0.2375x | 4.21x | 0.4751x | 0.4375x |
| AVG_HEAT5 | `W4BFPA3_B256` | HEATTopDegree10 | 10.00% | 35.91% | 0.2375x | 4.21x | 0.4751x | 0.4375x |
| AVG6 | `W4BFPA3_B256` | HEATTopDegree10 | 10.00% | 35.60% | 0.2375x | 4.21x | 0.4751x | 0.4375x |
| AVG_HEAT5 | `W4BFPA4_B256` | AllLow | 0.00% | 0.89% | 0.1250x | 8.00x | 0.2500x | 0.5000x |
| AVG6 | `W4BFPA4_B256` | AllLow | 0.00% | 1.03% | 0.1250x | 8.00x | 0.2500x | 0.5000x |
| AVG_HEAT5 | `W4BFPA4_B256` | RandomKey10 | 10.00% | 0.71% | 0.2375x | 4.21x | 0.4751x | 0.5500x |
| AVG6 | `W4BFPA4_B256` | RandomKey10 | 10.00% | 0.84% | 0.2375x | 4.21x | 0.4751x | 0.5500x |
| AVG_HEAT5 | `W4BFPA4_B256` | HEATTopDegree10 | 10.00% | 0.83% | 0.2375x | 4.21x | 0.4751x | 0.5500x |
| AVG6 | `W4BFPA4_B256` | HEATTopDegree10 | 10.00% | 0.85% | 0.2375x | 4.21x | 0.4751x | 0.5500x |

## Per-Task Drop

| Task | Low Proxy | Policy | Base | Score | Drop | Key Rate | HEAT Avg Bit-planes | HEAT Speedup vs INT8xINT8 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | `W4BFPA3_B256` | AllLow | 0.7007 | 0.2028 | 49.79% | 0.00% | 8.00 | 8.00x |
| CN | `W4BFPA3_B256` | HEATTopDegree10 | 0.7007 | 0.3399 | 36.07% | 10.01% | 15.21 | 4.21x |
| CN | `W4BFPA3_B256` | RandomKey10 | 0.7007 | 0.2713 | 42.94% | 10.01% | 15.21 | 4.21x |
| CN | `W4BFPA4_B256` | AllLow | 0.7007 | 0.6871 | 1.35% | 0.00% | 8.00 | 8.00x |
| CN | `W4BFPA4_B256` | HEATTopDegree10 | 0.7007 | 0.6902 | 1.05% | 10.01% | 15.21 | 4.21x |
| CN | `W4BFPA4_B256` | RandomKey10 | 0.7007 | 0.6918 | 0.89% | 10.01% | 15.21 | 4.21x |
| CL | `W4BFPA3_B256` | AllLow | 0.8922 | 0.6123 | 27.99% | 0.00% | 8.00 | 8.00x |
| CL | `W4BFPA3_B256` | HEATTopDegree10 | 0.8922 | 0.4061 | 48.62% | 10.01% | 15.21 | 4.21x |
| CL | `W4BFPA3_B256` | RandomKey10 | 0.8922 | 0.5660 | 32.63% | 10.01% | 15.21 | 4.21x |
| CL | `W4BFPA4_B256` | AllLow | 0.8922 | 0.8878 | 0.45% | 0.00% | 8.00 | 8.00x |
| CL | `W4BFPA4_B256` | HEATTopDegree10 | 0.8922 | 0.8853 | 0.70% | 10.01% | 15.21 | 4.21x |
| CL | `W4BFPA4_B256` | RandomKey10 | 0.8922 | 0.8895 | 0.28% | 10.01% | 15.21 | 4.21x |
| PN | `W4BFPA3_B256` | AllLow | 0.7522 | 0.4258 | 32.64% | 0.00% | 8.00 | 8.00x |
| PN | `W4BFPA3_B256` | HEATTopDegree10 | 0.7522 | 0.5076 | 24.46% | 10.00% | 15.20 | 4.21x |
| PN | `W4BFPA3_B256` | RandomKey10 | 0.7522 | 0.4619 | 29.03% | 10.00% | 15.20 | 4.21x |
| PN | `W4BFPA4_B256` | AllLow | 0.7522 | 0.7328 | 1.94% | 0.00% | 8.00 | 8.00x |
| PN | `W4BFPA4_B256` | HEATTopDegree10 | 0.7522 | 0.7369 | 1.53% | 10.00% | 15.20 | 4.21x |
| PN | `W4BFPA4_B256` | RandomKey10 | 0.7522 | 0.7347 | 1.75% | 10.00% | 15.20 | 4.21x |
| PL | `W4BFPA3_B256` | AllLow | 0.9192 | 0.6764 | 24.29% | 0.00% | 8.00 | 8.00x |
| PL | `W4BFPA3_B256` | HEATTopDegree10 | 0.9192 | 0.3801 | 53.92% | 10.00% | 15.20 | 4.21x |
| PL | `W4BFPA3_B256` | RandomKey10 | 0.9192 | 0.6213 | 29.79% | 10.00% | 15.20 | 4.21x |
| PL | `W4BFPA4_B256` | AllLow | 0.9192 | 0.9128 | 0.65% | 0.00% | 8.00 | 8.00x |
| PL | `W4BFPA4_B256` | HEATTopDegree10 | 0.9192 | 0.9114 | 0.78% | 10.00% | 15.20 | 4.21x |
| PL | `W4BFPA4_B256` | RandomKey10 | 0.9192 | 0.9132 | 0.60% | 10.00% | 15.20 | 4.21x |
| AR | `W4BFPA3_B256` | AllLow | 0.6781 | 0.1325 | 54.56% | 0.00% | 8.00 | 8.00x |
| AR | `W4BFPA3_B256` | HEATTopDegree10 | 0.6781 | 0.5133 | 16.48% | 10.00% | 15.20 | 4.21x |
| AR | `W4BFPA3_B256` | RandomKey10 | 0.6781 | 0.2637 | 41.44% | 10.00% | 15.20 | 4.21x |
| AR | `W4BFPA4_B256` | AllLow | 0.6781 | 0.6773 | 0.09% | 0.00% | 8.00 | 8.00x |
| AR | `W4BFPA4_B256` | HEATTopDegree10 | 0.6781 | 0.6772 | 0.09% | 10.00% | 15.20 | 4.21x |
| AR | `W4BFPA4_B256` | RandomKey10 | 0.6781 | 0.6776 | 0.05% | 10.00% | 15.20 | 4.21x |
| WK | `W4BFPA3_B256` | AllLow | 0.7699 | 0.1621 | 60.78% | 0.00% | 8.00 | 8.00x |
| WK | `W4BFPA3_B256` | HEATTopDegree10 | 0.7699 | 0.4296 | 34.03% | 10.00% | 15.20 | 4.21x |
| WK | `W4BFPA3_B256` | RandomKey10 | 0.7699 | 0.2368 | 53.32% | 10.00% | 15.20 | 4.21x |
| WK | `W4BFPA4_B256` | AllLow | 0.7699 | 0.7530 | 1.69% | 0.00% | 8.00 | 8.00x |
| WK | `W4BFPA4_B256` | HEATTopDegree10 | 0.7699 | 0.7602 | 0.97% | 10.00% | 15.20 | 4.21x |
| WK | `W4BFPA4_B256` | RandomKey10 | 0.7699 | 0.7552 | 1.47% | 10.00% | 15.20 | 4.21x |

## Interpretation Notes

- `AllLow` uses the low proxy pool for every node.
- `RandomKey10` protects the same number of nodes as HEAT but chooses them randomly.
- `HEATTopDegree10` follows HEAT's topology rule and protects the top-degree `alpha` fraction.
- HEAT bit-serial cost is independent from the proxy pool tags: with `alpha=0.1`, Fig. 6 gives about `15.2` average bit-plane GEMMs per multiply, or `0.2375x` of INT8xINT8.
- The proxy drop should be read as the effect of applying HEAT-style vertex routing to this repository's LLaMA2-7B BFPA pools, not as HEAT's official SentenceBERT W8A10/W4A2 accuracy.

## Raw Outputs

- Raw rows: `/home/zhangshangtong/Transformer/OFA/output/heat_style_bitserial_quant/raw.tsv`
- Summary rows: `/home/zhangshangtong/Transformer/OFA/output/heat_style_bitserial_quant/summary.tsv`
- Aggregate rows: `/home/zhangshangtong/Transformer/OFA/output/heat_style_bitserial_quant/aggregate.tsv`
- JSON: `/home/zhangshangtong/Transformer/OFA/output/heat_style_bitserial_quant/summary.json`
