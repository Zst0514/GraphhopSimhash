# 40% Reuse End-to-End Time Summary

## Experiment Configuration

- Tasks: CN, CL, PN, PL, AR, WK.
- Frontend reference: LLaMA2-7B `W4BFPA8_B128`; one full no-reuse encoder pass is normalized to `1.0`.
- Reuse point: selected TSER-full operating point near 40% final reuse.
- Dynamic encoder: `W4GraphBFPA4to6_B256`; BFPA4 base path with selected BFPA6 block lift.
- Online filter overhead: Hash-only `0.015`, TSER `0.020` normalized encoder-time units.
- Queue/compaction overhead: `0.005`; backend graph head overhead: `0.010`.
- CL reuses the Cora encoder trace; PL reuses the PubMed encoder trace because the link task shares the same node-text encoder work.
- AR currently uses the available `graphstress10` array trace; the other tasks use `graphstress20` traces.
- Wall-clock seconds are obtained by multiplying normalized time by the measured full-encoder LLaMA2-7B `W4BFPA4_B256` pool-generation encoding time on the local RTX4090. The normalized speedup is the primary architecture result.

## Measured Full-Encoder Timing Inputs

| Encoder Dataset | Full Encoder Time | Shared Tasks |
| --- | ---: | --- |
| Cora | 107.0s | CN, CL |
| PubMed | 1326.0s | PN, PL |
| OGBN-Arxiv | 7796.0s | AR |
| Wiki-CS | 791.0s | WK |

## Array Trace Inputs

| Encoder Dataset | Tag | Lifted Blocks | Eff. Bits | Dyn/BFPA8 Cycles |
| --- | --- | ---: | ---: | ---: |
| cora | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 0.556x |
| pubmed | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 0.556x |
| arxiv | `W4GraphBFPA4to6_B256_tser_graphstress10` | 10.00% | 4.200 | 0.528x |
| wikics | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 4.400 | 0.556x |

## Main Result

| Task | Reuse | Drop | Dyn Tag | Lifted Blocks | Dyn/BFPA8 | Norm. Time | Est. Time | Speedup |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CN | 39.90% | 0.98% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 0.556x | 0.366x | 39.5s | 2.73x |
| CL | 39.46% | 1.59% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 0.556x | 0.368x | 39.8s | 2.72x |
| PN | 39.90% | 1.67% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 0.556x | 0.366x | 489.7s | 2.73x |
| PL | 42.36% | 1.51% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 0.556x | 0.352x | 471.6s | 2.84x |
| AR | 39.69% | 1.47% | `W4GraphBFPA4to6_B256_tser_graphstress10` | 10.00% | 0.528x | 0.350x | 2756.0s | 2.86x |
| WK | 38.90% | 1.15% | `W4GraphBFPA4to6_B256_tser_graphstress20` | 20.00% | 0.556x | 0.371x | 296.5s | 2.69x |

Average normalized time for `TSER40+DynBFP`: `0.362x` of no-reuse BFPA8.
Average speedup over no-reuse BFPA8: `2.76x`.

## Full Policy Breakdown

| Task | Policy | Reuse | Filter | Queue | Encoder | Backend | Total | Seconds | Norm. | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | NoReuse+BFPA8 | 0.00% | 0.0000 | 0.0000 | 1.0000 | 0.0100 | 1.0100 | 108.1s | 1.000x | 1.00x |
| CN | Hash40+BFPA8 | 39.90% | 0.0150 | 0.0050 | 0.6010 | 0.0100 | 0.6310 | 67.5s | 0.625x | 1.60x |
| CN | TSER40+BFPA8 | 39.90% | 0.0200 | 0.0050 | 0.6010 | 0.0100 | 0.6360 | 68.1s | 0.630x | 1.59x |
| CN | TSER40+DynBFP | 39.90% | 0.0200 | 0.0050 | 0.3343 | 0.0100 | 0.3693 | 39.5s | 0.366x | 2.73x |
| CL | NoReuse+BFPA8 | 0.00% | 0.0000 | 0.0000 | 1.0000 | 0.0100 | 1.0100 | 108.1s | 1.000x | 1.00x |
| CL | Hash40+BFPA8 | 39.46% | 0.0150 | 0.0050 | 0.6054 | 0.0100 | 0.6354 | 68.0s | 0.629x | 1.59x |
| CL | TSER40+BFPA8 | 39.46% | 0.0200 | 0.0050 | 0.6054 | 0.0100 | 0.6404 | 68.5s | 0.634x | 1.58x |
| CL | TSER40+DynBFP | 39.46% | 0.0200 | 0.0050 | 0.3368 | 0.0100 | 0.3718 | 39.8s | 0.368x | 2.72x |
| PN | NoReuse+BFPA8 | 0.00% | 0.0000 | 0.0000 | 1.0000 | 0.0100 | 1.0100 | 1339.3s | 1.000x | 1.00x |
| PN | Hash40+BFPA8 | 39.90% | 0.0150 | 0.0050 | 0.6010 | 0.0100 | 0.6310 | 836.7s | 0.625x | 1.60x |
| PN | TSER40+BFPA8 | 39.90% | 0.0200 | 0.0050 | 0.6010 | 0.0100 | 0.6360 | 843.3s | 0.630x | 1.59x |
| PN | TSER40+DynBFP | 39.90% | 0.0200 | 0.0050 | 0.3343 | 0.0100 | 0.3693 | 489.7s | 0.366x | 2.73x |
| PL | NoReuse+BFPA8 | 0.00% | 0.0000 | 0.0000 | 1.0000 | 0.0100 | 1.0100 | 1339.3s | 1.000x | 1.00x |
| PL | Hash40+BFPA8 | 42.36% | 0.0150 | 0.0050 | 0.5764 | 0.0100 | 0.6064 | 804.1s | 0.600x | 1.67x |
| PL | TSER40+BFPA8 | 42.36% | 0.0200 | 0.0050 | 0.5764 | 0.0100 | 0.6114 | 810.7s | 0.605x | 1.65x |
| PL | TSER40+DynBFP | 42.36% | 0.0200 | 0.0050 | 0.3206 | 0.0100 | 0.3556 | 471.6s | 0.352x | 2.84x |
| AR | NoReuse+BFPA8 | 0.00% | 0.0000 | 0.0000 | 1.0000 | 0.0100 | 1.0100 | 7874.0s | 1.000x | 1.00x |
| AR | Hash40+BFPA8 | 39.69% | 0.0150 | 0.0050 | 0.6031 | 0.0100 | 0.6331 | 4935.6s | 0.627x | 1.60x |
| AR | TSER40+BFPA8 | 39.69% | 0.0200 | 0.0050 | 0.6031 | 0.0100 | 0.6381 | 4974.6s | 0.632x | 1.58x |
| AR | TSER40+DynBFP | 39.69% | 0.0200 | 0.0050 | 0.3185 | 0.0100 | 0.3535 | 2756.0s | 0.350x | 2.86x |
| WK | NoReuse+BFPA8 | 0.00% | 0.0000 | 0.0000 | 1.0000 | 0.0100 | 1.0100 | 798.9s | 1.000x | 1.00x |
| WK | Hash40+BFPA8 | 38.90% | 0.0150 | 0.0050 | 0.6110 | 0.0100 | 0.6410 | 507.0s | 0.635x | 1.58x |
| WK | TSER40+BFPA8 | 38.90% | 0.0200 | 0.0050 | 0.6110 | 0.0100 | 0.6460 | 511.0s | 0.640x | 1.56x |
| WK | TSER40+DynBFP | 38.90% | 0.0200 | 0.0050 | 0.3399 | 0.0100 | 0.3749 | 296.5s | 0.371x | 2.69x |
