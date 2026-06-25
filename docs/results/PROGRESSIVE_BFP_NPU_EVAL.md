# Progressive BFP NPU Policy Evaluation

Cycle proxies use effective mantissa bits: Cycle/BFPA4 = EffBits/4 and Cycle/BFPA6 = EffBits/6.

## Fixed 20% Lift Selector Comparison

| Task | BFPA4 | BFPA6 | Rand20 | Stress20 | Graph-only20 | GraphxStress20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 1.61% | -0.04% | 0.81% | 0.62% | 1.13% | 0.85% |
| CL | 0.70% | 0.00% | 0.70% | 0.52% | 0.47% | 0.49% |
| PN | 1.80% | 0.13% | 1.35% | 1.10% | 1.17% | 1.11% |
| PL | 0.62% | 0.02% | 0.40% | 0.37% | 0.53% | 0.48% |
| WK | 1.08% | 0.06% | 0.94% | 0.69% | 0.85% | 0.62% |

## Detailed Cost Rows

| Task | Policy | Drop | Lifted Blocks | Eff. Bits | Cycle/BFPA4 | Cycle/BFPA6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CN | BFPA4 | 1.61% | 0.00% | 4.000 | 1.000 | 0.667 |
| CN | BFPA6 | -0.04% | 100.00% | 6.000 | 1.500 | 1.000 |
| CN | Rand20 | 0.81% | 20.00% | 4.400 | 1.100 | 0.733 |
| CN | Stress20 | 0.62% | 20.00% | 4.400 | 1.100 | 0.733 |
| CN | Graph-only20 | 1.13% | 20.00% | 4.400 | 1.100 | 0.733 |
| CN | GraphxStress20 | 0.85% | 20.00% | 4.400 | 1.100 | 0.733 |
| CL | BFPA4 | 0.70% | 0.00% | 4.000 | 1.000 | 0.667 |
| CL | BFPA6 | 0.00% | 100.00% | 6.000 | 1.500 | 1.000 |
| CL | Rand20 | 0.70% | 20.00% | 4.400 | 1.100 | 0.733 |
| CL | Stress20 | 0.52% | 20.00% | 4.400 | 1.100 | 0.733 |
| CL | Graph-only20 | 0.47% | 20.00% | 4.400 | 1.100 | 0.733 |
| CL | GraphxStress20 | 0.49% | 20.00% | 4.400 | 1.100 | 0.733 |
| PN | BFPA4 | 1.80% | 0.00% | 4.000 | 1.000 | 0.667 |
| PN | BFPA6 | 0.13% | 100.00% | 6.000 | 1.500 | 1.000 |
| PN | Rand20 | 1.35% | 20.00% | 4.400 | 1.100 | 0.733 |
| PN | Stress20 | 1.10% | 20.00% | 4.400 | 1.100 | 0.733 |
| PN | Graph-only20 | 1.17% | 20.00% | 4.400 | 1.100 | 0.733 |
| PN | GraphxStress20 | 1.11% | 20.00% | 4.400 | 1.100 | 0.733 |
| PL | BFPA4 | 0.62% | 0.00% | 4.000 | 1.000 | 0.667 |
| PL | BFPA6 | 0.02% | 100.00% | 6.000 | 1.500 | 1.000 |
| PL | Rand20 | 0.40% | 20.00% | 4.400 | 1.100 | 0.733 |
| PL | Stress20 | 0.37% | 20.00% | 4.400 | 1.100 | 0.733 |
| PL | Graph-only20 | 0.53% | 20.00% | 4.400 | 1.100 | 0.733 |
| PL | GraphxStress20 | 0.48% | 20.00% | 4.400 | 1.100 | 0.733 |
| WK | BFPA4 | 1.08% | 0.00% | 4.000 | 1.000 | 0.667 |
| WK | BFPA6 | 0.06% | 100.00% | 6.000 | 1.500 | 1.000 |
| WK | Rand20 | 0.94% | 20.00% | 4.400 | 1.100 | 0.733 |
| WK | Stress20 | 0.69% | 20.00% | 4.400 | 1.100 | 0.733 |
| WK | Graph-only20 | 0.85% | 20.00% | 4.400 | 1.100 | 0.733 |
| WK | GraphxStress20 | 0.62% | 20.00% | 4.400 | 1.100 | 0.733 |

## Threshold-Style GraphxStress Gate

| Task | Policy | Drop | Lifted Blocks | Eff. Bits | Cycle/BFPA4 | Cycle/BFPA6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AR | GraphxStress-threshold | 0.19% | 19.28% | 4.386 | 1.096 | 0.731 |
| CN | GraphxStress-threshold | 1.11% | 21.42% | 4.428 | 1.107 | 0.738 |
| CL | GraphxStress-threshold | 0.60% | 21.42% | 4.428 | 1.107 | 0.738 |
| PN | GraphxStress-threshold | 1.25% | 18.44% | 4.369 | 1.092 | 0.728 |
| PL | GraphxStress-threshold | 0.62% | 18.44% | 4.369 | 1.092 | 0.728 |
| WK | GraphxStress-threshold | 0.84% | 23.63% | 4.473 | 1.118 | 0.745 |
