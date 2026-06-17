# TSER Equal-Reuse Ablation

This file records the TSER component ablation under comparable reuse rates.
The purpose is to avoid the unfair comparison where all policies use the same
threshold but produce different reuse rates.

Current status:

- Finished: Cora / LLaMA2-7B / W4BFPA8 target pool.
- Pending: PubMed, OGBN-Arxiv, Wiki-CS, Products subset, TAPE-Arxiv23.
- Output root: `output/llama7b_tser_equal_reuse_sweep_cora/`.

## Method

The SimHash-CAM candidate discovery configuration is fixed. For each risk
policy, the score threshold is swept, and the closest point to each target
reuse rate is selected.

Policies:

| Policy | Risk Terms | Meaning |
| --- | --- | --- |
| Hash only | none | Uses SimHash support/distance without graph-risk scoring. |
| P only | P | Uses propagation risk only. |
| P+C | P+C | Adds graph-context/boundary risk. |
| P+U | P+U | Adds low-degree uniqueness risk. |
| Full TSER | P+C+U | Uses all three TSER risk terms. |

## Cora: Complete Frontend Path

This table uses the `ResidualReuse` row, so it reflects the full lightweight
frontend path after TSER filtering and residual repair.

### Around 40% Reuse

| Policy | Actual Reuse | Drop | AvgErr | Selected T |
| --- | ---: | ---: | ---: | ---: |
| Hash only | 42.70% | 2.47% | 0.08724 | 35 |
| P only | 40.80% | 1.32% | 0.08420 | 28 |
| P+C | 40.20% | 1.76% | 0.08452 | 24 |
| P+U | 40.30% | 1.35% | 0.08239 | 31 |
| Full TSER | 39.90% | 0.98% | 0.08336 | 45 |

Main observation:

At roughly the same reuse rate, Full TSER gives the lowest accuracy drop. This
shows that the full `P+C+U` risk score is not merely lowering reuse to improve
accuracy; it selects safer candidates at a comparable reuse budget.

## Cora: TSER Filter Before Residual Repair

This table uses the `SoftDirectReuse` row, which isolates the TSER filtering
effect before residual repair.

### Around 50% Reuse

| Policy | Actual Reuse | Drop | AvgErr | Selected T |
| --- | ---: | ---: | ---: | ---: |
| Hash only | 59.70% | 4.11% | 0.14616 | 40 |
| P only | 55.60% | 3.19% | 0.13334 | 24 |
| P+C | 53.20% | 2.80% | 0.12489 | 24 |
| P+U | 52.10% | 2.64% | 0.12134 | 28 |
| Full TSER | 51.20% | 1.95% | 0.11915 | 31 |

This pre-repair view shows the same trend: adding graph-risk terms improves
the quality of accepted candidates, and Full TSER is the strongest filter at a
similar reuse rate.

## Source Files

- Summary markdown:
  `output/llama7b_tser_equal_reuse_sweep_cora/llama7b_tser_equal_reuse_sweep.md`
- Full frontier:
  `output/llama7b_tser_equal_reuse_sweep_cora/llama7b_tser_equal_reuse_frontier.tsv`
- Closest-point table:
  `output/llama7b_tser_equal_reuse_sweep_cora/llama7b_tser_equal_reuse_closest.tsv`

## Pending Plot Plan

After the remaining five datasets finish, draw a unified figure with one of the
following layouts:

1. Bar chart at fixed target reuse:
   x-axis is policy; y-axis is accuracy drop; one grouped bar per dataset.
2. Drop-vs-reuse frontier:
   x-axis is reuse rate; y-axis is accuracy drop; one curve per policy.

The current preferred paper figure is the fixed-target grouped bar chart at
around 30% or 40% reuse, because it directly answers which TSER component gives
the lowest drop at the same reuse budget.
