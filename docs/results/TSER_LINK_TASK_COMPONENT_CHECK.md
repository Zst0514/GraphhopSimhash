# TSER Link-Task Iso-Reuse Component Check

This note records the corrected protocol for adding link tasks to the TSER
component ablation.

## Correct Experimental Interpretation

The TSER component ablation is an iso-reuse experiment: each policy is tuned to
approximately the same final accepted reuse rate, then compared by downstream
drop. Therefore, `CL` and `PL` can be included as long as their support/gate
profile is fixed for the link-task experiment and each TSER metric variant is
tuned to the same accepted reuse budget.

The earlier conservative link-transfer setting was not an iso-reuse setting. It
reported the natural operating point of the original safety stack, which capped
`CL` around `30%` reuse. For the iso-reuse ablation, we use a unified relaxed
link profile:

```text
hard support >= 5
soft support >= 3
residual gate threshold = 0.00
target accepted reuse ~= 40%
```

Only the TSER metric components change across policies.

## Cora-Link 40% Iso-Reuse Search

With the relaxed link profile, all four policies reach the 40% reuse budget:

| Policy | T | Reuse |
| --- | ---: | ---: |
| P only | 28 | 40.18% |
| P+C | 36 | 40.10% |
| P+U | 36 | 40.29% |
| Full TSER | 36 | 40.32% |

Source:

```text
output/tser_component_link_iso40_relaxed/cora_reuse_grid_h5s3.tsv
output/tser_component_link_iso40_relaxed/cora_reuse_grid_h5s3_gate0_extra.tsv
```

## Cora-Link AUC at 40% Iso-Reuse

| Policy | T | Reuse | Base AUC | Reuse AUC | AUC Drop | Base AP | Reuse AP | AP Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P only | 28 | 39.57% | 0.8918 | 0.8750 | 1.68% | 0.8780 | 0.8751 | 0.28% |
| P+C | 36 | 39.49% | 0.8923 | 0.8717 | 2.06% | 0.8785 | 0.8733 | 0.52% |
| P+U | 36 | 39.07% | 0.8918 | 0.8773 | 1.49% | 0.8779 | 0.8775 | 0.10% |
| Full TSER | 36 | 39.46% | 0.8922 | 0.8727 | 1.96% | 0.8783 | 0.8735 | 0.48% |

## Cora-Link Full TSER Without Residual Repair

This row disables residual repair after Full TSER accepts an anchor. It keeps
the same `T=36`, `hard support >= 5`, `soft support >= 3`, and gate profile as
the Full TSER link-transfer point above.

| Policy | T | Reuse | Base AUC | Reuse AUC | AUC Drop | Base AP | Reuse AP | AP Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full TSER w/o repair | 36 | 39.46% | 0.8923 | 0.8726 | 1.96% | 0.8785 | 0.8733 | 0.52% |

Source:

```text
output/tser_component_link_iso40_relaxed/final/cora_p_only_T28_link_reuse.md
output/tser_component_link_iso40_relaxed/final/cora_p_c_T36_link_reuse.md
output/tser_component_link_iso40_relaxed/final/cora_p_u_T36_link_reuse.md
output/tser_component_link_iso40_relaxed/final/cora_full_tser_T36_link_reuse.md
output/tser_component_link_iso40_relaxed/final/cora_full_tser_T36_link_reuse_norepair.md
```

## PubMed-Link 40% Iso-Reuse Search

With the same relaxed link profile, all four policies now have matched
`~40%` reuse points:

| Policy | T | Reuse |
| --- | ---: | ---: |
| P only | 10 | 39.78% |
| P+C | 14 | 41.28% |
| P+U | 20 | 38.54% |
| Full TSER | 24 | 42.36% |

The `P+C` and `Full TSER` points were raised from the earlier low-reuse
settings so that the final comparison is not biased by different accepted
reuse budgets.

## PubMed-Link AUC at 40% Iso-Reuse

| Policy | T | Reuse | Base AUC | Reuse AUC | AUC Drop | Base AP | Reuse AP | AP Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P only | 10 | 39.78% | 0.9193 | 0.9041 | 1.52% | 0.9138 | 0.8981 | 1.56% |
| P+C | 14 | 41.28% | 0.9192 | 0.9039 | 1.53% | 0.9138 | 0.8980 | 1.58% |
| P+U | 20 | 38.54% | 0.9192 | 0.9030 | 1.62% | 0.9138 | 0.8975 | 1.62% |
| Full TSER | 24 | 42.36% | 0.9192 | 0.9025 | 1.67% | 0.9138 | 0.8967 | 1.71% |
| Full TSER w/o repair | 24 | 42.35% | 0.9191 | 0.9024 | 1.67% | 0.9137 | 0.8965 | 1.72% |

For `PL`, disabling residual repair barely changes AUC drop at the same reuse
budget. This suggests the link-transfer setting is dominated by TSER filtering
rather than residual correction.

Source:

```text
output/tser_component_link_iso40_relaxed/final/pubmed_p_only_T10_link_reuse.md
output/tser_component_link_iso40_relaxed/final/pubmed_p_c_T14_link_reuse.md
output/tser_component_link_iso40_relaxed/final/pubmed_p_u_T20_link_reuse.md
output/tser_component_link_iso40_relaxed/final/pubmed_full_tser_T24_link_reuse.md
output/tser_component_link_iso40_relaxed/final/pubmed_full_tser_T24_link_reuse_norepair.md
```

## Figure Usage

For Figure 8, `CL` and `PL` should use the 40% iso-reuse AUC-drop values above.
The caption should say `~40% target reuse` rather than exact `40%`, because
threshold discretization leaves small per-task deviations.
