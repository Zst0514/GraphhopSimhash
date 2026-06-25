# TSER No-Repair Iso-Reuse Ablation

This file records the updated TSER component ablation after removing residual
repair from the frontend path.  Older `ResidualReuse` rows should not be used
for the paper's TSER component figure.

## Protocol

The goal is to compare graph-risk scoring policies at the same accepted-reuse
budget.  Candidate discovery is fixed, and accepted anchors are reused directly
without residual correction.

Policies:

| Policy | Risk terms | Meaning |
| --- | --- | --- |
| Hash only | none | Candidate score from SimHash evidence only. |
| P | P | Propagation / fanout risk only. |
| P+C | P+C | Adds graph-context / boundary risk. |
| P+U | P+U | Adds low-degree uniqueness risk. |
| TSER full | P+C+U | Uses all three graph-risk terms. |

Node tasks (`CN`, `PN`, `AR`, `WK`) use exact 40% no-repair trace replay.  Link
tasks (`CL`, `PL`) use separate no-repair link-transfer evaluation, because
their metric is AUC and cannot be taken from node-classification replay.

## Main 40% No-Repair Table

Each cell reports metric drop at approximately 40% accepted reuse.  For `CN`,
`PN`, `AR`, and `WK`, the actual reuse is exactly 40% after replay.  For `CL`
and `PL`, `Hash only` uses exact 40% link evaluation; the graph-risk policies
use closest measured no-repair link-transfer runs around 40%.

| Dataset | Hash only | P | P+C | P+U | TSER full |
| --- | ---: | ---: | ---: | ---: | ---: |
| CN | 2.19% | 1.90% | 1.79% | 1.71% | 1.53% |
| CL | 2.13% | 1.95% | 2.69% | 1.49% | 1.95% |
| PN | 2.82% | 3.36% | 3.11% | 3.18% | 3.01% |
| PL | 6.09% | 1.39% | 1.51% | 1.60% | 1.60% |
| AR | 2.78% | 2.84% | 2.70% | 2.77% | 2.70% |
| WK | 1.94% | 1.85% | 1.59% | 1.74% | 1.49% |
| Avg. | 2.99% | 2.22% | 2.23% | 2.08% | 2.05% |

Interpretation:

- Removing residual repair raises the final TSER drop compared with the old
  repair-enabled table.
- `TSER full` remains the best average policy, but the margin is smaller:
  average drop falls from `2.99%` for hash-only to `2.05%`.
- `PN` is the main weak case under no-repair: graph-risk filtering alone does
  not recover the candidate errors as well as residual repair did.

## Source Details

Node-task exact 40% no-repair replay:

```text
output/llama7b_tser_no_repair_equal_budget_cpnwk/replay/equal_budget_replay.tsv
output/llama7b_tser_no_repair_equal_budget_arxiv/replay/equal_budget_replay.tsv
```

Node-task no-repair tradeoff curve, exact-budget replay:

```text
output/tser_reuse_drop_tradeoff_norepair_node_exact_budget.tsv
```

Link-task no-repair measured points:

```text
output/hash_only_equal_budget_40/link_norepair/cora_hash_only_equal_budget_link.md
output/hash_only_equal_budget_40/link_norepair/pubmed_hash_only_equal_budget_link.md
output/tser_component_link_iso40_norepair/cora_p_only_T28_link_reuse_norepair.md
output/tser_component_link_iso40_norepair/cora_p_c_T36_link_reuse_norepair.md
output/tser_component_link_iso40_norepair/cora_p_u_T36_link_reuse_norepair.md
output/tser_component_link_iso40_norepair/cora_full_tser_T36_link_reuse_norepair.md
output/tser_component_link_iso40_norepair/pubmed_p_only_T10_link_reuse_norepair.md
output/tser_component_link_iso40_norepair/pubmed_p_c_T14_link_reuse_norepair.md
output/tser_component_link_iso40_norepair/pubmed_p_u_T20_link_reuse_norepair.md
output/tser_component_link_iso40_norepair/pubmed_full_tser_T24_link_reuse_norepair.md
```

## Link-Task Values

| Dataset | Policy | T | Reuse | AUC drop |
| --- | --- | ---: | ---: | ---: |
| CL | Hash only | exact budget | 39.99% | 2.13% |
| CL | P | 28 | 39.92% | 1.95% |
| CL | P+C | 36 | 39.81% | 2.69% |
| CL | P+U | 36 | 39.06% | 1.49% |
| CL | TSER full | 36 | 39.45% | 1.95% |
| PL | Hash only | exact budget | 40.00% | 6.09% |
| PL | P | 10 | 39.83% | 1.39% |
| PL | P+C | 14 | 41.29% | 1.51% |
| PL | P+U | 20 | 38.28% | 1.60% |
| PL | TSER full | 24 | 42.16% | 1.60% |

## Copyable Table

```text
Dataset	hash only	P	P+C	P+U	TSER full
CN	2.19	1.90	1.79	1.71	1.53
CL	2.13	1.95	2.69	1.49	1.95
PN	2.82	3.36	3.11	3.18	3.01
PL	6.09	1.39	1.51	1.60	1.60
AR	2.78	2.84	2.70	2.77	2.70
WK	1.94	1.85	1.59	1.74	1.49
Avg.	2.99	2.22	2.23	2.08	2.05
```

## No-Repair Full-TSER Tradeoff Curve

This table reports exact-budget trace replay for node tasks only.  Link tasks
need separate link-prediction evaluation and are not included in this exact
node-task replay table.

```text
Dataset	10%	20%	30%	40%	50%	60%
CN	0.35	0.95	1.29	1.53	2.45	3.58
PN	0.74	1.50	2.22	3.01	3.61	4.15
AR	0.59	1.33	1.93	2.69	3.36	4.24
WK	0.29	0.66	1.00	1.49	1.92	2.80
```
