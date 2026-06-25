# Motivation Topology Risk Profiling

This note records the profiling design for Motivation 2B: semantic locality
creates reusable candidates, but graph position changes the downstream damage of
using an approximate anchor.

## Goal

The profiling question is:

```text
If two node groups are replaced with the same number of real SimHash-CAM
anchors, and those anchors have matched support / Hamming-distance quality,
does the graph position still change downstream accuracy loss?
```

This is deliberately a Motivation experiment. It should not introduce the final
TSER formula. It only proves that semantic distance is candidate evidence, not a
complete reuse decision.

## Script

Script:

```bash
GraphhopSimhash/scripts/profile_topology_risk_sensitivity.py
```

The script trains the normal downstream GNN once, executes SimHash-CAM candidate
discovery, and then replaces the same fraction of nodes selected by each
graph-side risk signal:

- `High-P` / `Low-P`: propagation position from log-degree.
- `High-C` / `Low-C`: graph-context boundary / neighborhood mismatch.
- `High-U` / `Low-U`: low-degree uniqueness / rare-tail position.
- `Random`: random node budget.

Perturbation modes:

- `anchor`: replace selected nodes with their discovered CAM anchor.
- `noise`: inject equal-norm random hidden-state noise.
- `zero`: zero selected hidden states.

The main paper-relevant mode is `anchor + matched_quality`, because it uses real
candidate anchors and matches high/low groups by lookup quality.

## Matched Replacement Design

For each run, the script first keeps nodes with valid SimHash-CAM anchors. For
each graph-side dimension, it selects a high-risk group and a low-risk group
under three controls:

1. The replacement budget is fixed. In the Cora sanity run, every group replaces
   `10.01%` of nodes.
2. The replacement uses real CAM anchors, not synthetic noise.
3. The high-risk and low-risk groups are matched by exact `(support, Hamming)`
   buckets, so the candidate-quality distribution is aligned.

Command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/profile_topology_risk_sensitivity.py \
  --datasets cora --runs 5 \
  --perturbation anchor --matched_quality \
  --replace_frac 0.10 --min_support 3 --risk_pool_frac 0.35 \
  --output_dir output/matched_replacement_cora_runs5
```

Outputs:

```text
output/matched_replacement_cora_runs5/topology_risk_sensitivity_summary.md
output/matched_replacement_cora_runs5/topology_risk_sensitivity_raw.tsv
output/matched_replacement_pubmed_runs5/topology_risk_sensitivity_summary.md
output/matched_replacement_pubmed_runs5/topology_risk_sensitivity_raw.tsv
```

## Cora Matched Results

The table reports 5-run averages. `Support` and `Ham.` are intentionally nearly
identical between each high/low pair because they are matched controls.

| Group | Replaced | Drop | Anchor Cos. | Support | Ham. | Label Agree. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| High-P | 10.01% | 0.74% | 0.7303 | 3.62 | 1.23 | 78.01% |
| Low-P | 10.01% | 0.75% | 0.7139 | 3.62 | 1.23 | 75.79% |
| High-C | 10.01% | 0.76% | 0.7146 | 3.60 | 1.24 | 73.36% |
| Low-C | 10.01% | 0.30% | 0.7539 | 3.60 | 1.24 | 80.52% |
| High-U | 10.01% | 0.76% | 0.7268 | 3.65 | 1.23 | 76.09% |
| Low-U | 10.01% | 0.59% | 0.7115 | 3.65 | 1.23 | 77.93% |
| Random | 10.01% | 0.18% | 0.7275 | 3.66 | 1.21 | 78.15% |

Interpretation:

- Matching works: high/low pairs have the same replacement rate and nearly the
  same CAM evidence, especially support and Hamming distance.
- Graph-context position shows a clear gap: `High-C` causes `0.76%` drop while
  matched `Low-C` causes `0.30%`.
- Rare-tail / uniqueness position also matters, though more mildly: `High-U`
  causes `0.76%` drop versus `0.59%` for matched `Low-U`.
- Propagation alone is not decisive on Cora under this exact matched setup:
  `High-P` and `Low-P` are nearly tied. This is useful rather than harmful; it
  says degree alone is insufficient and motivates multiple graph-side signals.

## PubMed Matched Results

Command:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/profile_topology_risk_sensitivity.py \
  --datasets pubmed --runs 5 \
  --perturbation anchor --matched_quality \
  --replace_frac 0.10 --min_support 3 --risk_pool_frac 0.35 \
  --output_dir output/matched_replacement_pubmed_runs5
```

The same controls hold: every group replaces `10.00%` of nodes, and the
high/low pairs have matched support and Hamming distance.

| Group | Replaced | Drop | Anchor Cos. | Support | Ham. | Label Agree. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| High-P | 10.00% | 0.62% | 0.8810 | 3.70 | 1.20 | 69.37% |
| Low-P | 10.00% | 0.98% | 0.8495 | 3.70 | 1.20 | 64.85% |
| High-C | 10.00% | 0.89% | 0.8550 | 3.66 | 1.20 | 64.53% |
| Low-C | 10.00% | 0.71% | 0.8740 | 3.66 | 1.20 | 70.20% |
| High-U | 10.00% | 0.84% | 0.8564 | 3.69 | 1.20 | 65.60% |
| Low-U | 10.00% | 0.65% | 0.8718 | 3.69 | 1.20 | 68.62% |
| Random | 10.00% | 0.65% | 0.8613 | 3.70 | 1.21 | 66.67% |

Interpretation:

- PubMed confirms the context and rare-tail trend: `High-C` and `High-U`
  produce larger drops than their matched low-risk counterparts.
- Propagation alone is again not a sufficient standalone predictor: `Low-P`
  drops more than `High-P` under this matched candidate distribution.
- Across Cora and PubMed, the robust Motivation conclusion is not "degree/hub
  always dominates." The stronger conclusion is that semantic match quality is
  not enough; graph-side vulnerability is multidimensional and must be checked
  before accepting reuse.

## Earlier Sanity Checks

Earlier single-run Cora and PubMed checks without candidate-quality matching
were not clean enough for Motivation. Candidate quality and label agreement were
entangled with topology, and synthetic `noise` / `zero` perturbations introduced
operator-specific artifacts. Those outputs remain useful for debugging but
should not be used as the main paper evidence.

Earlier output locations:

```text
output/topology_risk_sensitivity_*/topology_risk_sensitivity_summary.md
output/topology_risk_sensitivity_*/topology_risk_sensitivity_raw.tsv
```

## Motivation Text Guidance

For Motivation 2B, avoid naming the final scoring formula. A safe paper-level
statement is:

```text
We perform a matched replacement profile: each bar replaces the same fraction of
nodes using real SimHash-CAM anchors with matched support and Hamming-distance
distributions. Even under this control, the downstream drop changes with graph
position, especially for graph-context and rare-tail regions. Therefore,
semantic closeness is only candidate evidence; graph position must also be
considered before accepting encoder reuse.
```

The follow-up TSER component ablation should still be reported separately,
because it evaluates the actual reuse filter:

```text
output/llama7b_tser_trace_replay/replay_strict40/trace_replay_summary.md
output/llama7b_tser_trace_replay/equal_budget_40/equal_budget_replay.md
docs/results/TSER_EQUAL_REUSE_ABLATION.md
```
