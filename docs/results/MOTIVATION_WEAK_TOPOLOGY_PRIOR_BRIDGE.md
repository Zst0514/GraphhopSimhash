# Motivation Weak Topology-Prior Bridge

This note records the corrected Motivation-1 bridge experiment.

The bridge must not use the final TSER / residual policy. Otherwise the
Motivation section would use the proposed solution to justify itself. The
right profiling baseline should be deliberately weak:

1. keep the same fixed bypass budget, e.g., `30%`;
2. select anchors by SimHash / text distance;
3. optionally bias the candidate pool toward graph-neighbor anchors;
4. do not use TSER, residual repair, or graph-risk scoring.

This asks a narrower question:

> If we add only a simple topology prior to distance-based substitution, do the
> candidates become better than purely text-distance matches?

## Recommended Table

| Policy | Candidate rule | LLM bypass | Drop | Purpose |
| --- | --- | ---: | ---: | --- |
| Text-distance only | closest SimHash anchor | 30% | measured | shows raw semantic-reuse opportunity and risk |
| Neighbor-biased distance | closest one-hop graph-neighbor anchor | 30% | measured below | tests whether a weak graph prior improves substitution |

The second row should still be treated as a profiling control, not as the final
GRACE policy. It should not include TSER thresholds, P/C/U risk terms, or
residual repair.

## Existing Candidate-Level Evidence

The current candidate-discovery ablation already supports the direction:

| Method | Lookup Yield | Valid Anchor | Emb. Cos. | Label Agree. |
| --- | ---: | ---: | ---: | ---: |
| Self-only SimHash | 99.81% | 44.08% | 0.7488 | 24.90% |
| Graph-context SimHash (1H) | 99.86% | 49.74% | 0.7604 | 33.29% |
| Graph-context SimHash (8H) | 68.24% | 78.90% | 0.8449 | 63.73% |

This is candidate-quality evidence only. For Motivation Table I, the cleaner
downstream experiment is still the two-row `30%` bypass comparison above.

## Downstream 30% Bypass Result

This experiment uses a deliberately weak topology prior: each node selects the
most similar one-hop graph neighbor in the LLaMA embedding space, and exactly
the top `30%` nodes are substituted. It does not use TSER, P/C/U risk scoring,
or residual repair.

| Task | Metric | Base | Reuse | Drop | LLM Bypass | Anchor Cos. | Hash-only Drop | Delta vs Hash-only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | Acc. | 69.66% | 68.50% | 1.16% | 29.99% | 0.9244 | 1.64% | 0.48% |
| CL | AUC | 89.22% | 88.97% | 0.25% | 29.99% | 0.9244 | 2.01% | 1.76% |
| PN | Acc. | 75.22% | 74.86% | 0.36% | 30.00% | 0.9497 | 2.37% | 2.01% |
| PL | AUC | 91.93% | 91.43% | 0.49% | 30.00% | 0.9497 | 2.57% | 2.08% |
| AR | Acc. | 67.83% | 67.67% | 0.15% | 30.00% | 0.9594 | 2.11% | 1.96% |
| WK | Acc. | 77.09% | 76.99% | 0.10% | 30.00% | 0.9600 | 1.59% | 1.49% |

Result files:

- `output/motivation_weak_graph_bias_30/combined/weak_graph_bias_30_summary.tsv`
- `output/motivation_weak_graph_bias_30/no_arxiv/weak_graph_bias_30_raw.tsv`
- `output/motivation_weak_graph_bias_30/arxiv/weak_graph_bias_30_raw.tsv`

Interpretation: a weak local-topology prior substantially reduces the drop of
the hash-only `30%` substitution baseline on all six tasks. This closes the
Motivation-1 logic gap without using the final GRACE reuse policy.

## Paper Wording

Use language like:

```tex
This substitution is intentionally a weak profiling baseline rather than our
final reuse mechanism: it uses distance to select similar anchors and can be
constrained to favor local graph-neighbor candidates, but it does not apply
risk scoring or residual repair.
```

Avoid saying that the final graph-aware filter reduces drop in Motivation 1.
Those full-policy numbers belong in the Evaluation section.
