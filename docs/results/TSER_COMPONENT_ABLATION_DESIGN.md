# TSER Component Ablation Design

This experiment evaluates TSER as a graph-risk scoring mechanism, not as a
large parameter sweep. The SimHash-CAM candidate discovery frontend is fixed
across all policies. Each policy receives the same kind of candidate evidence
from the frontend: anchor ID, multi-head support, Hamming distance, and the
precomputed graph-risk fields. The experiment then changes only which graph
risk terms are used by the TSER score.

## Goal

The key question is whether TSER improves reuse safety beyond hash/support
matching and degree-only filtering. The ablation therefore compares five
semantically meaningful policies:

| Policy | Risk terms | Purpose |
| --- | --- | --- |
| Hash only | none | Candidate support and distance only; no graph-risk protection. |
| P only | propagation | Protects high-fanout nodes whose errors can spread widely. |
| P+C | propagation + context | Adds boundary / neighborhood mismatch risk. |
| P+U | propagation + uniqueness | Adds rare low-degree node protection. |
| Full TSER | propagation + context + uniqueness | Uses all three complementary risk terms. |

This avoids sweeping many weight tuples such as `211`, `322`, or `411`. The
main configuration uses the calibrated `3/1/1` weighting because it gives
propagation risk the dominant role while keeping context and uniqueness as
secondary guards.

## Dataset Scope

Cora and PubMed are the tuning/diagnostic datasets:

- Cora has cleaner fuzzy hits, so the ablation shows how much graph risk can
  preserve reuse while reducing avoidable drop.
- PubMed has noisier fuzzy hits, so the ablation stresses whether TSER can
  reject unsafe candidates.

After the Cora/PubMed diagnosis is fixed, six-dataset validation should use
only fixed policies:

| Validation policy | Meaning |
| --- | --- |
| Hash only | No graph-risk filtering. |
| P only | Degree/propagation-risk baseline. |
| Full TSER | Proposed graph-risk filter before residual repair. |
| Full TSER + residual | Final frontend path. |

The six-dataset run is not a parameter search; it checks whether the selected
policy transfers.

## What Row To Read

The component ablation reads the `SoftDirectReuse` row from each
`residual_reuse` log. This row is the cleanest TSER-only signal:

1. SimHash-CAM has already exposed candidates.
2. The selected score policy has filtered candidates.
3. Residual repair has not yet changed fuzzy embeddings.

The final `ResidualReuse` row is reported separately as the complete frontend
policy after fuzzy-hit repair.

## Trace Export

The script exports per-node reuse decision traces when `EXPORT_TRACE=1`
(default). Each trace contains metadata only, not embeddings:

- node ID
- candidate anchor ID
- multi-head support
- Hamming distance
- P / C / U graph-risk fields
- TSER risk fields and gate reason
- final candidate route kind

This makes later offline replay possible without rerunning SimHash-CAM.

## Commands

Primary Cora/PubMed component ablation:

```bash
DATASETS="cora pubmed" RUNS=3 FORCE=0 \
  OUT_DIR=/home/zhangshangtong/Transformer/OFA/output/llama7b_tser_score_ablation \
  bash scripts/run_llama7b_tser_score_ablation.sh
```

Fast smoke test:

```bash
DATASETS="cora" RUNS=1 FORCE=1 EXPORT_TRACE=1 \
  OUT_DIR=/home/zhangshangtong/Transformer/OFA/output/llama7b_tser_score_ablation_smoke \
  bash scripts/run_llama7b_tser_score_ablation.sh
```

Summarize existing logs:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  scripts/summarize_llama7b_tser_score_ablation.py \
  --log_dir /home/zhangshangtong/Transformer/OFA/output/llama7b_tser_score_ablation/logs \
  --output_dir /home/zhangshangtong/Transformer/OFA/output/llama7b_tser_score_ablation
```

## Output Files

Expected outputs:

- `output/llama7b_tser_score_ablation/logs/*.log`
- `output/llama7b_tser_score_ablation/traces/*.tsv`
- `output/llama7b_tser_score_ablation/llama7b_tser_score_ablation.tsv`
- `output/llama7b_tser_score_ablation/llama7b_tser_score_ablation.md`

## Paper Framing

The experiment should be described as:

```text
We fix the SimHash-CAM candidate discovery frontend and replay the same
candidate-generation configuration for all risk policies. This isolates the
TSER graph-risk terms from CAM lookup variance and avoids turning the study
into a parameter sweep.
```

The paper table should emphasize the trend:

```text
Hash-only exposes many candidates but has high drop.
P-only protects hubs.
P+C and P+U isolate the added value of boundary/context and rare-tail signals.
Full TSER provides the best stable tradeoff before residual repair.
Residual repair is reported afterward as the final fuzzy-hit recovery path.
```
