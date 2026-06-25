# Dual-Granularity BFP Oracle/Profile

This note records the current Node x Block mixed-precision profiling for the
progressive BFP encoder path.

## 1. BFPA Precision Boundary Tasks

The BFPA precision-boundary table has been re-measured for the current task set:

```text
CN = Cora node classification
CL = Cora link prediction
PN = PubMed node classification
PL = PubMed link prediction
AR = OGBN-Arxiv node classification
WK = Wiki-CS node classification
```

Reference is `W4BFPA8_B128`; target pools are `W4BFPA{6,5,4,3}_B256`.
Node tasks report accuracy; link tasks report sampled link AUC.

| Task | Ref. Score | BFPA6 | BFPA5 | BFPA4 | BFPA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CN | 0.7141 | -0.02% | 0.38% | 1.61% | 44.45% |
| CL | 0.8981 | 0.00% | 0.05% | 0.70% | 27.15% |
| PN | 0.7493 | 0.13% | 0.49% | 1.80% | 33.43% |
| PL | 0.9188 | 0.02% | 0.12% | 0.62% | 26.56% |
| AR | 0.6687 | 0.11% | 0.46% | 0.70% | 52.24% |
| WK | 0.7692 | 0.10% | 0.29% | 1.14% | 54.61% |

Source:

```text
output/bfpa_precision_tasks_cnclpnplarwk/summary.md
```

## 2. Dual-Granularity Oracle Setup

The oracle/profile uses BFPA4 as the base path and selectively lifts only
stressed activation blocks on high-risk nodes toward BFPA6.

The dynamic pool generator uses:

```text
node selector: top graph-risk fraction
block selector: activation stress threshold
base format: BFPA4
refine target: BFPA6
block size: 256
```

The activation stress signal is computed during BFP block exponent selection.
A block with a large max/median magnitude gap is more likely to suffer shared
exponent loss, so it is eligible for the extra BFPA6 mantissa work when the node
is graph-important.

## 3. Current Results

These are 5-run downstream evaluations for the current task set. The policy
`top25, threshold=0.20` is the first cross-dataset checkpoint.

| Dynamic Policy | Task | Ref. | BFPA4 Drop | Dynamic Drop | BFPA6 Drop | Lifted Blocks | Eff. Bits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top10, threshold=0.35 | CN | 0.7081 | 1.64% | 1.65% | -0.02% | 4.73% | 4.0946 |
| top10, threshold=0.35 | CL | 0.8943 | 0.70% | 0.78% | -0.01% | 4.73% | 4.0946 |
| top10, threshold=0.20 | CN | 0.7106 | 1.87% | 1.72% | -0.01% | 9.32% | 4.1864 |
| top10, threshold=0.20 | CL | 0.8945 | 0.69% | 0.82% | -0.02% | 9.32% | 4.1864 |
| top25, threshold=0.20 | CN | 0.7106 | 1.87% | 1.28% | -0.01% | 21.42% | 4.4285 |
| top25, threshold=0.20 | CL | 0.8945 | 0.68% | 0.57% | -0.02% | 21.42% | 4.4285 |
| top25, threshold=0.20 | PN | 0.7542 | 2.00% | 1.25% | 0.14% | 18.44% | 4.3687 |
| top25, threshold=0.20 | PL | 0.9187 | 0.65% | 0.62% | 0.02% | 18.44% | 4.3687 |
| top25, threshold=0.20 | AR | 0.6780 | 0.30% | 0.19% | 0.09% | 19.28% | 4.3856 |
| top25, threshold=0.20 | WK | 0.7651 | 1.54% | 0.84% | -0.01% | 23.63% | 4.4725 |

Sources:

```text
output/dual_granularity_bfp_oracle/cora_tser_top10_t035/summary.md
output/dual_granularity_bfp_oracle/cora_tser_top10_t020/summary.md
output/dual_granularity_bfp_oracle/cora_tser_top25_t020/summary.md
output/dual_granularity_bfp_oracle/pubmed_W4GraphBFPA4to6_B256_tser_top25_t0.2/summary.md
output/dual_granularity_bfp_oracle/wikics_W4GraphBFPA4to6_B256_tser_top25_t0.2/summary.md
output/dual_granularity_bfp_oracle/arxiv_W4GraphBFPA4to6_B256_tser_top25_t0.2/summary.md
```

## 4. Current Interpretation

The current implementation now measures the key hardware-facing metric:
the absolute ratio of lifted BFP blocks over all executed BFP blocks.

The first cross-dataset checkpoint shows:

1. `top10` node-only selection is too conservative. It lifts only 4.73% to
   9.32% of blocks, but the downstream recovery is weak.
2. `top25, threshold=0.20` starts to recover BFPA4 loss. CN improves from
   1.87% drop to 1.28%, PN improves from 2.00% to 1.25%, and WK improves from
   1.54% to 0.84%.
3. PL and AR have smaller BFPA4 drops, so their recoverable margin is smaller.
4. The current strongest point is not yet the final "tiny overhead fully
   restores accuracy" claim. It is evidence that Node x Block refinement works,
   but the selector budget still needs tuning.

The next useful sweep should search a smaller set of policy points:

```text
top_risk_frac = 0.15, 0.20, 0.25
threshold     = 0.15, 0.20, 0.25
tasks         = CN, CL, PN, PL, WK
```

The goal is to find the lowest lifted-block ratio that consistently recovers a
large fraction of the BFPA4 loss without approaching full BFPA6.

## 5. Reproduction

The queue script for regenerating the dynamic pools and profile summaries is:

```text
GraphhopSimhash/scripts/run_dual_granularity_bfp_oracle_queue.sh
```

The completed cross-dataset checkpoint used:

```bash
DATASETS="pubmed wikics arxiv" POINTS="0.25:0.20" RUNS=5 \
  GraphhopSimhash/scripts/run_dual_granularity_bfp_oracle_queue.sh
```
