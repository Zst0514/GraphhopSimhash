# Graph-Bit Validation Summary

Date: 2026-05-27

This summary records the validation pass for Graph-Bit: graph-conditioned bit-depth execution for LLaMA-7B encoder embeddings on Cora, PubMed, and Arxiv.

## Goal

Graph-Bit assumes that when a node must execute the LLM encoder, the NPU does not need to execute all A8 activation bit-planes for every node.

Instead, graph/task risk controls arithmetic effort:

- high-risk nodes use P8/P6,
- medium-risk nodes use P6/P5,
- low-risk nodes use P5/P4.

This is not module skipping. It is graph-conditioned bit-serial / bit-grained GEMM depth control.

## Commands

Pure Graph-Bit precision-depth budget sweep:

```bash
python -m GraphhopSimhash \
  --datasets cora pubmed \
  --runs 10 \
  --experiment_suite precision_depth_ablation \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4A8 \
  --precision_depth_tags W4A6 W4A5 W4A4 \
  --precision_depth_bits 6 5 4 \
  --precision_depth_reference_bits 8 \
  --precision_depth_high_ratio HIGH \
  --precision_depth_mid_ratio MID \
  --precision_depth_low_ratio LOW \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

Reuse + Graph-Bit combined experiment:

```bash
python -m GraphhopSimhash \
  --datasets cora pubmed \
  --runs 3 \
  --experiment_suite reuse_precision_depth \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4A8 \
  --precision_depth_tags W4A6 W4A5 W4A4 \
  --precision_depth_bits 6 5 4 \
  --precision_depth_reference_bits 8 \
  --precision_depth_high_ratio 0.20 \
  --precision_depth_mid_ratio 0.30 \
  --precision_depth_low_ratio 0.30 \
  --radius 2 \
  --hash_heads_per_route 4 \
  --main_hash_head_bits 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold T \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

Residual reuse + Graph-Bit full-stack experiment:

```bash
python -m GraphhopSimhash \
  --datasets cora pubmed \
  --runs 3 \
  --experiment_suite residual_precision_depth \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4A8 \
  --precision_depth_tags W4A6 W4A5 W4A4 \
  --precision_depth_bits 6 5 4 \
  --precision_depth_reference_bits 8 \
  --precision_depth_high_ratio 0.20 \
  --precision_depth_mid_ratio 0.30 \
  --precision_depth_low_ratio 0.30 \
  --radius 2 \
  --hash_heads_per_route 4 \
  --main_hash_head_bits 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold T \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --residual_fit_profile llama \
  --residual_rank 64 \
  --residual_epochs 120 \
  --residual_max_train_pairs 4096 \
  --residual_alpha_grid 0 0.125 0.25 0.5 \
  --residual_min_dist 1.0
```

Logs:

- `output/graph_bit_validation/precision_depth/*.log`
- `output/graph_bit_validation/precision_depth_aggressive_10runs/*.log`
- `output/graph_bit_validation/reuse_precision_depth/*.log`
- `output/graph_bit_validation/precision_depth_arxiv_10runs/*.log`
- `output/residual_precision_depth_manual/*_fullstack.log`

Debug / oracle policies are intentionally not part of the main strategy:

```text
PredictorDepthBudget:
    uses calibration nodes to fit a damage predictor.
    useful for debugging proxy quality, not the main deployable policy.

OracleDamageBudget:
    uses true reference-vs-low-precision embedding damage.
    useful as an upper bound, not deployable.
```

## Pure Graph-Bit Results

Each row uses the same P8/P6/P5/P4 budget across routing policies. Lower Drop is better. The table below is the 10-run LLaMA-7B result.

### Cora / LLaMA-7B

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 10/20/30/40 | 0.346 | 2.73 | 2.52 | 2.66 | 2.43 | 2.69 | Context |
| 20/30/30/20 | 0.378 | 1.75 | 1.43 | 1.76 | 1.52 | 1.60 | Degree |
| 30/40/20/10 | 0.404 | 1.07 | 0.64 | 1.19 | 1.03 | 0.98 | Degree |
| 50/30/20/0  | 0.436 | 0.37 | 0.50 | 0.51 | 0.33 | 0.55 | Context |

### PubMed / LLaMA-7B

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 10/20/30/40 | 0.346 | 2.46 | 2.22 | 2.33 | 2.52 | 2.73 | Degree |
| 20/30/30/20 | 0.378 | 1.74 | 1.53 | 1.66 | 1.74 | 1.95 | Degree |
| 30/40/20/10 | 0.404 | 1.25 | 1.03 | 1.21 | 1.24 | 1.51 | Degree |
| 50/30/20/0  | 0.436 | 0.72 | 0.64 | 0.74 | 0.70 | 0.86 | Degree |

### Arxiv / LLaMA-7B

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 10/20/30/40 | 0.346 | 0.82 | 0.59 | 0.66 | 0.75 | 0.87 | Degree |
| 20/30/30/20 | 0.378 | 0.52 | 0.36 | 0.38 | 0.45 | 0.57 | Degree |
| 30/40/20/10 | 0.404 | 0.36 | 0.22 | 0.26 | 0.34 | 0.43 | Degree |
| 50/30/20/0  | 0.436 | 0.21 | 0.12 | 0.13 | 0.18 | 0.21 | Degree |

Main observation:

Degree / propagation risk is the most stable deployable policy across PubMed and Arxiv, and remains competitive on Cora. Context is sometimes strong on Cora, but less stable across datasets.

Recommended operating points:

- Cost 0.404 (`P8/P6/P5/P4 = 30/40/20/10`): best balanced point. Degree gives 0.64% drop on Cora and 1.03% drop on PubMed.
- Cost 0.436 (`P8/P6/P5/P4 = 50/30/20/0`): near-lossless point. Context is best on Cora, Degree is best on PubMed.
- Cost 0.378 (`P8/P6/P5/P4 = 20/30/30/20`): aggressive but usable; Degree remains the most stable deployable policy.
- On Arxiv, Degree is best in all four budget points, with drops from 0.59% down to 0.12%.

## Aggressive Graph-Bit Sweep

This sweep pushes more nodes into P5/P4 to find the low-cost boundary. It uses Cora/PubMed, LLaMA-7B, 10 runs.

### Cora / LLaMA-7B

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 0/0/50/50   | 0.314 | 3.53 | 3.25 | 3.47 | 3.16 | 3.62 | Context |
| 0/0/60/40   | 0.319 | 3.14 | 3.08 | 3.12 | 2.86 | 3.09 | Context |
| 0/10/40/50  | 0.319 | 3.44 | 3.10 | 3.36 | 3.02 | 3.53 | Context |
| 0/20/30/50  | 0.325 | 3.40 | 3.09 | 3.41 | 3.06 | 3.52 | Context |
| 0/30/0/70   | 0.319 | 4.02 | 3.79 | 4.04 | 3.86 | 4.29 | Degree |
| 5/10/35/50  | 0.327 | 3.45 | 3.05 | 3.32 | 3.09 | 3.40 | Degree |
| 10/0/30/60  | 0.325 | 3.70 | 3.80 | 3.63 | 3.64 | 3.67 | TSER |
| 10/10/20/60 | 0.330 | 3.63 | 3.68 | 3.62 | 3.65 | 3.57 | LowUnique |

### PubMed / LLaMA-7B

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 0/0/50/50   | 0.314 | 3.22 | 3.17 | 3.21 | 3.21 | 3.30 | Degree |
| 0/0/60/40   | 0.319 | 2.96 | 2.93 | 3.00 | 3.03 | 3.10 | Degree |
| 0/10/40/50  | 0.319 | 3.12 | 2.93 | 3.01 | 3.12 | 3.22 | Degree |
| 0/20/30/50  | 0.325 | 3.02 | 2.82 | 2.89 | 2.99 | 3.13 | Degree |
| 0/30/0/70   | 0.319 | 3.32 | 3.09 | 3.16 | 3.37 | 3.55 | Degree |
| 5/10/35/50  | 0.327 | 2.98 | 2.68 | 2.77 | 2.98 | 3.10 | Degree |
| 10/0/30/60  | 0.325 | 3.19 | 2.79 | 2.92 | 3.21 | 3.34 | Degree |
| 10/10/20/60 | 0.330 | 3.06 | 2.70 | 2.80 | 3.08 | 3.24 | Degree |

Main observation:

- The aggressive region is viable: cost can go down to about 0.319-0.327 while keeping drops around 2.7%-3.1%.
- PubMed remains Degree-dominated. Degree is best for every aggressive budget.
- Cora is more context-sensitive. Context wins most low-cost budgets, while Degree remains competitive.
- Avoid abrupt P6/P4-only routing such as `0/30/0/70`. At the same cost, keeping a P5 middle tier is clearly better.
- Useful aggressive point: `5/10/35/50`, cost 0.327. Degree gives 3.05% drop on Cora and 2.68% drop on PubMed.
- Useful no-P8/no-P6 point: `0/0/60/40`, cost 0.319. Context gives 2.86% drop on Cora and Degree gives 2.93% drop on PubMed.

## Reuse + Graph-Bit Results

Here reuse hits are treated as direct cache reads. Only hash misses are routed to P8/P6/P5/P4 with a 20/30/30/20 miss-node budget.

This table is useful for isolating miss-node Graph-Bit behavior, but it is not the final full-stack policy because fuzzy hits are not residual-corrected.

### T = 20

| Dataset | Reuse | FullP8 Cost | FullP8 Drop | Graph-Bit Cost | Random | Degree | TSER | Context | LowUnique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 4.5 | 0.477 | 0.34 | 0.361 | 2.85 | 2.24 | 2.84 | 2.61 | 2.50 |
| PubMed | 31.3 | 0.344 | 2.50 | 0.260 | 4.01 | 3.82 | 3.90 | 4.12 | 4.40 |

### T = 30

| Dataset | Reuse | FullP8 Cost | FullP8 Drop | Graph-Bit Cost | Random | Degree | TSER | Context | LowUnique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 46.9 | 0.265 | 3.88 | 0.200 | 6.58 | 5.58 | 6.09 | 6.01 | 6.21 |
| PubMed | 77.1 | 0.114 | 6.03 | 0.086 | 7.34 | 7.00 | 7.14 | 7.48 | 7.81 |

Main observation:

When reuse is conservative, Graph-Bit behavior is visible and Degree protects better than Random. When reuse is aggressive, total drop is dominated by direct reuse error; Graph-Bit still saves miss-node cost, but it cannot fix bad reuse hits.

## Residual Reuse + Graph-Bit Full Stack

Correct full-stack policy:

```text
support >= 5 hit -> direct cache reuse
support == 4 hit -> residual correction
support < 4      -> Graph-Bit P8/P6/P5/P4
high-risk miss   -> P8 through the same budget router
```

Before the learned accept gate was added, the support-split reuse front-end for Graph-Bit experiments was:

```text
R = 2
8 heads x 16-bit
score threshold T = 40
hard direct reuse: support heads >= 5
residual correction: support heads == 4
compute / Graph-Bit: support heads < 4
```

This is the `h8_54_T40` operating point from the earlier residual reuse sweep:

```text
Cora:   reuse = 25.7%, drop = 0.45%
PubMed: reuse = 50.3%, drop = 2.52%
```

It is preferred over `h8_64_T40` because both have the same reuse, but routing support=5 hits to direct reuse gives lower PubMed drop than forcing them through residual correction.

The current ST/data.x pure residual-reuse front-end is stronger after adding the learned accept gate:

```text
T = 30
support >= 5   -> hard direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute
gate_accept_threshold = 0.575
```

3-run result:

| Dataset | Reuse | Drop |
|---|---:|---:|
| Cora | 46.5% | 0.93% |
| PubMed | 42.3% | 1.96% |

See `docs/archive/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md`.

For Cora/LLaMA, the learned-gate front-end has now been adapted to W4A8 LLaMA embeddings:

```text
T = 30
support >= 5   -> hard direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute
shared accept gate threshold = 0.60
```

3-run Cora/LLaMA W4A8 target result:

| Config | Reuse | Drop |
|---|---:|---:|
| DirectReuse | 16.1% | 0.53% |
| SoftDirectReuse | 50.4% | 2.97% |
| ResidualReuse | 29.7% | 1.47% |

The `residual_precision_depth` runner is now wired to this learned accept gate.  Gate-rejected fuzzy hits are removed from the accepted reuse set and routed to the miss-node Graph-Bit path.

Current Cora/LLaMA Graph-Bit smoke command:

```bash
DATASET=cora RUNS=1 RUN_ALGO=1 RUN_ONNXIM=0 BUDGET=p8heavy \
  bash scripts/run_graphbit_predictor_free_flow.sh
```

Smoke result, seed 42, `p8heavy` budget:

| Method | Reuse | P8 | P6 | P4 | Cost/Cycles Proxy | Drop |
|---|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 29.4% | 70.6% | 0.0% | 0.0% | 0.707 | 1.60% |
| Random static P8/P6/P4 | 29.4% | 56.5% | 14.1% | 0.0% | 0.672 | 2.27% |
| Degree static P8/P6/P4 | 29.4% | 56.5% | 14.1% | 0.0% | 0.672 | 2.18% |
| Degree predictor-free EarlyStop | 29.4% | 56.5% | 14.1% | 0.0% | 0.663 | 2.18% |

This confirms the new wiring: learned-gate rejected hits are counted as misses, and Degree Graph-Bit is applied only to the remaining encoder-executed nodes.

The Graph-Bit tables below that mention `h8_54_T40` are historical support-split baselines. They remain useful for NPU datapath validation and cost accounting, but they should not be presented as the latest learned-gate full-stack result.

The `FullP8` row below still includes reuse. It means "reuse hits use direct/residual, all misses use P8". It is the correct baseline for measuring how much extra error Graph-Bit adds on top of the reuse subsystem.

### Historical Cora / LLaMA-7B, fixed `h8_54_T40`, three-depth Graph-Bit

This is the current Cora full-stack main table.  It fixes the residual reuse front-end to the common robust setting:

```text
R = 2
heads = 8 x 16-bit
score threshold T = 40
hard direct reuse: support >= 5
residual correction: support == 4
compute / Graph-Bit: support < 4
```

For the miss nodes, this run uses a hardware-friendly three-depth Graph-Bit budget:

```text
P8: 20% of miss nodes
P6: 50% of miss nodes
P4: 30% of miss nodes
```

Reproduction command:

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 10 \
  --experiment_suite residual_precision_depth \
  --real_quant_model_name llama2_7b \
  --precision_depth_reference_tag W4A8 \
  --precision_depth_tags W4A6 W4A4 \
  --precision_depth_bits 6 4 \
  --precision_depth_reference_bits 8 \
  --precision_depth_high_ratio 0.20 \
  --precision_depth_mid_ratio 0.50 \
  --precision_depth_low_ratio 0.30 \
  --precision_depth_cost_scale 0.50 \
  --precision_depth_fixed_cost 0.15 \
  --radius 2 \
  --hash_heads_per_route 8 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 40 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --residual_fit_profile llama \
  --residual_rank 64 \
  --residual_epochs 120 \
  --residual_max_train_pairs 4096 \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 4 \
  --residual_alpha_grid 0 0.125 0.25 0.5 \
  --residual_min_dist 1.0
```

Log:

```text
output/residual_graphbit_main/cora_h8_54_T40/cora_h8_54_T40_runs10.log
```

Result:

| Config | Reuse | Direct | Residual | P8 | P6 | P4 | Cost | Acc | Drop | FinalErr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 | 40.0% | 21.9% | 18.0% | 60.0% | 0.0% | 0.0% | 0.301 | 0.7015 | 1.53% | 0.06888 |
| AllP6 | 40.0% | 21.9% | 18.0% | 0.0% | 60.0% | 0.0% | 0.237 | 0.6965 | 2.03% | 0.07283 |
| AllP4 | 40.0% | 21.9% | 18.0% | 0.0% | 0.0% | 60.0% | 0.173 | 0.6653 | 5.14% | 0.09536 |
| Random | 40.0% | 21.9% | 18.0% | 12.0% | 30.0% | 18.0% | 0.231 | 0.6888 | 2.79% | 0.07892 |
| Degree | 40.0% | 21.9% | 18.0% | 12.0% | 30.0% | 18.0% | 0.231 | 0.6928 | 2.39% | 0.07836 |
| TSER | 40.0% | 21.9% | 18.0% | 12.0% | 30.0% | 18.0% | 0.231 | 0.6891 | 2.77% | 0.07847 |
| Context | 40.0% | 21.9% | 18.0% | 12.0% | 30.0% | 18.0% | 0.231 | 0.6898 | 2.69% | 0.07837 |
| LowUnique | 40.0% | 21.9% | 18.0% | 12.0% | 30.0% | 18.0% | 0.231 | 0.6897 | 2.71% | 0.07909 |

Observation:

```text
FullP8 miss baseline:
    cost = 0.301, drop = 1.53%

Degree Graph-Bit:
    cost = 0.231, drop = 2.39%
```

At the same reuse set, Degree-based Graph-Bit is better than Random (`2.39%` vs `2.79%` drop) and gives about `23%` extra cost reduction relative to FullP8-miss (`0.301 -> 0.231`).  This supports the current design direction: reuse controls whether a node executes the encoder, and graph-risk-controlled bit-depth reduces NPU arithmetic effort for the remaining miss nodes.

### PubMed/LLaMA Support-Split Follow-Up

The ST residual-reuse sweep selected `h8_54_T40` as a good common front-end.
However, LLaMA-7B full-stack evaluation is stricter: the first check must be
`FullP8-miss`, where accepted hits use direct/residual reuse and all misses still
run P8. If `FullP8-miss` is already above the target drop, Graph-Bit cannot fix
the run because Graph-Bit only changes miss-node bit-depth.

PubMed/LLaMA, 10 runs, `T=40`, `R=2`, 8 heads:

| Front-end | Hard / Residual | Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | Degree Static Cost | Degree Static Drop | Degree Bound Cost | Degree Bound Drop | Bound AvgBit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `h8_54_T40` | `>=5 / ==4` | 54.1% | 42.8% | 11.3% | 0.230 | 3.01% | 0.176 | 3.80% | 0.184 | 3.54% | 6.10 |
| `h8_76_T40` | `>=7 / ==6` | 8.2% | 4.7% | 3.4% | 0.459 | 0.26% | 0.352 | 1.74% | 0.367 | 1.24% | 6.10 |

Runtime-bound NPU proxy:

| Front-end | Method | Cycles | Traffic | Energy | Saved AvgBit |
|---|---|---:|---:|---:|---:|
| `h8_54_T40` | FullP8-miss | 0.460 | 0.461 | 0.461 | 0.00 |
| `h8_54_T40` | Degree static | 0.334 | 0.417 | 0.371 | 2.20 |
| `h8_54_T40` | Degree runtime-bound | 0.351 | 0.423 | 0.383 | 1.90 |
| `h8_76_T40` | FullP8-miss | 0.918 | 0.918 | 0.918 | 0.00 |
| `h8_76_T40` | Degree static | 0.666 | 0.831 | 0.740 | 2.20 |
| `h8_76_T40` | Degree runtime-bound | 0.701 | 0.843 | 0.765 | 1.90 |

Current interpretation:

- `h8_54_T40` is still useful as the ST/data.x residual-reuse common point, but it is too loose for PubMed/LLaMA full-stack. Even when misses use full P8, drop is already `3.01%`.
- `h8_76_T40` is the current PubMed/LLaMA robust point: `FullP8-miss` is very safe (`0.26%` drop), and Degree runtime-bound stays under `2%` (`1.24%`). Its drawback is low reuse (`8.2%`).
- Runtime-bound behavior is visible in both front-ends: the low-risk bucket is not forced to P4; the bound pushes it to P5. This improves accuracy versus static Degree (`3.80% -> 3.54%` for `h8_54_T40`, `1.74% -> 1.24%` for `h8_76_T40`) at a small cost increase.
- Therefore, Graph-Bit should be evaluated only after the reuse/residual front-end keeps `FullP8-miss` below the desired accuracy budget.

Result files:

```text
output/graphbit_bound_runtime/pubmed_h8_54_T40_boundclean_runs10/predictor_free_main.txt
output/graphbit_bound_runtime/pubmed_h8_76_T40_boundclean_runs10/predictor_free_main.txt
```

### LLaMA-7B, T = 20

| Dataset | Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | Graph-Bit Cost | Random | Degree | TSER | Context | LowUnique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 4.5 | 0.5 | 4.0 | 0.477 | 0.27 | 0.361 | 2.74 | 2.22 | 2.77 | 2.55 | 2.50 |
| PubMed | 31.3 | 4.1 | 27.2 | 0.345 | 2.79 | 0.261 | 4.16 | 3.90 | 3.97 | 4.21 | 4.46 |

### LLaMA-7B, T = 30

| Dataset | Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | Graph-Bit Cost | Random | Degree | TSER | Context | LowUnique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 46.9 | 3.6 | 43.4 | 0.267 | 3.68 | 0.203 | 5.53 | 5.25 | 5.34 | 5.34 | 5.45 |
| PubMed | 77.2 | 8.3 | 68.9 | 0.118 | 6.02 | 0.090 | 6.75 | 6.60 | 6.65 | 6.78 | 7.00 |

### Backend sanity check: Cora/ST, T = 30

| Dataset/Backend | Reuse | Direct | Residual | FullP8 Cost | FullP8 Drop | Graph-Bit Cost | Random | Degree | TSER | Context | LowUnique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora/ST | 35.5 | 3.7 | 31.9 | 0.324 | 1.90 | 0.245 | 2.26 | 1.81 | 2.22 | 2.03 | 2.61 |

Main observation:

- The old Cora/ST residual result (`T=30`, drop about 2.43%) should not be directly compared with LLaMA-7B full-stack results. LLaMA raw embeddings are harder for CAM/residual reuse.
- For LLaMA-7B, `T=20` is the safer full-stack operating point. `T=30` is too aggressive on PubMed because reuse hit error dominates even when misses use P8.
- Once reuse error is controlled, Graph-Bit behaves as expected: Degree is consistently better than Random on PubMed and competitive on Cora.
- Graph-Bit is a miss-node NPU optimization. It cannot compensate for overly aggressive fuzzy reuse. The reuse gate and residual path must first keep `FullP8` drop acceptable.

## Implementation Note

The combined experiment now computes the CAM/hash reuse trace once per run and reuses the same `hit_mask/source_ids` for every P8/P6/P5/P4 policy. This matches the intended system behavior and avoids rerunning CAM for every routing baseline.

For `residual_precision_depth`, the trace is applied at raw embedding level:

```text
selected_raw = P8/P6/P5/P4 pool for miss nodes
exact hits   = reference_raw[source]
fuzzy hits   = residual_adapter(reference_raw[source], pair features)
final_raw    -> GNN encoder/classifier
```

## Takeaway

The pure Graph-Bit sweep supports the core mechanism: graph risk can control NPU arithmetic precision depth better than random assignment at the same cost.

For the full stack, reuse and Graph-Bit should be evaluated as two coupled but separate controls:

- reuse gate decides whether a node can skip encoder execution,
- Graph-Bit decides how deeply the NPU computes bit-planes for nodes that still execute the encoder.

## ONNXim Miss-Only Breakdown

To isolate whether predictor-free Graph-Bit actually saves NPU-internal bit-plane work, we also report a miss-only ONNXim breakdown:

```text
output/graphbit_predictor_free/cora_h8_54_T40/earlystop_sweep/miss_only_breakdown.txt
```

This table ignores direct reuse and residual reuse, and only evaluates nodes that must execute the encoder.

| Method | AvgD | BitComp | ActRd | ActSave | Traffic |
|---|---:|---:|---:|---:|---:|
| FullP8-miss | 8.00 | 1.000 | 1.000 | 0.0% | 1.000 |
| Static Degree P8/P6/P4 | 5.80 | 0.726 | 0.725 | 27.5% | 0.964 |
| EarlyStop conservative | 6.90 | 0.864 | 0.862 | 13.8% | 0.982 |
| EarlyStop balanced | 6.10 | 0.764 | 0.763 | 23.7% | 0.969 |
| EarlyStop aggressive | 5.80 | 0.726 | 0.725 | 27.5% | 0.964 |

Conclusion:

- Balanced early-stop reduces miss-node effective bit-serial compute to `76.4%` of FullP8.
- Activation/input reads also fall to `76.3%`, so the bit-plane mechanism is working.
- Total traffic only drops to `96.9%` because weight reads and output writes are unchanged.
- Therefore, full-stack cycle/traffic gains are modest because the NPU still pays fixed weight/output costs and because reuse already removes part of the encoder workload.

## Risk-Bucket Batching

A Graph-Bit NPU also needs graph-aware batching.  If high/mid/low risk miss nodes are randomly mixed inside the same bit-serial micro-batch, the shared bit-plane controller must execute to the maximum depth in that batch.

Command:

```bash
bash GraphhopSimhash/scripts/run_cora_graphbit_risk_bucket_batching.sh
```

Output:

```text
output/graphbit_predictor_free/cora_h8_54_T40/risk_bucket_batching/risk_bucket_batching.txt
```

Cora `h8_54_T40`, batch size 64:

| Method | Assign | Schedule | Mode | UsefulD | ExecD | Util | Waste | Cycles | BitComp | ActRd | Traffic | Drop |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomOrder Static | Degree | random mixed | static | 5.80 | 8.00 | 72.5% | 27.5% | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| DegreeBucket Static | Degree | risk bucket | static | 5.80 | 5.80 | 100.0% | 0.0% | 0.957 | 0.726 | 0.725 | 0.964 | 2.39% |
| RandomRisk Bucket | Random | risk bucket | static | 5.80 | 5.80 | 100.0% | 0.0% | 0.957 | 0.726 | 0.725 | 0.964 | 2.79% |
| RandomOrder EarlyStop | Degree | random mixed | early-stop | 6.10 | 8.00 | 76.3% | 23.7% | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| DegreeBucket EarlyStop | Degree | risk bucket | early-stop | 6.10 | 6.10 | 100.0% | 0.0% | 0.959 | 0.764 | 0.763 | 0.969 | 2.39% |

Takeaway:

- Random ordering can erase the arithmetic savings, because a single high-risk node can force the whole micro-batch to P8.
- Degree-risk buckets allow the NPU to realize the intended bit-depth reduction.
- Random-risk buckets have the same hardware cost as degree-risk buckets but worse accuracy, so the graph proxy matters for precision protection.

## Bit-Plane Demand-Fetch Model

The miss-only breakdown shows that bit-plane compute and activation reads can fall, but it does not by itself explain how much full-stack cycles/traffic improve.  We therefore added a demand-fetch model that separates:

```text
BitComp:
    bit-plane MAC work inside the PE array

ActRd:
    activation bit-plane reads

WgtRd / OutWr:
    fixed weight reads and output writes

Sched:
    whether the NPU executes risk buckets separately or randomly mixes risks
```

Command:

```bash
bash GraphhopSimhash/scripts/run_graphbit_demand_fetch_model.sh
```

Default output:

```text
output/graphbit_predictor_free/cora_h8_53_T30/demand_fetch_model/demand_fetch_model.txt
```

Historical balanced frontend output:

```bash
WORKLOAD=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_workload.json \
OUT_DIR=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/cora_h8_54_T40/demand_fetch_model \
bash GraphhopSimhash/scripts/run_graphbit_demand_fetch_model.sh
```

Key Cora/LLaMA results:

| Frontend | Method | Sched | UsefulD | ExecD | BitComp | ActRd | WgtRd | Full cycles | Full traffic | Drop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h8_53_T30 p8heavy | FullP8-miss | risk bucket | 8.00 | 8.00 | 1.000 | 1.000 | 1.000 | 0.707 | 0.707 | 1.60% |
| h8_53_T30 p8heavy | Degree demand-fetch | risk bucket | 7.60 | 7.60 | 0.950 | 0.950 | 1.000 | 0.700 | 0.703 | 2.18% |
| h8_54_T40 balanced | FullP8-miss | risk bucket | 8.00 | 8.00 | 1.000 | 1.000 | 1.000 | 0.601 | 0.602 | 1.53% |
| h8_54_T40 balanced | Degree compute-mask only | risk bucket | 5.80 | 5.80 | 0.726 | 1.000 | 1.000 | 0.601 | 0.602 | 2.39% |
| h8_54_T40 balanced | Degree random-mixed | random mixed | 6.10 | 8.00 | 1.000 | 1.000 | 1.000 | 0.601 | 0.602 | 2.39% |
| h8_54_T40 balanced | Degree demand-fetch | risk bucket | 6.10 | 6.10 | 0.764 | 0.762 | 1.000 | 0.576 | 0.583 | 2.39% |

Takeaway:

- `compute-mask only` saves bit-plane arithmetic but does not lower full-stack cycles/traffic.
- `random-mixed` loses the benefit because one high-risk node can force the whole bit-serial batch to P8.
- `Degree demand-fetch` is the actual NPU dataflow target: bit-plane-major activation layout plus risk-bucket scheduling.
- The latest `p8heavy` frontend is accuracy-first, so savings are small; the historical balanced frontend is useful for showing the hardware mechanism more clearly.

Detailed model:

```text
docs/npu/GRAPH_BIT_NPU_DESIGN.md
docs/archive/npu/GRAPH_BIT_TRACE_DRIVEN_SCHEDULER.md
```

## Closure Suite And Dynamic-Depth Follow-Up

The closure suite is the current shortest end-to-end check for the hardware story:

```bash
bash scripts/run_graphbit_closure_suite.sh
```

Output:

```text
output/graphbit_closure/cora/closure_table.txt
```

It compares the same Cora workload under four hardware interpretations:

```text
FullP8-miss:
    reuse/residual hits skip the encoder; all misses run P8.

compute-mask only:
    low-bit MACs are masked, but full A8 activations are still fetched.

random-mixed:
    low-depth nodes exist, but risk levels are mixed in the same bit-serial batch.

demand-fetch:
    low-bit activation planes are not fetched, and risk buckets are batched separately.
```

Key rows:

| Frontend | Budget | Method | UsefulD | ExecD | ActRd | FullC | FullT | Drop | Cycle Save |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| h8_53_T30 | p8heavy | FullP8-miss | 8.00 | 8.00 | 1.000 | 0.707 | 0.707 | 1.60% | 0.0% |
| h8_53_T30 | p8heavy | Degree demand-fetch | 7.60 | 7.60 | 0.950 | 0.700 | 0.703 | 2.18% | 0.9% |
| h8_54_T40 | balanced | FullP8-miss | 8.00 | 8.00 | 1.000 | 0.601 | 0.602 | 1.53% | 0.0% |
| h8_54_T40 | balanced | Degree compute-mask only | 5.80 | 5.80 | 1.000 | 0.601 | 0.602 | 2.39% | 0.0% |
| h8_54_T40 | balanced | Degree random-mixed | 6.10 | 8.00 | 1.000 | 0.601 | 0.602 | 2.39% | 0.0% |
| h8_54_T40 | balanced | Degree demand-fetch | 6.10 | 6.10 | 0.762 | 0.576 | 0.583 | 2.39% | 4.1% |

This is the main proof point for the NPU dataflow:

```text
Graph-Bit needs both bit-plane-major demand fetch and graph-risk bucket scheduling.
Otherwise the bit-depth decision does not translate into visible cycles/traffic savings.
```

### Dynamic P5 Proxy

True predictor-free early stop should not be forced to stop exactly at P6 or P4. Many low-risk nodes can stop around P5. The conservative proxy below maps the low-depth bucket to W4A5:

```bash
RUNS=3 bash scripts/run_cora_graphbit_dynamic_depth_accuracy.sh
```

Output:

```text
output/graphbit_predictor_free/cora_h8_54_T40_dynp5/
```

3-run result:

| Method | Reuse | P8 | P6 | P5 | P4 | Cost | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| FullP8 | 28.9% | 71.1% | 0.0% | 0.0% | 0.0% | 0.356 | 1.08% |
| Random | 28.9% | 14.2% | 35.5% | 21.3% | 0.0% | 0.284 | 2.30% |
| Degree | 28.9% | 14.2% | 35.5% | 21.3% | 0.0% | 0.284 | 1.93% |

This improves over the older P4 low-risk bucket, where Degree drop was about `2.39%`. It supports treating P6/P5/P4 as validation anchors for dynamic early stop rather than hard final datatypes.

### PubMed Lightweight Replay

For hardware-model changes, PubMed can be replayed from an existing workload without rerunning the expensive residual/GNN path:

```bash
bash scripts/run_pubmed_graphbit_demand_fetch_model.sh
```

Output:

```text
output/graphbit_predictor_free/pubmed_h8_76_T40/demand_fetch_model/demand_fetch_model.txt
```

Key rows:

| Method | Reuse | UsefulD | ExecD | ActRd | FullC | FullT | Drop | Cycle Save |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8-miss | 22.3% | 8.00 | 8.00 | 1.000 | 0.778 | 0.779 | 1.26% | 0.0% |
| Degree compute-mask only | 22.3% | 5.80 | 5.80 | 1.000 | 0.777 | 0.778 | 2.54% | 0.1% |
| Degree random-mixed | 22.3% | 6.10 | 8.00 | 1.000 | 0.777 | 0.778 | 2.54% | 0.1% |
| Degree demand-fetch | 22.3% | 6.10 | 6.10 | 0.762 | 0.745 | 0.753 | 2.54% | 4.2% |

The PubMed replay agrees with Cora: compute-mask alone gives almost no system-level benefit, while risk-bucket demand-fetch exposes the bit-plane savings as cycle/traffic reduction.

## FFN Block-Gating Hardware Probe

Activation bit-plane early-stop does not reduce weight reads.  As a next-stage Graph-Bit+ candidate, we probed FFN intermediate block gating in ONNXim:

```bash
bash GraphhopSimhash/scripts/run_onnxim_ffn_block_gating_microbench.sh
```

Output:

```text
output/onnxim_graphbit/ffn_block_gating/ffn_block_gating.txt
```

| Keep | Intermediate | Cycles | MatMul | Traffic | InRead | WeightRd | OutWr | GFLOPs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 74% | 8192 | 0.826 | 0.829 | 0.830 | 0.834 | 0.829 | 0.867 | 0.829 |
| 50% | 5504 | 0.629 | 0.654 | 0.669 | 0.687 | 0.666 | 0.741 | 0.666 |

This is hardware-only.  It shows that FFN block gating can reduce the weight bandwidth that activation bit-depth control cannot touch.  It is not yet an accuracy-safe policy; the next step is to validate whether low-risk nodes can tolerate a conservative FFN block keep ratio such as `74%`.

This FFN block-gating probe is not part of the current mainline.  The mainline Graph-Bit NPU uses predictor-free bit-serial early-stop and risk-bucket batching.

## Memory Dataflow Breakdown

To separate compute reduction from traffic reduction:

```bash
bash GraphhopSimhash/scripts/run_cora_graphbit_memory_dataflow.sh
```

Output:

```text
output/graphbit_predictor_free/cora_h8_54_T40/memory_dataflow/memory_dataflow.txt
```

| Method | AvgD | BitComp | Cycles | ActRd | WeightRd | OutWr | Traffic | Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullP8 miss | 8.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.53% |
| EarlyStop compute-only | 6.10 | 0.764 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.39% |
| EarlyStop + ActPack | 6.10 | 0.764 | 0.959 | 0.763 | 1.000 | 1.000 | 0.969 | 2.39% |
| EarlyStop + ActPack + FFNBypass | 6.10 | 0.764 | 0.959 | 0.763 | 1.000 | 1.000 | 0.938 | 2.39% |

Takeaway:

- Compute-only early stop is not enough if activations are still fetched as A8.
- Activation bit-plane packing is required to translate bit-depth reduction into memory reduction.
- FFNBypass is an exact dataflow upper bound: keeping FFN intermediate on chip can reduce traffic further without changing the model function.

## Batch-Size Amortization

Risk-bucket scheduling also helps weight-stationary reuse.  Larger same-risk micro-batches amortize weight reads across more node tokens:

```bash
bash GraphhopSimhash/scripts/run_onnxim_batch_amortization.sh
```

Output:

```text
output/onnxim_graphbit/batch_amortization/batch_amortization.txt
```

| Micro-batch | Cyc/Node norm | Traffic/Node norm | Weight/Node norm | Input/Node norm |
|---:|---:|---:|---:|---:|
| 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16 | 0.507 | 0.510 | 0.500 | 1.000 |
| 32 | 0.266 | 0.265 | 0.250 | 1.000 |
| 64 | 0.143 | 0.143 | 0.125 | 1.000 |
| 128 | 0.080 | 0.082 | 0.062 | 1.000 |

This supports the scheduler design: grouping miss nodes into degree-risk buckets enables both bit-depth coherence and weight-tile amortization.
