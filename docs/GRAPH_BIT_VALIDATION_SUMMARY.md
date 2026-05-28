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
exact hit      -> direct cache reuse
fuzzy hit      -> residual correction
reject / miss  -> Graph-Bit P8/P6/P5/P4
high-risk miss -> P8 through the same budget router
```

The `FullP8` row below still includes reuse. It means "reuse hits use direct/residual, all misses use P8". It is the correct baseline for measuring how much extra error Graph-Bit adds on top of the reuse subsystem.

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
