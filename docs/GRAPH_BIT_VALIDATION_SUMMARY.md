# Graph-Bit Validation Summary

Date: 2026-05-27

This summary records the first validation pass for Graph-Bit: graph-conditioned bit-depth execution for LLaMA-7B encoder embeddings on Cora and PubMed.

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
  --runs 3 \
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

Logs:

- `output/graph_bit_validation/precision_depth/*.log`
- `output/graph_bit_validation/reuse_precision_depth/*.log`

## Pure Graph-Bit Results

Each row uses the same P8/P6/P5/P4 budget across routing policies. Lower Drop is better.

### Cora / LLaMA-7B

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 10/20/30/40 | 0.346 | 3.61 | 3.30 | 3.55 | 3.27 | 3.27 | Context/LowUnique |
| 20/30/30/20 | 0.378 | 2.47 | 1.97 | 2.56 | 2.06 | 2.35 | Degree |
| 30/40/20/10 | 0.404 | 1.79 | 1.32 | 1.81 | 1.32 | 1.64 | Degree/Context |
| 50/30/20/0  | 0.436 | 0.92 | 0.77 | 0.76 | 0.60 | 0.98 | Context |

### PubMed / LLaMA-7B

| Budget P8/P6/P5/P4 | Cost | Random | Degree | TSER | Context | LowUnique | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| 10/20/30/40 | 0.346 | 2.52 | 2.15 | 2.26 | 2.52 | 2.78 | Degree |
| 20/30/30/20 | 0.378 | 1.80 | 1.52 | 1.61 | 1.78 | 2.15 | Degree |
| 30/40/20/10 | 0.404 | 1.35 | 1.03 | 1.18 | 1.31 | 1.75 | Degree |
| 50/30/20/0  | 0.436 | 0.84 | 0.64 | 0.76 | 0.74 | 1.05 | Degree |

Main observation:

Degree / propagation risk is the most stable deployable policy, especially on PubMed. Context is sometimes strong on Cora, but less stable across datasets.

## Reuse + Graph-Bit Results

Here reuse hits are treated as free cache reads. Only hash misses are routed to P8/P6/P5/P4 with a 20/30/30/20 miss-node budget.

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

## Implementation Note

The combined experiment now computes the CAM/hash reuse trace once per run and reuses the same `hit_mask/source_ids` for every P8/P6/P5/P4 policy. This matches the intended system behavior and avoids rerunning CAM for every routing baseline.

## Takeaway

The pure Graph-Bit sweep supports the core mechanism: graph risk can control NPU arithmetic precision depth better than random assignment at the same cost.

For the full stack, reuse and Graph-Bit should be evaluated as two coupled but separate controls:

- reuse gate decides whether a node can skip encoder execution,
- Graph-Bit decides how deeply the NPU computes bit-planes for nodes that still execute the encoder.

