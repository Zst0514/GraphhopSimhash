# GraphHop SimHash

Project-style refactor of `GraphAdaptiveMask.py`.

The package keeps the original experiment behavior by default. In particular,
the score gate is available but disabled unless `--enable_score_gate` is passed.

## Layout

- `cli.py`: command-line arguments and validation.
- `runner.py`: experiment orchestration, baseline training, route construction, evaluation.
- `controller.py`: SimHash cache, multi-route retrieval, structure checks, optional score gate.
- `scoring.py`: degree/context/rare-leaf sensitivity scoring and quantization policy.
- `real_quant.py`: real pre-generated FP/INT8/INT4 feature-pool policy evaluation.
- `features.py`: self/1-hop/2-hop hash feature construction.
- `projections.py`: raw and learned multi-head hash projections.
- `data.py`: OFA data loading and cheap-feature loading.
- `models.py`: lightweight GNN wrapper used for evaluation.
- `runtime.py`, `paths.py`, `config.py`: environment, paths, and dataset config helpers.

## Risk Gate

The default score gate keeps the original degree protection and adds a
low-degree uniqueness term:

```text
PropagationRisk_q      = quantized degree risk, 0..15
GraphContextRisk_q     = max(boundary risk, self-vs-context shift), 0..15
RarityRisk_q           = global SimHash bucket rarity from self cheap features, 0..15
LowDegreeUniqueRisk_q  = (15 - PropagationRisk_q) * RarityRisk_q / 15

Sensitivity_q =
  3 * PropagationRisk_q
  + 2 * GraphContextRisk_q
  + 2 * LowDegreeUniqueRisk_q

ReuseError_q = 1 for dist=0, 2 for dist=1, 4 for dist=2
Risk_q = Sensitivity_q * ReuseError_q
```

Decision rules:

- High-degree nodes are protected by `--score_hub_threshold`.
- Low-degree rare nodes block fuzzy reuse by default.
- Remaining candidates use `Risk_q <= --score_reuse_threshold`.

## Risk-Guided Quantization

The optional quantization policy reuses the same sensitivity score, but changes
the approximation error by action:

```text
INT4Risk_q = Sensitivity_q * --quant_int4_error
INT8Risk_q = Sensitivity_q * --quant_int8_error

INT4 if INT4Risk_q <= --quant_int4_threshold
else INT8 if INT8Risk_q <= --quant_int8_threshold
else Protected/full precision
```

This is an evaluation proxy: computed embeddings are fake-quantized per node
before they enter the cache, so later reuse also sees the quantization error.

## Run

Legacy-compatible baseline run:

```bash
python -m GraphhopSimhash --datasets cora --runs 1 --learned_hash_epochs 10 --learned_hash_dim 128 --hamming_only_acceptor
```

Fast smoke test:

```bash
python -m GraphhopSimhash --datasets cora --runs 1 --max_test 100 --no_learned_hash_projection
```

Score-gate run with the same base configuration:

```bash
python -m GraphhopSimhash --datasets cora --runs 1 --learned_hash_epochs 10 --learned_hash_dim 128 --hamming_only_acceptor --enable_score_gate
```

Risk-aware reuse plus quantization:

```bash
python -m GraphhopSimhash --datasets cora --runs 1 --learned_hash_epochs 10 --learned_hash_dim 128 --hamming_only_acceptor --enable_score_gate --enable_quant_policy
```

One-command ablation under the same trained baseline:

```bash
python -m GraphhopSimhash --datasets cora pubmed --runs 3 --learned_hash_epochs 10 --learned_hash_dim 128 --hamming_only_acceptor --experiment_suite quant_ablation
```

Real quantized feature-pool ablation:

```bash
python -m GraphhopSimhash --datasets cora pubmed --runs 3 --experiment_suite real_quant_ablation --real_quant_model_name llama2_7b --real_quant_fp_ratio 0.10 --real_quant_int8_ratio 0.20
```

See `REAL_QUANT.md` for required cache files and the exact DegreeTopK/TSERTopK
and DegreeCascade/TSERCascade policy definitions.

Useful score ablations:

```bash
python -m GraphhopSimhash --datasets cora --runs 1 --enable_score_gate --score_reuse_threshold 80
python -m GraphhopSimhash --datasets cora --runs 1 --enable_score_gate --allow_rare_fuzzy
python -m GraphhopSimhash --datasets cora --runs 1 --experiment_suite score_ablation
```
