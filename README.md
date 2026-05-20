# GraphHop SimHash

Project-style refactor of `GraphAdaptiveMask_Withonehop.py`.

## Layout

- `cli.py`: command-line arguments and validation.
- `runner.py`: experiment orchestration, baseline training, route construction, evaluation.
- `controller.py`: SimHash cache, multi-route retrieval, structure checks, score gate.
- `scoring.py`: degree/context/rare-leaf sensitivity scoring.
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

## Run

Fast smoke test:

```bash
python -m GraphhopSimhash --datasets cora --runs 1 --max_test 100 --no_learned_hash_projection
```

Default experiment:

```bash
python -m GraphhopSimhash --datasets cora --runs 1
```

Useful score ablations:

```bash
python -m GraphhopSimhash --datasets cora --runs 1 --disable_score_gate
python -m GraphhopSimhash --datasets cora --runs 1 --score_reuse_threshold 80
python -m GraphhopSimhash --datasets cora --runs 1 --allow_rare_fuzzy
```
