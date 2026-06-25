# GFMEngine Baseline

This folder contains a path-level reconstruction of the ASPDAC'25 GFMEngine
baseline for the local GraphHopSimhash frontend traces.

GFMEngine is not modeled as HEAT-style `W8A10/W4A2` bit-serial execution.  The
paper's core mechanism is PQ-based MatMul: offline codebook and activation-book
construction, followed by online centroid search and activation-book lookup.
The simulator follows the paper and excludes offline preparation time.

Run:

```bash
python GFMEngine/simulate_pq_frontend_paths.py
```

Main outputs:

- `GFMEngine/results/GFMENGINE_PQ_PATH_TIMING.md`
- `../output/gfmengine_pq_frontend_path_timing/pq_frontend_rows.tsv`
- `../output/gfmengine_pq_frontend_path_timing/pq_frontend_aggregate.tsv`
- `../output/gfmengine_pq_frontend_path_timing/pq_frontend_timing.json`

Important defaults are taken from the public paper text: `nc=256`, `1GHz`,
`16` PEs, `256KB` on-chip memory, and `256GB/s` HBM2.  The paper does not
publish enough per-layer cycle detail to reproduce exact simulator cycles, so
the missing microarchitectural choices are explicit command-line parameters.

The default simulator includes the attention residual (`QK^T` and `AV`) because
GFMEngine's PQ-based MatMul only replaces GEMMs involving model weights.  Use
`GFMEngine/results/GFMENGINE_PQ_OBJECTIVE_COMPARISON.md` for the current
comparison against TSER40.  Older peak-style reports are retained for
sensitivity/debugging, not as the primary claim.
