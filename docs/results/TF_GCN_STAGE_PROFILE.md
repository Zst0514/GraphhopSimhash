# TF Frontend vs GNN Backend Stage Profiling

This profiling supports the first motivation paragraph: in TAG inference with a LLaMA frontend, the Transformer encoding stage dominates the backend GNN stage.

## Measurement Scope

The frontend stage measures LLaMA-7B text encoding on sampled node texts and extrapolates to the full graph. The backend stage measures full-graph inference of a 3-layer GCN over cached LLaMA embeddings. Backend timing excludes training.

The script is:

```bash
GraphhopSimhash/scripts/profile_tf_gcn_stage_latency.py
```

## Recommended Command

Run this when the GPU is not occupied by long embedding generation:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/profile_tf_gcn_stage_latency.py \
  --datasets cora pubmed arxiv wikics tape_products tape_arxiv23 \
  --frontend-mode profile \
  --frontend-model llama2_7b \
  --frontend-config fp16 \
  --frontend-sample-nodes 256 \
  --frontend-batch-size 1 \
  --frontend-max-length 256 \
  --gcn-pool-tag W4BFPA8_B128 \
  --gcn-hidden-dim 256 \
  --gcn-warmup 5 \
  --gcn-repeats 20 \
  --output-dir output/tf_gcn_stage_profile_rtx4090
```

Outputs:

```text
output/tf_gcn_stage_profile_rtx4090/stage_profile.csv
output/tf_gcn_stage_profile_rtx4090/stage_profile.json
output/tf_gcn_stage_profile_rtx4090/stage_profile.md
```

## Smoke Test

The backend-only smoke test has passed on Cora:

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/profile_tf_gcn_stage_latency.py \
  --datasets cora \
  --frontend-mode skip \
  --gcn-warmup 1 \
  --gcn-repeats 3 \
  --output-dir output/tf_gcn_stage_profile_smoke
```

Observed Cora 3-layer GCN inference time is about 9 ms on the RTX 4090. Full frontend numbers should be generated after the current LLaMA-13B embedding job finishes, because the GPU is currently occupied.

## Paper Usage

Use the resulting table before the semantic-locality CDF in Motivation I. The intended logic is:

```text
LLaMA frontend dominates end-to-end TAG inference.
Therefore, reducing repeated frontend encoder execution matters more than optimizing the lightweight GNN backend alone.
Graph locality then motivates the SimHash-based encoder reuse profile.
```
