# LLaMA2-7B Frontend Reuse Experiments

This experiment evaluates the graph-aware embedding reuse frontend on six TAG
node-classification workloads using the same LLaMA2-7B target space.

## Target Pools

All runs use the existing `W4BFPA8_B128` LLaMA2-7B embedding pools as the
reference encoder output:

```text
cache_data/{dataset}_llama2_7b_oracle_W4BFPA8_B128.pt
```

The six datasets are:

```text
cora
pubmed
arxiv
wikics
tape_products
tape_arxiv23
```

## Command

```bash
cd /home/zhangshangtong/Transformer/OFA/GraphhopSimhash
OUT_DIR=/home/zhangshangtong/Transformer/OFA/output/llama7b_frontend_reuse_six_datasets \
RUNS=3 \
FORCE=0 \
bash scripts/run_llama7b_frontend_reuse_six_datasets.sh
```

The current run is launched in tmux:

```bash
tmux attach -t llama7b_reuse_six
```

## Output

```text
output/llama7b_frontend_reuse_six_datasets/logs/
output/llama7b_frontend_reuse_six_datasets/summary.tsv
output/llama7b_frontend_reuse_six_datasets/selected_operating_points.tsv
output/llama7b_frontend_reuse_six_datasets/summary.md
```

The summarizer selects, for each dataset, the highest-reuse `ResidualReuse`
operating point whose drop is no more than 2% when such a point exists.

## Current Partial Result

Cora has finished the 3-run T sweep. Two useful operating points are:

| Dataset | T | Residual Reuse | Residual Drop | SoftDirect Reuse | SoftDirect Drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 28 | 32.3% | 0.97% | 42.6% | 1.81% |
| Cora | 31 | 39.0% | 1.81% | 51.7% | 2.58% |

This already shows the expected behavior: accepting all fuzzy hits raises reuse
but costs more accuracy, while the residual path keeps a larger fraction of the
reuse opportunity within the low-drop regime.

## Experiment Role

This table supports the first contribution: graph-aware embedding reuse reduces
the number of LLaMA encoder invocations before dense Transformer execution.

The log table reports three rows per operating point:

```text
DirectReuse:
    only high-support anchors use cached embeddings.

SoftDirectReuse:
    medium-support anchors are accepted without repair.

ResidualReuse:
    medium-support anchors go through the TSER-guided residual path.
```

The expected paper table should include the selected `ResidualReuse` row for
each dataset and an ablation table comparing `DirectReuse`, `SoftDirectReuse`,
and `ResidualReuse` on representative operating points.
