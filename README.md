# GraphHopSimhash

GraphHopSimhash is a graph-aware LLM encoder execution prototype for text-attributed graph workloads such as Cora, PubMed, and OGBN-Arxiv.

The current mainline is:

```text
SimHash / LRU-CAM front-end
    -> skip encoder for high-confidence reusable nodes

Residual-Gate reuse
    -> repair or reject fuzzy CAM matches

Graph-Bit NPU
    -> for remaining miss nodes, use graph risk to control encoder GEMM execution
```

The project is organized as a full-stack encoder execution hierarchy, not as a standalone hash trick or a standalone quantization experiment.

## Execution Paths

```text
P0 Direct reuse
    high-confidence CAM hit
    read cached embedding

P1 Residual-Gate reuse
    fuzzy CAM hit
    MLP predicts residual delta and accept/reject score

P2 Graph-Bit NPU
    miss / rejected node
    run LLM encoder with graph-risk-guided bit-serial execution

P3 Full W4A8 encoder
    conservative fallback / reference path
```

Current shared online residual-gate front-end:

```text
8 heads x 16 bits
radius = 2
score gate = on
score weights = 3 / 1 / 1
score threshold T = 30

support >= 5   -> direct reuse
support = 3..4 -> residual candidate
support < 3    -> encoder / Graph-Bit
gate_accept_threshold = 0.575
```

Current ST 3-run result:

| Dataset | Reuse | Drop |
|---|---:|---:|
| Cora/ST | 46.5% | 0.93% |
| PubMed/ST | 42.3% | 1.96% |

Detailed results and commands are in:

```text
docs/results/GRAPH_BIT_MAIN_RESULTS.md
docs/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md
docs/results/ST_LLAMA_T31_SHARED_RETRIEVAL_RESULT.md
```

## Graph-Bit NPU

Graph-Bit handles nodes that cannot be safely reused. The core mechanism is:

```text
node tolerance:
    degree / propagation risk

runtime bound:
    A_low_bound(depth) * W_tile_abs_bound

op sensitivity:
    first version keeps this as 1
```

Graph risk does not directly assign a fixed P8/P6/P4 ratio. It sets node-level tolerance. The predictor-free numerical bound decides the actual stop depth for the current tile.

The NPU dataflow uses a weight-stationary systolic style:

```text
load W tile on chip
stream token rows from the same risk bucket
run bit-serial GEMM
stop low activation bits when the bound is satisfied
```

The two hardware effects are separated:

```text
Risk-bucket W-stationary scheduling:
    improves W tile service window and memory reuse.

Variable activation depth:
    reduces bit-serial PE / psum activity for miss nodes.
```

Main NPU docs:

```text
docs/npu/GRAPH_BIT_NPU_DESIGN.md
docs/npu/GRAPH_BIT_SYSTOLIC_FLASH_DATAFLOW.md
docs/npu/GRAPH_BIT_EARLY_STOP_IMPLEMENTATION.md
docs/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
docs/npu/LLAMA_ROOFLINE_PROFILE.md
```

## Repository Layout

```text
GraphhopSimhash/
    cli.py
        command-line argument definitions

    runner.py
        experiment suites and full-stack simulation flow

    controller.py
        SimHash/CAM lookup, reuse decision, structure/score checks

    scoring.py
        degree / TSER / graph-context scoring utilities

    residual_reuse.py
        residual adapter and accept-gate helpers

    real_quant.py
        real embedding pool loading

    precision_depth.py
        Graph-Bit precision-depth helpers

    generate_real_quant_pools.py
        W4A8/W4A6/W4A5/W4A4 embedding pool generator

    paths.py
        repo-relative model path resolution

    scripts/
        runnable experiment scripts and summary tools

    ONNXim/
        integrated ONNXim simulator and Graph-Bit microbench hooks

    CAM_sim/
        CAM frontend hardware simulator, reports, and hardware-focused docs

    docs/
        organized documentation
```

Documentation entry:

```text
docs/README.md
CAM_sim/README.md
```

## Running Location

Run experiments from the OFA repository root:

```bash
cd /home/zhangshangtong/Transformer/OFA
python -m GraphhopSimhash ...
```

If running elsewhere, add the OFA root to `PYTHONPATH`.

## Model Paths

Model paths are repo-relative by default and can be overridden by environment variables:

```bash
export GRAPHHOP_MODEL_ROOT=/path/to/models
export GRAPHHOP_LLAMA2_7B_PATH=/path/to/Llama-2-7b-ms
export GRAPHHOP_ST_PATH=/path/to/multi-qa-distilbert-cos-v1
export GRAPHHOP_BERT_PATH=/path/to/bert-base-uncased
```

Default logical paths:

```text
llama2_7b -> models/llama-7b/modelscope/Llama-2-7b-ms
ST        -> models/multi-qa-distilbert-cos-v1
BERT      -> models/bert-base-uncased
```

## Common Commands

Generate LLaMA-7B Graph-Bit embedding pools:

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A8 W4A6 W4A5 W4A4 \
  --batch_size 4 \
  --w4a_backend awq \
  --w4a_calib_samples 128 \
  --overwrite
```

Run the current Cora Graph-Bit trace replay flow:

```bash
cd /home/zhangshangtong/Transformer/OFA
RUNS=3 DATASET=cora bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

Run the shared online residual reuse experiment:

```bash
cd /home/zhangshangtong/Transformer/OFA
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite residual_reuse \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hash_heads_per_route 8 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --allow_rare_fuzzy \
  --score_reuse_threshold 30 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1 \
  --radius 2 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 3 \
  --residual_gate_accept_threshold 0.575
```

Full command variants are kept in:

```text
docs/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
docs/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md
docs/tools/量化+哈希命令.md
```

## Data And Output

The following are local artifacts and are kept out of git:

```text
cache_data/
models/
output/
```

Current result documents point to the exact output directories used for reported numbers.

## Push Helper

From the repo root:

```bash
./push.sh "commit message"
```

This stages tracked/untracked changes, commits if needed, and pushes the current branch to `origin`.
