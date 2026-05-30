# GraphHop SimHash

GraphhopSimhash is a graph-aware LLM encoder acceleration prototype for text-attributed graph workloads such as Cora, PubMed, and OGBN-Arxiv.

The current mainline is no longer just "SimHash reuse" or "degree-based quantization". The project now studies a full encoder execution hierarchy:

```text
P0: Exact hash reuse
    Reuse a cached embedding directly.

P1: Fuzzy hash reuse + residual correction
    Use CAM/SimHash to find an anchor embedding, then apply a lightweight residual adapter.

P2: Graph-Bit precision-depth encoder
    For nodes that must execute the LLM encoder, graph risk controls activation bit-plane depth inside the NPU.

P3: Full W4A8 encoder
    High-risk fallback path.
```

The key idea is:

```text
Graph structure controls how much LLM encoder work is necessary.
```

Reuse decides whether a node can skip the encoder. Graph-Bit decides how much bit-serial arithmetic the NPU should spend when the encoder cannot be skipped.

## Main Contributions

### 1. GraphHop SimHash / CAM Reuse

The system builds graph-context-aware hash signatures and uses multi-head SimHash/CAM lookup to find reusable anchor nodes.

Earlier ST/data.x residual-reuse support-split front-end:

```text
h8_54_T40
8 heads x 16 bits
radius R = 2
score threshold T = 40
support >= 5 -> direct reuse
support == 4 -> residual correction
support < 4  -> encoder / Graph-Bit
```

This front-end was the best common point found before adding the learned residual accept gate:

```text
Cora:   reuse 25.7%, drop 0.45%
PubMed: reuse 50.3%, drop 2.52%
```

The current pure residual-reuse recommendation adds an online shared support
split plus a learned accept gate inside the residual path:

```text
8 heads x 16 bits
radius R = 2
score threshold T = 30
support >= 5   -> direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute
gate_accept_threshold = 0.575
```

With dataset-specific offline residual/gate training and the same online control
flow, the 3-run result is:

```text
Cora:   reuse 46.5%, drop 0.93%
PubMed: reuse 42.3%, drop 1.96%
```

See [SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md](docs/results/SHARED_ONLINE_RESIDUAL_REUSE_RESULT.md).

For LLaMA-7B full-stack experiments, the front-end must first pass the
`FullP8-miss` sanity check: accepted hits use direct/residual reuse, while all
misses still run P8.  The latest Cora/LLaMA residual-gate front-end is:

```text
h8_53_T30 + shared accept gate
8 heads x 16 bits
radius R = 2
score threshold T = 30
support >= 5   -> direct reuse
support = 3..4 -> residual candidate
support < 3    -> encoder / Graph-Bit
gate_accept_threshold = 0.60
```

This uses LLaMA W4A8 embeddings as the residual target rather than the ST/data.x
target.  PubMed/LLaMA still needs a stricter split/gate validation before it can
be treated as final.

Older PubMed/LLaMA support-split checks showed that a stricter no-gate split can
be used as a safe fallback:

```text
h8_76_T40
8 heads x 16 bits
radius R = 2
score threshold T = 40
support >= 7 -> direct reuse
support == 6 -> residual correction
support < 6  -> encoder / Graph-Bit
```

### 2. TSER Score Gate

TSER means Topology-aware Semantic Error Risk. It scores reuse risk using:

```text
propagation risk
graph context risk
low-degree uniqueness risk
candidate confidence / support
```

In the current results, TSER is valuable for reuse filtering and analysis. For Graph-Bit precision-depth routing, degree / propagation risk is the most stable deployable proxy.

### 3. Residual-Corrected Reuse

Fuzzy reuse is not simply "copy the nearest embedding". The residual path uses:

```text
anchor embedding E_u
cheap feature delta
graph/context delta
low-rank adapter
```

to estimate a corrected embedding:

```text
E_v_hat = normalize(E_u + alpha * residual(v, u))
```

This path is only used for medium-confidence fuzzy hits. Exact/high-support hits use direct reuse, and unsafe hits are rejected into the encoder path.

### 4. AWQ-Based W4A8 / W4A6 / W4A5 / W4A4 Embedding Pools

The current embedding generation path uses the official llm-awq style weight quantization path plus activation fake quantization for multiple activation depths:

```text
W4A8 -> P8 reference
W4A6 -> P6 proxy
W4A5 -> P5 proxy
W4A4 -> P4 proxy
```

For LLaMA-7B, these pools are used to validate graph-conditioned precision-depth execution.

### 5. Graph-Bit NPU

Graph-Bit is the hardware-facing contribution. It does not just route nodes to cached W4A8/W4A4 pools. It maps graph risk into the NPU datapath:

```text
high-risk node:
    execute more activation bit-planes

low-risk node:
    allow early termination of low activation bit-planes
```

The current ONNXim-backed prototype supports:

```text
static precision-depth proxy:
    P8/P6/P5/P4 as fixed execution depths

predictor-free early stop:
    all nodes start from P8 high bits
    degree risk chooses min_depth and tolerance
    bit-level bound decides when lower bit-planes can stop
```

In the predictor-free version, P6/P4 are not fixed datatypes. They are safety floors / validation anchors.

The current hardware model separates four effects:

```text
bit-plane compute:
    PE work saved by early stopping low activation bits

activation demand fetch:
    bit-plane-major layout avoids fetching skipped low-bit planes

risk-bucket batching:
    high/mid/low risk nodes are scheduled separately so one P8 node does not drag a whole batch to P8

fixed traffic:
    weight reads and output writes remain unless a separate weight-stationary / fused-FFN design attacks them
```

The demand-fetch model is documented in:

```text
docs/npu/GRAPH_BIT_DEMAND_FETCH_MODEL.md
```

## Repository Layout

```text
GraphhopSimhash/
    cli.py
        command-line argument definitions

    runner.py
        main experiment suites

    controller.py
        SimHash/CAM lookup, reuse decision, structure/score checks

    scoring.py
        propagation, graph context, low-degree uniqueness, TSER components

    residual_reuse.py
        low-rank residual adapter and residual reuse helpers

    real_quant.py
        real embedding pool loading and quantization routing utilities

    precision_depth.py
        Graph-Bit precision-depth pool utilities

    generate_real_quant_pools.py
        FP/W4A16/W4A8/W4A6/W4A5/W4A4 embedding pool generator

    paths.py
        repo/model path resolution

    scripts/
        runnable experiment scripts and summary tools

    ONNXim/
        integrated ONNXim simulator with Graph-Bit microbenchmark hooks

    docs/
        organized project documentation
```

Documentation folders:

```text
docs/core/
    AWQ embedding generation, CAM design, score definitions, residual reuse

docs/npu/
    Graph-Bit NPU design, predictor-free bit-serial execution, proxy experiments

docs/results/
    main result summaries

docs/survey/
    LLM / Transformer accelerator survey

docs/tools/
    ONNXim guide and command notes
```

Start here:

```text
docs/README.md
docs/PROJECT_ROADMAP.md
docs/core/RESIDUAL_CORRECTED_REUSE.md
docs/npu/GRAPH_BIT_NPU_DESIGN.md
docs/npu/GRAPH_CONDITIONED_BIT_SERIAL_EXECUTION.md
docs/results/GRAPH_BIT_VALIDATION_SUMMARY.md
```

## Running Location

Run commands from the OFA repository root:

```bash
cd /home/zhangshangtong/Transformer/OFA
```

Then use:

```bash
python -m GraphhopSimhash ...
```

If running from another directory, make sure the OFA root is in `PYTHONPATH`.

## Model Paths

Model paths are now repo-relative by default and can be overridden by environment variables.

Default paths:

```text
llama2_7b -> models/llama-7b/modelscope/Llama-2-7b-ms
ST        -> models/multi-qa-distilbert-cos-v1
BERT      -> models/bert-base-uncased
```

Optional overrides:

```bash
export GRAPHHOP_MODEL_ROOT=/path/to/models
export GRAPHHOP_LLAMA2_7B_PATH=/path/to/Llama-2-7b-ms
export GRAPHHOP_ST_PATH=/path/to/multi-qa-distilbert-cos-v1
export GRAPHHOP_BERT_PATH=/path/to/bert-base-uncased
```

This keeps the code portable across machines without hard-coding one user's absolute paths.

## Generate Embedding Pools

### ST / Cora + PubMed

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora pubmed \
  --llm_name ST \
  --configs W4A16 W4A8 W4A4 \
  --batch_size 64 \
  --awq_calib_samples 16 \
  --awq_seqlen 128 \
  --overwrite
```

### LLaMA-7B / Cora

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name llama2_7b \
  --configs W4A8 W4A6 W4A5 W4A4 \
  --batch_size 4 \
  --awq_calib_samples 128 \
  --awq_seqlen 512 \
  --overwrite
```

Output format:

```text
cache_data/{dataset}_{model}_oracle_{tag}.pt
```

Examples:

```text
cache_data/cora_llama2_7b_oracle_W4A8.pt
cache_data/cora_llama2_7b_oracle_W4A6.pt
cache_data/cora_llama2_7b_oracle_W4A5.pt
cache_data/cora_llama2_7b_oracle_W4A4.pt
```

`cache_data/`, `models/`, and generated `output/` logs are not intended to be committed.

## Core Experiment Suites

### 1. Residual Reuse Front-End

This reproduces the current common front-end parameter point:

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite residual_reuse \
  --radius 2 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --score_reuse_threshold 40 \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --residual_fit_profile llama \
  --residual_hard_min_support_hits 5 \
  --residual_soft_min_support_hits 4 \
  --residual_rank 64 \
  --residual_epochs 120 \
  --residual_max_train_pairs 4096 \
  --residual_min_dist 1.0 \
  --residual_alpha_grid 0 0.125 0.25 0.5
```

More details:

```text
docs/core/RESIDUAL_CORRECTED_REUSE.md
```

### 2. Pure Graph-Bit Precision-Depth Routing

This isolates the question:

```text
If every node must execute the encoder, can graph risk decide P8/P6/P5/P4 better than random?
```

Command:

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
  --precision_depth_high_ratio 0.20 \
  --precision_depth_mid_ratio 0.30 \
  --precision_depth_low_ratio 0.30 \
  --score_propagation_weight 3 \
  --score_graph_context_weight 1 \
  --score_low_unique_weight 1
```

Main observation:

```text
Degree / propagation risk is the most stable deployable precision-depth proxy.
```

### 3. Residual Reuse + Graph-Bit Full Stack

This is the main full-stack software experiment:

```text
exact hit      -> direct reuse
fuzzy hit      -> residual correction
reject / miss  -> Graph-Bit P8/P6/P5/P4
```

Recommended Cora smoke run with the current LLaMA-aware learned gate:

```bash
DATASET=cora RUNS=1 RUN_ALGO=1 RUN_ONNXIM=0 BUDGET=p8heavy \
  bash scripts/run_graphbit_predictor_free_flow.sh
```

Historical Cora `h8_54_T40` 10-run result before the learned accept gate was
wired into `residual_precision_depth`:

```text
FullP8 miss baseline:
    cost = 0.301, drop = 1.53%

Degree Graph-Bit:
    cost = 0.231, drop = 2.39%
```

Interpretation:

```text
At the same reuse set, Degree Graph-Bit reduces normalized encoder cost
relative to the FullP8 miss path, with extra accuracy drop.
```

The "cost" here is a normalized software cost proxy, not measured wall-clock time.

### 4. ONNXim Predictor-Free Early-Stop Validation

This validates the hardware-facing bit-serial mechanism:

```text
All miss nodes start from max_depth=8.
Graph risk selects min_depth and tolerance.
The NPU stops low bit-planes when a predictor-free bound is satisfied.
```

Build ONNXim and run the Cora early-stop sweep:

```bash
bash GraphhopSimhash/scripts/build_onnxim.sh
FORCE_ONNXIM=1 bash GraphhopSimhash/scripts/run_cora_graphbit_earlystop_sweep.sh
```

Result file:

```text
output/graphbit_predictor_free/cora_h8_54_T40/earlystop_sweep/earlystop_sweep.txt
```

Current result:

```text
Method                     Reuse  AvgD  Saved  Stop   Cycles Traffic Energy Drop
FullP8-miss                 40.0%  8.00   0.00   0.0%   0.601   0.602  0.602  1.53%
Static Degree P8/P6/P4      40.0%  5.80   2.20   0.0%   0.575   0.581  0.578  2.39%
EarlyStop balanced          40.0%  6.10   1.90 100.0%   0.576   0.583  0.580  2.39%
EarlyStop aggressive        40.0%  5.80   2.20 100.0%   0.575   0.581  0.578  2.39%
```

Important:

```text
EarlyStop rows are hardware validation rows.
They no longer mean "fixed P6/P4".
They mean "start at P8, stop dynamically by bit-bound".
```

The drop is currently inherited from the static Degree proxy as a conservative accuracy estimate. A stricter next step is to generate dynamic-depth embeddings or nearest-depth conservative mappings for exact dynamic-depth accuracy.

### 5. Bit-Plane Demand-Fetch Modeling

The early-stop flow now has a demand-fetch model that distinguishes:

```text
compute-mask only:
    low bit MACs are masked, but A8 activations are still fetched

demand-fetch:
    skipped low bit-planes are not fetched

random-mixed batching:
    mixed-risk batches execute to the max depth in the batch

risk-bucket batching:
    high/mid/low risk nodes are batched separately
```

Run the default Cora/LLaMA learned-gate model:

```bash
bash GraphhopSimhash/scripts/run_graphbit_demand_fetch_model.sh
```

Historical balanced comparison:

```bash
WORKLOAD=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/cora_h8_54_T40/predictor_free_workload.json \
OUT_DIR=/home/zhangshangtong/Transformer/OFA/output/graphbit_predictor_free/cora_h8_54_T40/demand_fetch_model \
bash GraphhopSimhash/scripts/run_graphbit_demand_fetch_model.sh
```

Key conclusion:

```text
compute-mask only saves bit-plane arithmetic but not cycles/traffic;
demand-fetch + risk-bucket scheduling is required to turn early stop into NPU-visible savings.
```

## ONNXim Integration

ONNXim is integrated under:

```text
GraphhopSimhash/ONNXim/
```

Graph-Bit modifications currently touch:

```text
ONNXim/src/SystolicWS.cc
ONNXim/src/SystolicWS.h
ONNXim/src/operations/GemmWS.cc
scripts/onnxim_graphbit_microbench.py
```

The microbenchmark covers LLaMA-7B-style GEMM shapes:

```text
projection: 4096 x 4096
FFN up/gate: 4096 x 11008
FFN down: 11008 x 4096
```

Standalone microbenchmark:

```bash
python GraphhopSimhash/scripts/onnxim_graphbit_microbench.py \
  --seq-len 64 \
  --workspace output/onnxim_graphbit/microbench_s64_internal_p6 \
  --graphbit-depth 6 \
  --action all \
  --log-level info
```

Guide:

```text
docs/tools/ONNXIM_PROJECT_GUIDE.md
```

## Main Result Files

Useful output locations:

```text
output/residual_reuse/common_param_sweep_20260528/
    residual reuse common parameter sweep

output/residual_graphbit_main/cora_h8_54_T40/
    Cora residual + Graph-Bit main table

output/graphbit_predictor_free/cora_h8_54_T40/
    Cora ONNXim-backed predictor-free early-stop flow

output/llama7b_precision_depth_budget_sweep/
    LLaMA-7B precision-depth budget sweep

output/onnxim_graphbit/
    ONNXim Graph-Bit microbenchmarks
```

These output files are generated artifacts and are not part of the git-tracked source.

## Policy Boundaries

Deployable mainline policies:

```text
SimHash/CAM support
TSER score gate
Degree / propagation risk
Graph context risk
LowUnique risk
Predictor-free bit-serial early stop
```

Debug / upper-bound only:

```text
PredictorDepthBudget:
    requires calibration nodes to fit a damage predictor

OracleDamageBudget:
    requires true FP-vs-low-depth embedding errors

ErrorTopK / TSERErrorTopK:
    requires knowing real quantization error for every node
```

Do not present oracle/error-aware policies as deployable architecture mechanisms.

## Current Paper Direction

The cleanest current story is:

```text
Graph-aware hierarchical LLM encoder execution for text-attributed graphs.
```

System layers:

```text
1. CAM/SimHash finds reusable anchors.
2. TSER filters unsafe reuse.
3. Residual adapter corrects medium-confidence fuzzy reuse.
4. Degree-guided Graph-Bit controls encoder bit-plane effort for misses.
5. ONNXim validates the NPU datapath effect.
```

Hardware claim boundary:

```text
Safe to claim:
    Graph risk controls NPU bit-serial precision depth.
    ONNXim microbenchmarks estimate cycles / traffic / energy proxy.

Do not claim without further simulator evidence:
    exact wall-clock speedup
    exact silicon energy reduction
```

## Recommended Next Experiments

1. Validate a PubMed/LLaMA learned-gate front-end before running expensive Graph-Bit sweeps.
2. Add dynamic-depth accuracy validation for predictor-free early stop.
3. Sweep `min_depth/tolerance` for Cora first, then PubMed.
4. Add a weight-stationary / FFN-fusion model to attack the fixed weight/output traffic that demand-fetch cannot reduce.
5. Prepare the final paper figures:
   - reuse vs drop curve
   - Graph-Bit cost/drop curve
   - ONNXim cycles/traffic/energy table
   - full-stack path breakdown
