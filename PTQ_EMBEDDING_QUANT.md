# Calibration-Aware W4A8/W4A4 Embedding PTQ

This document describes the PTQ embedding-pool generation path implemented in
`generate_real_quant_pools.py` through:

```bash
--w4a_backend ptq
```

The goal is to generate low-precision Transformer-front-end embedding pools for
TAG/GFM experiments, especially `W4A8` and `W4A4`, with a calibration procedure
that is more meaningful than naive fake quantization.

## Motivation

The original `FakeQuantLinear` path only rounds weights and activations to low
bitwidth and immediately dequantizes them before an FP linear operation. This is
useful as a stress test, but it is too crude for a serious W4A4 embedding study:

```text
naive fake quant:
    W -> round/clip/dequant
    X -> round/clip/dequant
    Y = Linear(X_q, W_q)
```

The problem is that W4A4 is dominated by activation outliers. If the activation
range is chosen by raw max values, most 4-bit buckets are wasted on rare large
values, and the dense part of the activation distribution is poorly represented.
This can severely damage the final sentence/node embedding.

The new PTQ path uses a calibration-aware pipeline:

```text
Calibration-aware W4A4/W4A8 PTQ =
    group-wise W4 weight quantization
  + layer-wise activation-aware scale search
  + SmoothQuant/AWQ-style scale migration
  + percentile activation clipping
  + dynamic per-token activation quantization
  + output embedding affine alignment
  + damage check against FP16 embeddings
```

## High-Level Pipeline

For each dataset/model/config pair:

```text
1. Load the FP16 Transformer encoder.
2. Replace every nn.Linear with CalibratedPTQLinear.
3. Run a calibration forward pass on sampled node texts.
4. For each Linear layer:
      search scale mode, alpha, activation clipping threshold
      choose the candidate minimizing layer output MSE.
5. Extract all node embeddings using finalized quantized Linear layers.
6. Optionally fit output affine alignment on calibration embeddings.
7. Save the final embedding pool:
      cache_data/{dataset}_{model}_oracle_{tag}.pt
8. Print cosine damage statistics against FP16 embeddings.
```

The implementation is in:

```text
generate_real_quant_pools.py
    CalibratedPTQLinear
    groupwise_weight_quantize
    dynamic_activation_quantize
    fit_output_alignment
```

## Calibration Set

The calibration set is selected before quantization:

```bash
--w4a_calib_samples 128
--calibration_strategy random
--seed 42
```

Currently supported strategies:

```text
first:
    Use the first N raw texts.

random:
    Use N randomly sampled raw texts with a fixed seed.
```

During calibration, each `CalibratedPTQLinear` records a bounded number of input
rows from its incoming activation:

```bash
--ptq_sample_rows 256
```

Important detail: during this calibration pass, the layer still executes the
original FP Linear. The pass only records activation samples. This avoids
cascading quantization noise during statistics collection.

## Quantization Bounds

For a signed `b`-bit quantizer:

```text
q_min = -2^(b-1)
q_max =  2^(b-1) - 1
```

For 4-bit:

```text
q_min = -8
q_max =  7
```

For 8-bit:

```text
q_min = -128
q_max =  127
```

## Weight Quantization

Weights are quantized group-wise along the input-channel dimension.

For a Linear layer:

```text
Y = X W^T + b
W shape = [out_features, in_features]
```

The input channels are split into groups:

```bash
--ptq_group_size 128
```

For each output channel and each input-channel group:

```text
s_w = max(abs(W_group)) / q_max
Q_w = clamp(round(W_group / s_w), q_min, q_max)
W_hat = Q_w * s_w
```

This gives each group its own scale, which is much safer than one global scale
for the full weight matrix.

## Activation Quantization

Activations are quantized dynamically per token/row. For each activation vector
`x`:

```text
s_x = quantile(abs(x), p) / q_max
Q_x = clamp(round(x / s_x), q_min, q_max)
x_hat = Q_x * s_x
```

The quantile `p` is searched from:

```bash
--ptq_clip_grid 1.0 0.999 0.995
```

Interpretation:

```text
1.0:
    use the max absolute value

0.999 / 0.995:
    clip extreme activation outliers and allocate more 4-bit buckets
    to the dense region of the activation distribution
```

This is the main protection against activation outliers in A4.

## Scale Migration

Before quantization, the method searches a channel-wise scale vector `s` that
redistributes numerical range between activations and weights:

```text
X' = X / s
W' = W * s
```

Mathematically, this preserves the original FP linear operation before
quantization:

```text
X W^T = (X / s) (W * s)^T
```

After this transformation, both `X'` and `W'` are quantized. The purpose is to
move difficult activation outliers into weight scales, where group-wise W4
quantization can sometimes represent them more stably.

The implementation searches four scale families.

### Identity

```text
s = 1
```

This is the fallback candidate. It means no scale migration.

### SmoothQuant-Style

```text
s = act_stat^alpha / weight_stat^(1 - alpha)
```

This follows the SmoothQuant intuition: use activation and weight statistics to
balance the dynamic range between the two sides of the matrix multiply.

### AWQ-Style

```text
s = act_stat^alpha
```

This follows the AWQ intuition: activation statistics indicate which input
channels are more important; scaling can protect those channels before W4
weight quantization.

### Balanced

```text
s = act_stat^alpha * weight_stat^(1 - alpha)
```

This candidate gives another conservative way to combine activation and weight
statistics.

The searched alpha grid is controlled by:

```bash
--ptq_smooth_grid 0.0 0.25 0.5 0.75 1.0
```

Each scale candidate is sanitized:

```text
1. replace non-finite values
2. clamp to [ptq_scale_min, ptq_scale_max]
3. normalize by median(scale)
4. clamp again
```

This prevents a few pathological channels from producing extreme scales.

## Layer-Wise MSE Search

For each Linear layer, the calibration objective is:

```text
min over scale_mode, alpha, clip:

    MSE(Y_fp, Y_quant)

where:
    Y_fp    = Linear(X, W)
    Y_quant = Linear(Q_A(X / s), Q_W(W * s))
```

The best candidate is stored per layer:

```text
best_scale_mode
best_alpha
best_clip
best_mse
smooth_scale
quant_weight
```

After finalization, the layer uses the selected scale and quantized weight for
all subsequent full-dataset embedding extraction.

## Output Embedding Alignment

The Transformer output embedding can still have a distribution shift even when
each layer's local MSE is reduced. To reduce the TF-to-GNN boundary mismatch, the
PTQ path optionally fits a per-dimension affine alignment:

```bash
--ptq_align_output
```

On calibration embeddings:

```text
E_fp  = FP16 embeddings
E_q   = quantized embeddings
```

Fit:

```text
E_fp ~= gamma * E_q + beta
```

The closed-form per-dimension fit is:

```text
gamma = sum((E_q - mean(E_q)) * (E_fp - mean(E_fp)))
        / sum((E_q - mean(E_q))^2)

beta = mean(E_fp) - gamma * mean(E_q)
```

Then all quantized embeddings are aligned:

```text
E_aligned = normalize(gamma * E_q + beta)
```

This is similar in spirit to interface alignment: the quantized Transformer
frontend should produce embeddings in the numerical range expected by the
downstream GNN.

## Damage Check

After saving a quantized pool, the script compares it against the FP16 pool if
the FP16 pool exists:

```text
cache_data/{dataset}_{model}_oracle_FP16.pt
```

It reports cosine error:

```text
cos_err(v) = 1 - cosine(E_fp(v), E_quant(v))
```

Printed statistics:

```text
mean, p50, p90, p95, p99, max
```

This is a required sanity check. If W4A4 has very high cosine damage, the
downstream GNN accuracy drop is expected and should not be blamed on the
TSER/Degree policy.

## Example Commands

Generate FP16 ST embeddings:

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs fp16 \
  --batch_size 128 \
  --overwrite
```

Generate calibrated W4A8 and W4A4 embeddings:

```bash
python -m GraphhopSimhash.generate_real_quant_pools \
  --datasets cora \
  --llm_name ST \
  --configs W4A8 W4A4 \
  --batch_size 128 \
  --w4a_backend ptq \
  --w4a_calib_samples 128 \
  --calibration_strategy random \
  --seed 42 \
  --ptq_group_size 128 \
  --ptq_sample_rows 256 \
  --ptq_smooth_grid 0.0 0.25 0.5 0.75 1.0 \
  --ptq_clip_grid 1.0 0.999 0.995 \
  --ptq_align_output \
  --tag_suffix PTQ_TEST \
  --overwrite
```

Evaluate W4A8/W4A4 pools with the real quantization ablation:

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 3 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag FP16 \
  --real_quant_int8_tag W4A8_PTQ_TEST \
  --real_quant_int4_tag W4A4_PTQ_TEST \
  --real_quant_error_norm 1.0 \
  --real_quant_int8_ratio 0.20
```

## Current ST/Cora Sanity Results

With Sentence-BERT on Cora, the initial PTQ implementation produced:

```text
W4A8:
    embedding cosine error mean ~= 0.029
    downstream GNN drop ~= 0.50%

W4A4:
    embedding cosine error mean ~= 0.34
    downstream GNN drop ~= 14%
```

Interpretation:

```text
W4A8 is currently a stable aggressive low-precision point.
W4A4 is still too destructive if applied to all or most nodes.
```

Therefore, W4A4 should be treated as an extreme path assigned only to a small
low-risk subset of nodes, rather than as the default tail precision for 80-90%
of the graph.

## Relationship To TSER

The PTQ pool generation is intentionally independent of the routing policy:

```text
PTQ generator:
    produces FP16 / W4A8 / W4A4 embedding pools

TSER / Degree / Random policy:
    chooses which node reads which pool
```

This separation is important for fair comparison. Degree and TSER should use the
same FP16/W4A8/W4A4 pools. The only difference should be node assignment:

```text
DegreeTopK:
    high-degree nodes use W4A8 or FP

TSERTopK:
    high-risk nodes use W4A8 or FP

RandomTopK:
    random nodes use W4A8 or FP
```

This isolates the benefit of graph-semantic scoring from the quality of the
quantizer itself.

## Recommended Experimental Framing

Given the current W4A4 damage, the most reasonable experiment is not:

```text
80% W4A4 + 20% W4A8
```

Instead, use W4A8 as the safe low-precision baseline, then sweep W4A4 budget:

```text
0% W4A4 + 100% W4A8
5% W4A4 + 95% W4A8
10% W4A4 + 90% W4A8
20% W4A4 + 80% W4A8
```

At each fixed W4A4 budget, compare:

```text
Random
Degree
TSER
```

The paper claim should be:

```text
TSER is not a new quantizer by itself.
TSER is a risk-aware graph-semantic assignment policy that decides where
aggressive quantization is safe.
```

## Limitations

This implementation is a calibration-aware PTQ embedding generator, not a
production INT4 kernel:

```text
1. It simulates low-bit numerical effects through quantize-dequantize tensors.
2. It does not pack INT4 weights into a custom CUDA/NPU kernel.
3. It does not implement full industrial AWQ exactly.
4. It does not use Hessian/second-order information.
5. W4A4 remains fragile for sentence embeddings.
```

For architecture evaluation, this is still useful because it gives realistic
embedding damage trends and supports fair policy comparison. For a final systems
paper, the software PTQ should be paired with a hardware model that implements
the selected W4A8/W4A4 execution paths.

