# HEAT-Style Bit-Serial Quantization Baseline

This folder contains a local mechanism-level baseline inspired by HEAT
Sec. 5.2.1. It is not a full HEAT simulator reproduction.

## Scope

The script evaluates the part of HEAT that can be cleanly isolated from the
paper text:

- Select key vertices by topology: top `alpha` fraction by degree.
- Assign high precision to key vertices and low precision to the rest.
- Model bit-serial GEMM cost as `weight_bits * activation_bits` one-bit GEMMs.

The default HEAT-style bit widths follow Fig. 6 in the HEAT paper:

- key vertices: `10-bit activation/token x 8-bit weight`
- non-key vertices: `2-bit activation/token x 4-bit weight`
- default `alpha = 0.1`

Full HEAT also includes DIMM-NDP subgraph scheduling, hot buffers, and NPU-NDP
decoupled dataflow. Those are not reproduced here because the paper does not
publish the complete simulator and scheduling implementation.

## Accuracy Proxy

The local project does not have HEAT's official SentenceBERT W8A10/W4A2 pools.
For task-level drop, the script mixes the existing LLaMA2-7B BFPA embedding
pools:

- high proxy: `W4BFPA8_B128`
- low proxies: configurable, default `W4BFPA4_B256 W4BFPA3_B256`

This gives a faithful evaluation of the topology-aware high/low vertex routing
under the local GraphHopSimhash task pipeline, while the exact HEAT bit-serial
compute reduction is still reported from the published bit widths.

## Usage

Run a short sanity check:

```bash
python HEAT/evaluate_heat_style_quant.py --tasks CN PN --runs 1 --low_tags W4BFPA4_B256
```

Run the default six-task proxy evaluation:

```bash
python HEAT/evaluate_heat_style_quant.py
```

Outputs are written to:

```text
/home/zhangshangtong/Transformer/OFA/output/heat_style_bitserial_quant/
```

The main markdown report is also mirrored to:

```text
HEAT/results/HEAT_STYLE_BITSERIAL_QUANT.md
```

## Frontend Timing Comparison

For speedup comparison against the local fixed `W4BFPA4` encoder path, use:

```bash
python HEAT/model_frontend_timing.py
```

This writes:

```text
/home/zhangshangtong/Transformer/OFA/output/heat_frontend_timing_w4bfpa4/
HEAT/results/FRONTEND_TIMING_VS_HEAT.md
```

This model separates compute bit-plane cycles, streamed weight loading,
activation loading, output writes, and reused-embedding cache reads.  This is
the correct frontend-speed comparison point when the local accelerator keeps
weights fixed at W4 and activations at BFPA4.

For the more faithful path-level comparison that uses the existing local BFP
array traces and a separate HEAT bit-serial PE model, run:

```bash
python HEAT/simulate_frontend_paths.py
```

This writes:

```text
/home/zhangshangtong/Transformer/OFA/output/heat_frontend_path_timing/
HEAT/results/FRONTEND_PATH_TIMING_DETAILED.md
```

This is the preferred report for speedup claims.  It keeps the two datapaths
separate: GRACE uses measured `W4BFPA4` BFP-array cycles, while HEAT-style uses
W8A10/W4A2 bit-serial PE cycles plus explicit weight/activation/output loading.
