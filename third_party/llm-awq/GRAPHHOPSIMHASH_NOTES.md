# GraphhopSimhash AWQ Integration Notes

This directory vendors the core AWQ source extracted from:

```text
/home/qiumingzhi/Simhash-S/OneForAll/llm-awq-main.zip
```

Only the AWQ Python/CUDA source, README, and LICENSE are included. Large demo
assets from the upstream archive are intentionally excluded.

Local compatibility patches:

```text
awq/quantize/__init__.py
    Allows importing AWQ search/quantizer code when the optional compiled
    awq_inference_engine extension is not built.

awq/quantize/pre_quant.py
awq/quantize/auto_scale.py
    Make Qwen2 imports optional for the older transformers version used by the
    research3 environment.

awq/quantize/qmodule.py
    Keeps WQLinear importable without awq_inference_engine and raises a clear
    runtime error only if packed-kernel WQLinear inference is actually called.
```

GraphhopSimhash currently uses official AWQ for W4A16 pool generation through:

```text
GraphhopSimhash/generate_real_quant_pools.py
    apply_official_awq_w4a16
```
