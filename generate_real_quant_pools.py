import argparse
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .data import load_raw_texts
from .paths import ensure_repo_paths

ensure_repo_paths()


MODEL_SPECS = {
    "llama2_7b": {
        "path": "/home/zhangshangtong/Transformer/OFA/models/llama-7b/modelscope/Llama-2-7b-ms",
        "model_class": "llama",
    },
    "llama2_13b": {
        "path": "meta-llama/Llama-2-13b-hf",
        "model_class": "llama",
    },
    "ST": {
        "path": "/home/zhangshangtong/Transformer/OFA/models/multi-qa-distilbert-cos-v1",
        "model_class": "auto",
    },
    "BERT": {
        "path": "/home/zhangshangtong/Transformer/OFA/models/bert-base-uncased",
        "model_class": "auto",
    },
    "e5": {
        "path": "intfloat/e5-large-v2",
        "model_class": "auto",
    },
}

CONFIG_SPECS = {
    "fp16": {"tag": "FP16", "kind": "bnb", "w_bit": 16, "a_bit": 16},
    "int8": {"tag": "INT8", "kind": "bnb", "w_bit": 8, "a_bit": 8},
    "int4": {"tag": "INT4", "kind": "bnb", "w_bit": 4, "a_bit": 16},
    "W4A16": {"tag": "W4A16", "kind": "fake_wa", "w_bit": 4, "a_bit": 16},
    "W4A8": {"tag": "W4A8", "kind": "fake_wa", "w_bit": 4, "a_bit": 8},
    "W4A4": {"tag": "W4A4", "kind": "fake_wa", "w_bit": 4, "a_bit": 4},
}


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return F.normalize(summed / denom, p=2, dim=1)


def resolve_config(config_name):
    for key, spec in CONFIG_SPECS.items():
        if str(config_name).lower() == key.lower():
            return key, dict(spec)
    raise ValueError(f"Unknown config: {config_name}. Available: {sorted(CONFIG_SPECS)}")


def build_quant_config(config_name):
    _canonical, spec = resolve_config(config_name)
    if spec["kind"] != "bnb":
        return None, spec["tag"]

    if spec["tag"] == "FP16":
        return None, "FP16"

    from transformers import BitsAndBytesConfig

    if spec["tag"] == "INT8":
        return BitsAndBytesConfig(load_in_8bit=True), "INT8"
    if spec["tag"] == "INT4":
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "INT4",
        )
    raise ValueError(f"Unsupported bitsandbytes config: {config_name}")


def symmetric_fake_quantize(x, bit_width, mode="per_tensor", dim=-1):
    if int(bit_width) >= 16:
        return x

    q_min = -(2 ** (int(bit_width) - 1))
    q_max = (2 ** (int(bit_width) - 1)) - 1
    if mode == "per_tensor":
        abs_max = torch.max(torch.abs(x))
    elif mode == "per_channel":
        abs_max = torch.max(torch.abs(x), dim=dim, keepdim=True).values
    else:
        raise ValueError(f"Invalid fake quant mode: {mode}")

    scale = torch.clamp(abs_max, min=1e-8) / q_max
    quantized = torch.round(x / scale).clamp(q_min, q_max)
    return quantized * scale


class FakeQuantLinear(nn.Module):
    def __init__(self, original_linear, w_bit=4, a_bit=8, awq_grid=21):
        super().__init__()
        self.in_features = int(original_linear.in_features)
        self.out_features = int(original_linear.out_features)
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.w_bit = int(w_bit)
        self.a_bit = int(a_bit)
        self.awq_grid = max(1, int(awq_grid))
        self.calibrated = False
        self.register_buffer("awq_scales", torch.ones(self.in_features))

    def forward(self, x):
        if not self.calibrated:
            self._calibrate_awq_scale(x)

        view_shape = [1] * (x.dim() - 1) + [self.in_features]
        scales = self.awq_scales.to(device=x.device, dtype=x.dtype).view(*view_shape)
        weight_scales = self.awq_scales.to(device=self.weight.device, dtype=self.weight.dtype).view(1, -1)

        w_scaled = self.weight * weight_scales
        qw = symmetric_fake_quantize(w_scaled, self.w_bit, mode="per_channel", dim=1)

        x_scaled = x / scales
        qx = symmetric_fake_quantize(x_scaled, self.a_bit, mode="per_channel", dim=-1)
        return F.linear(qx, qw, self.bias)

    def _calibrate_awq_scale(self, x):
        x_flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        if x_flat.numel() == 0:
            self.calibrated = True
            return

        sx = torch.clamp(x_flat.abs().mean(dim=0), min=1e-5).to(device=self.weight.device)
        sw = torch.clamp(self.weight.detach().to(torch.float32).abs().max(dim=0)[0], min=1e-5)

        x_eval = x_flat[:128].to(device=self.weight.device, dtype=self.weight.dtype)
        y_orig = F.linear(x_eval, self.weight, self.bias)

        best_error = float("inf")
        best_scale = torch.ones_like(sx)
        for alpha in torch.linspace(0.0, 1.0, steps=self.awq_grid, device=sx.device):
            scale = torch.clamp(sx.pow(float(alpha.item())) * sw.pow(1.0 - float(alpha.item())), min=1e-5)
            scale = scale.to(device=self.weight.device, dtype=self.weight.dtype)
            w_scaled = self.weight * scale.view(1, -1)
            qw = symmetric_fake_quantize(w_scaled, self.w_bit, mode="per_channel", dim=1)
            w_recon = qw / scale.view(1, -1)
            y_quant = F.linear(x_eval, w_recon, self.bias)
            error = float((y_orig - y_quant).pow(2).mean().item())
            if error < best_error:
                best_error = error
                best_scale = scale.detach().to(device=self.awq_scales.device, dtype=self.awq_scales.dtype)

        self.awq_scales.copy_(best_scale)
        self.calibrated = True


def replace_linear_with_fake_quant(module, w_bit, a_bit, awq_grid=21, skip_names=("lm_head",)):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in set(skip_names):
            setattr(module, name, FakeQuantLinear(child, w_bit=w_bit, a_bit=a_bit, awq_grid=awq_grid))
        else:
            replace_linear_with_fake_quant(child, w_bit, a_bit, awq_grid=awq_grid, skip_names=skip_names)


def load_model_and_tokenizer(llm_name, config_name, cache_dir):
    from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

    if llm_name not in MODEL_SPECS:
        raise ValueError(f"Unknown llm_name={llm_name}. Available: {sorted(MODEL_SPECS)}")

    spec = MODEL_SPECS[llm_name]
    model_path = spec["path"]
    quant_config, tag = build_quant_config(config_name)

    if spec["model_class"] == "llama":
        model_cls = LlamaForCausalLM
        tokenizer_cls = LlamaTokenizer
    else:
        model_cls = AutoModel
        tokenizer_cls = AutoTokenizer

    kwargs = {
        "cache_dir": cache_dir,
        "output_hidden_states": True,
    }
    if spec["model_class"] == "llama":
        kwargs["device_map"] = "auto"
    if quant_config is None:
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["quantization_config"] = quant_config

    model = model_cls.from_pretrained(model_path, **kwargs)
    if "device_map" not in kwargs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    tokenizer = tokenizer_cls.from_pretrained(model_path, cache_dir=cache_dir, add_eos_token=False)
    if llm_name.startswith("llama2"):
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return model, tokenizer, tag


def _forward_hidden_states(model, tokens):
    target_model = model
    if hasattr(model, "model") and model.__class__.__name__.endswith("ForCausalLM"):
        target_model = model.model
    outputs = target_model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1].to(torch.float32)


def encode_texts(model, tokenizer, texts, batch_size, max_length, device):
    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = _forward_hidden_states(model, tokens)
            embs = mean_pool(hidden, tokens["attention_mask"]).cpu()
            all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def generate_pool(dataset, llm_name, config_name, args):
    _canonical, config_spec = resolve_config(config_name)
    tag = config_spec["tag"]

    out_path = args.output_path
    if out_path is None:
        out_path = os.path.join("cache_data", f"{dataset}_{llm_name}_oracle_{tag}.pt")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[Skip] {out_path} already exists. Use --overwrite to regenerate.")
        return

    print(f"\n{'=' * 72}")
    print(f"Generating {tag} pool | dataset={dataset} | llm={llm_name}")
    print(f"{'=' * 72}")
    texts = load_raw_texts(dataset)
    batch_size = args.batch_size
    if llm_name.startswith("llama2") and batch_size > 4:
        print("[Info] Reducing LLaMA batch_size to 4 to avoid OOM.")
        batch_size = 4

    load_config_name = "fp16" if config_spec["kind"] == "fake_wa" else config_name
    model, tokenizer, _tag = load_model_and_tokenizer(llm_name, load_config_name, args.cache_dir)
    device = next(model.parameters()).device

    if config_spec["kind"] == "fake_wa":
        print(
            "[FakeWA] Installing AWQ-style fake quant Linear wrappers "
            f"| W{config_spec['w_bit']}A{config_spec['a_bit']} "
            f"| calib_samples={args.w4a_calib_samples}"
        )
        replace_linear_with_fake_quant(
            model,
            w_bit=int(config_spec["w_bit"]),
            a_bit=int(config_spec["a_bit"]),
            awq_grid=int(args.w4a_awq_grid),
            skip_names=(),
        )
        calib_count = min(len(texts), int(args.w4a_calib_samples))
        if calib_count > 0:
            print(f"[FakeWA] Running calibration pass on {calib_count} node texts...")
            _ = encode_texts(model, tokenizer, texts[:calib_count], batch_size, args.max_length, device)

    embs = encode_texts(model, tokenizer, texts, batch_size, args.max_length, device)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(embs, out_path)
    print(f"[Saved] {out_path} | shape={tuple(embs.shape)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Generate FP16/INT8/INT4 or W4A16/W4A8/W4A4 node embedding pools.")
    parser.add_argument("--datasets", nargs="+", default=["cora"], choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--llm_name", type=str, default="llama2_7b", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--configs", nargs="+", default=["fp16", "int8", "int4"], choices=sorted(CONFIG_SPECS.keys()))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--cache_dir", type=str, default="cache_data/model")
    parser.add_argument("--output_path", type=str, default=None, help="Only valid with one dataset and one config.")
    parser.add_argument("--w4a_calib_samples", type=int, default=32)
    parser.add_argument("--w4a_awq_grid", type=int, default=21)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_path is not None and (len(args.datasets) != 1 or len(args.configs) != 1):
        parser.error("--output_path requires exactly one dataset and one config")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.max_length <= 0:
        parser.error("--max_length must be positive")
    if args.w4a_calib_samples < 0:
        parser.error("--w4a_calib_samples must be >= 0")
    if args.w4a_awq_grid <= 0:
        parser.error("--w4a_awq_grid must be positive")

    for dataset in args.datasets:
        for config_name in args.configs:
            generate_pool(dataset, args.llm_name, config_name, args)


if __name__ == "__main__":
    main()
