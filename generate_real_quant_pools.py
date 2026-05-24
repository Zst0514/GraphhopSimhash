import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager

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
    "W4A16": {"tag": "W4A16", "kind": "awq", "w_bit": 4, "a_bit": 16},
    "W4A8": {"tag": "W4A8", "kind": "awq_act", "w_bit": 4, "a_bit": 8},
    "W4A4": {"tag": "W4A4", "kind": "awq_act", "w_bit": 4, "a_bit": 4},
    "W4A16_FAKE": {"tag": "W4A16_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 16},
    "W4A8_FAKE": {"tag": "W4A8_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 8},
    "W4A4_FAKE": {"tag": "W4A4_FAKE", "kind": "fake_wa", "w_bit": 4, "a_bit": 4},
}

AWQ_SUPPORTED_CLASS_NAMES = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "OPTForCausalLM",
    "BloomForCausalLM",
    "DistilBertModel",
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


def affine_fake_quantize(x, bit_width, mode="per_tensor", dim=-1):
    if int(bit_width) >= 16:
        return x

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    q_min = 0
    q_max = (2 ** int(bit_width)) - 1
    if mode == "per_tensor":
        min_val = torch.min(x)
        max_val = torch.max(x)
    elif mode == "per_channel":
        min_val = torch.min(x, dim=dim, keepdim=True).values
        max_val = torch.max(x, dim=dim, keepdim=True).values
    else:
        raise ValueError(f"Invalid fake quant mode: {mode}")

    scale = torch.clamp(max_val - min_val, min=1e-8) / q_max
    zero_point = torch.round(q_min - min_val / scale).clamp(q_min, q_max)
    quantized = torch.round(x / scale + zero_point).clamp(q_min, q_max)
    return torch.nan_to_num((quantized - zero_point) * scale, nan=0.0, posinf=0.0, neginf=0.0)


def _clip_activation_with_channel_thresholds(x, clip_abs=None, channel_clip_abs=None):
    if clip_abs is None and channel_clip_abs is None:
        return x

    if channel_clip_abs is not None:
        channel_clip = channel_clip_abs.to(device=x.device, dtype=x.dtype)
        if channel_clip.numel() != x.shape[-1]:
            raise ValueError(
                f"activation channel clip size must match hidden dim {x.shape[-1]}, "
                f"got {channel_clip.numel()}"
            )
        view_shape = [1] * (x.dim() - 1) + [int(channel_clip.numel())]
        clip = channel_clip.view(*view_shape).clamp_min(1e-8)
        return torch.maximum(torch.minimum(x, clip), -clip)

    clip = torch.as_tensor(float(clip_abs), device=x.device, dtype=x.dtype).clamp_min(1e-8)
    return torch.clamp(x, min=-clip, max=clip)


def affine_fake_quantize_with_protected_channels(x, bit_width, protected_mask, protected_bit_width=8):
    if protected_mask is None or int(protected_mask.sum().item()) <= 0:
        return affine_fake_quantize(x, bit_width, mode="per_channel", dim=-1)
    if int(bit_width) >= 16:
        return x

    protected_mask = protected_mask.to(device=x.device, dtype=torch.bool)
    if protected_mask.numel() != x.shape[-1]:
        raise ValueError(
            f"protected channel mask size must match hidden dim {x.shape[-1]}, got {protected_mask.numel()}"
        )
    low_mask = ~protected_mask
    if int(low_mask.sum().item()) <= 0:
        return affine_fake_quantize(x, protected_bit_width, mode="per_channel", dim=-1)

    qx = x.clone()
    qx[..., low_mask] = affine_fake_quantize(x[..., low_mask], bit_width, mode="per_channel", dim=-1)
    if int(protected_bit_width) >= 16:
        qx[..., protected_mask] = x[..., protected_mask]
    else:
        qx[..., protected_mask] = affine_fake_quantize(
            x[..., protected_mask],
            protected_bit_width,
            mode="per_channel",
            dim=-1,
        )
    return qx


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


class ActivationQuantLinear(nn.Module):
    def __init__(self, original_linear, a_bit=8, module_name=None, clip_config=None):
        super().__init__()
        self.linear = original_linear
        self.a_bit = int(a_bit)
        self.in_features = int(original_linear.in_features)
        self.out_features = int(original_linear.out_features)
        self.module_name = module_name or ""
        clip_config = clip_config or {}
        clip_abs = clip_config.get("clip_abs")
        self.clip_abs = None if clip_abs is None else float(clip_abs)
        channel_clip_abs = clip_config.get("channel_clip_abs")
        if channel_clip_abs is None:
            self.register_buffer("channel_clip_abs", None)
        else:
            self.register_buffer("channel_clip_abs", torch.as_tensor(channel_clip_abs, dtype=torch.float32))
        outlier_channel_mask = clip_config.get("outlier_channel_mask")
        if outlier_channel_mask is None:
            self.register_buffer("outlier_channel_mask", None)
            self.has_outlier_channels = False
        else:
            outlier_channel_mask = torch.as_tensor(outlier_channel_mask, dtype=torch.bool)
            self.register_buffer("outlier_channel_mask", outlier_channel_mask)
            self.has_outlier_channels = bool(outlier_channel_mask.any().item())
        self.outlier_channel_a_bit = int(clip_config.get("outlier_channel_a_bit", 8))

    def forward(self, x):
        if self.has_outlier_channels:
            qx = affine_fake_quantize_with_protected_channels(
                x,
                self.a_bit,
                self.outlier_channel_mask,
                protected_bit_width=self.outlier_channel_a_bit,
            )
            return self.linear(qx)

        clipped = _clip_activation_with_channel_thresholds(
            x,
            clip_abs=self.clip_abs,
            channel_clip_abs=self.channel_clip_abs,
        )
        qx = affine_fake_quantize(clipped, self.a_bit, mode="per_channel", dim=-1)
        return self.linear(qx)


def replace_linear_with_activation_quant(module, a_bit, skip_names=("lm_head",), clip_config_by_name=None, prefix=""):
    clip_config_by_name = clip_config_by_name or {}
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and name not in set(skip_names):
            setattr(
                module,
                name,
                ActivationQuantLinear(
                    child,
                    a_bit=a_bit,
                    module_name=full_name,
                    clip_config=clip_config_by_name.get(full_name),
                ),
            )
        else:
            replace_linear_with_activation_quant(
                child,
                a_bit=a_bit,
                skip_names=skip_names,
                clip_config_by_name=clip_config_by_name,
                prefix=full_name,
            )


def load_model_and_tokenizer(llm_name, config_name, cache_dir, force_cpu=False):
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
    if spec["model_class"] == "llama" and not force_cpu:
        kwargs["device_map"] = "auto"
    if quant_config is None:
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["quantization_config"] = quant_config

    model = model_cls.from_pretrained(model_path, **kwargs)
    if force_cpu:
        model = model.to("cpu")
    elif "device_map" not in kwargs:
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
    hidden = outputs.hidden_states[-1].to(torch.float32)
    return torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)


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


def ensure_awq_project_on_path():
    awq_path = os.path.join(os.path.dirname(__file__), "third_party", "llm-awq")
    if not os.path.isdir(os.path.join(awq_path, "awq")):
        raise FileNotFoundError(
            "llm-awq source tree is missing. Expected official AWQ source at "
            f"{awq_path}. Extract llm-awq-main.zip into GraphhopSimhash/third_party/llm-awq."
        )
    if awq_path not in sys.path:
        sys.path.insert(0, awq_path)
    return awq_path


def _is_awq_supported_model(model):
    class_name = model.__class__.__name__
    class_text = str(model.__class__).lower()
    return (
        class_name in AWQ_SUPPORTED_CLASS_NAMES
        or "mpt" in class_text
        or "falcon" in class_text
        or "bigcode" in class_text
        or "neox" in class_text
    )


def build_local_awq_calib_getter(texts):
    def get_calib_dataset(data="graph_text", tokenizer=None, n_samples=128, block_size=512):
        del data
        samples = []
        for text in texts:
            line = str(text).strip()
            if not line:
                continue
            token_ids = tokenizer.encode(line)
            if len(token_ids) == 0 or len(token_ids) > block_size:
                continue
            samples.append(torch.tensor([token_ids], dtype=torch.long))
            if len(samples) >= int(n_samples):
                break
        if not samples:
            raise ValueError("No usable graph texts were found for AWQ calibration.")

        cat_samples = torch.cat(samples, dim=1)
        if cat_samples.shape[1] < block_size:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0
            pad = torch.full(
                (1, block_size - cat_samples.shape[1]),
                int(pad_id),
                dtype=torch.long,
            )
            cat_samples = torch.cat([cat_samples, pad], dim=1)

        n_split = max(1, cat_samples.shape[1] // block_size)
        print(f"[AWQ] Local graph-text calibration blocks={n_split} | samples={len(samples)} | block_size={block_size}")
        return [
            cat_samples[:, i * block_size : (i + 1) * block_size]
            for i in range(n_split)
        ]

    return get_calib_dataset


@contextmanager
def patch_awq_calibration_data(texts):
    ensure_awq_project_on_path()
    from awq.utils import calib_data as awq_calib_data

    original_getter = awq_calib_data.get_calib_dataset
    awq_calib_data.get_calib_dataset = build_local_awq_calib_getter(texts)
    try:
        yield
    finally:
        awq_calib_data.get_calib_dataset = original_getter


def _default_awq_results_path(dataset, llm_name, args):
    return os.path.join(
        "cache_data",
        "awq",
        f"{dataset}_{llm_name}_w4_g{int(args.awq_q_group_size)}_n{int(args.awq_calib_samples)}_s{int(args.awq_seqlen)}.pt",
    )


def apply_official_awq_w4(model, tokenizer, texts, dataset, llm_name, args, activation_bit=16):
    ensure_awq_project_on_path()
    if not _is_awq_supported_model(model):
        raise NotImplementedError(
            "Official llm-awq W4A16 currently supports causal-LM blocks "
            "(LLaMA/Qwen2/OPT/Bloom/MPT/Falcon/BigCode/NeoX) plus the "
            "GraphhopSimhash DistilBERT adapter. "
            f"Got model class {model.__class__.__name__}. "
            "Unsupported encoder models should use --configs fp16 as the reference pool."
        )

    from awq.quantize.pre_quant import apply_awq, run_awq
    from awq.quantize.quantizer import pseudo_quantize_model_weight

    q_config = {
        "zero_point": not bool(args.awq_no_zero_point),
        "q_group_size": int(args.awq_q_group_size),
    }
    awq_results_path = args.awq_results_path or _default_awq_results_path(dataset, llm_name, args)
    os.makedirs(os.path.dirname(awq_results_path) or ".", exist_ok=True)

    awq_already_applied = False
    mse_range = not bool(args.awq_disable_mse_clip)
    if model.__class__.__name__ == "DistilBertModel" and not bool(getattr(args, "awq_force_mse_clip", False)):
        if mse_range:
            print("[AWQ] DistilBERT adapter disables MSE clip by default to avoid large activation-clip memory spikes.")
        mse_range = False

    if os.path.exists(awq_results_path) and not bool(args.awq_overwrite_results):
        print(f"[AWQ] Loading cached AWQ search results from {awq_results_path}")
        awq_results = torch.load(awq_results_path, map_location="cpu")
    else:
        print(
            "[AWQ] Running official llm-awq search "
            f"| w_bit=4 | q_config={q_config} | samples={args.awq_calib_samples} | seqlen={args.awq_seqlen}"
        )
        with patch_awq_calibration_data(texts):
            awq_results = run_awq(
                model,
                tokenizer,
                w_bit=4,
                q_config=q_config,
                n_samples=int(args.awq_calib_samples),
                seqlen=int(args.awq_seqlen),
                auto_scale=not bool(args.awq_disable_auto_scale),
                mse_range=mse_range,
                calib_data="graph_text",
            )
        torch.save(awq_results, awq_results_path)
        print(f"[AWQ] Saved AWQ search results to {awq_results_path}")
        awq_already_applied = True

    if awq_already_applied:
        print(f"[AWQ] AWQ scales/clips were applied during search; pseudo-quantizing weights to W4A{int(activation_bit)}.")
    else:
        print(f"[AWQ] Applying cached AWQ scales/clips and pseudo-quantizing weights to W4A{int(activation_bit)}.")
        apply_awq(model, awq_results)
    pseudo_quantize_model_weight(model, w_bit=4, q_config=q_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


def apply_official_awq_w4a16(model, tokenizer, texts, dataset, llm_name, args):
    return apply_official_awq_w4(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        dataset=dataset,
        llm_name=llm_name,
        args=args,
        activation_bit=16,
    )


def _load_tensor(path):
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} did not contain a torch.Tensor")
    return tensor.to(dtype=torch.float32)


def _default_reference_path(out_path, dataset, llm_name):
    out_dir = os.path.dirname(out_path) or "."
    return os.path.join(out_dir, f"{dataset}_{llm_name}_oracle_FP16.pt")


def affine_align_to_reference(embs, reference_embs, sample_count):
    sample_count = min(int(sample_count), int(embs.size(0)), int(reference_embs.size(0)))
    if sample_count <= 0:
        raise ValueError("ptq_align_samples must leave at least one sample")

    quant_calib = embs[:sample_count].to(dtype=torch.float32)
    ref_calib = reference_embs[:sample_count].to(dtype=torch.float32)
    quant_mean = quant_calib.mean(dim=0)
    ref_mean = ref_calib.mean(dim=0)
    quant_centered = quant_calib - quant_mean
    ref_centered = ref_calib - ref_mean
    gamma = (quant_centered * ref_centered).sum(dim=0) / quant_centered.pow(2).sum(dim=0).clamp_min(1e-8)
    beta = ref_mean - gamma * quant_mean
    aligned = embs.to(dtype=torch.float32) * gamma + beta
    return F.normalize(aligned, p=2, dim=1), sample_count


def maybe_align_output_embeddings(embs, dataset, llm_name, tag, config_spec, out_path, args):
    if config_spec["kind"] != "fake_wa":
        return embs
    if not bool(getattr(args, "ptq_align_output", True)):
        return embs

    reference_path = getattr(args, "ptq_align_reference_path", None)
    if reference_path is None:
        reference_path = _default_reference_path(out_path, dataset, llm_name)
    if not os.path.exists(reference_path):
        print(f"[Align] Skipped {tag}: FP16 reference not found at {reference_path}")
        return embs

    reference_embs = _load_tensor(reference_path)
    if tuple(reference_embs.shape) != tuple(embs.shape):
        raise ValueError(
            f"Alignment reference shape must match generated embeddings: "
            f"reference={tuple(reference_embs.shape)}, generated={tuple(embs.shape)}"
        )

    before_err = (1.0 - F.cosine_similarity(reference_embs, embs.to(dtype=torch.float32), dim=1)).clamp(min=0.0)
    aligned, sample_count = affine_align_to_reference(
        embs,
        reference_embs,
        getattr(args, "ptq_align_samples", 512),
    )
    after_err = (1.0 - F.cosine_similarity(reference_embs, aligned, dim=1)).clamp(min=0.0)
    print(
        "[Align] Output affine alignment to FP16 "
        f"| tag={tag} | samples={sample_count} | reference={reference_path} "
        f"| err_mean {before_err.mean().item():.5f}->{after_err.mean().item():.5f} "
        f"| err_p95 {before_err.quantile(0.95).item():.5f}->{after_err.quantile(0.95).item():.5f}"
    )
    return aligned


def default_activation_outlier_report_path(dataset, llm_name, args):
    return os.path.join(
        "output",
        "graph_simhash",
        dataset.lower(),
        (
            f"{dataset.lower()}_{llm_name}_activation_outliers_"
            f"n{int(getattr(args, 'activation_outlier_calib_samples', 128))}_"
            f"seed{int(getattr(args, 'activation_outlier_seed', 42))}.json"
        ),
    )


def load_activation_outlier_clip_config(dataset, llm_name, a_bit, args):
    if not bool(getattr(args, "activation_outlier_clip", False)):
        return {}
    if int(a_bit) > int(getattr(args, "activation_outlier_apply_max_a_bit", 4)):
        return {}

    report_path = getattr(args, "activation_outlier_report_path", None)
    if not report_path:
        report_path = default_activation_outlier_report_path(dataset, llm_name, args)
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            "Activation outlier clipping was requested, but the report is missing: "
            f"{report_path}. Generate it with `python -m GraphhopSimhash.activation_outlier_calibration` "
            "or pass --activation_outlier_report_path."
        )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    layer_clip_ratio = float(getattr(args, "activation_outlier_clip_ratio", 1.0))
    channel_clip_ratio = float(getattr(args, "activation_outlier_channel_clip_ratio", 0.75))
    min_clip = float(getattr(args, "activation_outlier_min_clip", 1e-4))
    max_top_channels = int(getattr(args, "activation_outlier_max_top_channels", 64))
    mode = str(getattr(args, "activation_outlier_mode", "channel_protect")).lower()
    outlier_channel_a_bit = int(getattr(args, "activation_outlier_channel_a_bit", 8))

    clip_config = {}
    tuned_channels = 0
    for layer in report.get("layers", []):
        module_name = layer.get("module")
        feature_dim = int(layer.get("feature_dim", 0))
        threshold_abs = float(layer.get("threshold_abs", 0.0))
        if not module_name or feature_dim <= 0 or threshold_abs <= 0.0:
            continue

        layer_clip = max(min_clip, threshold_abs * layer_clip_ratio)
        channel_clip = torch.full((feature_dim,), float(layer_clip), dtype=torch.float32)
        outlier_channel_mask = torch.zeros(feature_dim, dtype=torch.bool)
        for channel in layer.get("top_channels", [])[:max_top_channels]:
            channel_idx = int(channel.get("channel", -1))
            if 0 <= channel_idx < feature_dim:
                if mode == "channel_protect":
                    outlier_channel_mask[channel_idx] = True
                else:
                    channel_clip[channel_idx] = max(min_clip, layer_clip * channel_clip_ratio)
                tuned_channels += 1
        if mode == "channel_protect":
            clip_config[module_name] = {
                "outlier_channel_mask": outlier_channel_mask,
                "outlier_channel_a_bit": outlier_channel_a_bit,
            }
        elif mode == "clip":
            clip_config[module_name] = {
                "clip_abs": layer_clip,
                "channel_clip_abs": channel_clip,
            }
        else:
            raise ValueError(f"Unknown activation_outlier_mode={mode}")

    print(
        "[ActOutlierClip] Loaded activation clip config "
        f"| report={report_path} | layers={len(clip_config)} | tuned_channels={tuned_channels} "
        f"| mode={mode} | layer_clip_ratio={layer_clip_ratio:.3f} "
        f"| channel_clip_ratio={channel_clip_ratio:.3f} | outlier_channel_a_bit={outlier_channel_a_bit}"
    )
    return clip_config


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

    load_config_name = "fp16" if config_spec["kind"] in ("fake_wa", "awq", "awq_act") else config_name
    model, tokenizer, _tag = load_model_and_tokenizer(
        llm_name,
        load_config_name,
        args.cache_dir,
        force_cpu=config_spec["kind"] in ("awq", "awq_act"),
    )
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
    elif config_spec["kind"] in ("awq", "awq_act"):
        print(
            "[AWQ] Installing official llm-awq W4 weight quantization path "
            f"| target=W4A{int(config_spec['a_bit'])}"
        )
        model, device = apply_official_awq_w4(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            dataset=dataset,
            llm_name=llm_name,
            args=args,
            activation_bit=int(config_spec["a_bit"]),
        )
        if config_spec["kind"] == "awq_act":
            act_clip_config = load_activation_outlier_clip_config(
                dataset,
                llm_name,
                int(config_spec["a_bit"]),
                args,
            )
            print(
                f"[AWQ] Installing activation fake quant wrappers | A{int(config_spec['a_bit'])} "
                f"| outlier_clip_layers={len(act_clip_config)}"
            )
            replace_linear_with_activation_quant(
                model,
                a_bit=int(config_spec["a_bit"]),
                clip_config_by_name=act_clip_config,
            )

    embs = encode_texts(model, tokenizer, texts, batch_size, args.max_length, device)
    embs = maybe_align_output_embeddings(embs, dataset, llm_name, tag, config_spec, out_path, args)

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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--cache_dir", type=str, default="cache_data/model")
    parser.add_argument("--output_path", type=str, default=None, help="Only valid with one dataset and one config.")
    parser.add_argument("--w4a_calib_samples", type=int, default=64)
    parser.add_argument("--w4a_awq_grid", type=int, default=21)
    parser.add_argument(
        "--ptq_align_output",
        dest="ptq_align_output",
        action="store_true",
        default=True,
        help="Align fake W/A quantized embeddings back to the FP16 pool with a per-dimension affine map.",
    )
    parser.add_argument(
        "--no_ptq_align_output",
        dest="ptq_align_output",
        action="store_false",
        help="Disable FP16 output affine alignment for fake W/A quantized pools.",
    )
    parser.add_argument("--ptq_align_samples", type=int, default=512)
    parser.add_argument("--ptq_align_reference_path", type=str, default=None)
    parser.add_argument("--awq_calib_samples", type=int, default=128)
    parser.add_argument("--awq_seqlen", type=int, default=512)
    parser.add_argument("--awq_q_group_size", type=int, default=128)
    parser.add_argument("--awq_no_zero_point", action="store_true")
    parser.add_argument("--awq_disable_auto_scale", action="store_true")
    parser.add_argument("--awq_disable_mse_clip", action="store_true")
    parser.add_argument("--awq_force_mse_clip", action="store_true")
    parser.add_argument("--awq_results_path", type=str, default=None)
    parser.add_argument("--awq_overwrite_results", action="store_true")
    parser.add_argument(
        "--activation_outlier_clip",
        action="store_true",
        help="Use a layer/channel activation outlier report to clip A4/A8 inputs before affine activation quantization.",
    )
    parser.add_argument("--activation_outlier_report_path", type=str, default=None)
    parser.add_argument("--activation_outlier_calib_samples", type=int, default=128)
    parser.add_argument("--activation_outlier_seed", type=int, default=42)
    parser.add_argument(
        "--activation_outlier_apply_max_a_bit",
        type=int,
        default=4,
        help="Apply outlier clipping only when activation bit width is <= this value. Default only affects A4.",
    )
    parser.add_argument(
        "--activation_outlier_clip_ratio",
        type=float,
        default=1.0,
        help="Layer-level clip multiplier applied to threshold_abs from the outlier report.",
    )
    parser.add_argument(
        "--activation_outlier_mode",
        type=str,
        default="channel_protect",
        choices=["channel_protect", "clip"],
        help=(
            "channel_protect excludes top outlier channels from A4 scale computation and quantizes them with "
            "--activation_outlier_channel_a_bit; clip applies hard layer/channel clipping."
        ),
    )
    parser.add_argument("--activation_outlier_channel_a_bit", type=int, default=8)
    parser.add_argument(
        "--activation_outlier_channel_clip_ratio",
        type=float,
        default=0.75,
        help="Extra multiplier for top outlier channels listed in the report.",
    )
    parser.add_argument("--activation_outlier_min_clip", type=float, default=1e-4)
    parser.add_argument("--activation_outlier_max_top_channels", type=int, default=64)
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
    if args.ptq_align_samples <= 0:
        parser.error("--ptq_align_samples must be positive")
    if args.awq_calib_samples <= 0:
        parser.error("--awq_calib_samples must be positive")
    if args.awq_seqlen <= 0:
        parser.error("--awq_seqlen must be positive")
    if args.awq_q_group_size == 0 or args.awq_q_group_size < -1:
        parser.error("--awq_q_group_size must be -1 or a positive integer")
    if args.activation_outlier_calib_samples <= 0:
        parser.error("--activation_outlier_calib_samples must be positive")
    if args.activation_outlier_seed < 0:
        parser.error("--activation_outlier_seed must be >= 0")
    if args.activation_outlier_apply_max_a_bit <= 0:
        parser.error("--activation_outlier_apply_max_a_bit must be positive")
    if args.activation_outlier_clip_ratio <= 0:
        parser.error("--activation_outlier_clip_ratio must be positive")
    if args.activation_outlier_channel_a_bit <= 0:
        parser.error("--activation_outlier_channel_a_bit must be positive")
    if args.activation_outlier_channel_clip_ratio <= 0:
        parser.error("--activation_outlier_channel_clip_ratio must be positive")
    if args.activation_outlier_min_clip <= 0:
        parser.error("--activation_outlier_min_clip must be positive")
    if args.activation_outlier_max_top_channels < 0:
        parser.error("--activation_outlier_max_top_channels must be >= 0")

    for dataset in args.datasets:
        for config_name in args.configs:
            generate_pool(dataset, args.llm_name, config_name, args)


if __name__ == "__main__":
    main()
