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


def _as_float_grid(values, default):
    if values is None:
        return list(default)
    if isinstance(values, str):
        values = values.split()
    return [float(v) for v in values]


def _quant_bounds(bit_width):
    bit_width = int(bit_width)
    q_min = -(2 ** (bit_width - 1))
    q_max = (2 ** (bit_width - 1)) - 1
    return q_min, q_max


def _quantile_abs_max(x, percentile, dim=-1, keepdim=True):
    abs_x = x.abs().to(torch.float32)
    percentile = float(percentile)
    if percentile >= 0.99999:
        return abs_x.amax(dim=dim, keepdim=keepdim)
    if percentile <= 0.0:
        percentile = 1.0
    return torch.quantile(abs_x, percentile, dim=dim, keepdim=keepdim)


def dynamic_activation_quantize(x, bit_width, percentile=1.0):
    if int(bit_width) >= 16:
        return x
    q_min, q_max = _quant_bounds(bit_width)
    abs_clip = _quantile_abs_max(x, percentile, dim=-1, keepdim=True).to(dtype=x.dtype, device=x.device)
    scale = torch.clamp(abs_clip, min=1e-8) / float(q_max)
    qx = torch.round(x / scale).clamp(q_min, q_max)
    return qx * scale


def mixed_outlier_activation_quantize(x, base_bit, outlier_bit, outlier_mask, percentile=1.0):
    if outlier_mask is None or outlier_mask.numel() == 0 or not bool(outlier_mask.any()):
        return dynamic_activation_quantize(x, base_bit, percentile)
    q_base = dynamic_activation_quantize(x, base_bit, percentile)
    if int(outlier_bit) <= int(base_bit):
        return q_base
    q_outlier = dynamic_activation_quantize(x, outlier_bit, percentile)
    view_shape = [1] * (x.dim() - 1) + [x.shape[-1]]
    mask = outlier_mask.to(device=x.device).view(*view_shape)
    return torch.where(mask, q_outlier, q_base)


def groupwise_weight_quantize(weight, bit_width, group_size):
    if int(bit_width) >= 16:
        return weight
    q_min, q_max = _quant_bounds(bit_width)
    out_features, in_features = weight.shape
    group_size = int(group_size)
    if group_size <= 0 or group_size >= in_features:
        abs_max = weight.abs().amax(dim=1, keepdim=True)
        scale = torch.clamp(abs_max, min=1e-8) / float(q_max)
        qw = torch.round(weight / scale).clamp(q_min, q_max)
        return qw * scale

    chunks = []
    for start in range(0, in_features, group_size):
        end = min(start + group_size, in_features)
        group = weight[:, start:end]
        abs_max = group.abs().amax(dim=1, keepdim=True)
        scale = torch.clamp(abs_max, min=1e-8) / float(q_max)
        qw = torch.round(group / scale).clamp(q_min, q_max)
        chunks.append(qw * scale)
    return torch.cat(chunks, dim=1)


class CalibratedPTQLinear(nn.Module):
    def __init__(
        self,
        original_linear,
        w_bit=4,
        a_bit=4,
        group_size=128,
        smooth_grid=None,
        clip_grid=None,
        sample_rows=256,
        scale_min=1e-3,
        scale_max=1e3,
        output_clip_percentile=0.999,
        output_clip_multiplier=0.0,
        outlier_ratio=0.0,
        outlier_a_bit=8,
    ):
        super().__init__()
        self.in_features = int(original_linear.in_features)
        self.out_features = int(original_linear.out_features)
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.w_bit = int(w_bit)
        self.a_bit = int(a_bit)
        self.group_size = int(group_size)
        self.smooth_grid = list(smooth_grid or [0.0, 0.25, 0.5, 0.75, 1.0])
        self.clip_grid = list(clip_grid or [1.0, 0.999, 0.995])
        self.sample_rows = int(sample_rows)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.output_clip_percentile = float(output_clip_percentile)
        self.output_clip_multiplier = float(output_clip_multiplier)
        self.outlier_ratio = float(outlier_ratio)
        self.outlier_a_bit = int(outlier_a_bit)
        self.finalized = False
        self._samples = []
        self._sample_count = 0
        self.best_alpha = 0.0
        self.best_clip = 1.0
        self.best_mse = float("inf")
        self.best_scale_mode = "identity"
        self.register_buffer("smooth_scale", torch.ones(self.in_features))
        self.register_buffer("quant_weight", torch.empty(0))
        self.register_buffer("output_clip", torch.empty(0))
        self.register_buffer("outlier_mask", torch.zeros(self.in_features, dtype=torch.bool))

    def forward(self, x):
        if not self.finalized:
            self._record_sample(x)
            return F.linear(x, self.weight, self.bias)

        view_shape = [1] * (x.dim() - 1) + [self.in_features]
        scale = self.smooth_scale.to(device=x.device, dtype=x.dtype).view(*view_shape)
        x_scaled = x / scale
        qx = mixed_outlier_activation_quantize(
            x_scaled,
            self.a_bit,
            self.outlier_a_bit,
            self.outlier_mask,
            self.best_clip,
        )
        qweight = self.quant_weight.to(device=x.device, dtype=x.dtype)
        y = F.linear(qx, qweight, self.bias)
        return self._apply_output_guard(y)

    def _apply_output_guard(self, y):
        if self.output_clip.numel() == 0:
            return y
        clip = self.output_clip.to(device=y.device, dtype=y.dtype)
        view_shape = [1] * (y.dim() - 1) + [self.out_features]
        clip = clip.view(*view_shape)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.maximum(torch.minimum(y, clip), -clip)

    def _record_sample(self, x):
        if self.sample_rows <= 0 or self._sample_count >= self.sample_rows:
            return
        flat = x.detach().reshape(-1, x.shape[-1])
        if flat.numel() == 0:
            return
        remaining = self.sample_rows - self._sample_count
        if flat.size(0) > remaining:
            flat = flat[:remaining]
        self._samples.append(flat.to(torch.float32).cpu())
        self._sample_count += int(flat.size(0))

    def _sanitize_scale(self, scale):
        finite = torch.isfinite(scale)
        if not bool(finite.all()):
            scale = torch.where(finite, scale, torch.ones_like(scale))
        scale = torch.clamp(scale, min=self.scale_min, max=self.scale_max)
        median = torch.clamp(torch.median(scale), min=1e-8)
        scale = torch.clamp(scale / median, min=self.scale_min, max=self.scale_max)
        return scale

    def _iter_scale_candidates(self, x_eval, weight):
        act_stat = torch.clamp(x_eval.abs().amax(dim=0), min=1e-5)
        weight_stat = torch.clamp(weight.abs().amax(dim=0), min=1e-5)
        yielded = set()

        identity = torch.ones_like(act_stat)
        yield "identity", 0.0, identity
        yielded.add(("identity", 0.0))

        for alpha in self.smooth_grid:
            alpha = float(alpha)
            smooth = act_stat.pow(alpha) / weight_stat.pow(1.0 - alpha)
            awq = act_stat.pow(alpha)
            balanced = act_stat.pow(alpha) * weight_stat.pow(1.0 - alpha)
            for mode, scale in (("smooth", smooth), ("awq", awq), ("balanced", balanced)):
                key = (mode, round(alpha, 6))
                if key in yielded:
                    continue
                yielded.add(key)
                yield mode, alpha, self._sanitize_scale(scale)

    def finalize(self):
        if self.finalized:
            return
        if not self._samples:
            self.quant_weight = groupwise_weight_quantize(
                self.weight.detach().to(torch.float32),
                self.w_bit,
                self.group_size,
            ).to(device=self.weight.device, dtype=self.weight.dtype)
            self.finalized = True
            return

        x_eval = torch.cat(self._samples, dim=0)
        if x_eval.size(0) > self.sample_rows:
            x_eval = x_eval[: self.sample_rows]
        x_eval = x_eval.to(device=self.weight.device, dtype=torch.float32)
        weight = self.weight.detach().to(torch.float32)
        bias = self.bias.detach().to(torch.float32) if self.bias is not None else None
        y_ref = F.linear(x_eval, weight, bias)
        outlier_mask = self._build_outlier_mask(x_eval)
        y_clip = None
        if self.output_clip_multiplier > 0:
            y_abs = y_ref.abs()
            p = min(max(self.output_clip_percentile, 0.0), 1.0)
            if p >= 0.99999:
                y_clip = y_abs.amax(dim=0)
            else:
                y_clip = torch.quantile(y_abs, p, dim=0)
            y_clip = torch.clamp(
                y_clip * self.output_clip_multiplier,
                min=1e-5,
                max=self.scale_max,
            )

        best = None
        for mode, alpha, smooth_scale in self._iter_scale_candidates(x_eval, weight):
            w_scaled = weight * smooth_scale.view(1, -1)
            qweight = groupwise_weight_quantize(w_scaled, self.w_bit, self.group_size)
            x_scaled = x_eval / smooth_scale.view(1, -1)
            for clip in self.clip_grid:
                qx = mixed_outlier_activation_quantize(
                    x_scaled,
                    self.a_bit,
                    self.outlier_a_bit,
                    outlier_mask,
                    clip,
                )
                y_quant = F.linear(qx, qweight, bias)
                if y_clip is not None:
                    y_quant = torch.nan_to_num(y_quant, nan=0.0, posinf=0.0, neginf=0.0)
                    y_quant = torch.maximum(torch.minimum(y_quant, y_clip.view(1, -1)), -y_clip.view(1, -1))
                mse = float((y_ref - y_quant).pow(2).mean().item())
                if best is None or mse < best[0]:
                    best = (mse, str(mode), float(alpha), float(clip), smooth_scale.detach(), qweight.detach())

        mse, mode, alpha, clip, smooth_scale, qweight = best
        self.best_mse = float(mse)
        self.best_scale_mode = str(mode)
        self.best_alpha = float(alpha)
        self.best_clip = float(clip)
        self.smooth_scale.copy_(smooth_scale.to(device=self.smooth_scale.device, dtype=self.smooth_scale.dtype))
        self.quant_weight = qweight.to(device=self.weight.device, dtype=self.weight.dtype)
        self.outlier_mask.copy_(outlier_mask.to(device=self.outlier_mask.device))
        if y_clip is not None:
            self.output_clip = y_clip.to(device=self.weight.device, dtype=self.weight.dtype)
        self.finalized = True
        self._samples = []

    def _build_outlier_mask(self, x_eval):
        mask = torch.zeros(self.in_features, dtype=torch.bool, device=x_eval.device)
        if self.outlier_ratio <= 0:
            return mask
        k = int(round(float(self.in_features) * self.outlier_ratio))
        k = max(0, min(self.in_features, k))
        if k == 0:
            return mask
        act_stat = x_eval.abs().amax(dim=0)
        top_idx = torch.topk(act_stat, k=k, largest=True, sorted=False).indices
        mask[top_idx] = True
        return mask


def replace_linear_with_ptq(module, w_bit, a_bit, args, skip_names=("lm_head",)):
    layers = []
    skip_names = set(skip_names)
    smooth_grid = _as_float_grid(args.ptq_smooth_grid, [0.0, 0.25, 0.5, 0.75, 1.0])
    clip_grid = _as_float_grid(args.ptq_clip_grid, [1.0, 0.999, 0.995])

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name not in skip_names:
                wrapped = CalibratedPTQLinear(
                    child,
                    w_bit=w_bit,
                    a_bit=a_bit,
                    group_size=int(args.ptq_group_size),
                    smooth_grid=smooth_grid,
                    clip_grid=clip_grid,
                    sample_rows=int(args.ptq_sample_rows),
                    scale_min=float(args.ptq_scale_min),
                    scale_max=float(args.ptq_scale_max),
                    output_clip_percentile=float(args.ptq_output_clip_percentile),
                    output_clip_multiplier=float(args.ptq_output_clip_multiplier),
                    outlier_ratio=float(args.ptq_outlier_ratio),
                    outlier_a_bit=int(args.ptq_outlier_a_bit),
                )
                setattr(parent, name, wrapped)
                layers.append(wrapped)
            else:
                _replace(child)

    _replace(module)
    return layers


def finalize_ptq_layers(layers):
    if not layers:
        return
    mses = []
    alphas = []
    clips = []
    modes = {}
    for layer in tqdm(layers, desc="[PTQ] Finalizing layers"):
        layer.finalize()
        mses.append(layer.best_mse)
        alphas.append(layer.best_alpha)
        clips.append(layer.best_clip)
        modes[layer.best_scale_mode] = modes.get(layer.best_scale_mode, 0) + 1
    mse_mean = sum(mses) / max(1, len(mses))
    alpha_mean = sum(alphas) / max(1, len(alphas))
    clip_mean = sum(clips) / max(1, len(clips))
    print(
        f"[PTQ] finalized {len(layers)} Linear layers "
        f"| mse_mean={mse_mean:.6e} | alpha_mean={alpha_mean:.3f} | clip_mean={clip_mean:.5f}"
        f" | modes={modes}"
    )


def fit_output_alignment(fp_embs, quant_embs, eps=1e-6):
    fp = fp_embs.to(torch.float32)
    quant = quant_embs.to(torch.float32)
    q_mean = quant.mean(dim=0, keepdim=True)
    fp_mean = fp.mean(dim=0, keepdim=True)
    q_center = quant - q_mean
    fp_center = fp - fp_mean
    gamma = (q_center * fp_center).sum(dim=0, keepdim=True) / torch.clamp(
        q_center.pow(2).sum(dim=0, keepdim=True),
        min=eps,
    )
    beta = fp_mean - gamma * q_mean
    return gamma, beta


def apply_output_alignment(embs, gamma, beta):
    aligned = embs.to(torch.float32) * gamma.to(embs.device) + beta.to(embs.device)
    return F.normalize(aligned, p=2, dim=1).cpu()


def cosine_damage_stats(fp_embs, quant_embs):
    fp = F.normalize(fp_embs.to(torch.float32), p=2, dim=1)
    quant = F.normalize(quant_embs.to(torch.float32), p=2, dim=1)
    err = 1.0 - F.cosine_similarity(fp, quant, dim=1)
    return {
        "mean": float(err.mean().item()),
        "p50": float(torch.quantile(err, 0.50).item()),
        "p90": float(torch.quantile(err, 0.90).item()),
        "p95": float(torch.quantile(err, 0.95).item()),
        "p99": float(torch.quantile(err, 0.99).item()),
        "max": float(err.max().item()),
    }


def print_damage_check(dataset, llm_name, tag, embs):
    fp_path = os.path.join("cache_data", f"{dataset}_{llm_name}_oracle_FP16.pt")
    if not os.path.exists(fp_path):
        return
    fp = torch.load(fp_path, map_location="cpu")
    if isinstance(fp, (tuple, list)):
        fp = fp[0]
    if not isinstance(fp, torch.Tensor) or fp.shape != embs.shape:
        return
    stats = cosine_damage_stats(fp, embs.cpu())
    print(
        f"[DamageCheck:{tag} vs FP16] "
        f"mean={stats['mean']:.6f} | p50={stats['p50']:.6f} | p90={stats['p90']:.6f} "
        f"| p95={stats['p95']:.6f} | p99={stats['p99']:.6f} | max={stats['max']:.6f}"
    )


def select_calibration_texts(texts, count, strategy, seed):
    count = min(len(texts), max(0, int(count)))
    if count == 0:
        return []
    if strategy == "first":
        indices = list(range(count))
    elif strategy == "random":
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        indices = torch.randperm(len(texts), generator=gen)[:count].tolist()
    else:
        raise ValueError(f"Unknown calibration strategy: {strategy}")
    return [texts[idx] for idx in indices]


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
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"

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

    alignment = None
    if config_spec["kind"] == "fake_wa":
        calib_texts = select_calibration_texts(
            texts,
            int(args.w4a_calib_samples),
            args.calibration_strategy,
            int(args.seed),
        )
        if args.w4a_backend == "fake":
            print(
                "[FakeWA] Installing AWQ-style fake quant Linear wrappers "
                f"| W{config_spec['w_bit']}A{config_spec['a_bit']} "
                f"| calib_samples={len(calib_texts)}"
            )
            replace_linear_with_fake_quant(
                model,
                w_bit=int(config_spec["w_bit"]),
                a_bit=int(config_spec["a_bit"]),
                awq_grid=int(args.w4a_awq_grid),
                skip_names=(),
            )
            if calib_texts:
                print(f"[FakeWA] Running calibration pass on {len(calib_texts)} node texts...")
                _ = encode_texts(model, tokenizer, calib_texts, batch_size, args.max_length, device)
        elif args.w4a_backend in ("ptq", "ptq_outlier"):
            if args.w4a_backend == "ptq":
                args.ptq_outlier_ratio = 0.0
            elif float(args.ptq_outlier_ratio) <= 0:
                args.ptq_outlier_ratio = 0.01
            print(
                "[PTQ-WA] Installing calibrated PTQ Linear wrappers "
                f"| W{config_spec['w_bit']}A{config_spec['a_bit']} "
                f"| calib_samples={len(calib_texts)} "
                f"| group_size={args.ptq_group_size} "
                f"| outlier_ratio={float(args.ptq_outlier_ratio):.4f} "
                f"| outlier_a_bit={int(args.ptq_outlier_a_bit)}"
            )
            layers = replace_linear_with_ptq(
                model,
                w_bit=int(config_spec["w_bit"]),
                a_bit=int(config_spec["a_bit"]),
                args=args,
                skip_names=(),
            )
            fp_calib_embs = None
            if calib_texts:
                print(f"[PTQ-WA] Collecting FP activations on {len(calib_texts)} calibration texts...")
                fp_calib_embs = encode_texts(model, tokenizer, calib_texts, batch_size, args.max_length, device)
            finalize_ptq_layers(layers)
            if bool(args.ptq_align_output) and calib_texts and fp_calib_embs is not None:
                print("[PTQ-WA] Fitting output affine alignment on calibration embeddings...")
                quant_calib_embs = encode_texts(model, tokenizer, calib_texts, batch_size, args.max_length, device)
                if torch.isfinite(quant_calib_embs).all():
                    alignment = fit_output_alignment(fp_calib_embs, quant_calib_embs)
                else:
                    print("[PTQ-WA] Skipping output alignment because calibration embeddings contain non-finite values.")
        else:
            raise ValueError(f"Unknown --w4a_backend={args.w4a_backend}")

    embs = encode_texts(model, tokenizer, texts, batch_size, args.max_length, device)
    if alignment is not None:
        gamma, beta = alignment
        embs = apply_output_alignment(embs, gamma, beta)

    finite_rows = torch.isfinite(embs).all(dim=1)
    if not bool(finite_rows.all()):
        bad_count = int((~finite_rows).sum().item())
        bad_path = f"{out_path}.bad_rows.pt"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(torch.nonzero(~finite_rows, as_tuple=False).view(-1).cpu(), bad_path)
        if not bool(args.allow_nonfinite_save):
            raise RuntimeError(
                f"{bad_count} non-finite embedding rows detected for {tag}. "
                f"Refusing to save an invalid pool. Bad row indices saved to {bad_path}. "
                "Use safer PTQ settings or --allow_nonfinite_save only for debugging."
            )
        print(f"[Warning] Saving invalid pool for debugging; bad row indices saved to {bad_path}.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(embs, out_path)
    print(f"[Saved] {out_path} | shape={tuple(embs.shape)}")
    print_damage_check(dataset, llm_name, tag, embs)

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
    parser.add_argument("--w4a_backend", type=str, default="fake", choices=["fake", "ptq", "ptq_outlier"])
    parser.add_argument("--tag_suffix", type=str, default="")
    parser.add_argument("--calibration_strategy", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ptq_group_size", type=int, default=128)
    parser.add_argument("--ptq_sample_rows", type=int, default=256)
    parser.add_argument("--ptq_smooth_grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--ptq_clip_grid", nargs="+", type=float, default=[1.0, 0.999, 0.995])
    parser.add_argument("--ptq_scale_min", type=float, default=1e-3)
    parser.add_argument("--ptq_scale_max", type=float, default=1e3)
    parser.add_argument("--ptq_output_clip_percentile", type=float, default=0.999)
    parser.add_argument("--ptq_output_clip_multiplier", type=float, default=0.0)
    parser.add_argument("--ptq_outlier_ratio", type=float, default=0.0)
    parser.add_argument("--ptq_outlier_a_bit", type=int, default=8)
    parser.add_argument("--ptq_align_output", action="store_true")
    parser.add_argument("--allow_nonfinite_save", action="store_true")
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
    if args.ptq_group_size <= 0:
        parser.error("--ptq_group_size must be positive")
    if args.ptq_sample_rows < 0:
        parser.error("--ptq_sample_rows must be >= 0")
    if args.ptq_scale_min <= 0 or args.ptq_scale_max <= 0:
        parser.error("--ptq_scale_min/--ptq_scale_max must be positive")
    if args.ptq_scale_min > args.ptq_scale_max:
        parser.error("--ptq_scale_min must be <= --ptq_scale_max")
    if not 0.0 <= args.ptq_output_clip_percentile <= 1.0:
        parser.error("--ptq_output_clip_percentile must be in [0, 1]")
    if args.ptq_output_clip_multiplier < 0:
        parser.error("--ptq_output_clip_multiplier must be >= 0")
    if not 0.0 <= args.ptq_outlier_ratio <= 1.0:
        parser.error("--ptq_outlier_ratio must be in [0, 1]")
    if args.ptq_outlier_a_bit < 4 or args.ptq_outlier_a_bit > 16:
        parser.error("--ptq_outlier_a_bit must be in [4, 16]")

    for dataset in args.datasets:
        for config_name in args.configs:
            generate_pool(dataset, args.llm_name, config_name, args)


if __name__ == "__main__":
    main()
