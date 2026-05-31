import argparse
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import load_raw_texts
from .generate_real_quant_pools import MODEL_SPECS, _forward_hidden_states, load_model_and_tokenizer


class FirstPassLayerStats:
    def __init__(self, max_samples):
        self.max_samples = int(max_samples)
        self.value_count = 0
        self.sum_abs = 0.0
        self.sum_sq_abs = 0.0
        self.max_abs = 0.0
        self.feature_dim = 0
        self.channel_count = 0
        self.channel_sum_abs = None
        self.channel_max_abs = None
        self.samples = []
        self.sample_count = 0
        self.forward_calls = 0

    def update(self, x):
        if x is None or x.numel() == 0:
            return

        x_abs = torch.nan_to_num(x.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).abs()
        feature_dim = int(x_abs.shape[-1])
        flat = x_abs.reshape(-1, feature_dim)

        self.forward_calls += 1
        self.feature_dim = feature_dim
        self.value_count += int(flat.numel())
        self.sum_abs += float(flat.sum().item())
        self.sum_sq_abs += float(flat.square().sum().item())
        self.max_abs = max(self.max_abs, float(flat.max().item()))

        channel_sum = flat.sum(dim=0).cpu()
        channel_max = flat.max(dim=0).values.cpu()
        if self.channel_sum_abs is None:
            self.channel_sum_abs = channel_sum
            self.channel_max_abs = channel_max
        else:
            self.channel_sum_abs += channel_sum
            self.channel_max_abs = torch.maximum(self.channel_max_abs, channel_max)
        self.channel_count += int(flat.shape[0])

        room = self.max_samples - self.sample_count
        if room <= 0:
            return

        values = flat.reshape(-1)
        take = min(room, int(values.numel()))
        if take <= 0:
            return
        if values.numel() > take:
            stride = max(1, values.numel() // take)
            picked = values[::stride][:take]
        else:
            picked = values
        self.samples.append(picked.cpu())
        self.sample_count += int(picked.numel())

    def threshold_report(self, percentile, std_factor, min_abs_threshold):
        if self.value_count <= 0:
            return {
                "mean_abs": 0.0,
                "std_abs": 0.0,
                "p_abs": 0.0,
                "threshold_abs": float(min_abs_threshold),
            }

        mean_abs = self.sum_abs / self.value_count
        sq_mean = self.sum_sq_abs / self.value_count
        std_abs = math.sqrt(max(0.0, sq_mean - mean_abs * mean_abs))
        if self.samples:
            sample = torch.cat(self.samples)
            q = max(0.0, min(1.0, float(percentile) / 100.0))
            p_abs = float(torch.quantile(sample, q).item())
        else:
            p_abs = 0.0
        threshold = max(float(min_abs_threshold), p_abs, mean_abs + float(std_factor) * std_abs)
        return {
            "mean_abs": mean_abs,
            "std_abs": std_abs,
            "p_abs": p_abs,
            "threshold_abs": threshold,
        }


class SecondPassLayerStats:
    def __init__(self, feature_dim):
        self.feature_dim = int(feature_dim)
        self.value_count = 0
        self.outlier_count = 0
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.forward_calls = 0
        self.channel_count = 0
        self.channel_sum_abs = torch.zeros(self.feature_dim, dtype=torch.float64)
        self.channel_max_abs = torch.zeros(self.feature_dim, dtype=torch.float32)
        self.channel_outlier_count = torch.zeros(self.feature_dim, dtype=torch.long)

    def update(self, x, threshold):
        if x is None or x.numel() == 0:
            return None

        x_abs = torch.nan_to_num(x.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).abs()
        feature_dim = int(x_abs.shape[-1])
        if feature_dim != self.feature_dim:
            return None

        flat = x_abs.reshape(-1, feature_dim)
        mask = flat > float(threshold)

        self.forward_calls += 1
        self.value_count += int(flat.numel())
        self.outlier_count += int(mask.sum().item())
        self.sum_abs += float(flat.sum().item())
        self.max_abs = max(self.max_abs, float(flat.max().item()))
        self.channel_count += int(flat.shape[0])
        self.channel_sum_abs += flat.sum(dim=0).cpu().to(torch.float64)
        self.channel_max_abs = torch.maximum(self.channel_max_abs, flat.max(dim=0).values.cpu())
        self.channel_outlier_count += mask.sum(dim=0).cpu().to(torch.long)

        return x_abs


class ActivationOutlierCollector:
    def __init__(
        self,
        model,
        percentile=99.9,
        std_factor=6.0,
        min_abs_threshold=0.0,
        max_abs_samples_per_layer=65536,
        skip_name_tokens=None,
        module_name_contains=None,
    ):
        self.model = model
        self.percentile = float(percentile)
        self.std_factor = float(std_factor)
        self.min_abs_threshold = float(min_abs_threshold)
        self.max_abs_samples_per_layer = int(max_abs_samples_per_layer)
        self.skip_name_tokens = tuple(skip_name_tokens or ("lm_head",))
        self.module_name_contains = module_name_contains
        self.mode = "first"
        self.handles = []
        self.first = defaultdict(lambda: FirstPassLayerStats(self.max_abs_samples_per_layer))
        self.second = {}
        self.thresholds = {}
        self.current_node_indices = []
        self.node_stats = defaultdict(lambda: {"outlier_count": 0, "max_abs": 0.0, "sum_mean_abs": 0.0, "hook_count": 0})

    def _should_hook(self, name):
        if any(token and token in name for token in self.skip_name_tokens):
            return False
        if self.module_name_contains and self.module_name_contains not in name:
            return False
        return True

    def register(self):
        self.remove()
        hooked = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._should_hook(name):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
                hooked.append(name)
        if not hooked:
            raise RuntimeError("No nn.Linear modules were hooked. Check --module_name_contains or model type.")
        return hooked

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, name):
        def hook(_module, inputs, _outputs):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.numel() == 0:
                return
            if x.dim() < 2:
                return

            if self.mode == "first":
                self.first[name].update(x)
                return

            threshold = self.thresholds.get(name)
            layer_stats = self.second.get(name)
            if threshold is None or layer_stats is None:
                return
            x_abs = layer_stats.update(x, threshold)
            if x_abs is None:
                return
            self._update_node_stats(x_abs, threshold)

        return hook

    def _update_node_stats(self, x_abs, threshold):
        node_indices = self.current_node_indices
        if not node_indices or x_abs.dim() < 3:
            return
        batch_size = int(x_abs.shape[0])
        if batch_size != len(node_indices):
            return

        per_sample = x_abs.reshape(batch_size, -1)
        sample_outliers = (per_sample > float(threshold)).sum(dim=1).cpu()
        sample_max = per_sample.max(dim=1).values.cpu()
        sample_mean = per_sample.mean(dim=1).cpu()

        for offset, node_idx in enumerate(node_indices):
            stats = self.node_stats[int(node_idx)]
            stats["outlier_count"] += int(sample_outliers[offset].item())
            stats["max_abs"] = max(stats["max_abs"], float(sample_max[offset].item()))
            stats["sum_mean_abs"] += float(sample_mean[offset].item())
            stats["hook_count"] += 1

    def finalize_first_pass(self):
        thresholds = {}
        second = {}
        for name, stats in s                                elf.first.items():
            report = stats.threshold_report(self.percentile, self.std_factor, self.min_abs_threshold)
            thresholds[name] = report["threshold_abs"]
            second[name] = SecondPassLayerStats(stats.feature_dim)
        self.thresholds = thresholds
        self.second = second
        return thresholds

    def layer_reports(self, top_k_channels):
        reports = []
        for name in sorted(self.second):
            first = self.first[name]
            second = self.second[name]
            threshold_info = first.threshold_report(self.percentile, self.std_factor, self.min_abs_threshold)
            outlier_ratio = second.outlier_count / second.value_count if second.value_count else 0.0
            mean_abs = second.sum_abs / second.value_count if second.value_count else 0.0
            channel_ratio = second.channel_outlier_count.to(torch.float64) / max(1, second.channel_count)
            channel_mean = second.channel_sum_abs / max(1, second.channel_count)
            order = torch.argsort(channel_ratio, descending=True)[: int(top_k_channels)]
            top_channels = []
            for channel_idx in order.tolist():
                top_channels.append(
                    {
                        "channel": int(channel_idx),
                        "outlier_count": int(second.channel_outlier_count[channel_idx].item()),
                        "outlier_ratio": float(channel_ratio[channel_idx].item()),
                        "max_abs": float(second.channel_max_abs[channel_idx].item()),
                        "mean_abs": float(channel_mean[channel_idx].item()),
                    }
                )

            reports.append(
                {
                    "module": name,
                    "forward_calls": int(second.forward_calls),
                    "feature_dim": int(second.feature_dim),
                    "value_count": int(second.value_count),
                    "outlier_count": int(second.outlier_count),
                    "outlier_ratio": float(outlier_ratio),
                    "max_abs": float(second.max_abs),
                    "mean_abs": float(mean_abs),
                    "first_pass_mean_abs": float(threshold_info["mean_abs"]),
                    "first_pass_std_abs": float(threshold_info["std_abs"]),
                    "first_pass_p_abs": float(threshold_info["p_abs"]),
                    "threshold_abs": float(threshold_info["threshold_abs"]),
                    "top_channels": top_channels,
                }
            )
        reports.sort(key=lambda item: (item["outlier_ratio"], item["outlier_count"]), reverse=True)
        return reports

    def node_reports(self, top_k_nodes):
        rows = []
        for node_idx, stats in self.node_stats.items():
            hook_count = max(1, int(stats["hook_count"]))
            rows.append(
                {
                    "node_idx": int(node_idx),
                    "outlier_count": int(stats["outlier_count"]),
                    "max_abs": float(stats["max_abs"]),
                    "mean_abs": float(stats["sum_mean_abs"] / hook_count),
                    "hook_count": int(stats["hook_count"]),
                }
            )
        rows.sort(key=lambda item: (item["outlier_count"], item["max_abs"]), reverse=True)
        return rows[: int(top_k_nodes)]


def select_calibration_indices(num_nodes, calib_samples, seed, strategy):
    count = min(int(calib_samples), int(num_nodes))
    if count <= 0:
        return []
    if strategy == "first":
        return list(range(count))
    if strategy != "random":
        raise ValueError(f"Unknown calibration strategy: {strategy}")
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(int(num_nodes)), count))


def batched_forward(model, tokenizer, texts, indices, batch_size, max_length, device, collector, desc):
    with torch.inference_mode():
        for start in tqdm(range(0, len(indices), batch_size), desc=desc):
            batch_indices = indices[start : start + batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]
            tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            collector.current_node_indices = batch_indices
            _ = _forward_hidden_states(model, tokens)
            collector.current_node_indices = []


def default_output_path(dataset, llm_name, calib_samples, seed):
    return os.path.join(
        "output",
        "graph_simhash",
        dataset.lower(),
        f"{dataset.lower()}_{llm_name}_activation_outliers_n{int(calib_samples)}_seed{int(seed)}.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Select activation outliers from a graph-text calibration set.")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--llm_name", type=str, default="ST", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--cache_dir", type=str, default="cache_data/hf_cache")
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_strategy", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--outlier_percentile", type=float, default=99.9)
    parser.add_argument("--std_factor", type=float, default=6.0)
    parser.add_argument("--min_abs_threshold", type=float, default=0.0)
    parser.add_argument("--max_abs_samples_per_layer", type=int, default=65536)
    parser.add_argument("--top_k_nodes", type=int, default=64)
    parser.add_argument("--top_k_channels", type=int, default=16)
    parser.add_argument("--module_name_contains", type=str, default=None)
    parser.add_argument("--skip_name_tokens", nargs="*", default=["lm_head"])
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.calib_samples <= 0:
        raise ValueError("--calib_samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")

    texts = load_raw_texts(args.dataset)
    selected_indices = select_calibration_indices(len(texts), args.calib_samples, args.seed, args.calib_strategy)
    model, tokenizer, _tag = load_model_and_tokenizer(args.llm_name, "fp16", args.cache_dir, force_cpu=args.force_cpu)
    device = next(model.parameters()).device

    collector = ActivationOutlierCollector(
        model=model,
        percentile=args.outlier_percentile,
        std_factor=args.std_factor,
        min_abs_threshold=args.min_abs_threshold,
        max_abs_samples_per_layer=args.max_abs_samples_per_layer,
        skip_name_tokens=args.skip_name_tokens,
        module_name_contains=args.module_name_contains,
    )
    hooked_modules = collector.register()

    print(
        "[ActivationOutlierCalib] "
        f"dataset={args.dataset} | llm={args.llm_name} | samples={len(selected_indices)}/{len(texts)} "
        f"| batch_size={args.batch_size} | max_length={args.max_length} | hooked={len(hooked_modules)}"
    )

    try:
        collector.mode = "first"
        batched_forward(
            model,
            tokenizer,
            texts,
            selected_indices,
            args.batch_size,
            args.max_length,
            device,
            collector,
            desc="Outlier pass 1/2",
        )
        collector.finalize_first_pass()

        collector.mode = "second"
        batched_forward(
            model,
            tokenizer,
            texts,
            selected_indices,
            args.batch_size,
            args.max_length,
            device,
            collector,
            desc="Outlier pass 2/2",
        )
    finally:
        collector.remove()

    layer_reports = collector.layer_reports(args.top_k_channels)
    node_reports = collector.node_reports(args.top_k_nodes)

    output_path = args.output_path or default_output_path(args.dataset, args.llm_name, len(selected_indices), args.seed)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result = {
        "meta": {
            "dataset": args.dataset,
            "llm_name": args.llm_name,
            "model_path": MODEL_SPECS[args.llm_name]["path"],
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "num_nodes": len(texts),
            "calib_samples": len(selected_indices),
            "calib_strategy": args.calib_strategy,
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "device": str(device),
            "outlier_percentile": float(args.outlier_percentile),
            "std_factor": float(args.std_factor),
            "min_abs_threshold": float(args.min_abs_threshold),
            "max_abs_samples_per_layer": int(args.max_abs_samples_per_layer),
            "hooked_module_count": len(hooked_modules),
            "module_name_contains": args.module_name_contains,
            "skip_name_tokens": list(args.skip_name_tokens),
        },
        "selected_indices": [int(idx) for idx in selected_indices],
        "selected_outlier_indices": [int(row["node_idx"]) for row in node_reports],
        "top_outlier_nodes": node_reports,
        "layers": layer_reports,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"[ActivationOutlierCalib] report={output_path}")
    print("[ActivationOutlierCalib] top layers:")
    for row in layer_reports[:5]:
        print(
            f"  {row['module']} | outlier_ratio={row['outlier_ratio']:.6f} "
            f"| threshold={row['threshold_abs']:.4f} | max={row['max_abs']:.4f}"
        )
    print("[ActivationOutlierCalib] top nodes:")
    for row in node_reports[:10]:
        print(
            f"  node={row['node_idx']} | hits={row['outlier_count']} "
            f"| max={row['max_abs']:.4f} | mean={row['mean_abs']:.4f}"
        )


if __name__ == "__main__":
    main()
