#!/usr/bin/env python3
"""Generate and evaluate graph-aware dynamic BFPA4/BFPA6 embeddings.

This is the first full encoder-side validation for the policy:

    default: BFPA4 activation blocks
    refine: BFPA6 activation blocks when graph_risk(node) * activation_stress(block)
            exceeds a threshold

Unlike pool-level routing, this script performs the BFPA4/BFPA6 decision inside
each Linear wrapper during the LLaMA forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.data import load_run_state  # noqa: E402
from GraphhopSimhash.generate_real_quant_pools import (  # noqa: E402
    MODEL_SPECS,
    apply_official_awq_w4,
    bfp_fake_quantize,
    load_model_and_tokenizer,
    load_raw_texts,
    load_text_selection_edge_index,
    mean_pool,
)
from GraphhopSimhash.real_quant import default_pool_path, load_tensor_pool  # noqa: E402
from GraphhopSimhash.runner import evaluate_gnn_embeddings, train_baseline_model  # noqa: E402


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    lo = torch.quantile(x, 0.01)
    hi = torch.quantile(x, 0.99)
    return ((x - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)


def build_degree_risk(dataset: str, num_nodes: int) -> torch.Tensor:
    edge_index = load_text_selection_edge_index(dataset)
    if edge_index is None:
        raise FileNotFoundError(f"Could not load edge_index for {dataset}")
    deg = torch.zeros(int(num_nodes), dtype=torch.float32)
    src, dst = edge_index
    valid_src = src[(src >= 0) & (src < num_nodes)].to(torch.long)
    valid_dst = dst[(dst >= 0) & (dst < num_nodes)].to(torch.long)
    deg.scatter_add_(0, valid_src, torch.ones_like(valid_src, dtype=torch.float32))
    deg.scatter_add_(0, valid_dst, torch.ones_like(valid_dst, dtype=torch.float32))
    return _normalize(torch.log1p(deg))


def _bfp_quantize_grouped(grouped: torch.Tensor, mantissa_bit: int) -> torch.Tensor:
    q_min = -(2 ** (int(mantissa_bit) - 1))
    q_max = (2 ** (int(mantissa_bit) - 1)) - 1
    abs_max = grouped.detach().abs().amax(dim=-1, keepdim=True)
    safe_abs = abs_max.to(torch.float32).clamp_min(1e-30)
    exponent = torch.ceil(torch.log2(safe_abs / float(q_max))).clamp(min=-30.0, max=30.0)
    scale = torch.pow(torch.full_like(exponent, 2.0), exponent).to(dtype=grouped.dtype)
    quantized = torch.round(grouped / scale).clamp(q_min, q_max)
    return quantized * scale


class GraphAwareBFPController:
    def __init__(
        self,
        node_risk: torch.Tensor,
        threshold: float = 0.35,
        stress_scale: float = 8.0,
        block_size: int = 128,
        base_mantissa: int = 4,
        refine_mantissa: int = 6,
    ) -> None:
        self.node_risk_cpu = node_risk.detach().to(torch.float32).cpu()
        self.threshold = float(threshold)
        self.stress_scale = float(stress_scale)
        self.block_size = int(block_size)
        self.base_mantissa = int(base_mantissa)
        self.refine_mantissa = int(refine_mantissa)
        self.current_node_ids: torch.Tensor | None = None
        self.total_blocks = 0
        self.refined_blocks = 0

    def set_batch_node_ids(self, node_ids: torch.Tensor) -> None:
        self.current_node_ids = node_ids.detach().to(torch.long).cpu()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_node_ids is None or x.dim() < 2:
            return bfp_fake_quantize(x, self.base_mantissa, self.block_size, dim=-1)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        orig_shape = x.shape
        hidden = int(orig_shape[-1])
        pad = (self.block_size - (hidden % self.block_size)) % self.block_size
        x_work = F.pad(x, (0, pad)) if pad else x
        grouped = x_work.reshape(*x_work.shape[:-1], -1, self.block_size)

        abs_grouped = grouped.detach().abs()
        max_abs = abs_grouped.amax(dim=-1)
        median_abs = abs_grouped.median(dim=-1).values.clamp_min(1e-12)
        stress = torch.log2((max_abs / median_abs).clamp_min(1.0))
        stress_norm = (stress / max(1e-6, self.stress_scale)).clamp(0.0, 1.0)

        batch = int(grouped.shape[0]) if grouped.dim() >= 3 else int(self.current_node_ids.numel())
        node_ids = self.current_node_ids[:batch]
        risk = self.node_risk_cpu[node_ids].to(device=x.device, dtype=torch.float32)
        view_shape = [batch] + [1] * (stress_norm.dim() - 1)
        priority = risk.view(*view_shape) * stress_norm
        refine_mask = priority >= self.threshold

        q_base = _bfp_quantize_grouped(grouped, self.base_mantissa)
        q_refine = _bfp_quantize_grouped(grouped, self.refine_mantissa)
        out = torch.where(refine_mask.unsqueeze(-1), q_refine, q_base)
        self.total_blocks += int(refine_mask.numel())
        self.refined_blocks += int(refine_mask.sum().item())

        out = out.reshape(*x_work.shape)
        if pad:
            out = out[..., :hidden]
        return torch.nan_to_num(out.reshape(orig_shape), nan=0.0, posinf=0.0, neginf=0.0)


class GraphAwareBFPActivationLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, controller: GraphAwareBFPController) -> None:
        super().__init__()
        self.linear = original_linear
        self.controller = controller

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.controller.quantize(x))


def replace_linear_with_graph_bfp(module: nn.Module, controller: GraphAwareBFPController, skip_names=("lm_head",)) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in set(skip_names):
            setattr(module, name, GraphAwareBFPActivationLinear(child, controller))
        else:
            replace_linear_with_graph_bfp(child, controller, skip_names=skip_names)


def _forward_last_hidden_state(model, tokens):
    target_model = model.model if hasattr(model, "model") and model.__class__.__name__.endswith("ForCausalLM") else model
    outputs = target_model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        output_hidden_states=False,
        return_dict=True,
    )
    return torch.nan_to_num(outputs.last_hidden_state.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)


def encode_texts_dynamic(model, tokenizer, texts, node_ids, batch_size, max_length, device, controller):
    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[start : start + batch_size]
            ids = torch.as_tensor(node_ids[start : start + batch_size], dtype=torch.long)
            controller.set_batch_node_ids(ids)
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = _forward_last_hidden_state(model, tokens)
            all_embs.append(mean_pool(hidden, tokens["attention_mask"]).cpu())
    return torch.cat(all_embs, dim=0)


def _make_eval_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        llm_name="ST",
        emb_dim=768,
        radius=2,
        max_test=None,
        standard_eval_baseline=False,
        hash_view="mix",
        hash_mix_weights=[0.30, 0.70, 0.0],
        sketch_bits=14,
        controller_seed=seed,
        run_seed=seed,
        score_rarity_bits=12,
        score_rarity_seed=12345,
        score_propagation_weight=3.0,
        score_graph_context_weight=1.0,
        score_low_unique_weight=1.0,
    )


def evaluate_pool(dataset: str, pool: torch.Tensor, reference: torch.Tensor, runs: int, seed: int, device: torch.device) -> dict[str, float]:
    drops = []
    accs = []
    baselines = []
    for run_idx in range(int(runs)):
        run_seed = int(seed) + run_idx
        run_args = _make_eval_args(run_seed)
        _conf, data, _verify_features, run_device = load_run_state(dataset, run_args, run_seed)
        data.x = reference.to(run_device)
        model, baseline_acc, _hidden, _logits = train_baseline_model(data, run_args, run_device)
        with torch.no_grad():
            mixed_hidden = model.encoder(pool.to(run_device))
        acc = evaluate_gnn_embeddings(model, data, mixed_hidden)
        baselines.append(float(baseline_acc))
        accs.append(float(acc))
        drops.append(float(baseline_acc - acc))
    return {
        "baseline": float(sum(baselines) / len(baselines)),
        "acc": float(sum(accs) / len(accs)),
        "drop": float(sum(drops) / len(drops)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--llm_name", default="llama2_7b", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--stress_scale", type=float, default=8.0)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--base_mantissa", type=int, default=4)
    parser.add_argument("--refine_mantissa", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="cache_data/model")
    parser.add_argument("--output_dir", default="output/graphbfp_dynamic_pool")
    parser.add_argument(
        "--save_to_cache",
        action="store_true",
        help="Save the generated pool to the standard cache_data path so other suites can load it by tag.",
    )
    parser.add_argument(
        "--cache_tag",
        default=None,
        help="Optional explicit cache tag. Defaults to W4GraphBFPA{base}to{refine}_B{block}_deg_t{threshold}.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--awq_calib_samples", type=int, default=128)
    parser.add_argument("--awq_seqlen", type=int, default=512)
    parser.add_argument("--awq_q_group_size", type=int, default=128)
    parser.add_argument("--awq_no_zero_point", action="store_true")
    parser.add_argument("--awq_disable_auto_scale", action="store_true")
    parser.add_argument("--awq_disable_mse_clip", action="store_true")
    parser.add_argument("--awq_force_mse_clip", action="store_true")
    parser.add_argument("--awq_results_path", type=str, default=None)
    parser.add_argument("--awq_overwrite_results", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    default_tag = f"W4GraphBFPA{args.base_mantissa}to{args.refine_mantissa}_B{args.block_size}_deg_t{args.threshold:g}"
    tag = args.cache_tag or default_tag
    if args.save_to_cache:
        out_path = Path(default_pool_path(args.dataset, args.llm_name, tag))
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = out_dir / f"{args.dataset}_{args.llm_name}_{tag}.pt"
    meta_path = out_dir / f"{tag}_metadata.json"
    cache_meta_path = Path(default_pool_path(args.dataset, args.llm_name, tag)).with_suffix(".json")
    metadata: dict[str, Any] = {}
    if out_path.exists() and not args.overwrite:
        print(f"[Skip] {out_path} exists. Use --overwrite to regenerate.")
        embs = torch.load(out_path, map_location="cpu").to(torch.float32)
        for candidate in (cache_meta_path, meta_path):
            if candidate.exists():
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
                break
    else:
        texts = load_raw_texts(args.dataset)
        risk = build_degree_risk(args.dataset, len(texts))
        model, tokenizer, _tag = load_model_and_tokenizer(args.llm_name, "fp16", args.cache_dir, force_cpu=True)
        model, device = apply_official_awq_w4(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            dataset=args.dataset,
            llm_name=args.llm_name,
            args=args,
            activation_bit=int(args.refine_mantissa),
        )
        controller = GraphAwareBFPController(
            risk,
            threshold=float(args.threshold),
            stress_scale=float(args.stress_scale),
            block_size=int(args.block_size),
            base_mantissa=int(args.base_mantissa),
            refine_mantissa=int(args.refine_mantissa),
        )
        print(
            "[GraphAwareBFP] Installing dynamic wrappers "
            f"| base=A{args.base_mantissa} | refine=A{args.refine_mantissa} "
            f"| block={args.block_size} | threshold={args.threshold} | stress_scale={args.stress_scale}"
        )
        replace_linear_with_graph_bfp(model, controller, skip_names=())
        node_ids = list(range(len(texts)))
        embs = encode_texts_dynamic(
            model,
            tokenizer,
            texts,
            node_ids,
            int(args.batch_size),
            int(args.max_length),
            device,
            controller,
        )
        torch.save(embs, out_path)
        refined_ratio = controller.refined_blocks / max(1, controller.total_blocks)
        effective_bits = float(args.base_mantissa) + refined_ratio * float(
            int(args.refine_mantissa) - int(args.base_mantissa)
        )
        metadata = {
            "dataset": args.dataset,
            "llm_name": args.llm_name,
            "tag": tag,
            "default_tag": default_tag,
            "pool_path": str(out_path),
            "threshold": float(args.threshold),
            "stress_scale": float(args.stress_scale),
            "block_size": int(args.block_size),
            "base_mantissa": int(args.base_mantissa),
            "refine_mantissa": int(args.refine_mantissa),
            "total_blocks": int(controller.total_blocks),
            "refined_blocks": int(controller.refined_blocks),
            "refined_ratio": float(refined_ratio),
            "effective_bits": float(effective_bits),
            "policy": "degree_risk(node) * activation_stress(block) >= threshold",
        }
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.save_to_cache:
            cache_meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"[Saved] {out_path} | shape={tuple(embs.shape)} "
            f"| refined_blocks={controller.refined_blocks}/{controller.total_blocks} ({refined_ratio:.2%}) "
            f"| effective_bits={effective_bits:.3f}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = load_tensor_pool(default_pool_path(args.dataset, args.llm_name, "W4BFPA8_B128"), device).cpu()
    result = evaluate_pool(args.dataset, embs, reference, int(args.runs), int(args.seed), device)
    note = (
        f"Graph-aware dynamic BFP pool | dataset={args.dataset} | tag={tag}\n"
        f"output={out_path}\n"
        f"metadata={meta_path}\n"
        f"refined_ratio={metadata.get('refined_ratio', 'unknown')}\n"
        f"effective_bits={metadata.get('effective_bits', 'unknown')}\n"
        f"Baseline Acc: {result['baseline']:.4f}\n"
        f"Dynamic Acc:  {result['acc']:.4f}\n"
        f"Dynamic Drop: {result['drop']:.2%}\n"
    )
    (out_dir / f"{tag}_summary.txt").write_text(note, encoding="utf-8")
    print(note)


if __name__ == "__main__":
    main()
