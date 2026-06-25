#!/usr/bin/env python3
"""Evaluate sampled link prediction under a frontend reuse operating point.

This is a lightweight task-transfer check for TAG node-classification graphs.
The script reconstructs residual-reuse embeddings, trains one link predictor on
baseline node representations, and evaluates the same predictor on baseline vs.
reused representations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.cli import build_parser, validate_args  # noqa: E402
from GraphhopSimhash.runner import (  # noqa: E402
    apply_soft_cosine_gate,
    build_controller,
    build_residual_correction_mask,
    build_route_bundle,
    build_support_split_masks,
    load_residual_target_features,
    make_run_args,
    resolve_residual_fit_config,
    resolve_residual_gate_threshold_config,
    train_baseline_model,
)
from GraphhopSimhash.residual_reuse import (  # noqa: E402
    apply_residual_adapter,
    train_residual_adapter,
)
from GraphhopSimhash.data import load_run_state  # noqa: E402


def quiet(_msg: str) -> None:
    return None


def accept_args_for_dataset(dataset: str, ablation_profile: bool = False) -> list[str]:
    if dataset == "pubmed":
        neg_anchors = "1" if ablation_profile else "4"
        return [
            "--residual_accept_mode",
            "shared",
            "--residual_positive_error_max",
            "0.40",
            "--residual_offline_negative_anchors_per_node",
            neg_anchors,
            "--residual_negative_error_min",
            "0.45",
            "--residual_negative_gate_weight",
            "1.0",
            "--residual_accept_loss_weight",
            "0.0",
            "--residual_gate_sparsity_weight",
            "0.02",
            "--residual_gate_accept_threshold",
            "0.91",
        ]
    neg_anchors = "1" if ablation_profile else "4"
    return [
        "--residual_accept_mode",
        "separate",
        "--residual_positive_error_max",
        "0.40",
        "--residual_offline_negative_anchors_per_node",
        neg_anchors,
        "--residual_negative_error_min",
        "0.45",
        "--residual_negative_gate_weight",
        "2.0",
        "--residual_accept_loss_weight",
        "2.0",
        "--residual_gate_sparsity_weight",
        "0.02",
        "--residual_classifier_accept_gate",
        "--residual_classifier_accept_mode",
        "both",
        "--residual_classifier_accept_max_kl",
        "0.2",
        "--residual_classifier_accept_after_residual",
        "--residual_classifier_accept_probe_alpha",
        "0.125",
        "--residual_gate_accept_threshold",
        "0.40",
    ]


def policy_args(policy: str) -> list[str]:
    if policy == "hash_only":
        return ["--disable_score_gate"]
    weights = {
        "p_only": ("3", "0", "0"),
        "p_c": ("3", "1", "0"),
        "p_u": ("3", "0", "1"),
        "full_tser": ("3", "1", "1"),
    }
    if policy not in weights:
        raise ValueError(f"Unknown policy: {policy}")
    p, c, u = weights[policy]
    return [
        "--enable_score_gate",
        "--score_propagation_weight",
        p,
        "--score_graph_context_weight",
        c,
        "--score_low_unique_weight",
        u,
    ]


def make_reuse_args(
    seed: int,
    runs: int,
    dataset: str,
    threshold: int,
    policy: str,
    ablation_profile: bool,
    hard_support: int,
    soft_support: int,
    gate_threshold: float | None,
) -> SimpleNamespace:
    residual_rank = "16" if ablation_profile else "64"
    residual_epochs = "1" if ablation_profile else "200"
    residual_max_pairs = "128" if ablation_profile else "4096"
    extra_anchors = "0" if ablation_profile else "8"
    extra_nodes = "0" if ablation_profile else "4096"
    args_list = [
        "--datasets",
        dataset,
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--experiment_suite",
        "residual_reuse",
        "--learned_hash_epochs",
        "10",
        "--learned_hash_dim",
        "128",
        "--hash_heads_per_route",
        "8",
        "--hamming_only_acceptor",
        "--disable_structure_check",
        "--allow_rare_fuzzy",
        "--score_pair_confidence_discount",
        "1",
        "--radius",
        "2",
        "--main_hash_head_bits",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "--residual_hard_min_support_hits",
        str(hard_support),
        "--residual_soft_min_support_hits",
        str(soft_support),
        "--residual_rank",
        residual_rank,
        "--residual_epochs",
        residual_epochs,
        "--residual_max_train_pairs",
        residual_max_pairs,
        "--residual_min_dist",
        "1.0",
        "--residual_alpha_grid",
        "0",
        "0.03125",
        "0.0625",
        "0.125",
        "0.25",
        "0.5",
        "--residual_support_aware_alpha",
        "--residual_adapter_type",
        "mlp",
        "--residual_dropout",
        "0.05",
        "--residual_loss_cosine_weight",
        "1.0",
        "--residual_loss_mse_weight",
        "0.5",
        "--residual_loss_delta_weight",
        "0.75",
        "--residual_bucket_mode",
        "support_dist",
        "--residual_offline_extra_anchors_per_node",
        extra_anchors,
        "--residual_offline_extra_query_nodes",
        extra_nodes,
        "--residual_train_split",
        "train_val",
        "--residual_gate_loss_weight",
        "0.5",
        "--residual_gate_error_scale",
        "0.25",
        "--residual_gate_error_max",
        "0.45",
        "--residual_embedding_source",
        "real_quant_fp",
        "--real_quant_model_name",
        "llama2_7b",
        "--real_quant_fp_tag",
        "W4BFPA8_B128",
        "--residual_fit_profile",
        "llama",
        "--score_reuse_threshold",
        str(threshold),
    ]
    args_list.extend(policy_args(policy))
    args_list.extend(accept_args_for_dataset(dataset, ablation_profile=ablation_profile))
    if gate_threshold is not None:
        args_list.extend(["--residual_gate_accept_threshold", str(gate_threshold)])
    parser = build_parser()
    parsed = parser.parse_args(args_list)
    validate_args(parser, parsed)
    return parsed


def make_unique_edges(edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index.detach().cpu()
    mask = row != col
    row = row[mask]
    col = col[mask]
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    edges = torch.stack([lo, hi], dim=1)
    return torch.unique(edges, dim=0)


def sample_negative_edges(num_nodes: int, positives: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    pos_set = {tuple(map(int, e)) for e in positives.tolist()}
    neg = []
    seen = set()
    while len(neg) < count:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key in pos_set or key in seen:
            continue
        seen.add(key)
        neg.append(key)
    return torch.tensor(neg, dtype=torch.long)


def split_edges(edge_index: torch.Tensor, num_nodes: int, seed: int):
    positives = make_unique_edges(edge_index)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(positives.size(0), generator=gen)
    positives = positives[order]
    n = positives.size(0)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    pos_train = positives[:n_train]
    pos_val = positives[n_train : n_train + n_val]
    pos_test = positives[n_train + n_val :]
    neg_all = sample_negative_edges(num_nodes, positives, n, seed + 991)
    neg_train = neg_all[:n_train]
    neg_val = neg_all[n_train : n_train + n_val]
    neg_test = neg_all[n_train + n_val :]
    return (pos_train, neg_train), (pos_val, neg_val), (pos_test, neg_test)


class LinkPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        src = z[edges[:, 0]]
        dst = z[edges[:, 1]]
        feat = torch.cat([torch.abs(src - dst), src * dst], dim=1)
        return self.net(feat).view(-1)


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ap_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order].astype(np.float64)
    if y.sum() == 0:
        return float("nan")
    precision = np.cumsum(y) / (np.arange(len(y)) + 1.0)
    return float((precision * y).sum() / y.sum())


def eval_link(model: LinkPredictor, z: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> tuple[float, float]:
    edges = torch.cat([pos, neg], dim=0).to(z.device)
    labels = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))], dim=0).to(z.device)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(z, edges)).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    return auc_score(y, scores), ap_score(y, scores)


def train_link_predictor(
    z: torch.Tensor,
    train_edges,
    val_edges,
    seed: int,
    epochs: int,
) -> LinkPredictor:
    torch.manual_seed(seed)
    model = LinkPredictor(z.size(1)).to(z.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_train, neg_train = (x.to(z.device) for x in train_edges)
    pos_val, neg_val = (x.to(z.device) for x in val_edges)
    train_edges_all = torch.cat([pos_train, neg_train], dim=0)
    train_labels = torch.cat([torch.ones(pos_train.size(0)), torch.zeros(neg_train.size(0))], dim=0).to(z.device)
    best_state = None
    best_val = -1.0
    for _epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(z, train_edges_all)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward()
        opt.step()
        val_auc, _ = eval_link(model, z, pos_val, neg_val)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_reuse_features(args, dataset: str, seed: int, no_residual_repair: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_args = make_run_args(args, seed)
    _conf, data, verify_features, device = load_run_state(dataset, run_args, seed)
    target_features = load_residual_target_features(dataset, data, run_args, device, quiet)
    fit_cfg = resolve_residual_fit_config(run_args, target_features.size(1))
    data.x = target_features
    node_model, node_acc, baseline_hidden, oracle_logits = train_baseline_model(data, run_args, device)
    route_bundle = build_route_bundle(verify_features, data, baseline_hidden, oracle_logits, run_args, quiet, device)
    controller = build_controller(
        data,
        verify_features,
        route_bundle,
        {"name": "ResidualReuse", "overrides": {}},
        run_args,
        device,
    )
    direct_features, _hits = controller.query_full_batch(
        route_bundle["hash_route_features"],
        verify_features,
        target_features,
    )
    trace = controller.last_query_trace
    correction_mask, _correction_info = build_residual_correction_mask(
        trace,
        controller.risk_gate,
        run_args.residual_direct_threshold,
        device,
        min_route_hits=run_args.residual_min_route_hits,
        min_base_hits=run_args.residual_min_base_hits,
    )
    split_info = build_support_split_masks(
        trace,
        run_args.residual_soft_min_support_hits,
        run_args.residual_hard_min_support_hits,
        device,
    )
    split_info = apply_soft_cosine_gate(trace, split_info, run_args.residual_soft_min_cosine, device)
    residual_hit_mask = split_info["residual_hit_mask"]
    soft_mask = split_info["soft_mask"]
    residual_base_features = target_features.clone()
    residual_base_features[residual_hit_mask] = direct_features[residual_hit_mask]
    correction_mask = correction_mask & soft_mask

    effective_reuse_mask = residual_hit_mask.clone()
    if no_residual_repair:
        residual_features = residual_base_features
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        adapter, _train_info = train_residual_adapter(
            target_embeddings=target_features,
            verify_features=verify_features,
            edge_index=data.edge_index,
            trace=trace,
            data=data,
            risk_scores=controller.node_risk_scores,
            rank=fit_cfg["rank"],
            epochs=fit_cfg["epochs"],
            lr=run_args.residual_lr,
            weight_decay=run_args.residual_weight_decay,
            residual_l2=run_args.residual_l2,
            train_split=run_args.residual_train_split,
            max_pairs=fit_cfg["max_pairs"],
            correction_mask=correction_mask,
            min_dist=run_args.residual_min_dist,
            controller=controller,
            hash_route_features=route_bundle["hash_route_features"],
            extra_anchors_per_node=run_args.residual_offline_extra_anchors_per_node,
            extra_query_nodes=run_args.residual_offline_extra_query_nodes,
            positive_error_max=run_args.residual_positive_error_max,
            extra_negative_anchors_per_node=run_args.residual_offline_negative_anchors_per_node,
            negative_error_min=run_args.residual_negative_error_min,
            negative_gate_weight=run_args.residual_negative_gate_weight,
            adapter_type=run_args.residual_adapter_type,
            accept_mode=run_args.residual_accept_mode,
            hidden_dim=run_args.residual_hidden_dim,
            hidden_layers=run_args.residual_hidden_layers,
            dropout=run_args.residual_dropout,
            cosine_weight=run_args.residual_loss_cosine_weight,
            mse_weight=run_args.residual_loss_mse_weight,
            delta_weight=run_args.residual_loss_delta_weight,
            bucket_mode=run_args.residual_bucket_mode,
            gate_loss_weight=run_args.residual_gate_loss_weight,
            accept_loss_weight=run_args.residual_accept_loss_weight,
            gate_error_scale=run_args.residual_gate_error_scale,
            gate_error_max=run_args.residual_gate_error_max,
            gate_sparsity_weight=run_args.residual_gate_sparsity_weight,
            class_aware_accept=run_args.residual_class_aware_accept,
            classifier_accept_gate=run_args.residual_classifier_accept_gate,
            classifier_model=node_model,
            classifier_reference_logits=oracle_logits,
            classifier_accept_mode=run_args.residual_classifier_accept_mode,
            classifier_accept_scope=run_args.residual_classifier_accept_scope,
            classifier_accept_after_residual=run_args.residual_classifier_accept_after_residual,
            classifier_accept_probe_alpha=run_args.residual_classifier_accept_probe_alpha,
            classifier_accept_max_kl=run_args.residual_classifier_accept_max_kl,
        )
        gate_threshold = resolve_residual_gate_threshold_config(run_args)
        if gate_threshold is None:
            gate_threshold = max(0.0, float(run_args.residual_gate_accept_threshold))
        residual_features, apply_info = apply_residual_adapter(
            direct_embeddings=residual_base_features,
            target_embeddings=target_features,
            verify_features=verify_features,
            edge_index=data.edge_index,
            trace=trace,
            adapter=adapter,
            risk_scores=controller.node_risk_scores,
            alpha=0.0,
            gate_accept_threshold=gate_threshold,
            min_dist=run_args.residual_min_dist,
            correction_mask=correction_mask,
            bucket_mode=run_args.residual_bucket_mode,
        )
        rejected_nodes = apply_info.get("rejected_nodes", None)
        if rejected_nodes is not None and int(rejected_nodes.numel()) > 0:
            effective_reuse_mask[rejected_nodes] = False
    reuse_rate = float(effective_reuse_mask.float().mean().item())

    node_model.eval()
    with torch.no_grad():
        baseline_z = node_model.encoder(target_features).detach()
        reuse_z = node_model.encoder(residual_features).detach()
    return data, baseline_z, reuse_z, float(node_acc), reuse_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "pubmed"], default="cora")
    parser.add_argument("--threshold", type=int, default=45)
    parser.add_argument(
        "--policy",
        choices=["hash_only", "p_only", "p_c", "p_u", "full_tser"],
        default="full_tser",
    )
    parser.add_argument(
        "--ablation_profile",
        action="store_true",
        help="Use the lightweight TSER component-ablation residual profile.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hard_support", type=int, default=5)
    parser.add_argument("--soft_support", type=int, default=3)
    parser.add_argument("--gate_threshold", type=float, default=None)
    parser.add_argument(
        "--reuse_only",
        action="store_true",
        help="Only rebuild the reuse embeddings and report reuse rate; skip link predictor training.",
    )
    parser.add_argument(
        "--no_residual_repair",
        action="store_true",
        help="Use TSER-filtered anchors directly without residual correction or residual accept gating.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("output/node_link_reuse_transfer"))
    args = parser.parse_args()

    reuse_args = make_reuse_args(
        args.seed,
        args.runs,
        args.dataset,
        args.threshold,
        args.policy,
        args.ablation_profile,
        args.hard_support,
        args.soft_support,
        args.gate_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        data, baseline_z, reuse_z, node_acc, reuse_rate = build_reuse_features(
            reuse_args,
            args.dataset,
            seed,
            no_residual_repair=args.no_residual_repair,
        )
        if args.reuse_only:
            rows.append(
                {
                    "run": run_idx + 1,
                    "seed": seed,
                    "node_acc": node_acc,
                    "reuse_rate": reuse_rate,
                    "base_auc": float("nan"),
                    "reuse_auc": float("nan"),
                    "auc_drop": float("nan"),
                    "base_ap": float("nan"),
                    "reuse_ap": float("nan"),
                    "ap_drop": float("nan"),
                }
            )
            print(f"run={run_idx + 1} seed={seed} reuse={reuse_rate:.2%}", flush=True)
            continue
        train_edges, val_edges, test_edges = split_edges(data.edge_index, data.num_nodes, seed)
        link_model = train_link_predictor(baseline_z, train_edges, val_edges, seed, args.epochs)
        pos_test, neg_test = test_edges
        base_auc, base_ap = eval_link(link_model, baseline_z, pos_test, neg_test)
        reuse_auc, reuse_ap = eval_link(link_model, reuse_z, pos_test, neg_test)
        rows.append(
            {
                "run": run_idx + 1,
                "seed": seed,
                "node_acc": node_acc,
                "reuse_rate": reuse_rate,
                "base_auc": base_auc,
                "reuse_auc": reuse_auc,
                "auc_drop": base_auc - reuse_auc,
                "base_ap": base_ap,
                "reuse_ap": reuse_ap,
                "ap_drop": base_ap - reuse_ap,
            }
        )
        print(
            f"run={run_idx + 1} seed={seed} reuse={reuse_rate:.2%} "
            f"AUC={base_auc:.4f}->{reuse_auc:.4f} drop={base_auc - reuse_auc:.2%} "
            f"AP={base_ap:.4f}->{reuse_ap:.4f} drop={base_ap - reuse_ap:.2%}",
            flush=True,
        )

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    lines = [
        f"# {args.dataset} {args.policy} T{args.threshold} Link Prediction Transfer Check",
        "",
        f"Operating point: policy={args.policy}, T={args.threshold}, LLaMA2-7B W4BFPA8 target.",
        f"Residual repair: {'disabled' if args.no_residual_repair else 'enabled'}.",
        "The link predictor is trained once on baseline node representations and evaluated on baseline vs reused representations.",
        "",
        "| Run | Reuse | Base AUC | Reuse AUC | AUC Drop | Base AP | Reuse AP | AP Drop |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['reuse_rate']:.2%} | {r['base_auc']:.4f} | {r['reuse_auc']:.4f} | "
            f"{r['auc_drop']:.2%} | {r['base_ap']:.4f} | {r['reuse_ap']:.4f} | {r['ap_drop']:.2%} |"
        )
    lines.extend(
        [
            "| **Mean** | "
            f"**{mean('reuse_rate'):.2%}** | **{mean('base_auc'):.4f}** | **{mean('reuse_auc'):.4f}** | "
            f"**{mean('auc_drop'):.2%}** | **{mean('base_ap'):.4f}** | **{mean('reuse_ap'):.4f}** | "
            f"**{mean('ap_drop'):.2%}** |",
            "",
        ]
    )
    suffix = "_norepair" if args.no_residual_repair else ""
    out_md = args.output_dir / f"{args.dataset}_{args.policy}_T{args.threshold}_link_reuse{suffix}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
