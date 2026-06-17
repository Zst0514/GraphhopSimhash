#!/usr/bin/env python3
"""Profile frontend LLaMA encoding and backend 3-layer GCN inference.

This script is intentionally narrow: it measures the stage-level latency split
used in the paper motivation.  The frontend is sampled LLaMA text encoding and
is extrapolated to the full graph.  The backend is full-graph 3-layer GCN
inference over cached LLaMA embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GraphhopSimhash.config import DATASET_CONFIGS  # noqa: E402
from GraphhopSimhash.data import load_raw_texts, load_run_state  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ThreeLayerGCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_dim, hidden_dim, cached=False, normalize=True)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, cached=False, normalize=True)
        self.conv3 = GCNConv(hidden_dim, out_dim, cached=False, normalize=True)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)


@dataclass
class DatasetProfile:
    dataset: str
    nodes: int
    edges: int
    frontend_sample_nodes: int
    frontend_sample_s: float | None
    frontend_nodes_per_s: float | None
    frontend_est_full_s: float | None
    gcn_full_s: float
    gcn_repeats: int
    tf_share_est: float | None
    gcn_share_est: float | None
    tf_over_gcn_est: float | None
    embedding_pool: str


def default_pool_path(dataset: str, model_name: str, tag: str) -> Path:
    return ROOT / "cache_data" / f"{dataset}_{model_name}_oracle_{tag}.pt"


def load_embedding_pool(dataset: str, model_name: str, tag: str, device: torch.device) -> tuple[torch.Tensor, Path]:
    path = default_pool_path(dataset, model_name, tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding pool: {path}")
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        for key in ("embeddings", "x", "features"):
            if key in x:
                x = x[key]
                break
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} did not contain a tensor embedding pool")
    return x.to(device=device, dtype=torch.float32), path


def load_graph(dataset: str, device: torch.device):
    run_args = SimpleNamespace(llm_name="ST", emb_dim=768, radius=2, max_test=None)
    _conf, data, _cheap, run_device = load_run_state(dataset, run_args, seed=42)
    if torch.device(run_device).type != device.type:
        data = data.to(device)
    if data.y.dim() > 1:
        data.y = data.y.squeeze()
    return data


def profile_gcn(
    dataset: str,
    device: torch.device,
    model_name: str,
    pool_tag: str,
    hidden_dim: int,
    warmup: int,
    repeats: int,
) -> tuple[float, int, int, Path]:
    data = load_graph(dataset, device)
    x, pool_path = load_embedding_pool(dataset, model_name, pool_tag, device)
    if x.size(0) != int(data.num_nodes):
        raise ValueError(f"{dataset}: pool nodes={x.size(0)} graph nodes={int(data.num_nodes)}")
    out_dim = int(data.y.max().item()) + 1
    model = ThreeLayerGCN(x.size(1), hidden_dim, out_dim).to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(x, data.edge_index)
        sync(device)
        times = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            _ = model(x, data.edge_index)
            sync(device)
            times.append(time.perf_counter() - start)
    return float(sum(times) / len(times)), int(data.num_nodes), int(data.edge_index.size(1)), pool_path


def select_sample_texts(texts: list[str], sample_nodes: int, seed: int) -> list[str]:
    sample_nodes = min(len(texts), max(1, int(sample_nodes)))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(len(texts), generator=gen)[:sample_nodes].tolist()
    return [texts[int(i)] for i in idx]


class FrontendProfiler:
    def __init__(self, model_name: str, config_name: str):
        from GraphhopSimhash.generate_real_quant_pools import (
            MODEL_SPECS,
            canonical_model_name,
            load_model_and_tokenizer,
        )

        self._encode_texts = None
        from GraphhopSimhash.generate_real_quant_pools import encode_texts

        self._encode_texts = encode_texts
        self.model, self.tokenizer, _tag = load_model_and_tokenizer(
            model_name,
            config_name,
            str(ROOT / "cache_data" / "model"),
        )
        canonical = canonical_model_name(model_name)
        self.model_spec = MODEL_SPECS[canonical]
        self.device = next(self.model.parameters()).device

    def profile(
        self,
        dataset: str,
        sample_nodes: int,
        batch_size: int,
        max_length: int,
        warmup_batches: int,
        seed: int,
    ) -> tuple[int, float, float, float]:
        texts = load_raw_texts(dataset)
        sample_texts = select_sample_texts(texts, sample_nodes, seed)
        warmup_texts = sample_texts[: min(len(sample_texts), max(1, batch_size * max(0, warmup_batches)))]
        if warmup_texts:
            _ = self._encode_texts(
                self.model,
                self.tokenizer,
                warmup_texts,
                batch_size,
                max_length,
                self.device,
                model_spec=self.model_spec,
            )
        sync(torch.device(self.device))
        start = time.perf_counter()
        _ = self._encode_texts(
            self.model,
            self.tokenizer,
            sample_texts,
            batch_size,
            max_length,
            self.device,
            model_spec=self.model_spec,
        )
        sync(torch.device(self.device))
        elapsed = time.perf_counter() - start
        nodes_per_s = len(sample_texts) / max(elapsed, 1e-9)
        est_full = len(texts) / max(nodes_per_s, 1e-9)
        return len(sample_texts), float(elapsed), float(nodes_per_s), float(est_full)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_outputs(rows: list[DatasetProfile], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {"args": args_dict, "rows": [asdict(row) for row in rows]}
    (output_dir / "stage_profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with (output_dir / "stage_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    lines = [
        "# TF Frontend vs 3-layer GCN Backend Profiling",
        "",
        f"GPU/device: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}`",
        f"Frontend: `{args.frontend_model}` sampled LLaMA text encoding, config `{args.frontend_config}`.",
        f"Backend: 3-layer GCN full-graph inference over `{args.gcn_pool_tag}` embeddings.",
        "",
        "| Dataset | Nodes | Edges | TF sample | TF est. full (s) | 3-layer GCN (s) | TF share | TF/GNN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        tf_est = "-" if row.frontend_est_full_s is None else f"{row.frontend_est_full_s:.2f}"
        tf_share = "-" if row.tf_share_est is None else f"{100.0 * row.tf_share_est:.2f}%"
        tf_gnn = "-" if row.tf_over_gcn_est is None else f"{row.tf_over_gcn_est:.1f}x"
        lines.append(
            f"| {row.dataset} | {row.nodes:,} | {row.edges:,} | {row.frontend_sample_nodes:,} | "
            f"{tf_est} | {row.gcn_full_s:.4f} | {tf_share} | {tf_gnn} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- TF full-graph latency is extrapolated from sampled node-text encoding throughput.",
            "- GCN latency is measured as full-graph inference, excluding training.",
            "- The split is intended for motivation profiling, not final accelerator speedup.",
        ]
    )
    (output_dir / "stage_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "arxiv", "wikics"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "tf_gcn_stage_profile")
    parser.add_argument("--frontend-mode", choices=["profile", "skip"], default="profile")
    parser.add_argument("--frontend-model", default="llama2_7b")
    parser.add_argument("--frontend-config", default="fp16")
    parser.add_argument("--frontend-sample-nodes", type=int, default=256)
    parser.add_argument("--frontend-batch-size", type=int, default=1)
    parser.add_argument("--frontend-max-length", type=int, default=256)
    parser.add_argument("--frontend-warmup-batches", type=int, default=2)
    parser.add_argument("--gcn-model-name", default="llama2_7b")
    parser.add_argument("--gcn-pool-tag", default="W4BFPA8_B128")
    parser.add_argument("--gcn-hidden-dim", type=int, default=256)
    parser.add_argument("--gcn-warmup", type=int, default=5)
    parser.add_argument("--gcn-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    invalid = [ds for ds in args.datasets if ds not in DATASET_CONFIGS]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[DatasetProfile] = []
    frontend = None
    try:
        if args.frontend_mode == "profile":
            frontend = FrontendProfiler(args.frontend_model, args.frontend_config)
        for dataset in args.datasets:
            print(f"\n[StageProfile] dataset={dataset}")
            gcn_s, nodes, edges, pool_path = profile_gcn(
                dataset,
                device,
                args.gcn_model_name,
                args.gcn_pool_tag,
                args.gcn_hidden_dim,
                args.gcn_warmup,
                args.gcn_repeats,
            )

            sample_n = 0
            frontend_s = None
            nodes_per_s = None
            frontend_full_s = None
            if frontend is not None:
                sample_n, frontend_s, nodes_per_s, frontend_full_s = frontend.profile(
                    dataset,
                    args.frontend_sample_nodes,
                    args.frontend_batch_size,
                    args.frontend_max_length,
                    args.frontend_warmup_batches,
                    args.seed,
                )

            total_est = None if frontend_full_s is None else frontend_full_s + gcn_s
            tf_share = None if total_est is None else frontend_full_s / max(total_est, 1e-9)
            gcn_share = None if total_est is None else gcn_s / max(total_est, 1e-9)
            tf_over_gcn = None if frontend_full_s is None else frontend_full_s / max(gcn_s, 1e-9)
            row = DatasetProfile(
                dataset=dataset,
                nodes=nodes,
                edges=edges,
                frontend_sample_nodes=sample_n,
                frontend_sample_s=frontend_s,
                frontend_nodes_per_s=nodes_per_s,
                frontend_est_full_s=frontend_full_s,
                gcn_full_s=gcn_s,
                gcn_repeats=int(args.gcn_repeats),
                tf_share_est=tf_share,
                gcn_share_est=gcn_share,
                tf_over_gcn_est=tf_over_gcn,
                embedding_pool=str(pool_path),
            )
            rows.append(row)
            print(
                f"[StageProfile] {dataset}: nodes={nodes} edges={edges} "
                f"tf_est={frontend_full_s if frontend_full_s is not None else 'skip'} "
                f"gcn={gcn_s:.4f}s"
            )
    finally:
        if frontend is not None:
            frontend.close()

    write_outputs(rows, args.output_dir, args)
    print(f"\n[StageProfile] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
