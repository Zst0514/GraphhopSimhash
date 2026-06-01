#!/usr/bin/env python3
"""Profile LLaMA linear-layer W-tile strength for Graph-Bit bound sweeps.

The bound estimator needs a lightweight proxy for how much omitted activation
low bits can affect the current GEMM tile. For a Linear layer with PyTorch
weight shape [N, K], a W tile covers output rows N_tile and reduction columns
K_tile. For each output channel j in that tile:

    omitted_error_j <= A_low_bound(depth) * sum_k |W[j, k]|

This script reports normalized tile strength values:

    strength = tile_col_l1_stat / (layer_mean_abs(W) * actual_tile_k)

where tile_col_l1_stat is mean/p95/max over output channels in the tile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_MODEL_PATH = "models/llama-7b/modelscope/Llama-2-7b-ms"
LINEAR_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def resolve_model_path(path: str) -> Path:
    text = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(text):
        return Path(text)
    repo_root = Path(__file__).resolve().parents[2]
    if text.startswith("models/"):
        model_root = Path(os.environ.get("GRAPHHOP_MODEL_ROOT", str(repo_root / "models")))
        return model_root / text[len("models/") :]
    return repo_root / text


def load_weight_index(model_path: Path) -> dict[str, str]:
    safetensors_index = model_path / "model.safetensors.index.json"
    pytorch_index = model_path / "pytorch_model.bin.index.json"
    if safetensors_index.exists():
        with safetensors_index.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload.get("weight_map", {}))
    if pytorch_index.exists():
        with pytorch_index.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload.get("weight_map", {}))
    raise FileNotFoundError(f"No model index json found under {model_path}")


def wanted_tensor(name: str, include_all_linear: bool) -> bool:
    if include_all_linear:
        return name.endswith(".weight")
    return any(name.endswith(suffix) for suffix in LINEAR_SUFFIXES)


def iter_shard_tensors(model_path: Path, names_by_file: dict[str, list[str]]):
    for filename, names in sorted(names_by_file.items()):
        shard_path = model_path / filename
        if filename.endswith(".safetensors"):
            from safetensors.torch import load_file

            shard = load_file(str(shard_path), device="cpu")
        else:
            shard = torch.load(str(shard_path), map_location="cpu")
        for name in names:
            tensor = shard.get(name)
            if tensor is not None:
                yield name, tensor
        del shard


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("count", "mean", "p50", "p75", "p90", "p95", "p99", "max")}
    return {
        "count": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def module_kind(name: str) -> str:
    for suffix in LINEAR_SUFFIXES:
        if name.endswith(suffix):
            return suffix.replace(".weight", "")
    parts = name.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else name


def profile_weight(name: str, weight: torch.Tensor, tile_k: int, tile_n: int):
    if weight.ndim != 2:
        return [], {}
    # PyTorch Linear weight is [out_features, in_features] = [N, K].
    n_dim, k_dim = int(weight.shape[0]), int(weight.shape[1])
    abs_w = weight.detach().to(dtype=torch.float32).abs()
    layer_mean_abs = float(abs_w.mean().item())
    if layer_mean_abs <= 0:
        return [], {"name": name, "kind": module_kind(name), "N": n_dim, "K": k_dim, "tiles": 0}

    tile_rows: list[dict[str, float | int | str]] = []
    mean_values: list[float] = []
    p95_values: list[float] = []
    max_values: list[float] = []
    for n0 in range(0, n_dim, tile_n):
        n1 = min(n0 + tile_n, n_dim)
        for k0 in range(0, k_dim, tile_k):
            k1 = min(k0 + tile_k, k_dim)
            tile = abs_w[n0:n1, k0:k1]
            actual_k = max(1, k1 - k0)
            denom = layer_mean_abs * float(actual_k)
            col_l1 = tile.sum(dim=1)
            strength_mean = float(col_l1.mean().item() / denom)
            strength_p95 = float(torch.quantile(col_l1, 0.95).item() / denom)
            strength_max = float(col_l1.max().item() / denom)
            row = {
                "module": name,
                "kind": module_kind(name),
                "n0": n0,
                "n1": n1,
                "k0": k0,
                "k1": k1,
                "tile_k": actual_k,
                "tile_n": n1 - n0,
                "strength_mean": strength_mean,
                "strength_p95": strength_p95,
                "strength_max": strength_max,
            }
            tile_rows.append(row)
            mean_values.append(strength_mean)
            p95_values.append(strength_p95)
            max_values.append(strength_max)
    summary = {
        "name": name,
        "kind": module_kind(name),
        "N": n_dim,
        "K": k_dim,
        "tiles": len(tile_rows),
        "layer_mean_abs": layer_mean_abs,
        **{f"mean_{k}": v for k, v in summarize_values(mean_values).items()},
        **{f"p95_{k}": v for k, v in summarize_values(p95_values).items()},
        **{f"max_{k}": v for k, v in summarize_values(max_values).items()},
    }
    return tile_rows, summary


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=os.environ.get("GRAPHHOP_LLAMA2_7B_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--output_dir", default="output/graphbit_w_tile_strength/llama2_7b_k128_n128")
    parser.add_argument("--include_all_linear", action="store_true")
    parser.add_argument("--max_tensors", type=int, default=0, help="Debug only: stop after this many tensors.")
    parser.add_argument("--write_tiles", action="store_true", help="Write per-tile TSV. This can be large.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    weight_map = load_weight_index(model_path)
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        if wanted_tensor(name, args.include_all_linear):
            names_by_file[filename].append(name)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tile_rows: list[dict] = []
    module_rows: list[dict] = []
    by_kind: dict[str, list[float]] = defaultdict(list)
    global_mean_values: list[float] = []
    global_p95_values: list[float] = []
    global_max_values: list[float] = []

    for name, tensor in iter_shard_tensors(model_path, names_by_file):
        if tensor.ndim != 2:
            continue
        tile_rows, summary = profile_weight(name, tensor, args.tile_k, args.tile_n)
        if not summary:
            continue
        module_rows.append(summary)
        if args.write_tiles:
            all_tile_rows.extend(tile_rows)
        for row in tile_rows:
            global_mean_values.append(float(row["strength_mean"]))
            global_p95_values.append(float(row["strength_p95"]))
            global_max_values.append(float(row["strength_max"]))
            by_kind[str(row["kind"])].append(float(row["strength_p95"]))
        print(
            f"[WTile] {name} shape=({summary['N']},{summary['K']}) "
            f"tiles={summary['tiles']} p95_strength_p75={summary['p95_p75']:.3f} "
            f"p95_strength_p95={summary['p95_p95']:.3f}"
        )
        del tensor
        if args.max_tensors > 0 and len(module_rows) >= args.max_tensors:
            break

    global_rows = []
    for stat_name, values in (
        ("strength_mean", global_mean_values),
        ("strength_p95", global_p95_values),
        ("strength_max", global_max_values),
    ):
        row = {"scope": "all", "metric": stat_name, **summarize_values(values)}
        global_rows.append(row)
    for kind, values in sorted(by_kind.items()):
        global_rows.append({"scope": kind, "metric": "strength_p95", **summarize_values(values)})

    write_tsv(out_dir / "module_summary.tsv", module_rows)
    write_tsv(out_dir / "global_summary.tsv", global_rows)
    if args.write_tiles:
        write_tsv(out_dir / "tile_strength.tsv", all_tile_rows)

    manifest = {
        "model_path": str(model_path),
        "tile_k": args.tile_k,
        "tile_n": args.tile_n,
        "modules": len(module_rows),
        "tile_rows_written": bool(args.write_tiles),
        "recommended_scalar_sweep": {
            "optimistic": round(quantile(global_p95_values, 0.50), 4),
            "balanced": round(quantile(global_p95_values, 0.75), 4),
            "conservative": round(quantile(global_p95_values, 0.90), 4),
            "strict": round(quantile(global_p95_values, 0.95), 4),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[WTile] wrote {out_dir}")
    print("[WTile] recommended scalar sweep:", manifest["recommended_scalar_sweep"])


if __name__ == "__main__":
    main()
