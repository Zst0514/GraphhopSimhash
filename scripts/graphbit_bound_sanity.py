#!/usr/bin/env python3
"""Inspect Graph-Bit predictor-free stopping depths.

This mirrors the ONNXim GemmWS bound logic and is meant for quick sanity checks
before running the heavier simulator.  It does not use calibration nodes,
reference embeddings, or learned damage predictors.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundConfig:
    mode: str
    full_depth: int
    config_depth: int
    min_depth: int
    tolerance: float
    scale: float
    weight_abs_mean: float
    weight_abs_max: float
    partial_norm_scale: float
    partial_norm_floor: float
    safety_factor: float


def remaining_bound(depth: int, tile_k: int, cfg: BoundConfig) -> float:
    full_depth = max(1, cfg.full_depth)
    depth = min(depth, full_depth)
    if depth >= full_depth:
        return 0.0

    full_range = (2.0**full_depth) - 1.0
    omitted_range = (2.0 ** (full_depth - depth)) - 1.0
    normalized_omitted = omitted_range / max(1.0, full_range)
    k_scale = math.sqrt(max(1, tile_k)) / math.sqrt(128.0)

    if cfg.mode == "range":
        return normalized_omitted * k_scale * cfg.scale

    tile = float(max(1, tile_k))
    weight_mean = max(1.0e-9, cfg.weight_abs_mean)
    weight_max = max(weight_mean, cfg.weight_abs_max)
    remaining_weight = weight_max if cfg.mode == "tile_max" else weight_mean
    remaining = normalized_omitted * tile * remaining_weight * max(0.0, cfg.safety_factor)
    high_range = max(0.0, 1.0 - normalized_omitted)
    partial_norm = high_range * tile * weight_mean * max(0.0, cfg.partial_norm_scale)
    partial_norm = max(partial_norm, cfg.partial_norm_floor)
    return remaining / max(1.0e-12, partial_norm + remaining) * k_scale * cfg.scale


def select_depth(tile_k: int, cfg: BoundConfig) -> tuple[int, float]:
    min_depth = max(1, min(cfg.min_depth, cfg.config_depth, cfg.full_depth))
    for depth in range(min_depth, cfg.config_depth + 1):
        bound = remaining_bound(depth, tile_k, cfg)
        if bound <= cfg.tolerance:
            return depth, bound
    return cfg.config_depth, remaining_bound(cfg.config_depth, tile_k, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["range", "tile_mean", "tile_max"], default="tile_mean")
    parser.add_argument("--full-depth", type=int, default=8)
    parser.add_argument("--config-depth", type=int, default=8)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--weight-abs-mean", type=float, default=0.50)
    parser.add_argument("--weight-abs-max", type=float, default=1.00)
    parser.add_argument("--partial-norm-scale", type=float, default=1.0)
    parser.add_argument("--partial-norm-floor", type=float, default=1.0e-6)
    parser.add_argument("--safety-factor", type=float, default=1.0)
    parser.add_argument("--tile-k", type=int, nargs="+", default=[32, 64, 128, 256])
    args = parser.parse_args()

    cfg = BoundConfig(
        mode=args.mode,
        full_depth=args.full_depth,
        config_depth=args.config_depth,
        min_depth=args.min_depth,
        tolerance=args.tolerance,
        scale=args.scale,
        weight_abs_mean=args.weight_abs_mean,
        weight_abs_max=args.weight_abs_max,
        partial_norm_scale=args.partial_norm_scale,
        partial_norm_floor=args.partial_norm_floor,
        safety_factor=args.safety_factor,
    )
    print(
        "mode={mode} full={full_depth} config={config_depth} min={min_depth} "
        "tol={tolerance} scale={scale}".format(**cfg.__dict__)
    )
    print(f"{'tile_k':>8s} {'stop':>6s} " + " ".join(f"d{d:02d}" for d in range(cfg.min_depth, cfg.config_depth + 1)))
    for tile_k in args.tile_k:
        stop, _ = select_depth(tile_k, cfg)
        vals = [remaining_bound(depth, tile_k, cfg) for depth in range(cfg.min_depth, cfg.config_depth + 1)]
        print(f"{tile_k:8d} {stop:6d} " + " ".join(f"{v:0.4f}" for v in vals))


if __name__ == "__main__":
    main()
