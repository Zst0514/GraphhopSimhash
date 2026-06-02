"""Utilities for graph-guided progressive BFP encoder routing.

This module keeps the software-facing contract small: the SimHash/residual
front-end decides which nodes still need the encoder, and this module describes
how those miss nodes are routed through a BFPA4 base path plus optional
mantissa-plane refinement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class ProgressiveBFPConfig:
    """Execution model for a W4 + progressive BFP activation encoder path."""

    reference_bit: int = 8
    base_bit: int = 4
    refine_bit: int = 6
    weight_bit: int = 4
    hbm_container_bit: int = 8
    block_size: int = 128
    exponent_bits: int = 8
    cost_scale: float = 0.50
    fixed_cost: float = 0.15

    def validate(self) -> None:
        if self.reference_bit <= 0:
            raise ValueError("reference_bit must be positive")
        if self.base_bit <= 0 or self.refine_bit <= 0:
            raise ValueError("base_bit/refine_bit must be positive")
        if self.base_bit > self.refine_bit:
            raise ValueError("base_bit must be <= refine_bit")
        if self.refine_bit > self.reference_bit:
            raise ValueError("refine_bit must be <= reference_bit")
        if self.hbm_container_bit < self.refine_bit:
            raise ValueError("hbm_container_bit must cover the largest execution bit")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.exponent_bits <= 0:
            raise ValueError("exponent_bits must be positive")
        if not (0.0 <= self.fixed_cost <= 1.0):
            raise ValueError("fixed_cost must be in [0, 1]")
        if self.cost_scale <= 0.0:
            raise ValueError("cost_scale must be positive")


@dataclass(frozen=True)
class EncoderInterfaceStats:
    """Compact summary for the miss-node progressive BFP encoder interface."""

    total_nodes: int
    encoder_nodes: int
    base_nodes: int
    refine_nodes: int
    reference_nodes: int
    average_execution_bit: float
    encoder_cost: float
    base_cost: float
    refine_cost: float
    reference_cost: float
    metadata_overhead_base: float
    metadata_overhead_refine: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def bfp_metadata_overhead(mantissa_bit: int, cfg: ProgressiveBFPConfig) -> float:
    """Return block-exponent metadata overhead relative to mantissa storage."""

    cfg.validate()
    mantissa_bit = int(mantissa_bit)
    if mantissa_bit <= 0:
        raise ValueError("mantissa_bit must be positive")
    return float(cfg.exponent_bits) / float(cfg.block_size * mantissa_bit)


def execution_cost(bit_depth: int, cfg: ProgressiveBFPConfig) -> float:
    """Cost proxy for executing one encoder node at a given activation bit-depth."""

    cfg.validate()
    bit_depth = int(bit_depth)
    ratio = 1.0 if bit_depth >= cfg.reference_bit else float(bit_depth) / float(cfg.reference_bit)
    return float(cfg.cost_scale) * (float(cfg.fixed_cost) + (1.0 - float(cfg.fixed_cost)) * ratio)


def progressive_extra_cost(cfg: ProgressiveBFPConfig) -> float:
    """Incremental cost of refining from base_bit to refine_bit."""

    return execution_cost(cfg.refine_bit, cfg) - execution_cost(cfg.base_bit, cfg)


def select_refinement_mask(
    priority: torch.Tensor,
    eligible_mask: torch.Tensor,
    refine_ratio: float,
) -> torch.Tensor:
    """Select top-priority eligible nodes for BFP refinement.

    `priority` can be TSER, degree/propagation risk, or another online score.
    The returned mask is always a subset of `eligible_mask`.
    """

    if priority.dim() != 1:
        raise ValueError("priority must be a 1-D tensor")
    eligible_mask = torch.as_tensor(eligible_mask, dtype=torch.bool, device=priority.device)
    if eligible_mask.numel() != priority.numel():
        raise ValueError("eligible_mask length must match priority length")

    refine_ratio = max(0.0, min(1.0, float(refine_ratio)))
    selected = torch.zeros_like(eligible_mask, dtype=torch.bool)
    eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).flatten()
    if int(eligible_idx.numel()) == 0 or refine_ratio <= 0.0:
        return selected

    count = int(round(float(eligible_idx.numel()) * refine_ratio))
    count = max(0, min(count, int(eligible_idx.numel())))
    if count <= 0:
        return selected

    order = eligible_idx[torch.argsort(priority[eligible_idx].to(dtype=torch.float32), descending=True)]
    selected[order[:count]] = True
    return selected


def build_progressive_actions(
    total_nodes: int,
    encoder_mask: torch.Tensor,
    refine_mask: torch.Tensor,
    cfg: ProgressiveBFPConfig,
    reference_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build node-level execution-depth actions.

    Non-encoder nodes keep reference_bit as a placeholder because their embedding
    comes from direct/residual reuse. Encoder nodes default to base_bit; selected
    encoder nodes use refine_bit; optional reference_mask can force P8.
    """

    cfg.validate()
    actions = torch.full((int(total_nodes),), int(cfg.reference_bit), dtype=torch.int64, device=encoder_mask.device)
    encoder_mask = torch.as_tensor(encoder_mask, dtype=torch.bool, device=actions.device)
    refine_mask = torch.as_tensor(refine_mask, dtype=torch.bool, device=actions.device)
    if encoder_mask.numel() != actions.numel() or refine_mask.numel() != actions.numel():
        raise ValueError("encoder/refine masks must match total_nodes")

    actions[encoder_mask] = int(cfg.base_bit)
    actions[encoder_mask & refine_mask] = int(cfg.refine_bit)
    if reference_mask is not None:
        reference_mask = torch.as_tensor(reference_mask, dtype=torch.bool, device=actions.device)
        if reference_mask.numel() != actions.numel():
            raise ValueError("reference_mask length must match total_nodes")
        actions[encoder_mask & reference_mask] = int(cfg.reference_bit)
    return actions


def summarize_progressive_interface(
    actions: torch.Tensor,
    encoder_mask: torch.Tensor,
    cfg: ProgressiveBFPConfig,
) -> EncoderInterfaceStats:
    """Summarize the front-end-to-encoder progressive BFP interface."""

    cfg.validate()
    actions = torch.as_tensor(actions, dtype=torch.int64)
    encoder_mask = torch.as_tensor(encoder_mask, dtype=torch.bool, device=actions.device)
    if actions.numel() != encoder_mask.numel():
        raise ValueError("actions and encoder_mask must have the same length")

    total_nodes = int(actions.numel())
    encoder_nodes = int(encoder_mask.sum().item())
    base_nodes = int((encoder_mask & (actions == int(cfg.base_bit))).sum().item())
    refine_nodes = int((encoder_mask & (actions == int(cfg.refine_bit))).sum().item())
    reference_nodes = int((encoder_mask & (actions == int(cfg.reference_bit))).sum().item())

    if encoder_nodes > 0:
        avg_bit = float(actions[encoder_mask].to(dtype=torch.float32).mean().item())
    else:
        avg_bit = 0.0

    base_cost = execution_cost(cfg.base_bit, cfg)
    refine_cost = execution_cost(cfg.refine_bit, cfg)
    reference_cost = execution_cost(cfg.reference_bit, cfg)
    encoder_cost = (
        base_nodes * base_cost
        + refine_nodes * refine_cost
        + reference_nodes * reference_cost
    ) / float(max(1, total_nodes))

    return EncoderInterfaceStats(
        total_nodes=total_nodes,
        encoder_nodes=encoder_nodes,
        base_nodes=base_nodes,
        refine_nodes=refine_nodes,
        reference_nodes=reference_nodes,
        average_execution_bit=avg_bit,
        encoder_cost=float(encoder_cost),
        base_cost=float(base_cost),
        refine_cost=float(refine_cost),
        reference_cost=float(reference_cost),
        metadata_overhead_base=bfp_metadata_overhead(cfg.base_bit, cfg),
        metadata_overhead_refine=bfp_metadata_overhead(cfg.refine_bit, cfg),
    )


def format_interface_stats(stats: EncoderInterfaceStats) -> str:
    """Human-readable one-line summary for logs or scripts."""

    total = max(1, int(stats.total_nodes))
    return (
        f"encoder={stats.encoder_nodes / total:.1%} "
        f"base={stats.base_nodes / total:.1%} "
        f"refine={stats.refine_nodes / total:.1%} "
        f"ref={stats.reference_nodes / total:.1%} "
        f"avg_bit={stats.average_execution_bit:.2f} "
        f"cost={stats.encoder_cost:.3f}"
    )


def config_from_mapping(values: Optional[Mapping[str, object]] = None) -> ProgressiveBFPConfig:
    """Build a config from a plain mapping, keeping defaults for missing keys."""

    values = dict(values or {})
    fields = ProgressiveBFPConfig.__dataclass_fields__
    kwargs = {key: values[key] for key in fields if key in values}
    cfg = ProgressiveBFPConfig(**kwargs)
    cfg.validate()
    return cfg
