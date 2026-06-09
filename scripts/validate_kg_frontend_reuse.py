#!/usr/bin/env python3
"""Lightweight KG frontend-reuse validation for FB15K237 / WN18RR.

This script validates whether entity-text SimHash reuse is plausible on KG
link-prediction datasets. It is intentionally separate from the node
classification runner because KG workloads have triplet splits rather than
node labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent
if str(OFA_ROOT) not in sys.path:
    sys.path.insert(0, str(OFA_ROOT))

from utils import SentenceEncoder  # noqa: E402


@dataclass
class KGData:
    name: str
    texts: list[str]
    entity2id: dict[str, int]
    relation2id: dict[str, int]
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor


@dataclass
class MatchResult:
    anchor: torch.Tensor
    support: torch.Tensor
    anchors: int


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def _load_entity_texts(name: str) -> tuple[list[str], dict[str, int]]:
    kg_root = OFA_ROOT / "data" / "KG" / name
    if name == "WN18RR":
        entities: list[str] = []
        texts: list[str] = []
        with (kg_root / "entity2text.txt").open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    entities.append(parts[0])
                    texts.append(f"feature node. entity and entity description: {parts[1]}")
        return texts, {entity: idx for idx, entity in enumerate(entities)}

    if name == "FB15K237":
        with (kg_root / "entity2wikidata.json").open("r", encoding="utf-8") as f:
            obj = json.load(f)
        entities = []
        texts = []
        for entity, meta in obj.items():
            label = meta.get("label") or entity
            alternatives = ", ".join(meta.get("alternatives") or [])
            description = meta.get("description") or "None"
            entities.append(entity)
            texts.append(
                "feature node. entity and entity description: "
                f"entity names: {label}, entity alternatives: {alternatives}. "
                f"entity descriptions: {description}"
            )
        return texts, {entity: idx for idx, entity in enumerate(entities)}

    raise ValueError(f"unsupported KG dataset: {name}")


def _convert_triples(
    triples: Iterable[tuple[str, str, str]],
    texts: list[str],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> torch.Tensor:
    converted: list[list[int]] = []
    for head, rel, tail in triples:
        if head not in entity2id:
            entity2id[head] = len(texts)
            texts.append("feature node. entity and entity description: Unknown")
        if tail not in entity2id:
            entity2id[tail] = len(texts)
            texts.append("feature node. entity and entity description: Unknown")
        if rel not in relation2id:
            relation2id[rel] = len(relation2id)
        converted.append([entity2id[head], relation2id[rel], entity2id[tail]])
    return torch.tensor(converted, dtype=torch.long)


def load_kg(name: str) -> KGData:
    kg_root = OFA_ROOT / "data" / "KG" / name
    texts, entity2id = _load_entity_texts(name)
    relation2id: dict[str, int] = {}
    train = _convert_triples(_read_triples(kg_root / "train.txt"), texts, entity2id, relation2id)
    valid = _convert_triples(_read_triples(kg_root / "valid.txt"), texts, entity2id, relation2id)
    test = _convert_triples(_read_triples(kg_root / "test.txt"), texts, entity2id, relation2id)
    return KGData(name=name, texts=texts, entity2id=entity2id, relation2id=relation2id, train=train, valid=valid, test=test)


def encode_or_load(
    kg: KGData,
    encoder_name: str,
    batch_size: int,
    overwrite: bool,
) -> torch.Tensor:
    cache = OFA_ROOT / "cache_data" / "kg_frontend_reuse" / f"{kg.name.lower()}_{encoder_name}_entity_emb.pt"
    if cache.exists() and not overwrite:
        print(f"[Cache] loading {cache}")
        return torch.load(cache, map_location="cpu").float()

    print(f"[Encode] {kg.name} entities={len(kg.texts)} encoder={encoder_name}")
    encoder = SentenceEncoder(encoder_name, batch_size=batch_size)
    emb = encoder.encode(kg.texts).float().cpu()
    encoder.flush_model()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, cache)
    print(f"[Cache] saved {cache}")
    return emb


def build_codes(emb: torch.Tensor, heads: int, bits: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    norm = F.normalize(emb.float(), dim=1)
    projections = torch.randn(heads, norm.size(1), bits, generator=g)
    codes = torch.einsum("nd,hdb->hnb", norm, projections) >= 0
    return codes


def match_anchor_cache(
    emb: torch.Tensor,
    heads: int,
    bits: int,
    radius: int,
    max_anchors: int,
    chunk_size: int,
    seed: int,
) -> MatchResult:
    n = emb.size(0)
    anchor_count = min(max_anchors, n)
    query_ids = torch.arange(anchor_count, n, dtype=torch.long)
    codes = build_codes(emb, heads=heads, bits=bits, seed=seed)
    anchor_codes = codes[:, :anchor_count, :]
    best_anchor_all = torch.zeros(n, dtype=torch.long)
    best_support_all = torch.zeros(n, dtype=torch.int16)

    for start in range(0, query_ids.numel(), chunk_size):
        cur_ids = query_ids[start : start + chunk_size]
        q_codes = codes[:, cur_ids, :]
        support = torch.zeros(cur_ids.numel(), anchor_count, dtype=torch.int16)
        dist_sum = torch.zeros(cur_ids.numel(), anchor_count, dtype=torch.int16)
        for head in range(heads):
            dist = (q_codes[head, :, None, :] ^ anchor_codes[head, None, :, :]).sum(dim=-1).to(torch.int16)
            support += (dist <= radius).to(torch.int16)
            dist_sum += dist
        score = support.to(torch.int32) * (bits * heads + 1) - dist_sum.to(torch.int32)
        best_score, best_anchor = score.max(dim=1)
        best_support = support.gather(1, best_anchor.view(-1, 1)).view(-1)
        best_anchor_all[cur_ids] = best_anchor
        best_support_all[cur_ids] = best_support

    return MatchResult(anchor=best_anchor_all, support=best_support_all, anchors=anchor_count)


def reuse_stats(emb: torch.Tensor, recon: torch.Tensor, reused: torch.Tensor) -> dict[str, float]:
    errors = 1.0 - F.cosine_similarity(F.normalize(recon, dim=1), F.normalize(emb, dim=1), dim=1)
    reuse_rate = reused.float().mean().item()
    hit_err = errors[reused].mean().item() if reused.any() else 0.0
    avg_err = errors.mean().item()
    return {
        "entities": float(emb.size(0)),
        "reuse": reuse_rate,
        "reused": float(reused.sum().item()),
        "avg_err": avg_err,
        "hit_err": hit_err,
    }


def direct_reconstruction(emb: torch.Tensor, match: MatchResult, threshold: int) -> tuple[torch.Tensor, torch.Tensor]:
    reused = match.support >= threshold
    recon = emb.clone()
    if reused.any():
        recon[reused] = emb[match.anchor[reused]]
    return recon, reused


def fit_bucket_delta(
    emb: torch.Tensor,
    match: MatchResult,
    soft_mask: torch.Tensor,
    max_pairs: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    ids = soft_mask.nonzero(as_tuple=False).view(-1)
    if ids.numel() > max_pairs:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        ids = ids[torch.randperm(ids.numel(), generator=g)[:max_pairs]]
    deltas: dict[int, list[torch.Tensor]] = {}
    for support in sorted(match.support[ids].unique().tolist()):
        support_i = int(support)
        cur = ids[match.support[ids] == support_i]
        if cur.numel() == 0:
            continue
        delta = emb[cur] - emb[match.anchor[cur]]
        deltas[support_i] = [delta.mean(dim=0)]
    return {support: vals[0] for support, vals in deltas.items()}


def residual_reconstruction(
    emb: torch.Tensor,
    match: MatchResult,
    hard_threshold: int,
    soft_threshold: int,
    bucket_delta: dict[int, torch.Tensor],
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hard = match.support >= hard_threshold
    soft = (match.support >= soft_threshold) & (match.support < hard_threshold)
    reused = hard | soft
    recon = emb.clone()
    if hard.any():
        recon[hard] = emb[match.anchor[hard]]
    for support, delta in bucket_delta.items():
        cur = soft & (match.support == support)
        if cur.any():
            recon[cur] = emb[match.anchor[cur]] + alpha * delta
    return F.normalize(recon, dim=1), reused


def relation_prototypes(emb: torch.Tensor, train: torch.Tensor, num_relations: int) -> torch.Tensor:
    rel = torch.zeros(num_relations, emb.size(1), dtype=torch.float32)
    counts = torch.zeros(num_relations, dtype=torch.float32)
    for start in range(0, train.size(0), 65536):
        batch = train[start : start + 65536]
        delta = emb[batch[:, 2]] - emb[batch[:, 0]]
        rel.index_add_(0, batch[:, 1], delta)
        counts.index_add_(0, batch[:, 1], torch.ones(batch.size(0)))
    rel = rel / counts.clamp_min(1.0).view(-1, 1)
    return rel


def _transe_score(emb: torch.Tensor, rel: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    pred = emb[triples[:, 0]] + rel[triples[:, 1]]
    target = emb[triples[:, 2]]
    return -((pred - target) ** 2).sum(dim=1)


def sampled_link_auc(
    emb: torch.Tensor,
    rel: torch.Tensor,
    triples: torch.Tensor,
    num_entities: int,
    max_triples: int,
    negatives: int,
    seed: int,
    chunk_size: int = 512,
) -> float:
    if triples.size(0) > max_triples:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randperm(triples.size(0), generator=g)[:max_triples]
        triples = triples[idx]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 13)
    total = 0.0
    count = 0
    for start in range(0, triples.size(0), chunk_size):
        batch = triples[start : start + chunk_size]
        pos = _transe_score(emb, rel, batch)
        neg_tails = torch.randint(0, num_entities, (batch.size(0), negatives), generator=g)
        neg = batch[:, None, :].repeat(1, negatives, 1).reshape(-1, 3)
        neg[:, 2] = neg_tails.reshape(-1)
        neg_scores = _transe_score(emb, rel, neg).view(batch.size(0), negatives)
        total += (pos[:, None] > neg_scores).float().sum().item()
        count += batch.size(0) * negatives
    return total / max(count, 1)


def write_outputs(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "kg_frontend_reuse_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    headers = [
        "Dataset",
        "Config",
        "Entities",
        "Relations",
        "Triples",
        "Anchors",
        "Reuse",
        "AvgErr",
        "HitErr",
        "BaseAUC",
        "ReuseAUC",
        "AUCDrop",
        "Alpha",
    ]
    lines = ["# KG Frontend Reuse Validation", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(h, "")) for h in headers)
            + " |"
        )
    lines.append("")
    lines.append("AUC is a sampled TransE-style proxy using relation prototypes from train triples; it is not official KG MRR.")
    (output_dir / "kg_frontend_reuse_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tsv_lines = ["\t".join(headers)]
    for row in rows:
        tsv_lines.append("\t".join(str(row.get(h, "")) for h in headers))
    (output_dir / "kg_frontend_reuse_summary.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def make_row(
    dataset: str,
    kg: KGData,
    emb: torch.Tensor,
    rel: torch.Tensor,
    match: MatchResult,
    config: str,
    recon: torch.Tensor,
    reused: torch.Tensor,
    base_auc: float,
    args: argparse.Namespace,
    alpha: float | None = None,
) -> dict[str, object]:
    stats = reuse_stats(emb, recon, reused)
    reuse_auc = sampled_link_auc(
        F.normalize(recon.float(), dim=1),
        rel,
        kg.test,
        num_entities=emb.size(0),
        max_triples=args.max_eval_triples,
        negatives=args.negatives,
        seed=args.seed,
    )
    return {
        "Dataset": dataset,
        "Config": config,
        "Entities": emb.size(0),
        "Relations": len(kg.relation2id),
        "Triples": int(kg.train.size(0) + kg.valid.size(0) + kg.test.size(0)),
        "Anchors": int(match.anchors),
        "Reuse": format_percent(stats["reuse"]),
        "AvgErr": f"{stats['avg_err']:.5f}",
        "HitErr": f"{stats['hit_err']:.5f}",
        "BaseAUC": f"{base_auc:.4f}",
        "ReuseAUC": f"{reuse_auc:.4f}",
        "AUCDrop": f"{100.0 * (base_auc - reuse_auc):.2f}%",
        "Alpha": "-" if alpha is None else f"{alpha:.3f}",
    }


def tune_residual_alpha(
    emb: torch.Tensor,
    kg: KGData,
    rel: torch.Tensor,
    match: MatchResult,
    hard_threshold: int,
    soft_threshold: int,
    bucket_delta: dict[int, torch.Tensor],
    alpha_grid: list[float],
    args: argparse.Namespace,
) -> float:
    best_alpha = 0.0
    best_auc = -math.inf
    for alpha in alpha_grid:
        recon, _reused = residual_reconstruction(
            emb,
            match,
            hard_threshold=hard_threshold,
            soft_threshold=soft_threshold,
            bucket_delta=bucket_delta,
            alpha=alpha,
        )
        auc = sampled_link_auc(
            recon,
            rel,
            kg.valid,
            num_entities=emb.size(0),
            max_triples=args.max_eval_triples,
            negatives=args.negatives,
            seed=args.seed + 101,
        )
        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha
    return best_alpha


def run_dataset(args: argparse.Namespace, dataset: str) -> list[dict[str, object]]:
    kg = load_kg(dataset)
    emb = encode_or_load(kg, args.encoder, args.batch_size, args.overwrite_embeddings)
    emb = F.normalize(emb.float(), dim=1)
    match = match_anchor_cache(
        emb,
        heads=args.hash_heads,
        bits=args.head_bits,
        radius=args.radius,
        max_anchors=args.max_anchors,
        chunk_size=args.match_chunk_size,
        seed=args.seed,
    )
    rel = relation_prototypes(emb, kg.train, len(kg.relation2id))
    base_auc = sampled_link_auc(
        emb,
        rel,
        kg.test,
        num_entities=emb.size(0),
        max_triples=args.max_eval_triples,
        negatives=args.negatives,
        seed=args.seed,
    )
    rows: list[dict[str, object]] = []

    direct_recon, direct_reused = direct_reconstruction(emb, match, args.hard_support_threshold)
    rows.append(
        make_row(
            dataset,
            kg,
            emb,
            rel,
            match,
            "DirectReuse",
            direct_recon,
            direct_reused,
            base_auc,
            args,
        )
    )

    soft_recon, soft_reused = direct_reconstruction(emb, match, args.soft_support_threshold)
    rows.append(
        make_row(
            dataset,
            kg,
            emb,
            rel,
            match,
            "SoftDirectReuse",
            soft_recon,
            soft_reused,
            base_auc,
            args,
        )
    )

    soft_mask = (match.support >= args.soft_support_threshold) & (match.support < args.hard_support_threshold)
    bucket_delta = fit_bucket_delta(
        emb,
        match,
        soft_mask,
        max_pairs=args.residual_max_train_pairs,
        seed=args.seed,
    )
    alpha = tune_residual_alpha(
        emb,
        kg,
        rel,
        match,
        hard_threshold=args.hard_support_threshold,
        soft_threshold=args.soft_support_threshold,
        bucket_delta=bucket_delta,
        alpha_grid=args.residual_alpha_grid,
        args=args,
    )
    residual_recon, residual_reused = residual_reconstruction(
        emb,
        match,
        hard_threshold=args.hard_support_threshold,
        soft_threshold=args.soft_support_threshold,
        bucket_delta=bucket_delta,
        alpha=alpha,
    )
    rows.append(
        make_row(
            dataset,
            kg,
            emb,
            rel,
            match,
            "ResidualReuse",
            residual_recon,
            residual_reused,
            base_auc,
            args,
            alpha=alpha,
        )
    )

    for row in rows:
        print(
            f"[KGReuse] {dataset}/{row['Config']} reuse={row['Reuse']} "
            f"avg_err={row['AvgErr']} hit_err={row['HitErr']} "
            f"auc={row['BaseAUC']}->{row['ReuseAUC']} alpha={row['Alpha']}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["FB15K237", "WN18RR"], choices=["FB15K237", "WN18RR"])
    parser.add_argument("--encoder", default="ST")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--hash-heads", type=int, default=8)
    parser.add_argument("--head-bits", type=int, default=16)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--hard-support-threshold", type=int, default=5)
    parser.add_argument("--soft-support-threshold", type=int, default=3)
    parser.add_argument("--max-anchors", type=int, default=2048)
    parser.add_argument("--match-chunk-size", type=int, default=512)
    parser.add_argument("--max-eval-triples", type=int, default=5000)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--residual-max-train-pairs", type=int, default=4096)
    parser.add_argument(
        "--residual-alpha-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OFA_ROOT / "output" / "kg_frontend_reuse")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        rows.extend(run_dataset(args, dataset))
    write_outputs(args.output_dir, rows)
    print(f"[KGReuse] wrote {args.output_dir / 'kg_frontend_reuse_summary.md'}")


if __name__ == "__main__":
    main()
