#!/usr/bin/env python3
"""Inspect readiness of extra graph datasets used by the paper.

The script is intentionally lightweight by default. It summarizes local files
for Wiki-CS, FB15K237, WN18RR, and ogbn-products without downloading large
datasets unless --download-products is explicitly provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for line in f if line.strip())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Dataset",
        "Task",
        "Nodes/Entities",
        "Edges/Triples",
        "Classes/Relations",
        "Local status",
        "Main use",
    ]
    lines = ["# Extra Graph Dataset Readiness", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(key, "")).replace("\n", "<br>") for key in headers
            )
            + " |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Wiki-CS, FB15K237, and WN18RR already have OFA-side dataset definitions.")
    lines.append("- ogbn-products is intentionally not downloaded by default because it is a 2.4M-node graph.")
    lines.append("- Link-prediction KGs need a different evaluation path from node-classification TAG datasets.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_wikics() -> dict[str, Any]:
    root = DATA_ROOT / "single_graph" / "wikics"
    metadata = root / "metadata.json"
    local_nodes = None
    local_labels = None
    if metadata.exists():
        with metadata.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        local_nodes = len(obj.get("nodes", []))
        local_labels = len(obj.get("labels", {}))
    return {
        "Dataset": "Wiki-CS",
        "Task": "Node classification",
        "Nodes/Entities": local_nodes or 11701,
        "Edges/Triples": 216123,
        "Classes/Relations": local_labels or 10,
        "Local status": "metadata+generator present" if metadata.exists() else "missing metadata",
        "Main use": "medium-size text-rich node graph",
        "path": str(root),
    }


def inspect_kg(name: str, official_entities: int, official_triples: int, official_relations: int) -> dict[str, Any]:
    root = DATA_ROOT / "KG" / name
    train = root / "train.txt"
    valid = root / "valid.txt"
    test = root / "test.txt"
    train_n = _count_lines(train)
    valid_n = _count_lines(valid)
    test_n = _count_lines(test)
    rels = set()
    ents = set()
    for path in (train, valid, test):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    ents.add(parts[0])
                    ents.add(parts[2])
                    rels.add(parts[1])
    local_triples = train_n + valid_n + test_n
    return {
        "Dataset": name,
        "Task": "Link prediction",
        "Nodes/Entities": len(ents) or official_entities,
        "Edges/Triples": local_triples or official_triples,
        "Classes/Relations": len(rels) or official_relations,
        "Local status": "train/valid/test present" if local_triples else "missing triples",
        "Main use": "relation-heavy KG link prediction",
        "path": str(root),
        "splits": {"train": train_n, "valid": valid_n, "test": test_n},
    }


def inspect_products(download: bool) -> dict[str, Any]:
    root = DATA_ROOT
    processed = root / "ogbn_products" / "processed"
    status = "not downloaded"
    local_nodes = None
    local_edges = None
    local_classes = 47
    if download:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset

            dataset = PygNodePropPredDataset(name="ogbn-products", root=str(root))
            data = dataset[0]
            local_nodes = int(data.num_nodes)
            local_edges = int(data.edge_index.size(1)) if hasattr(data, "edge_index") else None
            split = dataset.get_idx_split()
            status = "downloaded via OGB"
            local_classes = int(data.y.max().item()) + 1 if getattr(data, "y", None) is not None else 47
            split_sizes = {k: int(v.numel()) for k, v in split.items()}
        except Exception as exc:  # noqa: BLE001
            status = f"download failed: {exc}"
            split_sizes = {}
    else:
        split_sizes = {}
        if processed.exists():
            status = "processed directory exists"
    return {
        "Dataset": "ogbn-products",
        "Task": "Node classification",
        "Nodes/Entities": local_nodes or 2449029,
        "Edges/Triples": local_edges or 61859140,
        "Classes/Relations": local_classes,
        "Local status": status,
        "Main use": "million-scale scalability / batching stress test",
        "path": str(root / "ogbn_products"),
        "splits": split_sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-products", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "dataset_readiness",
    )
    args = parser.parse_args()

    rows = [
        inspect_wikics(),
        inspect_kg("FB15K237", 14541, 310116, 237),
        inspect_kg("WN18RR", 40943, 93003, 11),
        inspect_products(args.download_products),
    ]
    _write_json(args.output_dir / "extra_graph_datasets.json", rows)
    _write_markdown(args.output_dir / "extra_graph_datasets.md", rows)
    print(f"wrote {args.output_dir / 'extra_graph_datasets.json'}")
    print(f"wrote {args.output_dir / 'extra_graph_datasets.md'}")
    for row in rows:
        print(
            f"{row['Dataset']}: task={row['Task']} | nodes={row['Nodes/Entities']} "
            f"| edges/triples={row['Edges/Triples']} | status={row['Local status']}"
        )


if __name__ == "__main__":
    sys.exit(main())
