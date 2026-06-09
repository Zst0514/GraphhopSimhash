#!/usr/bin/env python3
"""Prepare TAPE ogbn-products subset text for GraphhopSimhash experiments.

TAPE provides a text-attributed subset of ogbn-products with columns:
uid, nid, title, content.  The standard OGB products release only exposes
100-d node features, so this script converts the TAPE subset into an explicit
node-text table that can later be used by ST/LLaMA embedding generation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OFA_ROOT = REPO_ROOT.parent
DEFAULT_URL = (
    "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/"
    "dataset/ogbn_products_orig/ogbn-products_subset.csv"
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TAPE ogbn-products Text Subset",
        "",
        "This file records the local conversion from TAPE's ogbn-products text subset to a node-text table.",
        "",
        "## Files",
        "",
        f"- Source CSV: `{payload['source_csv']}`",
        f"- Output TSV: `{payload['output_tsv']}`",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Rows | {payload['rows']} |",
        f"| Unique node ids | {payload['unique_nids']} |",
        f"| Unique ASINs | {payload['unique_uids']} |",
        f"| Non-empty title | {payload['non_empty_title']} |",
        f"| Non-empty content | {payload['non_empty_content']} |",
        f"| Non-empty title or content | {payload['non_empty_text']} |",
        f"| Missing both title/content | {payload['missing_text']} |",
        f"| Valid OGB node id rows | {payload['valid_nid_rows']} |",
        f"| ASIN mapping checked rows | {payload['asin_checked_rows']} |",
        f"| ASIN match rate | {payload['asin_match_rate_pct']:.2f}% |",
        "",
        "## Text Format",
        "",
        "```text",
        "Product: <title>; Description: <content>",
        "```",
        "",
        "This is a subset of ogbn-products, not the full 2.45M-node graph.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_download(url: str, target: Path, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", url, "-o", str(target)], check=True)


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def load_asin_mapping(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    mapping = pd.read_csv(path, compression="gzip")
    if "node idx" in mapping.columns:
        mapping = mapping.rename(columns={"node idx": "nid"})
    return mapping[["nid", "asin"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=OFA_ROOT / "data" / "tape_ogbn_products_orig" / "ogbn-products_subset.csv",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download TAPE's ogbn-products_subset.csv before conversion.",
    )
    parser.add_argument("--download-url", default=DEFAULT_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=OFA_ROOT / "data" / "tape_ogbn_products_orig" / "ogbn-products_subset_text.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OFA_ROOT / "output" / "tape_products_text",
    )
    parser.add_argument(
        "--asin-map",
        type=Path,
        default=OFA_ROOT / "data" / "ogbn_products" / "mapping" / "nodeidx2asin.csv.gz",
    )
    parser.add_argument("--num-ogb-nodes", type=int, default=2_449_029)
    args = parser.parse_args()

    if args.download:
        maybe_download(args.download_url, args.source_csv, args.force_download)
    if not args.source_csv.exists():
        raise FileNotFoundError(
            f"Missing {args.source_csv}. Re-run with --download or provide --source-csv."
        )

    df = pd.read_csv(args.source_csv)
    required = {"uid", "nid", "title", "content"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Source CSV missing columns: {missing}")

    df = df[["uid", "nid", "title", "content"]].copy()
    df["title"] = df["title"].map(normalize_text)
    df["content"] = df["content"].map(normalize_text)
    df["raw_text"] = [
        f"Product: {title}; Description: {content}"
        for title, content in zip(df["title"], df["content"])
    ]
    df["nid"] = df["nid"].astype(int)
    df["uid"] = df["uid"].astype(str)

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    df[["nid", "uid", "raw_text", "title", "content"]].to_csv(
        args.output_tsv, sep="\t", index=False
    )

    asin_checked_rows = 0
    asin_match_rate = 0.0
    mapping = load_asin_mapping(args.asin_map)
    if mapping is not None:
        merged = df[["nid", "uid"]].merge(mapping, on="nid", how="left")
        checked = merged["asin"].notna()
        asin_checked_rows = int(checked.sum())
        if asin_checked_rows:
            asin_match_rate = float((merged.loc[checked, "uid"] == merged.loc[checked, "asin"]).mean())

    valid_nid = (df["nid"] >= 0) & (df["nid"] < args.num_ogb_nodes)
    non_empty_title = df["title"].astype(bool)
    non_empty_content = df["content"].astype(bool)
    non_empty_text = non_empty_title | non_empty_content
    payload = {
        "source_csv": str(args.source_csv),
        "output_tsv": str(args.output_tsv),
        "rows": int(len(df)),
        "unique_nids": int(df["nid"].nunique()),
        "unique_uids": int(df["uid"].nunique()),
        "non_empty_title": int(non_empty_title.sum()),
        "non_empty_content": int(non_empty_content.sum()),
        "non_empty_text": int(non_empty_text.sum()),
        "missing_text": int((~non_empty_text).sum()),
        "valid_nid_rows": int(valid_nid.sum()),
        "asin_checked_rows": asin_checked_rows,
        "asin_match_rate_pct": asin_match_rate * 100.0,
    }
    write_json(args.output_dir / "summary.json", payload)
    write_markdown(args.output_dir / "summary.md", payload)
    print(f"wrote {args.output_tsv}")
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'summary.md'}")
    print(
        "TAPE products text | "
        f"rows={payload['rows']} non_empty={payload['non_empty_text']} "
        f"asin_match={payload['asin_match_rate_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
