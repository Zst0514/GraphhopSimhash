#!/usr/bin/env python3
"""Profile DistilBERT layer-1 cheap-feature extraction time.

The script uses the same raw text loader and DistilBERT model path as
``GraphhopSimhash.data`` but does not overwrite the existing feature caches.
It reports unique dataset times and maps them to the six paper tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]


TASKS = {
    "CN": "cora",
    "CL": "cora",
    "PN": "pubmed",
    "PL": "pubmed",
    "AR": "arxiv",
    "WK": "wikics",
}


def resolve_st_model_path() -> str:
    env_path = os.environ.get("GRAPHHOP_ST_PATH")
    if env_path:
        return env_path
    local = ROOT / "models" / "multi-qa-distilbert-cos-v1"
    if local.exists():
        return str(local)
    return "sentence-transformers/multi-qa-distilbert-cos-v1"


def load_raw_texts(ds_key: str) -> list[str]:
    ds_key = ds_key.lower()
    if ds_key == "cora":
        data = torch.load(ROOT / "data" / "single_graph" / "Cora" / "cora.pt", map_location="cpu")
        return [str(x) for x in data.raw_texts]
    if ds_key == "pubmed":
        data = torch.load(ROOT / "data" / "single_graph" / "Pubmed" / "pubmed.pt", map_location="cpu")
        return [str(x) for x in data.raw_texts]
    if ds_key == "arxiv":
        import pandas as pd

        path = ROOT / "data" / "single_graph" / "arxiv"
        nodeidx2paperid = pd.read_csv(path / "nodeidx2paperid.csv.gz", index_col="node idx")
        nodeidx2paperid = nodeidx2paperid.sort_index()
        titleabs = pd.read_csv(
            path / "titleabs.tsv",
            sep="\t",
            names=["paper id", "title", "abstract"],
            index_col="paper id",
            on_bad_lines="skip",
            quoting=3,
        )
        titleabs = nodeidx2paperid.join(titleabs, on="paper id").fillna("")
        text = "feature node. paper title and abstract: " + titleabs["title"] + ". " + titleabs["abstract"]
        return text.astype(str).tolist()
    if ds_key == "wikics":
        import functools
        import json

        path = ROOT / "data" / "single_graph" / "wikics" / "metadata.json"
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
        texts = []
        for node in raw_data["nodes"]:
            content = functools.reduce(lambda x, y: x + " " + y, node["tokens"])
            texts.append(
                (
                    "feature node. wikipedia entry name: "
                    + node["title"]
                    + ". entry content: "
                    + content
                )
                .lower()
                .strip()
            )
        return texts
    raise ValueError(f"Dataset {ds_key} not supported for raw text loading.")


def distilbert_early_exit(
    model,
    encoded: dict[str, torch.Tensor],
    layer_idx: int,
) -> Optional[torch.Tensor]:
    model_type = getattr(model.config, "model_type", "")
    n_layers = int(getattr(model.config, "n_layers", getattr(model.config, "num_hidden_layers", 0)))
    if model_type != "distilbert" or layer_idx <= 0 or layer_idx > n_layers:
        return None

    hidden = model.embeddings(input_ids=encoded["input_ids"])
    attention_mask = encoded.get("attention_mask")
    for layer_id in range(layer_idx):
        hidden = model.transformer.layer[layer_id](
            hidden,
            attn_mask=attention_mask,
            head_mask=None,
            output_attentions=False,
        )[0]
    return hidden


def get_distilbert_embeddings(
    texts: list[str],
    device: str,
    layer_idx: int,
    batch_size: int,
    early_exit: bool = True,
) -> torch.Tensor:
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    model_name = resolve_st_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Inferencing Layer {layer_idx}"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = distilbert_early_exit(model, encoded, layer_idx) if early_exit else None
            if hidden is None:
                outputs = model(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx]
            all_embs.append(hidden[:, 0, :].detach().cpu())
    return torch.cat(all_embs, dim=0)


def fmt_s(x: float) -> str:
    return f"{x:.3f}"


def has_gpu_process() -> bool:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return any(line.strip() for line in out.splitlines())


def wait_for_gpu_idle(poll_s: int) -> None:
    while has_gpu_process():
        print(f"[wait] GPU has active compute process; sleeping {poll_s}s", flush=True)
        time.sleep(poll_s)


def profile_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    texts = load_raw_texts(dataset)
    load_text_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    embs = get_distilbert_embeddings(
        texts,
        args.device,
        layer_idx=1,
        batch_size=args.batch_size,
        early_exit=not args.disable_early_exit,
    )
    extract_s = time.perf_counter() - t0

    return {
        "dataset": dataset,
        "nodes": len(texts),
        "batch_size": args.batch_size,
        "device": str(args.device),
        "early_exit": not args.disable_early_exit,
        "load_text_s": load_text_s,
        "distilbert_l1_extract_s": extract_s,
        "embedding_shape": list(embs.shape),
        "embedding_bytes": int(embs.numel() * embs.element_size()),
    }


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_path = out_dir / "distilbert_l1_time_unique_datasets.tsv"
    task_path = out_dir / "distilbert_l1_time_six_tasks.tsv"
    json_path = out_dir / "distilbert_l1_time_raw.json"
    md_path = out_dir / "distilbert_l1_time_six_tasks.md"

    columns = [
        "dataset",
        "nodes",
        "batch_size",
        "device",
        "load_text_s",
        "distilbert_l1_extract_s",
        "embedding_bytes",
    ]
    with unique_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in columns) + "\n")

    by_dataset = {row["dataset"]: row for row in rows}
    with task_path.open("w", encoding="utf-8") as f:
        f.write("task\t" + "\t".join(columns) + "\n")
        for task, dataset in TASKS.items():
            row = by_dataset[dataset]
            f.write(task + "\t" + "\t".join(str(row[c]) for c in columns) + "\n")

    payload = {
        "config": {
            "datasets": args.datasets,
            "batch_size": args.batch_size,
            "device": str(args.device),
            "early_exit": not args.disable_early_exit,
            "output_dir": str(args.output_dir),
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "grap_hop_st_path": os.environ.get("GRAPHHOP_ST_PATH", ""),
        },
        "unique_datasets": rows,
        "six_tasks": [{**by_dataset[dataset], "task": task} for task, dataset in TASKS.items()],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# DistilBERT Layer-1 Cheap Feature Extraction Time",
        "",
        "Scope: one-time extraction of DistilBERT layer-1 cheap semantic proxy features. This excludes graph-context key construction, SimHash table construction, P/C/U risk construction, LLaMA encoding, and online inference.",
        "",
        f"- device: `{args.device}`",
        f"- batch size: `{args.batch_size}`",
        f"- early exit: `{not args.disable_early_exit}`",
        f"- measured on: `{platform.node()}`",
        "",
        "## Six Evaluation Tasks",
        "",
        "| Task | Dataset | Nodes | Text Load (s) | DistilBERT-L1 Extract (s) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for task, dataset in TASKS.items():
        row = by_dataset[dataset]
        lines.append(
            f"| {task} | {dataset} | {row['nodes']} | "
            f"{fmt_s(row['load_text_s'])} | {fmt_s(row['distilbert_l1_extract_s'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- CN and CL share the same Cora cheap-feature artifact.",
            "- PN and PL share the same PubMed cheap-feature artifact.",
            "- Existing `cache_data/*_distilbert_l1.pt` files are not overwritten by this profiling script.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed", "arxiv", "wikics"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--disable-early-exit",
        action="store_true",
        help="Run the full DistilBERT forward pass and then read hidden_states[1].",
    )
    parser.add_argument("--wait-gpu-idle", action="store_true")
    parser.add_argument("--poll-s", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing_time_six_tasks"))
    args = parser.parse_args()

    if args.wait_gpu_idle and str(args.device).startswith("cuda"):
        wait_for_gpu_idle(args.poll_s)

    rows = []
    for dataset in args.datasets:
        print(f"[profile] {dataset}", flush=True)
        rows.append(profile_dataset(dataset, args))
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
