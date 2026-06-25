#!/usr/bin/env python3
"""Profile official AWQ search time without generating embedding pools."""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFA_DIR = os.path.dirname(REPO_DIR)
if OFA_DIR not in sys.path:
    sys.path.insert(0, OFA_DIR)

from GraphhopSimhash.data import load_raw_texts  # noqa: E402
from GraphhopSimhash.generate_real_quant_pools import (  # noqa: E402
    apply_official_awq_w4,
    canonical_model_name,
    load_model_and_tokenizer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cora", "pubmed", "arxiv", "wikics", "tape_products", "tape_arxiv23"],
    )
    parser.add_argument("--llm_name", default="llama2_7b")
    parser.add_argument("--cache_dir", default="cache_data/model")
    parser.add_argument("--awq_results_path", required=True)
    parser.add_argument("--awq_calib_samples", type=int, default=128)
    parser.add_argument("--awq_seqlen", type=int, default=512)
    parser.add_argument("--awq_q_group_size", type=int, default=128)
    parser.add_argument("--awq_no_zero_point", action="store_true")
    parser.add_argument("--awq_disable_auto_scale", action="store_true")
    parser.add_argument("--awq_disable_mse_clip", action="store_true")
    parser.add_argument("--awq_force_mse_clip", action="store_true")
    parser.add_argument(
        "--awq_force_cpu",
        action="store_true",
        help="Use the old CPU-loading AWQ path. Default profiles the GPU/auto-device AWQ path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    llm_name = canonical_model_name(args.llm_name)
    texts = load_raw_texts(args.dataset)
    model, tokenizer, _ = load_model_and_tokenizer(
        llm_name,
        "fp16",
        args.cache_dir,
        force_cpu=bool(args.awq_force_cpu),
    )
    awq_args = SimpleNamespace(
        awq_results_path=args.awq_results_path,
        awq_calib_samples=args.awq_calib_samples,
        awq_seqlen=args.awq_seqlen,
        awq_q_group_size=args.awq_q_group_size,
        awq_no_zero_point=args.awq_no_zero_point,
        awq_disable_auto_scale=args.awq_disable_auto_scale,
        awq_disable_mse_clip=args.awq_disable_mse_clip,
        awq_force_mse_clip=args.awq_force_mse_clip,
        awq_force_cpu=args.awq_force_cpu,
        awq_overwrite_results=args.overwrite,
    )

    start = time.perf_counter()
    apply_official_awq_w4(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        dataset=args.dataset,
        llm_name=llm_name,
        args=awq_args,
        activation_bit=8,
    )
    elapsed_s = time.perf_counter() - start

    result = {
        "dataset": args.dataset,
        "llm_name": llm_name,
        "awq_calib_samples": args.awq_calib_samples,
        "awq_seqlen": args.awq_seqlen,
        "awq_q_group_size": args.awq_q_group_size,
        "elapsed_s": elapsed_s,
        "awq_results_path": args.awq_results_path,
    }
    print("[AWQProfile] " + json.dumps(result, sort_keys=True))
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
