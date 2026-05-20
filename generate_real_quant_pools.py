import argparse
import gc
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import load_raw_texts
from .paths import ensure_repo_paths

ensure_repo_paths()


MODEL_SPECS = {
    "llama2_7b": {
        "path": "/home/zhangshangtong/Transformer/OFA/models/llama-7b/modelscope/Llama-2-7b-ms",
        "model_class": "llama",
    },
    "llama2_13b": {
        "path": "meta-llama/Llama-2-13b-hf",
        "model_class": "llama",
    },
    "ST": {
        "path": "/home/zhangshangtong/Transformer/OFA/models/multi-qa-distilbert-cos-v1",
        "model_class": "auto",
    },
    "BERT": {
        "path": "/home/zhangshangtong/Transformer/OFA/models/bert-base-uncased",
        "model_class": "auto",
    },
    "e5": {
        "path": "intfloat/e5-large-v2",
        "model_class": "auto",
    },
}


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return F.normalize(summed / denom, p=2, dim=1)


def build_quant_config(config_name):
    if config_name == "fp16":
        return None, "FP16"

    from transformers import BitsAndBytesConfig

    if config_name == "int8":
        return BitsAndBytesConfig(load_in_8bit=True), "INT8"
    if config_name == "int4":
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "INT4",
        )
    raise ValueError(f"Unknown config: {config_name}")


def load_model_and_tokenizer(llm_name, config_name, cache_dir):
    from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

    if llm_name not in MODEL_SPECS:
        raise ValueError(f"Unknown llm_name={llm_name}. Available: {sorted(MODEL_SPECS)}")

    spec = MODEL_SPECS[llm_name]
    model_path = spec["path"]
    quant_config, tag = build_quant_config(config_name)

    if spec["model_class"] == "llama":
        model_cls = LlamaForCausalLM
        tokenizer_cls = LlamaTokenizer
    else:
        model_cls = AutoModel
        tokenizer_cls = AutoTokenizer

    kwargs = {
        "cache_dir": cache_dir,
        "device_map": "auto",
        "output_hidden_states": True,
    }
    if quant_config is None:
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["quantization_config"] = quant_config

    model = model_cls.from_pretrained(model_path, **kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    tokenizer = tokenizer_cls.from_pretrained(model_path, cache_dir=cache_dir, add_eos_token=False)
    if llm_name.startswith("llama2"):
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return model, tokenizer, tag


def encode_texts(model, tokenizer, texts, batch_size, max_length, device):
    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1].to(torch.float32)
            embs = mean_pool(hidden, tokens["attention_mask"]).cpu()
            all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def generate_pool(dataset, llm_name, config_name, args):
    quant_config, tag = build_quant_config(config_name)
    del quant_config

    out_path = args.output_path
    if out_path is None:
        out_path = os.path.join("cache_data", f"{dataset}_{llm_name}_oracle_{tag}.pt")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[Skip] {out_path} already exists. Use --overwrite to regenerate.")
        return

    print(f"\n{'=' * 72}")
    print(f"Generating {tag} pool | dataset={dataset} | llm={llm_name}")
    print(f"{'=' * 72}")
    texts = load_raw_texts(dataset)
    batch_size = args.batch_size
    if llm_name.startswith("llama2") and batch_size > 4:
        print("[Info] Reducing LLaMA batch_size to 4 to avoid OOM.")
        batch_size = 4

    model, tokenizer, _tag = load_model_and_tokenizer(llm_name, config_name, args.cache_dir)
    device = next(model.parameters()).device
    embs = encode_texts(model, tokenizer, texts, batch_size, args.max_length, device)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(embs, out_path)
    print(f"[Saved] {out_path} | shape={tuple(embs.shape)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Generate FP16/INT8/INT4 node embedding pools.")
    parser.add_argument("--datasets", nargs="+", default=["cora"], choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--llm_name", type=str, default="llama2_7b", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--configs", nargs="+", default=["fp16", "int8", "int4"], choices=["fp16", "int8", "int4"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--cache_dir", type=str, default="cache_data/model")
    parser.add_argument("--output_path", type=str, default=None, help="Only valid with one dataset and one config.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_path is not None and (len(args.datasets) != 1 or len(args.configs) != 1):
        parser.error("--output_path requires exactly one dataset and one config")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.max_length <= 0:
        parser.error("--max_length must be positive")

    for dataset in args.datasets:
        for config_name in args.configs:
            generate_pool(dataset, args.llm_name, config_name, args)


if __name__ == "__main__":
    main()
