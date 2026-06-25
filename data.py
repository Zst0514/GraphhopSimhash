import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .config import DATASET_CONFIGS
from .paths import ensure_repo_paths, resolve_model_path
from .runtime import get_available_devices, load_yaml, merge_mod, set_random_seed, setup_exp

ensure_repo_paths()
from task_constructor import UnifiedTaskConstructor  # noqa: E402
from utils import SentenceEncoder  # noqa: E402


def normalize_node_masks(data):
    for name in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, name):
            mask = getattr(data, name)
            if isinstance(mask, torch.Tensor) and mask.dim() > 1:
                setattr(data, name, mask[:, 0].contiguous())
    if hasattr(data, "train_masks") and not hasattr(data, "train_mask"):
        data.train_mask = data.train_masks[0]
        data.val_mask = data.val_masks[0]
        data.test_mask = data.test_masks[0]
    return data


def ensure_arxiv_masks(data, ds_key, device):
    if hasattr(data, "train_mask") or ds_key.lower() != "arxiv":
        return False

    print("[Info] Generating masks for Arxiv...")
    node_year_path = os.path.join("data", "ogbn_arxiv", "raw", "node_year.csv.gz")
    if os.path.exists(node_year_path):
        try:
            import gzip
            import pandas as pd

            print(f"[Info] Found {node_year_path}. Generating Standard Time Split (<2018, 2018, >=2019)...")
            with gzip.open(node_year_path, "rt") as f:
                node_years = pd.read_csv(f, header=None).values.flatten()

            node_years = torch.tensor(node_years, device=device)
            data.train_mask = node_years <= 2017
            data.val_mask = node_years == 2018
            data.test_mask = node_years >= 2019
            print(
                f"[Success] Generated Time-Split Masks. "
                f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}"
            )
            return True
        except Exception as e:
            print(f"[Warning] Manual split failed: {e}. Falling back...")

    try:
        import builtins
        from ogb.nodeproppred import PygNodePropPredDataset

        original_input = builtins.input
        try:
            builtins.input = lambda *args: "n"
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="./data")
        finally:
            builtins.input = original_input

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        data.train_mask[train_idx] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        data.val_mask[valid_idx] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        data.test_mask[test_idx] = True
        return True
    except Exception as e:
        print(f"[Warning] OGB Split failed: {e}. Fallback to random splits...")
        n = data.num_nodes
        indices = torch.randperm(n, device=device)
        data.train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        data.train_mask[indices[: int(n * 0.6)]] = True
        data.val_mask = torch.zeros(n, dtype=torch.bool, device=device)
        data.val_mask[indices[int(n * 0.6) : int(n * 0.8)]] = True
        data.test_mask = torch.zeros(n, dtype=torch.bool, device=device)
        data.test_mask[indices[int(n * 0.8) :]] = True
        return True


def build_tape_products_data(cache_path):
    import gzip

    import pandas as pd
    from torch_geometric.data import Data

    text_path = os.path.join("data", "tape_ogbn_products_orig", "ogbn-products_subset_text.tsv")
    products_path = os.path.join("data", "ogbn_products", "processed", "geometric_data_processed.pt")
    split_dir = os.path.join("data", "ogbn_products", "split", "sales_ranking")
    if not os.path.exists(text_path):
        raise FileNotFoundError(
            f"{text_path} missing. Run GraphhopSimhash/scripts/prepare_tape_products_text.py first."
        )
    if not os.path.exists(products_path):
        raise FileNotFoundError(f"{products_path} missing. Download/process ogbn-products first.")

    text_df = pd.read_csv(text_path, sep="\t")
    nids = torch.tensor(text_df["nid"].astype("int64").values, dtype=torch.long)

    loaded = torch.load(products_path, map_location="cpu")
    full = loaded[0] if isinstance(loaded, tuple) else loaded
    num_full_nodes = int(full.num_nodes)
    local_of_global = torch.full((num_full_nodes,), -1, dtype=torch.long)
    local_of_global[nids] = torch.arange(nids.numel(), dtype=torch.long)

    src, dst = full.edge_index
    edge_mask = (local_of_global[src] >= 0) & (local_of_global[dst] >= 0)
    edge_index = torch.stack((local_of_global[src[edge_mask]], local_of_global[dst[edge_mask]]), dim=0)

    def read_split(name):
        path = os.path.join(split_dir, f"{name}.csv.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing")
        with gzip.open(path, "rt") as f:
            values = pd.read_csv(f, header=None).values.reshape(-1)
        return torch.tensor(values, dtype=torch.long)

    data = Data(
        x=full.x[nids].to(torch.float32),
        edge_index=edge_index.contiguous(),
        y=full.y[nids].view(-1).to(torch.long),
        num_nodes=int(nids.numel()),
    )
    data.global_nid = nids
    for split_name, attr_name in (("train", "train_mask"), ("valid", "val_mask"), ("test", "test_mask")):
        split_idx = read_split(split_name)
        mask = torch.isin(nids, split_idx)
        setattr(data, attr_name, mask)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    print(
        "[TAPEProducts] Built induced subgraph "
        f"nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"train={int(data.train_mask.sum())} val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}"
    )
    return data


def build_tape_arxiv23_data(cache_path):
    source_path = os.path.join(
        "data",
        "TAPE_repo",
        "dataset",
        "arxiv_2023",
        "graph.pt",
    )
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"{source_path} missing. Clone https://github.com/XiaoxinHe/TAPE under data/TAPE_repo first."
        )
    data = torch.load(source_path, map_location="cpu")
    data.num_nodes = int(data.y.numel())
    if not all(hasattr(data, name) for name in ("train_mask", "val_mask", "test_mask")):
        raise ValueError(f"{source_path} must contain TAPE train/val/test masks")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    print(
        "[TAPEArxiv23] Built graph "
        f"nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"train={int(data.train_mask.sum())} val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}"
    )
    return data


def load_data_pipeline(ds_key, params, device):
    batch_size = 1 if "llama2" in params.llm_name.lower() else params.batch_size
    encoder = SentenceEncoder(params.llm_name, batch_size=batch_size)

    st_data_path = os.path.join("cache_data", params.data_dir, params.llm_name, "processed", "geometric_data_processed.pt")
    print(f"\n[Debug] Checking existence of: {st_data_path}")
    print(f"[Debug] Current Working Directory: {os.getcwd()}")
    print(f"[Debug] Exists? {os.path.exists(st_data_path)}")

    if os.path.exists(st_data_path):
        print(f"Loading cached HQ features from {st_data_path}")
        loaded = torch.load(st_data_path)
        data = loaded[0] if isinstance(loaded, tuple) else loaded

        if ds_key == "tape_arxiv23" and int(getattr(data, "num_nodes", data.y.numel())) != 46198:
            print("[TAPEArxiv23] Cached graph is not the official 46,198-node TAPE split. Rebuilding.")
            data = build_tape_arxiv23_data(st_data_path)

        if ensure_arxiv_masks(data, ds_key, device):
            torch.save(data.cpu(), st_data_path)

        normalize_node_masks(data)
        return data.to(device), encoder

    if ds_key in ("tape_products", "products_text"):
        print(f"[TAPEProducts] Cache miss. Building {st_data_path}")
        data = build_tape_products_data(st_data_path)
        normalize_node_masks(data)
        return data.to(device), encoder

    if ds_key == "tape_arxiv23":
        print(f"[TAPEArxiv23] Cache miss. Building {st_data_path}")
        data = build_tape_arxiv23_data(st_data_path)
        normalize_node_masks(data)
        return data.to(device), encoder

    task_config = load_yaml(os.path.join("configs", "task_config.yaml"))
    data_config = load_yaml(os.path.join("configs", "data_config.yaml"))
    tasks = UnifiedTaskConstructor(
        params.task_names,
        params.load_texts,
        encoder,
        task_config,
        data_config,
        batch_size=params.batch_size,
        sample_size=-1,
    )
    tasks.construct_exp()

    if hasattr(tasks, "dataset"):
        print("Fallback to UnifiedTaskConstructor dataset")
        dataset_name = DATASET_CONFIGS[ds_key]["data_dir"]
        data = tasks.dataset[dataset_name].data if dataset_name in tasks.dataset else list(tasks.dataset.values())[0].data

    normalize_node_masks(data)

    ensure_arxiv_masks(data, ds_key, device)

    if not os.path.exists(st_data_path):
        print(f"[Cache] Saving generated Features to {st_data_path}")
        os.makedirs(os.path.dirname(st_data_path), exist_ok=True)
        torch.save(data.cpu(), st_data_path)
        data = data.to(device)

    return data.to(device), encoder


def load_raw_texts(ds_key):
    ds_key = ds_key.lower()
    if ds_key == "cora":
        path = os.path.join("data", "single_graph", "Cora", "cora.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing")
        data = torch.load(path)
        return data.raw_texts
    if ds_key == "pubmed":
        path = os.path.join("data", "single_graph", "Pubmed", "pubmed.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing")
        data = torch.load(path)
        return data.raw_texts
    if ds_key == "arxiv":
        import pandas as pd

        path = os.path.join("data", "single_graph", "arxiv")
        local_titleabs_path = os.path.join(path, "titleabs.tsv")
        nodeidx_path = os.path.join(path, "nodeidx2paperid.csv.gz")
        if not os.path.exists(local_titleabs_path):
            raise FileNotFoundError(f"{local_titleabs_path} missing. Please download OGB Arxiv data.")

        nodeidx2paperid = pd.read_csv(nodeidx_path, index_col="node idx")
        nodeidx2paperid = nodeidx2paperid.sort_index()

        titleabs = pd.read_csv(
            local_titleabs_path,
            sep="\t",
            names=["paper id", "title", "abstract"],
            index_col="paper id",
            on_bad_lines="skip",
            quoting=3,
        )
        titleabs = nodeidx2paperid.join(titleabs, on="paper id")
        titleabs = titleabs.fillna("")
        text = "feature node. paper title and abstract: " + titleabs["title"] + ". " + titleabs["abstract"]
        return text.values.tolist()
    if ds_key == "wikics":
        import functools
        import json

        path = os.path.join("data", "single_graph", "wikics", "metadata.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing")
        with open(path, "r", encoding="utf-8") as f:
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
    if ds_key in ("tape_products", "products_text"):
        import pandas as pd

        path = os.path.join("data", "tape_ogbn_products_orig", "ogbn-products_subset_text.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing. Run scripts/prepare_tape_products_text.py before generating pools."
            )
        df = pd.read_csv(path, sep="\t").fillna("")
        if "raw_text" not in df.columns:
            raise ValueError(f"{path} must contain a raw_text column")
        return df["raw_text"].astype(str).tolist()
    if ds_key == "tape_arxiv23":
        import pandas as pd

        path = os.path.join("data", "TAPE_repo", "dataset", "arxiv_2023_orig", "paper_info.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing. Clone https://github.com/XiaoxinHe/TAPE under data/TAPE_repo first."
            )
        df = pd.read_csv(path).fillna("")
        if "node_id" in df.columns:
            df = df.sort_values("node_id")
        return ("Title: " + df["title"].astype(str) + "\nAbstract: " + df["abstract"].astype(str)).tolist()
    raise ValueError(f"Dataset {ds_key} not supported for raw text loading.")


def _distilbert_early_exit(model, encoded, layer_idx):
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


def get_distilbert_embeddings(texts, device, layer_idx=-1, batch_size=128):
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    model_name = resolve_model_path("models/multi-qa-distilbert-cos-v1", "GRAPHHOP_ST_PATH")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError:
        print(f"[Warning] Local model at {model_name} not found. Trying huggingface hub...")
        model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()
    all_embs = []
    layer_desc = f"Layer {layer_idx}" if layer_idx != -1 else "Last Layer"
    print(f"[CheapFeature] Extracting DistilBERT {layer_desc} features for {len(texts)} nodes...")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Inferencing {layer_desc}"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            early_hidden = _distilbert_early_exit(model, encoded, layer_idx)
            if early_hidden is not None:
                batch_embs = early_hidden[:, 0, :]
            else:
                outputs = model(**encoded, output_hidden_states=(layer_idx != -1))
                if layer_idx == -1:
                    batch_embs = outputs.last_hidden_state[:, 0, :]
                else:
                    batch_embs = outputs.hidden_states[layer_idx][:, 0, :]
            all_embs.append(batch_embs.cpu())

    return torch.cat(all_embs, dim=0).to(device)


def load_bert_features(ds_key, data, device, layer_idx):
    del data
    suffix = layer_idx if layer_idx != -1 else "last"
    cache_path = os.path.join("cache_data", f"{ds_key}_distilbert_l{suffix}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)

    print(f"[CheapFeature] Cache miss for Layer-{layer_idx}. Generating...")
    texts = load_raw_texts(ds_key)
    embs = get_distilbert_embeddings(texts, device, layer_idx=layer_idx)
    os.makedirs("cache_data", exist_ok=True)
    torch.save(embs.cpu(), cache_path)
    return embs


def load_cheap_features(ds_key, data, device):
    if ds_key in ("tape_products", "products_text"):
        embs = data.x.detach().to(device=device, dtype=torch.float32)
        print("[CheapFeature] Using ogbn-products node features for TAPE-products subset.")
        return F.normalize(embs - embs.mean(dim=0, keepdim=True), p=2, dim=1)
    if ds_key == "tape_arxiv23":
        embs = data.x.detach().to(device=device, dtype=torch.float32)
        print("[CheapFeature] Using TAPE-Arxiv23 provided node features.")
        return F.normalize(embs - embs.mean(dim=0, keepdim=True), p=2, dim=1)

    cache_path = os.path.join("cache_data", f"{ds_key}_distilbert_l1.pt")
    if os.path.exists(cache_path):
        print(f"[CheapFeature] Loading cached DistilBERT Layer-1 features from {cache_path}")
        embs = torch.load(cache_path, map_location=device)
    else:
        embs = load_bert_features(ds_key, data, device, layer_idx=1)

    if isinstance(embs, torch.Tensor):
        print("[CheapFeature] Applying Anisotropy Correction (Centering & Normalization)...")
        mean_vec = embs.mean(dim=0, keepdim=True)
        embs = embs - mean_vec
        embs = F.normalize(embs, p=2, dim=1)
    return embs
def ensure_edge_features(data, device):
    if not hasattr(data, "edge_type") or data.edge_type is None:
        data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long, device=device)
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        data.edge_attr = torch.zeros((data.edge_index.size(1), 64), device=device)
    if data.edge_attr.size(1) != 64:
        edge_dim = data.edge_attr.size(1)
        if edge_dim > 64:
            data.edge_attr = data.edge_attr[:, :64]
        else:
            data.edge_attr = F.pad(data.edge_attr, (0, 64 - edge_dim))
def load_run_state(ds_key, args, seed):
    conf = DATASET_CONFIGS[ds_key]
    base_config = load_yaml(os.path.join("configs", "default_config.yaml"))
    override_dict = {
        "task_names": conf["task_names"],
        "llm_name": args.llm_name,
        "batch_size": conf["batch_size"],
        "eval_batch_size": conf["batch_size"],
        "num_workers": 0,
        "load_texts": False,
        "emb_dim": args.emb_dim,
        "num_layers": 3,
        "JK": "none",
        "dropout": 0.0,
        "seed": seed,
        "radius": args.radius,
        "data_dir": conf["data_dir"],
    }
    mod_params = merge_mod(base_config, [])
    for key, value in override_dict.items():
        mod_params[key] = value
    params = SimpleNamespace(**mod_params)
    setup_exp(vars(params))
    set_random_seed(seed)
    device = get_available_devices()[0]

    data, _encoder = load_data_pipeline(ds_key, params, device)
    verify_features = load_cheap_features(ds_key, data, device)

    if ds_key.lower() == "arxiv":
        from torch_geometric.transforms import ToUndirected

        data = ToUndirected()(data)

    ensure_edge_features(data, device)
    return conf, data, verify_features, device


def maybe_limit_test_mask(data, max_test):
    if max_test is None or data.test_mask.sum().item() <= max_test:
        return
    test_indices = data.test_mask.nonzero(as_tuple=False).view(-1)
    limited_test_mask = torch.zeros_like(data.test_mask)
    limited_test_mask[test_indices[:max_test]] = True
    data.test_mask = limited_test_mask
    print(f"[Debug] Limited test mask to {max_test} nodes.")
