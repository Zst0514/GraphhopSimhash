DATASET_CONFIGS = {
    "cora": {"name": "Cora", "task_names": ["cora_node"], "data_dir": "Cora", "batch_size": 128, "max_test": None},
    "pubmed": {"name": "Pubmed", "task_names": ["pubmed_node"], "data_dir": "Pubmed", "batch_size": 128, "max_test": None},
    "arxiv": {"name": "Arxiv", "task_names": ["arxiv"], "data_dir": "arxiv", "batch_size": 128, "max_test": 2000},
    "wikics": {"name": "Wiki-CS", "task_names": ["wikics"], "data_dir": "wikics", "batch_size": 128, "max_test": None},
    "tape_products": {"name": "TAPE-products", "task_names": ["tape_products"], "data_dir": "tape_products", "batch_size": 128, "max_test": None},
    "tape_arxiv23": {"name": "TAPE-Arxiv23", "task_names": ["tape_arxiv23"], "data_dir": "tape_arxiv23", "batch_size": 128, "max_test": None},
}
