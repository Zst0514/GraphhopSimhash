import os
import os.path as osp
import random
from datetime import datetime

import numpy as np
import torch
import yaml

def load_yaml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_exp(params):
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")
    curtime = datetime.now()
    exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(exp_dir)
    with open(osp.join(exp_dir, "command"), "w") as f:
        yaml.dump(params, f)
    params["exp_dir"] = exp_dir


def merge_mod(params, mod_args):
    for i in range(0, len(mod_args), 2):
        if mod_args[i + 1].isdigit():
            val = int(mod_args[i + 1])
        elif mod_args[i + 1].replace(".", "", 1).isdigit():
            val = float(mod_args[i + 1])
        elif mod_args[i + 1].lower() == "true":
            val = True
        elif mod_args[i + 1].lower() == "false":
            val = False
        else:
            val = mod_args[i + 1]
        params[mod_args[i]] = val
    return params


def get_available_devices():
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device, gpu_ids
