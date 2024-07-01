import random
import numpy as np

import torch

# TODO: add the func referencing /share/src/utils/common.py
def setup_device(cfg):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.default.device_id}")
        
        if not cfg.default.deterministic:
            torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    print("CUDA is available:", torch.cuda.is_available())
    print(f"using device: {device}")
    return device


def fixed_r_seed(cfg):
    random.seed(cfg.default.seed)
    np.random.seed(cfg.default.seed)
    torch.manual_seed(cfg.default.seed)
    torch.cuda.manual_seed(cfg.default.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
