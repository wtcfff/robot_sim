"""Small shared configuration and reproducibility helpers."""
import copy
import random
import numpy as np


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {k: class_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(v) for v in obj]
    if not hasattr(obj, '__dict__'):
        return obj
    return {k: class_to_dict(getattr(obj, k)) for k in dir(obj)
            if not k.startswith('_') and not callable(getattr(obj, k))}


def update_config(obj, values):
    for key, value in values.items():
        if isinstance(value, dict) and hasattr(obj, key) and hasattr(getattr(obj, key), '__dict__'):
            update_config(getattr(obj, key), value)
        else:
            setattr(obj, key, copy.deepcopy(value))
    return obj


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
