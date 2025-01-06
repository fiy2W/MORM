from typing import List, Dict, Tuple

import numpy as np

import torch

__all__ = [
    'patch_weight', 'load_weights',
]


def patch_weight(size: List[int], axis: List[int], func: str='exp'):
    if func=='exp':
        func = lambda x: np.exp(-np.abs(x))
    elif func=='square':
        func = lambda x: 1-np.square(x)
    
    ws = []
    for i in range(len(size)):
        if i not in axis:
            w = np.array([1.0 for _ in range(size[i])])
        else:
            x = np.linspace(-1, 1, size[i])
            w = func(x)
        ws.append(w.reshape([size[i] if i==idx else 1 for idx in range(len(size))]))
    
    weights = np.array([1.0])
    for w in ws:
        weights = np.multiply(weights, w)
    return weights


def load_weights(model, weight_path, device='cpu'):
    load_dict = torch.load(weight_path, map_location=device)
    model_dict = model.state_dict()

    load_dict = {k: v for k, v in load_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    return model