import torch

def to_float_tensor(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t