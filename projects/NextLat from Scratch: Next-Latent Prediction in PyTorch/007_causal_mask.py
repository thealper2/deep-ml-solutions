import torch

def causal_mask(T: int):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))
