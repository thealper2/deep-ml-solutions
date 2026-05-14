import torch

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    t = x + b
    return t
