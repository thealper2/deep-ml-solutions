import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    return x.flatten().reshape(new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-1, -2)
