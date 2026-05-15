import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training:
        return x

    keep_prob = 1 - p
    mask = (torch.rand(*x.shape) < keep_prob)
    mask = mask / keep_prob
    out = x * mask
    return out