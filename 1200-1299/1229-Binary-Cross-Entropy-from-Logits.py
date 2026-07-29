import torch

def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    x = logits
    y = targets
    loss = torch.max(x, torch.tensor(0.0)) - x * y + torch.log(1 + torch.exp(-torch.abs(x)))
    return round(loss.mean().item(), 4)
