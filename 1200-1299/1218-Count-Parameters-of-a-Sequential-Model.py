import torch
import torch.nn as nn

def count_params():
    """Build Sequential(Linear(4,8), ReLU, Linear(8,2)) and return trainable param count.

    Returns:
        int: total number of trainable parameters
    """
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total
