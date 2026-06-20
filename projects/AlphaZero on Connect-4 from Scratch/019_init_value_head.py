import torch
import torch.nn as nn

def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.Flatten(),
        nn.Linear(6 * 7, 1),
        nn.Tanh(),
    )
