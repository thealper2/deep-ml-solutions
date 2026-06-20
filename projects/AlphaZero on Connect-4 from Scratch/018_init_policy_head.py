import torch
import torch.nn as nn

def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.Flatten(),
        nn.Linear(6 * 7, num_columns),
    )
