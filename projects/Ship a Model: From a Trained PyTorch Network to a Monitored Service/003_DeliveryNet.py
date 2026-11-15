import torch
from torch import nn


class DeliveryNet(nn.Module):
    """MLP that standardizes raw inputs with stored mean/std buffers and returns 2 logits."""

    def __init__(self, mean, std, hidden=32):
        super().__init__()
        self.register_buffer("mean", torch.tensor(np.array(mean), dtype=torch.float32).clone())
        self.register_buffer("std", torch.tensor(np.array(std), dtype=torch.float32).clone())
        n_features = len(mean)
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.layers(x)