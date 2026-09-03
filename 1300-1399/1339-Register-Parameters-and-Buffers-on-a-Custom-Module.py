import torch
import torch.nn as nn


class ScaledShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.register_buffer('calls', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            self.calls += 1

        return x * self.scale + self.shift