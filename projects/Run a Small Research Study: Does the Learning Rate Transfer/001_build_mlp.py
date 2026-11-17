# ---- provided: the task for the study (do not edit)
import math
import numpy as np
import torch
import torch.nn as nn

D_IN, D_OUT, BASE_WIDTH = 16, 4, 16


def load_task(n_train=2048, n_val=1024, seed=0):
    """Inputs and noisy outputs of a fixed random teacher network: (X_train, Y_train, X_val, Y_val)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_train + n_val, D_IN, generator=g)
    W1 = torch.randn(D_IN, 64, generator=g) / math.sqrt(D_IN)
    W2 = torch.randn(64, 64, generator=g) / math.sqrt(64)
    W3 = torch.randn(64, D_OUT, generator=g) / math.sqrt(64)
    Y = torch.tanh(torch.tanh(X @ W1 * 2) @ W2 * 2) @ W3
    Y = Y + 0.1 * torch.randn(Y.shape, generator=g)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def build_mlp(width, param, seed, base_width=BASE_WIDTH):
    """A 3-hidden-layer ReLU MLP initialized under the standard parametrization or muP."""
    if param not in ("sp", "mup"):
        raise ValueError(f"unknown parametrization {param!r}")

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(D_IN, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, D_OUT, bias=False),
    )

    linears = [m for m in model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        nn.init.normal_(linears[0].weight, mean=0, std=1 / D_IN ** 0.5)
        nn.init.normal_(linears[1].weight, mean=0, std=1 / width ** 0.5)
        nn.init.normal_(linears[2].weight, mean=0, std=1 / width ** 0.5)
        if param == "sp":
            readout_std = 1 / width ** 0.5
        else:
            readout_std = (1 / width ** 0.5) * (base_width / width) ** 0.5
        nn.init.normal_(linears[3].weight, mean=0, std=readout_std)

    return model