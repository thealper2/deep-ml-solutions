import numpy as np
import torch
from torch import nn


def train_model(X, y, seed=0, epochs=8, lr=1e-2, batch_size=64, hidden=32):
    """Seeded training of a DeliveryNet; returns (model, per-epoch mean losses)."""
    set_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-6)

    model = DeliveryNet(mean, std, hidden=hidden)
    loader = make_loader(X, y, batch_size=batch_size, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            n = len(yb)
            total += float(loss.item()) * n
            count += n
        epoch_losses.append(total / count)
    return model, epoch_losses
