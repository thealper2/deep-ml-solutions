import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    if y.dim() == 2 and y.shape[1] == 1:
        y = y.squeeze(1)

    D = X.shape[1]
    w = (torch.randn(D) * 0.01).requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    for _ in range(steps):
        pred = X @ w + b
        loss = torch.mean((pred - y) ** 2)

        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

    return w.detach(), b.detach().squeeze()
