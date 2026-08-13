import numpy as np

def batch_size_scaled_sgd_loss(X: np.ndarray, y: np.ndarray, w0: np.ndarray,
                               base_lr: float, base_bs: int,
                               batch_size: int, epochs: int) -> float:
    """
    Run mini-batch SGD for linear regression with the linear LR scaling rule
    (effective_lr = base_lr * batch_size / base_bs) and return the final MSE loss.
    """
    n, d = X.shape
    w = w0.copy()
    lr = base_lr * (batch_size / base_bs)

    for epoch in range(epochs):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            y_batch = y[start:end]
            pred = X_batch @ w
            m = len(y_batch)
            grad = (2 / m) * X_batch.T @ (pred - y_batch)
            w -= lr * grad

    pred = X @ w
    mse = np.mean((pred - y) ** 2)
    return mse
