import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    n = X.shape[0]
    weights = w0.copy()
    losses = []

    for k in range(n_steps):
        if n_steps == 1:
            lr = 10 ** a
        else:
            lr = 10 ** (a + (b - a) * k / (n_steps - 1))

        pred = X @ weights
        gradient = (2 / n) * X.T @ (pred - y)

        weights = weights - lr * gradient

        pred = X @ weights
        loss = np.mean((pred - y) ** 2)
        losses.append(float(loss)

    return losses