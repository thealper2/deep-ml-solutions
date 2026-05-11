import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    m = len(batch_indices)

    predictions = X_batch @ weights + bias
    errors = predictions - y_batch

    grad_w = (2.0 / m) * X_batch.T @ errors
    grad_b = (2.0 / m) * np.sum(errors)

    weights_new = weights - lr * grad_w
    bias_new = bias - lr * grad_b
    return np.append(weights_new, bias_new)