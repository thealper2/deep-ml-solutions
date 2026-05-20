import numpy as np

def batchnorm1d_forward(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    BatchNorm1d forward pass with Bessel's correction (unbiased variance).

    Args:
        X:     input batch of shape (N, D)
        gamma: scale parameter of shape (D,)
        beta:  shift parameter of shape (D,)
        eps:   numerical stability constant

    Returns:
        Output array of shape (N, D)
    """
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=1)
    x_hat = (X - mean) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y.tolist()
