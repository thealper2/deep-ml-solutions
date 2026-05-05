import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    B, C, H, W = X.shape
    mean = np.mean(X, axis=(2, 3), keepdims=True)
    variance = np.var(X, axis=(2, 3), keepdims=True)
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    gamma_res = gamma.reshape(1, C, 1, 1)
    beta_res = beta.reshape(1, C, 1, 1)
    output = gamma_res * X_norm + beta_res
    return output