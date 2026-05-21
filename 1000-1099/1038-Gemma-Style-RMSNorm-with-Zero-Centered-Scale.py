import numpy as np

def gemma_rmsnorm(x, weight, eps: float = 1e-6):
    """Gemma-style RMSNorm with zero-centered scale.

    Args:
        x: numpy array of shape (..., D)
        weight: numpy array of shape (D,), zero-centered scale
        eps: small constant for numerical stability

    Returns:
        numpy array of the same shape and dtype as x
    """
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(mean_sq + eps)
    mul = np.multiply(x_normed, 1 + weight)
    return mul