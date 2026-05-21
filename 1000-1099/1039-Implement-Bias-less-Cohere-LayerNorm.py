import numpy as np

def cohere_layernorm(x, weight, eps: float = 1e-5):
    """Bias-less LayerNorm computed in float32.

    Args:
        x: numpy array of shape (..., D)
        weight: numpy array of shape (D,)
        eps: small constant for numerical stability
    Returns:
        numpy array of same shape and dtype as x
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True, ddof=0)
    x_hat = (x - mean) / np.sqrt(var + eps)
    mul = np.multiply(x_hat, weight)
    return mul