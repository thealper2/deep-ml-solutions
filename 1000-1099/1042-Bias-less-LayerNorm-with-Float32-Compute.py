import numpy as np

def biasless_layernorm(x, gamma, eps=1e-5):
    """
    Bias-less LayerNorm over the last dimension with float32 compute.

    Args:
        x: numpy array of shape (..., D)
        gamma: numpy array of shape (D,)
        eps: float, numerical stability constant

    Returns:
        numpy array of same shape and dtype as x
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, ddof=0, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    mul = np.multiply(x_norm, gamma)
    result = np.array(mul, dtype=x.dtype)
    return mul
