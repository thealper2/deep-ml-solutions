import numpy as np

def gemma_rmsnorm(x, eps=1e-6, scale=None):
    """
    Gemma-style RMSNorm along the last axis.

    Args:
        x: numpy array of shape (..., D)
        eps: small constant for numerical stability
        scale: optional 1D numpy array of shape (D,). If provided, multiply
               the normalized output by (1 + scale).

    Returns:
        numpy array of the same shape as x.
    """
    mean = np.mean(x ** 2, axis=-1, keepdims=True)
    rms_x = np.sqrt(mean + eps)
    if scale is None:
        y = x / rms_x
    else:
        y = (x / rms_x) * (1 + scale)

    return y.tolist()