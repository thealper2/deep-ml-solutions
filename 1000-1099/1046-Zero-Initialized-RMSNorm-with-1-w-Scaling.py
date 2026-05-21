import numpy as np

def zero_init_rmsnorm(x, weight, eps=1e-5):
    """
    RMSNorm with (1 + weight) scaling, computed in float32 and cast back.

    Args:
        x: numpy array of shape (..., D)
        weight: numpy array of shape (D,)
        eps: small constant for numerical stability

    Returns:
        A nested Python list with the same shape as x.
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x ** 2, axis=-1, keepdims=True)
    rms = np.sqrt(mean + eps)
    x_normed = x / rms
    y = x_normed * (1 + weight)
    y = np.array(y, dtype=x.dtype)
    return y.tolist()
