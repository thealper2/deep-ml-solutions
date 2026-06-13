import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    return np.mean(x, axis=-1, keepdims=True)
