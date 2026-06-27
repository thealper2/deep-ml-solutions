import numpy as np

def compute_mean_variance(x, eps=1e-5):
    """Compute per-feature mean and variance along the last axis of x."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return mean, var
