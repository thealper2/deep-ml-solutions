import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    x = cache['x']
    mean = cache['mean']
    B, D = dy.shape
    dy_mean = np.mean(dy, axis=1, keepdims=True)
    dx = dy - dy_mean
    return dx
