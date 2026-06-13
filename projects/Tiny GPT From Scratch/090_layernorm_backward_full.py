import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    x = cache['x']
    x_hat = cache['x_hat']
    mean = cache['mean']
    var = cache['var']
    gamma = cache['gamma']
    eps = cache['eps']

    B, D = x.shape

    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)

    dy_hat = dy * gamma

    std = np.sqrt(var + eps)
    dx_norm = dy_hat / std

    dx_centered = dx_norm - np.mean(dx_norm, axis=1, keepdims=True)

    dy_hat_mean = np.mean(dy_hat, axis=1, keepdims=True)
    dy_hat_x_hat_mean = np.mean(dy_hat * x_hat, axis=1, keepdims=True)

    dx = (1 / std) * (dy_hat - dy_hat_mean - x_hat * dy_hat_x_hat_mean)
    return {
        'dx': dx,
        'dgamma': dgamma,
        'dbeta': dbeta,
    }
