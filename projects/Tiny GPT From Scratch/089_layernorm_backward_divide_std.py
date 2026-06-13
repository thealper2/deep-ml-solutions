def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    x_hat = cache['x_hat']
    var = cache['var']
    eps = cache['eps']

    std = np.sqrt(var + eps)
    dx = dy / std
    return dx
