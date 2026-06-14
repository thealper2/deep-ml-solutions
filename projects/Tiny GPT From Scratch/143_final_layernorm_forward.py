def final_layernorm_forward(x, gamma, beta):
    """Apply LayerNorm to a (B, T, d_model) tensor with affine params gamma, beta.

    Returns (y, cache) where cache has keys 'x', 'mean', 'var', 'x_hat', 'gamma'.
    """
    eps = 1e-5
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    cache = {
        'x': x,
        'mean': mean,
        'var': var,
        'x_hat': x_hat,
        'gamma': gamma,
        'eps': eps,
    }
    return y, cache
